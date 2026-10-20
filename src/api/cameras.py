# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Oct-20
# ==============================================================================
"""Camera discovery and registration endpoints.

The console needs three things from the API to let an operator point the service at a real camera:

* **What is out there** - ``GET /cameras/discover``: local devices (opened once, then released) and
  ONVIF cameras that answer WS-Discovery.
* **Register it, with a password if it needs one** - ``POST /cameras``: validates the source, strips
  any pasted ``user:password@`` into its own columns, encrypts the password.
* **Know whether it actually works, then switch to it** - ``POST /cameras/{id}/probe`` reads a
  single frame without touching the running pipeline; ``POST /cameras/{id}/activate`` restarts the
  capture pipeline on that camera and only then marks the row active.

Failure modes are chosen to be *actionable*: a source the pipeline cannot parse is 422 with the
reason, a password the vault cannot decrypt is 503 with the key name, and a camera that does not
answer is a probe result (``ok: false``) rather than a 500 - because "the camera is off" is a
normal condition in a plant, not a server error.

The routes are synchronous on purpose: ``VideoCapture`` and WS-Discovery are blocking, and FastAPI
runs sync handlers in a threadpool, which keeps the event loop (and the MJPEG streams) responsive.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api import runtime
from src.api.prompts.dependencies import get_db
from src.api.storage import camera_repository
from src.camera.credentials import CredentialError, CredentialVault
from src.camera.discovery import discover_onvif, list_local_devices
from src.camera.probe import probe_source
from src.camera.registry import (
    InvalidSourceError,
    parse_source,
    slugify,
    split_credentials,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cameras"], prefix="/cameras")

#: How many device indices to probe. Four covers a laptop (one webcam, sometimes an IR twin) and a
#: small rig; more costs a failed open per index on Windows.
DEFAULT_MAX_INDEX = 4


class CameraIn(BaseModel):
    """A camera to register. Omitted fields are left as they are on an existing camera."""

    id: Optional[str] = Field(default=None, max_length=64)
    label: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=512)
    username: Optional[str] = Field(default=None, max_length=128)
    password: Optional[str] = Field(default=None, max_length=256)
    activate: bool = Field(
        default=False,
        description="Start capturing this camera as soon as it is registered.",
    )


class ProbeRequest(BaseModel):
    """An ad-hoc probe, before anything is registered."""

    source: str = Field(min_length=1, max_length=512)
    username: Optional[str] = Field(default=None, max_length=128)
    password: Optional[str] = Field(default=None, max_length=256)


def _vault() -> CredentialVault:
    return CredentialVault.from_environment()


def _capture_config():  # noqa: ANN202 - CaptureConfig, imported lazily
    return runtime.capture_config()


def _source_with_credentials(payload: ProbeRequest) -> str:
    """Validate a probe source and add credentials, refusing device sources with a password."""
    try:
        spec = parse_source(payload.source)
    except InvalidSourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if spec.kind != "network":
        if payload.username or payload.password:
            raise HTTPException(
                status_code=422,
                detail=f"a {spec.kind} source does not take a username and password",
            )
        return spec.raw

    return spec.with_credentials(payload.username, payload.password)


@router.get("/discover", summary="Local devices and ONVIF cameras on the LAN")
def discover(
    include_network: bool = True,
    timeout: float = 2.0,
    max_index: int = DEFAULT_MAX_INDEX,
    session: Session = Depends(get_db),
) -> dict:
    """What the service could point at right now.

    Local devices are only visible when the API process can open them: in Kubernetes that means the
    edge node's ``/dev/video*`` mounted into the pod (see doc/camera.md), otherwise the list is
    empty and the console says so rather than pretending there are no cameras.
    """
    devices = [device.to_dict() for device in list_local_devices(max_index=max_index)]
    network = []
    if include_network:
        network = [camera.to_dict() for camera in discover_onvif(timeout=timeout)]

    registered = [
        camera_repository.public_view(camera) for camera in camera_repository.list_cameras(session)
    ]
    return {
        "devices": devices,
        "network": network,
        "registered": registered,
        "credentials_configured": _vault().available,
        "hint": (
            "no local devices are visible to the API process; in Kubernetes mount the "
            "host's /dev/video* into the api container"
            if not devices
            else ""
        ),
    }


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Register or update a camera",
)
def register_camera(payload: CameraIn, session: Session = Depends(get_db)) -> dict:
    """Create or update a camera.

    A source pasted with ``user:password@`` is normalised: the URL is stored clean and the password
    encrypted, so it never appears in the registry listing or a log line.
    """
    clean_source, embedded_user, embedded_password = split_credentials(payload.source)
    try:
        spec = parse_source(clean_source)
    except InvalidSourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    username = payload.username if payload.username is not None else embedded_user
    password_given = payload.password is not None or embedded_password is not None
    password = payload.password if payload.password is not None else embedded_password

    if password_given and spec.kind != "network":
        raise HTTPException(
            status_code=422,
            detail=f"a {spec.kind} source does not take a username and password",
        )
    if password and not _vault().available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "camera credentials cannot be stored: CAMERA_CRED_KEY is not configured. "
                "See doc/camera.md; the key must be a Fernet key."
            ),
        )

    camera_id = payload.id or slugify(payload.label, fallback="camera")

    encrypted: Optional[str] = None
    if password:
        try:
            encrypted = _vault().encrypt(password)
        except CredentialError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

    camera, created = camera_repository.upsert_camera(
        session,
        camera_id=camera_id,
        label=payload.label.strip(),
        source=clean_source,
        username=username,
        password_encrypted=encrypted,
        update_password=password_given,
        update_username=payload.username is not None or embedded_user is not None,
    )
    # Commit before activating: a camera that fails to start must still exist, so the operator can
    # fix the address or password instead of retyping everything.
    session.commit()

    body = camera_repository.public_view(camera)
    if payload.activate:
        body = _activate(camera, session)

    logger.info(
        "camera %s %s (source %s, kind %s)",
        camera_id,
        "registered" if created else "updated",
        body["source"],
        body["kind"],
    )
    return body


@router.post("/{camera_id}/probe", summary="Read one frame from a registered camera")
def probe_registered(camera_id: str, session: Session = Depends(get_db)) -> dict:
    """Open the camera, read one frame, report resolution and latency.

    Does not disturb the running pipeline. This is the check to run before activating a camera
    whose address or password is uncertain.
    """
    camera = camera_repository.get_camera(session, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"no camera '{camera_id}'")

    try:
        source = camera_repository.openable_source(camera, _vault())
    except CredentialError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    result = probe_source(source, _capture_config())
    return {**result.to_dict(source), "camera_id": camera_id}


@router.post("/probe", summary="Probe an address before registering it")
def probe_adhoc(payload: ProbeRequest) -> dict:
    """Same check as ``/cameras/{id}/probe`` for a source that is not in the registry yet."""
    source = _source_with_credentials(payload)
    result = probe_source(source, _capture_config())
    return result.to_dict(source)


@router.post("/{camera_id}/activate", summary="Capture this camera")
def activate(camera_id: str, session: Session = Depends(get_db)) -> dict:
    """Restart the capture pipeline on this camera and mark it active.

    The database is updated only after the pipeline reports that it started, so the registry cannot
    claim a camera is live while the service is capturing something else.
    """
    camera = camera_repository.get_camera(session, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"no camera '{camera_id}'")
    body = _activate(camera, session)
    return {**body, "pipeline": runtime.pipeline_status()}


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove(camera_id: str, session: Session = Depends(get_db)) -> None:
    """Forget a camera. Deleting the active one stops capture rather than leaving it orphaned."""
    camera = camera_repository.get_camera(session, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"no camera '{camera_id}'")

    if camera.active:
        runtime.shutdown_pipeline()
    camera_repository.delete_camera(session, camera_id)


def _activate(camera, session: Session) -> dict:  # noqa: ANN001 - Camera
    """Shared activation path for register-with-activate and the activate route."""
    try:
        source = camera_repository.openable_source(camera, _vault())
    except CredentialError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    started = runtime.restart_pipeline(source=source, camera_id=camera.id)
    if not started:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "the capture pipeline did not start on this camera: "
                f"{runtime.start_error() or 'unknown error'}. Probing the camera usually shows why."
            ),
        )

    camera_repository.activate_camera(session, camera.id)
    session.commit()
    return camera_repository.public_view(camera, running=True)
