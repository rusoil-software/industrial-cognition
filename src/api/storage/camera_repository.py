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
"""Camera registry persistence.

The rules that matter here are about *what is stored*:

* A source string is stored **without credentials**. An operator who pastes
  ``rtsp://user:pass@host/stream`` gets ``source = rtsp://host/stream`` and the password encrypted in
  its own column, so the plaintext never lands in the row, in a backup, or in an error message that
  quotes the source.
* Activating a camera deactivates every other one in the same transaction. The partial unique
  index then makes a second active row impossible, so the database and the single capture pipeline
  cannot disagree.
* Reads for the API go through `public_view`, which never includes the password - only whether one
  is stored. A route that needs the password asks `openable_source` and gets a URL for OpenCV.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.api.storage.models import Camera
from src.camera.credentials import CredentialError, CredentialVault
from src.camera.registry import parse_source, redact_source
from src.camera.sources import REPLAY_PREFIXES

__all__ = [
    "CameraConflict",
    "CameraNotFound",
    "activate_camera",
    "active_camera",
    "delete_camera",
    "get_camera",
    "list_cameras",
    "openable_source",
    "public_view",
    "upsert_camera",
]

logger = logging.getLogger(__name__)


class CameraNotFound(LookupError):
    """No camera has that id."""


class CameraConflict(ValueError):
    """The write would leave the registry inconsistent (e.g. two active cameras)."""


def list_cameras(session: Session) -> list[Camera]:
    """Every registered camera, in label order."""
    return list(session.scalars(select(Camera).order_by(Camera.label, Camera.id)))


def get_camera(session: Session, camera_id: str) -> Optional[Camera]:
    return session.get(Camera, camera_id)


def active_camera(session: Session) -> Optional[Camera]:
    return session.scalars(select(Camera).where(Camera.active.is_(True))).first()


def upsert_camera(
    session: Session,
    *,
    camera_id: str,
    label: str,
    source: str,
    username: Optional[str],
    password_encrypted: Optional[str],
    update_password: bool,
    update_username: bool,
) -> tuple[Camera, bool]:
    """Create or replace a camera. Returns ``(camera, created)``.

    ``update_password`` and ``update_username`` distinguish "leave it alone" from "clear it": the
    API sets them only for fields the operator actually sent, so a console edit that changes the
    label does not wipe the credentials.

    Raises:
        CameraNotFound: when creating with an id that belongs to another camera (guarded by the
            caller's uniqueness check, which needs the label in the message).
    """
    camera = get_camera(session, camera_id)
    created = camera is None
    if camera is None:
        camera = Camera(
            id=camera_id,
            label=label,
            source=source,
            username=username,
            password_encrypted=password_encrypted,
        )
        session.add(camera)
    else:
        camera.label = label
        camera.source = source
        if update_username:
            camera.username = username
        if update_password:
            camera.password_encrypted = password_encrypted

    session.flush()
    return camera, created


def activate_camera(session: Session, camera_id: str) -> Camera:
    """Make *camera_id* the single active camera.

    Raises:
        CameraNotFound: no such camera.
        CameraConflict: another transaction activated a camera first (the partial unique index
            rejected the write).
    """
    camera = get_camera(session, camera_id)
    if camera is None:
        raise CameraNotFound(camera_id)

    # Clear first, then set: two statements in one transaction, in that order, so the partial
    # unique index never sees two active rows. An unconditional UPDATE also deactivates a camera
    # whose row is active but whose pipeline is dead, which is the state after a crash.
    session.execute(update(Camera).where(Camera.active.is_(True)).values(active=False))
    camera.active = True
    try:
        session.flush()
    except IntegrityError as exc:  # pragma: no cover - needs a real race
        raise CameraConflict(
            "another camera was activated at the same time; reload and try again"
        ) from exc
    return camera


def deactivate_all(session: Session) -> None:
    """Stop pointing at any camera (used when activation fails after a partial start)."""
    session.execute(update(Camera).where(Camera.active.is_(True)).values(active=False))
    session.flush()


def delete_camera(session: Session, camera_id: str) -> bool:
    """Remove a camera; returns whether a row existed."""
    result = session.execute(delete(Camera).where(Camera.id == camera_id))
    session.flush()
    return bool(result.rowcount)


def openable_source(camera: Camera, vault: CredentialVault) -> str:
    """The string to hand to ``VideoCapture``, credentials included.

    The password is decrypted only here and only for the duration of building the URL; it is not
    stored on the ORM object, not returned, and not logged.

    Raises:
        CredentialError: the row has a password the vault cannot decrypt (rotated key), or the
            vault is unavailable.
    """
    password: Optional[str] = None
    if camera.password_encrypted:
        password = vault.decrypt(camera.password_encrypted)

    return parse_source(camera.source).with_credentials(camera.username, password)


def public_view(camera: Camera, running: bool = False) -> dict:
    """A camera as the API may report it: everything except the password."""
    spec = None
    try:
        spec = parse_source(camera.source).kind
    except ValueError:  # pragma: no cover - a stored source is validated on write
        spec = "unknown"

    return {
        "camera_id": camera.id,
        "label": camera.label,
        "source": redact_source(camera.source),
        "kind": spec,
        "username": camera.username,
        "has_credentials": bool(camera.password_encrypted),
        "active": bool(camera.active),
        "running": running,
    }
