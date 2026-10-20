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
"""Point capture at whichever camera the registry says is active.

The registry is written by the API but capture runs in a different process - the
`camera` Deployment in K8s, or the API itself in the development overlay. A shared
database is the only channel between them, so this supervisor polls it and restarts
capture when the active camera changes.

Polling rather than a message is deliberate: the state that matters ("which camera
is active") is one row, a restart is seconds of work, and a poll that misses a change
catches it on the next tick. A queue would add a broker hop and a failure mode -
activation succeeded, notification lost - for a latency saving nobody can see.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Tuple

from src.api.storage import camera_repository
from src.api.storage.database import session_scope
from src.camera.credentials import CredentialError, CredentialVault

logger = logging.getLogger(__name__)

__all__ = ["RegistrySupervisor", "registry_scan"]

# (camera_id, openable source) or None when no camera is active.
Target = Tuple[str, str]
Scan = Callable[[], Optional[Target]]


def registry_scan(vault: CredentialVault, session_factory=None) -> Optional[Target]:  # noqa: ANN001
    """Read the active camera and build a source string capture can open.

    Returns ``None`` when there is no active camera or its stored password cannot be
    decrypted. Both are "keep doing what you are doing" rather than errors: an
    operator who has not picked a camera yet must not stop an already running capture,
    and a rotated key must be reported once, not restart-loop the pipeline.
    """
    with session_scope(session_factory) as session:
        camera = camera_repository.active_camera(session)
        if camera is None:
            return None
        try:
            source = camera_repository.openable_source(camera, vault)
        except CredentialError as exc:
            logger.warning("active camera %s cannot be opened: %s", camera.id, exc)
            return None
        return camera.id, source


class RegistrySupervisor:
    """Poll the registry and restart capture when the active camera changes."""

    def __init__(
        self,
        scan: Scan,
        on_activate: Callable[..., bool],
        poll_seconds: float = 5.0,
        interval: Optional[Callable[[float], bool]] = None,
        current: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._scan = scan
        self._on_activate = on_activate
        self._poll_seconds = max(0.5, float(poll_seconds))
        # Injectable so tests do not sleep; defaults to Event.wait, which also makes
        # stop() immediate instead of waiting out the current interval.
        self._interval = interval
        # Which camera capture is actually on right now. Activation tears this supervisor
        # down and builds a fresh one, so without this the replacement sees "the registry
        # says replay-cam" and restarts a pipeline that is already on replay-cam - forever.
        self._current = current
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._applied: Optional[Target] = None
        self.failures = 0

    def poll_once(self) -> Optional[Target]:
        """One poll. Returns the target when a restart was applied.

        Never raises: a broken database connection or a camera that stopped answering
        must not kill the supervisor thread, because nothing would restart it.
        """
        try:
            target = self._scan()
        except Exception as exc:  # noqa: BLE001 - the thread must survive anything
            logger.warning("camera registry poll failed: %s", exc)
            return None
        if target is None or target == self._applied:
            return None
        if self._current is not None:
            try:
                running = self._current()
            except Exception:  # noqa: BLE001 - status is a hint, not a dependency
                running = None
            if running == target[0]:
                # Already capturing what the registry names: adopt it, do not restart.
                # This is the normal path immediately after an activation.
                self._applied = target
                self.failures = 0
                return None
        camera_id, source = target
        try:
            activated = bool(self._on_activate(source=source, camera_id=camera_id))
        except Exception as exc:  # noqa: BLE001 - see above
            logger.warning("camera registry activation of %s failed: %s", camera_id, exc)
            activated = False
        if not activated:
            # Do not record the target: the next poll retries it. Recording it here
            # would make a single failed start permanent until the camera changes.
            self.failures += 1
            return None
        self.failures = 0
        self._applied = target
        logger.info("capture now follows camera %s from the registry", camera_id)
        return target

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="camera-registry", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopped.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            # Activation restarts capture, and a restart tears the supervisor down. When
            # that happens on the supervisor's own thread, joining would raise; setting
            # the flag is enough - the loop checks it before its next poll.
            thread.join(timeout=timeout)
        self._stopped.clear()

    def _wait(self) -> None:
        if self._interval is None:
            self._stopped.wait(self._poll_seconds)
        else:
            self._interval(self._poll_seconds)

    def _run(self) -> None:
        while not self._stopped.is_set():
            self.poll_once()
            self._wait()

    def snapshot(self) -> dict:
        applied = self._applied
        return {
            "running": self._thread is not None,
            "camera_id": applied[0] if applied else None,
            "failures": self.failures,
            "poll_seconds": self._poll_seconds,
        }
