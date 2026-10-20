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
"""The registry supervisor restarts capture exactly when it must, and never dies."""

from __future__ import annotations

import threading

from src.camera.supervisor import RegistrySupervisor


class Recorder:
    """Records activations and answers with a scripted sequence of results."""

    def __init__(self, *results: bool) -> None:
        self.results = list(results) or [True]
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


def test_the_active_camera_is_activated_once():
    recorder = Recorder(True)
    supervisor = RegistrySupervisor(
        scan=lambda: ("cam1", "rtsp://10.0.0.5:554/stream1"),
        on_activate=recorder,
    )

    assert supervisor.poll_once() == ("cam1", "rtsp://10.0.0.5:554/stream1")
    # Second poll sees the same target: no restart, capture keeps running.
    assert supervisor.poll_once() is None
    assert recorder.calls == [{"source": "rtsp://10.0.0.5:554/stream1", "camera_id": "cam1"}]


def test_a_new_active_camera_restarts_capture():
    targets = iter([("cam1", "rtsp://a/1"), ("cam1", "rtsp://a/1"), ("cam2", "rtsp://b/2")])
    recorder = Recorder(True)
    supervisor = RegistrySupervisor(scan=lambda: next(targets), on_activate=recorder)

    supervisor.poll_once()
    supervisor.poll_once()
    assert supervisor.poll_once() == ("cam2", "rtsp://b/2")
    assert [call["camera_id"] for call in recorder.calls] == ["cam1", "cam2"]


def test_a_failed_activation_is_retried_not_remembered():
    recorder = Recorder(False, True)
    supervisor = RegistrySupervisor(
        scan=lambda: ("cam1", "rtsp://10.0.0.5:554/stream1"), on_activate=recorder
    )

    assert supervisor.poll_once() is None
    assert supervisor.failures == 1
    # The same target is attempted again: a camera that was busy comes back.
    assert supervisor.poll_once() == ("cam1", "rtsp://10.0.0.5:554/stream1")
    assert supervisor.failures == 0


def test_no_active_camera_leaves_capture_alone():
    recorder = Recorder(True)
    supervisor = RegistrySupervisor(scan=lambda: None, on_activate=recorder)

    assert supervisor.poll_once() is None
    assert recorder.calls == []


def test_a_raising_scan_does_not_kill_the_supervisor():
    def scan():
        raise RuntimeError("database is down")

    recorder = Recorder(True)
    supervisor = RegistrySupervisor(scan=scan, on_activate=recorder)

    assert supervisor.poll_once() is None
    assert recorder.calls == []
    assert supervisor.snapshot()["failures"] == 0


def test_a_raising_activation_is_reported_as_a_failure():
    def activate(**kwargs):
        raise RuntimeError("device busy")

    supervisor = RegistrySupervisor(scan=lambda: ("cam1", "0"), on_activate=activate)

    assert supervisor.poll_once() is None
    assert supervisor.failures == 1


def test_stop_ends_the_thread():
    activated = threading.Event()

    def activate(**kwargs):
        activated.set()
        return True

    supervisor = RegistrySupervisor(
        scan=lambda: ("cam1", "0"), on_activate=activate, poll_seconds=0.5
    )
    supervisor.start()
    supervisor.start()  # idempotent: one thread, not two
    try:
        assert activated.wait(5.0), "the supervisor never polled"
        assert supervisor.snapshot()["running"] is True
        assert supervisor.snapshot()["camera_id"] == "cam1"
    finally:
        supervisor.stop(timeout=5.0)
    assert supervisor.snapshot()["running"] is False


def test_stopping_from_the_supervisor_thread_does_not_join_itself():
    # Activation restarts capture, and a restart stops the supervisor. That call happens
    # on the supervisor thread, where join() would raise.
    holder: dict[str, RegistrySupervisor] = {}

    def activate(**kwargs):
        holder["supervisor"].stop(timeout=5.0)
        return True

    supervisor = RegistrySupervisor(
        scan=lambda: ("cam1", "0"), on_activate=activate, poll_seconds=0.5
    )
    holder["supervisor"] = supervisor
    supervisor.start()
    try:
        for _ in range(100):
            if supervisor.snapshot()["camera_id"] is not None:
                break
            threading.Event().wait(0.05)
        assert supervisor.snapshot()["camera_id"] == "cam1"
    finally:
        supervisor.stop(timeout=5.0)


def test_a_camera_already_capturing_is_adopted_not_restarted():
    # Activation recreates the supervisor. Without adoption the replacement restarts a
    # pipeline that is already on the registry's camera, and does it again forever.
    recorder = Recorder(True)
    supervisor = RegistrySupervisor(
        scan=lambda: ("cam1", "rtsp://10.0.0.5:554/stream1"),
        on_activate=recorder,
        current=lambda: "cam1",
    )

    assert supervisor.poll_once() is None
    assert supervisor.poll_once() is None
    assert recorder.calls == []
    assert supervisor.snapshot()["camera_id"] == "cam1"


def test_a_different_active_camera_still_restarts_capture():
    recorder = Recorder(True)
    supervisor = RegistrySupervisor(
        scan=lambda: ("cam2", "rtsp://10.0.0.6:554/stream1"),
        on_activate=recorder,
        current=lambda: "cam1",
    )

    assert supervisor.poll_once() == ("cam2", "rtsp://10.0.0.6:554/stream1")
    assert [call["camera_id"] for call in recorder.calls] == ["cam2"]


def test_a_raising_current_lookup_still_activates():
    def current():
        raise RuntimeError("status is not ready")

    recorder = Recorder(True)
    supervisor = RegistrySupervisor(
        scan=lambda: ("cam1", "0"), on_activate=recorder, current=current
    )

    assert supervisor.poll_once() == ("cam1", "0")
