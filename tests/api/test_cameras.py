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
"""Camera route behaviour, without a database.

These tests call the handlers directly with a stub session and stubbed repository functions. That is
deliberate: the value of the routes is in their *decisions* - which failure becomes which status
code, what a pasted password does to the stored source, what happens to the registry when the
pipeline refuses to start - and none of that needs Postgres. The SQL itself is covered where it
belongs, in ``tests/db`` against a real database.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("pydantic")

from fastapi import HTTPException  # noqa: E402  (after importorskip)

from src.api import cameras, runtime  # noqa: E402
from src.api.prompts import service as prompt_service  # noqa: E402
from src.api.storage import camera_repository  # noqa: E402
from src.camera.credentials import generate_key  # noqa: E402

VALID_KEY = generate_key()


class StubSession:
    """Just enough session to let the handlers call ``commit()``."""

    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


def _camera(**overrides) -> SimpleNamespace:
    values = {
        "id": "gate",
        "label": "Gate",
        "source": "rtsp://10.0.0.5:554/stream1",
        "username": "operator",
        "password_encrypted": None,
        "active": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture()
def vault_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMERA_CRED_KEY", VALID_KEY)


@pytest.fixture()
def no_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_CRED_KEY", raising=False)


def _payload(**overrides) -> cameras.CameraIn:
    values = {"label": "Gate", "source": "rtsp://10.0.0.5:554/stream1"}
    values.update(overrides)
    return cameras.CameraIn(**values)


def test_an_unparseable_source_is_rejected_with_422() -> None:
    with pytest.raises(HTTPException) as caught:
        cameras.register_camera(_payload(source="ftp://10.0.0.5/stream"), StubSession())
    assert caught.value.status_code == 422
    assert "rtsp" in str(caught.value.detail)


def test_a_device_source_may_not_carry_a_password() -> None:
    with pytest.raises(HTTPException) as caught:
        cameras.register_camera(_payload(source="0", password="secret"), StubSession())
    assert caught.value.status_code == 422
    assert "device" in str(caught.value.detail)


def test_storing_a_password_without_a_key_is_a_503_that_names_the_variable(
    no_vault: None,
) -> None:
    with pytest.raises(HTTPException) as caught:
        cameras.register_camera(
            _payload(password="secret"), StubSession()
        )
    assert caught.value.status_code == 503
    assert "CAMERA_CRED_KEY" in str(caught.value.detail)


def test_a_pasted_userinfo_url_is_stored_clean_and_encrypted(
    vault_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_upsert(session, **kwargs):  # noqa: ANN001, ANN003
        captured.update(kwargs)
        return _camera(**{"source": kwargs["source"]}), True

    monkeypatch.setattr(camera_repository, "upsert_camera", fake_upsert)
    session = StubSession()

    body = cameras.register_camera(
        _payload(
            source="rtsp://operator:p%40ss@10.0.0.5:554/stream1",
            password=None,
        ),
        session,
    )

    assert captured["source"] == "rtsp://10.0.0.5:554/stream1"
    assert captured["username"] == "operator"
    assert captured["password_encrypted"] and "p@ss" not in captured["password_encrypted"]
    assert body["source"] == "rtsp://10.0.0.5:554/stream1"
    assert "password" not in body
    assert session.commits == 1


def test_activating_an_unknown_camera_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_repository, "get_camera", lambda session, camera_id: None)
    with pytest.raises(HTTPException) as caught:
        cameras.activate("missing", StubSession())
    assert caught.value.status_code == 404


def test_activation_reports_why_the_pipeline_did_not_start(
    vault_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        camera_repository, "get_camera", lambda session, camera_id: _camera()
    )
    monkeypatch.setattr(runtime, "restart_pipeline", lambda **kwargs: False)
    monkeypatch.setattr(
        runtime, "start_error", lambda: "cannot open rtsp://10.0.0.5:554/stream1"
    )

    with pytest.raises(HTTPException) as caught:
        cameras.activate("gate", StubSession())

    assert caught.value.status_code == 503
    # The operator needs the reason, not just "it failed".
    assert "cannot open" in str(caught.value.detail)


def test_activation_marks_the_camera_active_only_after_the_pipeline_starts(
    vault_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    camera = _camera()

    monkeypatch.setattr(camera_repository, "get_camera", lambda session, camera_id: camera)
    monkeypatch.setattr(
        runtime, "restart_pipeline", lambda **kwargs: order.append("pipeline") or True
    )
    monkeypatch.setattr(
        camera_repository,
        "activate_camera",
        lambda session, camera_id: order.append("database") or camera,
    )

    body = cameras.activate("gate", StubSession())

    assert order == ["pipeline", "database"]
    assert body["running"] is True
    assert body["camera_id"] == "gate"


def test_a_stored_password_that_cannot_be_decrypted_is_a_503(
    no_vault: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row written under a key that is now gone: the camera exists, but it cannot be opened.
    camera = _camera(password_encrypted="gAAAAA-not-really-a-token")
    monkeypatch.setattr(camera_repository, "get_camera", lambda session, camera_id: camera)

    with pytest.raises(HTTPException) as caught:
        cameras.probe_registered("gate", StubSession())

    assert caught.value.status_code == 503
    assert "CAMERA_CRED_KEY" in str(caught.value.detail)


def _stub_delete(monkeypatch: pytest.MonkeyPatch, prompt_existed: bool) -> list[str]:
    """Stub the prompt deletion and return the list it records ids into."""
    deleted_prompts: list[str] = []

    def _delete_prompt(session, camera_id):  # noqa: ANN001, ANN202
        deleted_prompts.append(camera_id)
        return prompt_existed

    monkeypatch.setattr(prompt_service, "delete_camera_prompt", _delete_prompt)
    return deleted_prompts


def test_deleting_the_active_camera_stops_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[bool] = []
    removed: list[str] = []
    deleted_prompts = _stub_delete(monkeypatch, prompt_existed=False)
    monkeypatch.setattr(
        camera_repository,
        "get_camera",
        lambda session, camera_id: _camera(active=True),
    )
    monkeypatch.setattr(runtime, "shutdown_pipeline", lambda: stopped.append(True))
    monkeypatch.setattr(
        camera_repository,
        "delete_camera",
        lambda session, camera_id: removed.append(camera_id) or True,
    )

    cameras.remove("gate", StubSession())

    assert stopped == [True]
    assert removed == ["gate"]
    # The console lists tasks too, so the delete must take the task with it - otherwise
    # the row survives the delete and the next attempt answers "no camera".
    assert deleted_prompts == ["gate"]


def test_deleting_an_inactive_camera_does_not_stop_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[bool] = []
    _stub_delete(monkeypatch, prompt_existed=False)
    monkeypatch.setattr(
        camera_repository, "get_camera", lambda session, camera_id: _camera(active=False)
    )
    monkeypatch.setattr(runtime, "shutdown_pipeline", lambda: stopped.append(True))
    monkeypatch.setattr(
        camera_repository, "delete_camera", lambda session, camera_id: True
    )

    cameras.remove("gate", StubSession())

    assert stopped == []


def test_a_task_without_a_camera_row_is_deletable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exactly the report: the registry row was already gone, the console still listed the
    # camera because its detection task remained, and the delete answered 404.
    stopped: list[bool] = []
    deleted_prompts = _stub_delete(monkeypatch, prompt_existed=True)
    removed: list[str] = []
    monkeypatch.setattr(camera_repository, "get_camera", lambda session, camera_id: None)
    monkeypatch.setattr(runtime, "shutdown_pipeline", lambda: stopped.append(True))
    monkeypatch.setattr(
        camera_repository,
        "delete_camera",
        lambda session, camera_id: removed.append(camera_id) or True,
    )

    cameras.remove("orphan", StubSession())

    assert deleted_prompts == ["orphan"]
    assert removed == []  # nothing to delete in the registry
    assert stopped == []  # and it was not the camera being captured


def test_deleting_the_camera_that_is_capturing_stops_capture_even_if_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A pipeline can be running on a camera whose row says inactive (a manual start, or a
    # row that predates the activation flag). Deleting it must not leave capture orphaned.
    stopped: list[bool] = []
    _stub_delete(monkeypatch, prompt_existed=False)
    monkeypatch.setattr(
        camera_repository, "get_camera", lambda session, camera_id: _camera(active=False)
    )
    monkeypatch.setattr(
        runtime,
        "pipeline_status",
        lambda: {"running": True, "pipeline": {"camera_id": "gate"}},
    )
    monkeypatch.setattr(runtime, "shutdown_pipeline", lambda: stopped.append(True))
    monkeypatch.setattr(camera_repository, "delete_camera", lambda session, camera_id: True)

    cameras.remove("gate", StubSession())

    assert stopped == [True]


def test_an_unknown_camera_is_a_404(monkeypatch: pytest.MonkeyPatch) -> None:
    removed: list[str] = []
    _stub_delete(monkeypatch, prompt_existed=False)
    monkeypatch.setattr(camera_repository, "get_camera", lambda session, camera_id: None)
    monkeypatch.setattr(
        camera_repository,
        "delete_camera",
        lambda session, camera_id: removed.append(camera_id) or True,
    )

    with pytest.raises(HTTPException) as caught:
        cameras.remove("ghost", StubSession())

    assert caught.value.status_code == 404
    assert "no camera 'ghost'" in str(caught.value.detail)
    assert removed == []
