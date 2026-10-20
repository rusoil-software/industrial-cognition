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
"""Single-frame camera probe.

The probe is the operator's answer to "will this camera work?", so the tests are about answers:
a source that cannot open yields ``ok: false`` with a reason, a synthetic source yields a frame and
its size, and a probe never raises - an unreachable camera is a normal plant condition, not a
server error.

No test here needs a real camera; ``synthetic`` is the frame source the pipeline itself uses in
tests, so the probe path is exercised end to end without hardware.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from src.camera.capture import CaptureConfig  # noqa: E402
from src.camera.probe import probe_source  # noqa: E402


def test_a_source_that_cannot_open_is_an_answer_not_an_exception() -> None:
    result = probe_source("replay:/definitely/not/here.webp", CaptureConfig())

    assert result.ok is False
    assert result.detail, "the operator needs a reason"
    assert result.width == 0 and result.height == 0


def test_a_readable_source_reports_its_frame_size() -> None:
    result = probe_source(
        "replay:assets/images/roboflow_test_detect_human_robot.webp",
        CaptureConfig(width=320, height=240),
    )

    assert result.ok is True
    assert result.width > 0 and result.height > 0
    assert result.latency_ms >= 0.0


def test_the_probe_never_leaks_a_password_into_its_report() -> None:
    result = probe_source(
        "rtsp://operator:secret@127.0.0.1:1/nothing",
        CaptureConfig(),
    )

    report = result.to_dict("rtsp://operator:secret@127.0.0.1:1/nothing")
    assert "secret" not in str(report)
    assert "****" in report["source"]


def test_an_unparseable_source_is_reported_without_raising() -> None:
    result = probe_source("ftp://example/stream", CaptureConfig())

    assert result.ok is False
    assert "ftp" in result.detail or "scheme" in result.detail
