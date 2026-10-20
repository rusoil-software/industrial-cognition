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
"""The `scripts/rtsp_camera.py` command line, without needing a camera."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rtsp_camera.py"


def load_script():
    spec = importlib.util.spec_from_file_location("rtsp_camera_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


def test_device_defaults_to_zero(script):
    args = script.build_parser().parse_args([])

    assert args.device == 0
    assert args.port == 8554
    assert args.path == "cam"
    assert args.list is False


def test_a_device_number_and_options_are_accepted(script):
    args = script.build_parser().parse_args(
        ["2", "--port", "9554", "--path", "line1", "--fps", "10", "--quality", "70"]
    )

    assert (args.device, args.port, args.path) == (2, 9554, "line1")
    assert (args.fps, args.quality) == (10.0, 70)


def test_urls_cover_every_candidate_address(script):
    urls = script.format_urls(8554, "cam", ["192.168.1.20", "172.19.0.1", "127.0.0.1"])

    assert urls == [
        "rtsp://192.168.1.20:8554/cam",
        "rtsp://172.19.0.1:8554/cam",
        "rtsp://127.0.0.1:8554/cam",
    ]


def test_a_missing_path_falls_back_to_the_default(script):
    assert script.format_urls(8554, " ", ["127.0.0.1"]) == ["rtsp://127.0.0.1:8554/cam"]


def test_urls_are_built_from_the_machine_when_no_addresses_are_given(script):
    urls = script.format_urls(8554, "cam")

    assert urls
    assert all(url.startswith("rtsp://") for url in urls)
    assert any("127.0.0.1" in url for url in urls)


def test_list_devices_reports_nothing_as_a_failure(script, monkeypatch):
    monkeypatch.setattr(script, "list_local_devices", lambda max_index=4: [])

    assert script.list_devices(2) == 1


def test_list_devices_prints_what_it_found(script, monkeypatch, capsys):
    class Device:
        index = 0
        label = "Camera 0"
        width = 640
        height = 480
        fps = 30.0
        backend = "CAP_DSHOW"

    monkeypatch.setattr(script, "list_local_devices", lambda max_index=4: [Device()])

    assert script.list_devices(2) == 0
    output = capsys.readouterr().out
    assert "index 0: Camera 0 (640x480 @ 30.0 fps, CAP_DSHOW)" in output


def test_an_out_of_range_quality_is_refused_before_opening_a_camera(script):
    with pytest.raises(SystemExit) as error:
        script.main(["0", "--quality", "0"])

    assert "--quality" in str(error.value)
