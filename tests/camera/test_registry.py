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
# ==============================================================================
"""Camera source rules.

These decide what gets stored, what gets handed to OpenCV, and - most importantly - what is
allowed to appear in a log line.
"""

from __future__ import annotations

import pytest

from src.camera.registry import (
    InvalidSourceError,
    parse_source,
    redact_source,
    slugify,
    split_credentials,
)


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("0", "device"),
        ("3", "device"),
        ("synthetic", "synthetic"),
        ("replay:/app/assets/images/x.webp", "replay"),
        ("rtsp://192.168.1.64:554/Streaming/Channels/101", "network"),
        ("http://cam.local/snapshot", "network"),
    ],
)
def test_sources_are_classified(raw: str, kind: str) -> None:
    assert parse_source(raw).kind == kind


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "ftp://host/file", "rtsp://", "nonsense", "replay:"],
)
def test_bad_sources_are_refused_with_a_readable_message(raw: str) -> None:
    with pytest.raises(InvalidSourceError):
        parse_source(raw)


def test_credentials_are_percent_encoded_into_the_url() -> None:
    """A password containing '@' or ':' must not restructure the URL."""
    spec = parse_source("rtsp://10.0.0.5:554/stream")
    url = spec.with_credentials("operator", "p@ss:word")

    assert url == "rtsp://operator:p%40ss%3Aword@10.0.0.5:554/stream"


def test_a_username_without_a_password_still_works() -> None:
    spec = parse_source("rtsp://10.0.0.5/stream")
    assert spec.with_credentials("viewer", None) == "rtsp://viewer@10.0.0.5/stream"


def test_no_credentials_leaves_the_source_untouched() -> None:
    spec = parse_source("rtsp://10.0.0.5/stream")
    assert spec.with_credentials(None, None) == "rtsp://10.0.0.5/stream"


def test_credentials_on_a_device_are_refused() -> None:
    """A device index has no authentication; accepting it would store a password that is never used."""
    with pytest.raises(InvalidSourceError, match="cannot take a username"):
        parse_source("0").with_credentials("user", "secret")


def test_credentials_already_in_the_url_are_not_doubled() -> None:
    spec = parse_source("rtsp://operator:old@10.0.0.5/stream")
    assert spec.with_credentials("operator", "new") == spec.raw


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("rtsp://operator:secret@10.0.0.5:554/stream", "rtsp://operator:****@10.0.0.5:554/stream"),
        ("rtsp://operator@10.0.0.5/stream", "rtsp://operator:****@10.0.0.5/stream"),
        ("rtsp://10.0.0.5/stream", "rtsp://10.0.0.5/stream"),
        ("0", "0"),
        ("replay:/tmp/x.webp", "replay:/tmp/x.webp"),
    ],
)
def test_redaction_removes_the_password_and_keeps_the_rest(source: str, expected: str) -> None:
    assert redact_source(source) == expected


def test_redaction_keeps_the_port_and_path_usable() -> None:
    """The redacted form is for logs and responses; an operator must still recognise the camera."""
    redacted = redact_source("rtsp://user:pass@cam.local:8554/live/main?channel=1")
    assert redacted == "rtsp://user:****@cam.local:8554/live/main?channel=1"


def test_no_secret_survives_redaction() -> None:
    redacted = redact_source("rtsp://user:sup3rS3cret@cam.local/stream")
    assert "sup3rS3cret" not in redacted


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Entrance Gate", "entrance-gate"),
        ("  Camera 01  ", "camera-01"),
        ("Machine/Vision #2", "machine-vision-2"),
        ("", "camera"),
        ("!!!", "camera"),
    ],
)
def test_ids_are_slugs(label: str, expected: str) -> None:
    assert slugify(label) == expected


@pytest.mark.parametrize(
    ("source", "expected_source", "expected_user", "expected_password"),
    [
        # The common case: an operator copies the RTSP URL out of the camera's web page.
        (
            "rtsp://operator:p%40ss%3Aword@10.0.0.5:554/stream1",
            "rtsp://10.0.0.5:554/stream1",
            "operator",
            "p@ss:word",
        ),
        ("rtsp://operator@10.0.0.5/stream1", "rtsp://10.0.0.5/stream1", "operator", None),
        ("rtsp://10.0.0.5/stream1", "rtsp://10.0.0.5/stream1", None, None),
        (
            "https://user:pass@cam.local/snapshot?x=1",
            "https://cam.local/snapshot?x=1",
            "user",
            "pass",
        ),
        ("0", "0", None, None),
        ("replay:/tmp/frame.webp", "replay:/tmp/frame.webp", None, None),
    ],
)
def test_pasted_userinfo_is_split_out_of_the_source(
    source: str,
    expected_source: str,
    expected_user: str | None,
    expected_password: str | None,
) -> None:
    """A password must never survive inside the stored source string."""
    clean, username, password = split_credentials(source)
    assert clean == expected_source
    assert username == expected_user
    assert password == expected_password


def test_splitting_a_clean_source_is_a_no_op() -> None:
    source = "rtsp://10.0.0.5:554/stream1"
    assert split_credentials(source) == (source, None, None)
