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
"""Camera source specifications: what an operator may point the pipeline at.

A source string is the whole contract between the registry and `src/camera/sources.py`:

* ``"0"``, ``"1"`` - a locally attached device, by index (what OpenCV's ``VideoCapture`` takes).
* ``"rtsp://host:554/path"`` - an IP camera stream. Credentials are added at open time, never
  stored inside the URL.
* ``"replay:/path/or/url"`` - a still image replayed as a camera; what the dev overlay uses.
* ``"synthetic"`` - generated frames for load tests.

Everything here is pure: validation, credential injection and redaction. Keeping it out of the
routes means the rules - the part that decides whether a camera password ends up in a log line -
are covered by tests that need no database, no OpenCV and no network.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

__all__ = [
    "CameraSourceSpec",
    "InvalidSourceError",
    "parse_source",
    "redact_source",
    "slugify",
    "split_credentials",
]

#: Schemes that carry a host and may carry credentials.
NETWORK_SCHEMES = ("rtsp", "rtsps", "http", "https")

#: Schemes that name a local file or a synthetic source - credentials make no sense there.
LOCAL_PREFIXES = ("replay:", "image:", "still:")


class InvalidSourceError(ValueError):
    """The source string cannot be opened, and the operator should be told why."""


@dataclass(frozen=True)
class CameraSourceSpec:
    """A validated source, plus the operations the registry needs on it."""

    raw: str
    kind: str  # "device" | "network" | "replay" | "synthetic"

    @property
    def is_network(self) -> bool:
        return self.kind == "network"

    def with_credentials(
        self, username: Optional[str], password: Optional[str]
    ) -> str:
        """The string to hand to ``VideoCapture``, with credentials inserted if any.

        Credentials go in the userinfo part of the URL, percent-encoded: a password containing
        ``@`` or ``:`` would otherwise change the URL's structure and silently point the pipeline
        at a different host - the kind of bug that presents as "the camera rejects our password".

        Raises:
            InvalidSourceError: if credentials are supplied for a device or a file.
        """
        if not username and password is None:
            return self.raw

        if self.kind != "network":
            raise InvalidSourceError(
                f"a {self.kind} source cannot take a username and password"
            )

        parts = urlsplit(self.raw)
        if parts.username:
            # Already embedded by the operator; leave it alone rather than doubling it up.
            return self.raw

        user = quote(username or "", safe="")
        secret = quote(password or "", safe="")
        userinfo = f"{user}:{secret}" if secret else user
        netloc = f"{userinfo}@{parts.netloc}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def parse_source(raw: str) -> CameraSourceSpec:
    """Validate *raw* and classify it.

    Raises:
        InvalidSourceError: with a message aimed at the operator, not a stack trace.
    """
    text = (raw or "").strip()
    if not text:
        raise InvalidSourceError("source is empty")

    lowered = text.lower()

    for prefix in LOCAL_PREFIXES:
        if lowered.startswith(prefix):
            path = text[len(prefix) :].strip()
            if not path:
                raise InvalidSourceError(f"{prefix} needs a path after it")
            return CameraSourceSpec(raw=text, kind="replay")

    if lowered in ("synthetic", "fake", "test"):
        return CameraSourceSpec(raw="synthetic", kind="synthetic")

    if text.isdigit():
        return CameraSourceSpec(raw=text, kind="device")

    if "://" in text:
        parts = urlsplit(text)
        if parts.scheme.lower() not in NETWORK_SCHEMES:
            raise InvalidSourceError(
                f"unsupported scheme '{parts.scheme}'; expected one of "
                + ", ".join(NETWORK_SCHEMES)
            )
        if not parts.hostname:
            raise InvalidSourceError(f"'{text}' has no host")
        return CameraSourceSpec(raw=text, kind="network")

    raise InvalidSourceError(
        "a source must be a device index (0, 1, ...), an rtsp:// or http:// URL, "
        "replay:/path, or 'synthetic'"
    )


def split_credentials(source: str) -> tuple[str, Optional[str], Optional[str]]:
    """Split any userinfo out of a URL so it can be stored in its own columns.

    Operators copy RTSP URLs out of a camera's web page, and those URLs usually carry
    ``user:password@``. Storing that verbatim would put a plaintext password in the ``source``
    column - a column that is displayed, logged and indexed - so a pasted URL is normalised into
    ``(source_without_userinfo, username, password)`` and the password is encrypted separately.

    A non-URL source is returned unchanged with no credentials.
    """
    if "://" not in source:
        return source, None, None

    parts = urlsplit(source)
    if not parts.username and not parts.password:
        return source, None, None

    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    clean = urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))

    username = unquote(parts.username) if parts.username else None
    password = unquote(parts.password) if parts.password else None
    return clean, username, password


def redact_source(source: str) -> str:
    """Remove any embedded password so a source string is safe in logs and API responses.

    Discovery results and operator input both end up in logs. A camera URL with ``user:pass@`` is
    one careless ``logger.info`` away from a plaintext credential in a log aggregator, so every
    path that reports a source passes through here.
    """
    if "://" not in source:
        return source

    parts = urlsplit(source)
    if parts.password is None and parts.username is None:
        return source

    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    if parts.username:
        netloc = f"{parts.username}:****@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def slugify(label: str, fallback: str = "camera") -> str:
    """A stable, URL-safe camera id from a human label.

    Camera ids appear in prompt rows, WebSocket messages and URLs, so they are lower-case ASCII
    with dashes; a label of "Entrance Gate" becomes ``entrance-gate``. Accents are transliterated
    rather than dropped, so "Вход" keeps a chance of producing something readable instead of an
    empty slug.
    """
    normalised = unicodedata.normalize("NFKD", label or "")
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug or fallback
