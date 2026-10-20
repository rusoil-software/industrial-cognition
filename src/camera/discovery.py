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
"""Find cameras: locally attached devices, and ONVIF cameras on the LAN.

Two different problems wearing the same word:

* **Local devices.** A USB/UVC camera is a number, not an address. On Linux it is
  ``/dev/videoN`` with a name in sysfs; on Windows it is a DirectShow index with no name at all.
  The only reliable way to know whether an index is a camera is to open it and read a frame, which
  is why probing is bounded (an index that is not a camera costs a failed ``VideoCapture`` open,
  and on Windows that is not instant) and why the backend list is platform-specific: DirectShow is
  what WDM webcams expect, V4L2 is what Linux cameras expect, and letting OpenCV guess is how you
  get a 10-second hang on the wrong backend.
* **IP cameras.** ONVIF defines WS-Discovery: a SOAP ``Probe`` sent to the multicast group
  ``239.255.255.250:3702``, answered by every ONVIF device with its service addresses. That gives
  the RTSP URL without asking the operator to read it off a label. Many cameras answer only on
  their own subnet, so the multicast send is best-effort and an explicit address is always allowed.

Nothing here imports ``cv2`` at module scope: parsing a discovery response must be testable on a
machine without OpenCV, and the device probe imports it lazily.

RTSP credentials are **not** part of discovery. ONVIF may report a stream URL, but the password for
it is stored per camera by the registry (`src/camera/credentials.py`), never in the discovered
metadata and never in a log line.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = [
    "DiscoveredCamera",
    "LocalDevice",
    "ONVIF_MULTICAST_ADDRESS",
    "ONVIF_PORT",
    "build_probe",
    "discover_onvif",
    "list_local_devices",
    "parse_ws_discovery_response",
]

logger = logging.getLogger(__name__)

#: WS-Discovery multicast group and port (ONVIF Core specification).
ONVIF_MULTICAST_ADDRESS = "239.255.255.250"
ONVIF_PORT = 3702

#: How many indices to try. Cameras are normally 0-2; the default stops a probe from costing a
#: minute on a machine with many virtual capture devices.
DEFAULT_MAX_INDEX = 4


@dataclass(frozen=True)
class LocalDevice:
    """A camera that can be opened by index on the machine running the probe."""

    index: int
    label: str
    width: int
    height: int
    fps: float
    backend: str

    def as_source(self) -> str:
        return str(self.index)

    def to_dict(self) -> dict:
        return {
            "kind": "device",
            "index": self.index,
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "backend": self.backend,
            "source": self.as_source(),
        }


@dataclass(frozen=True)
class DiscoveredCamera:
    """An ONVIF device that answered a WS-Discovery probe."""

    addresses: tuple[str, ...]
    name: str
    types: str = ""
    scopes: tuple[str, ...] = ()

    @property
    def stream_address(self) -> Optional[str]:
        """The address the pipeline should open.

        Preference matters: ONVIF devices publish the device-service URL (HTTP) *and* the stream
        URL (RTSP). Opening the HTTP one yields a SOAP endpoint, not video, so RTSP wins whenever
        it is offered - the first version of this property returned the HTTP address because it
        matched on the first entry instead of on the scheme.
        """
        for scheme in ("rtsp://", "rtsps://", "http://", "https://"):
            for address in self.addresses:
                if address.lower().startswith(scheme):
                    return address
        return self.addresses[0] if self.addresses else None

    def to_dict(self) -> dict:
        return {
            "kind": "onvif",
            "label": self.name or (self.stream_address or "ONVIF camera"),
            "addresses": list(self.addresses),
            "source": self.stream_address,
            "types": self.types,
            "scopes": list(self.scopes),
        }


# --------------------------------------------------------------------------- #
# Local devices
# --------------------------------------------------------------------------- #
def _backend_candidates():  # noqa: ANN202 - returns a list of cv2 backend ids
    """Platform-appropriate capture backends, most likely first.

    Imported lazily so the rest of this module is usable without OpenCV.
    """
    import cv2

    system = platform.system().lower()
    if system == "windows":
        return [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]
    if system == "linux":
        return [("CAP_V4L2", cv2.CAP_V4L2)]
    if system == "darwin":
        return [("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION)]
    return [("CAP_ANY", cv2.CAP_ANY)]


def _linux_device_name(index: int) -> Optional[str]:
    """The human name from sysfs, e.g. ``Integrated Camera`` - Linux only."""
    path = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def list_local_devices(max_index: int = DEFAULT_MAX_INDEX) -> list[LocalDevice]:
    """Probe indices ``0..max_index-1`` and return the ones that produce a frame.

    Opening is not enough: a device can open and never deliver a frame (busy, exclusive access,
    a virtual device with no source), so a frame is the test that matters.

    Never raises: a machine without OpenCV or without cameras returns an empty list, which the API
    reports as "no local devices found" rather than as an error.
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001 - discovery is optional, the pipeline is not
        logger.warning("OpenCV is unavailable; local camera discovery is skipped")
        return []

    devices: list[LocalDevice] = []
    for index in range(max(0, max_index)):
        for backend_name, backend_id in _backend_candidates():
            capture = cv2.VideoCapture(index, backend_id)
            try:
                if not capture.isOpened():
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                height, width = frame.shape[:2]
                devices.append(
                    LocalDevice(
                        index=index,
                        label=_linux_device_name(index) or f"Camera {index}",
                        width=int(width),
                        height=int(height),
                        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                        backend=backend_name,
                    )
                )
                break
            finally:
                capture.release()

    return devices


# --------------------------------------------------------------------------- #
# ONVIF / WS-Discovery
# --------------------------------------------------------------------------- #
_PROBE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{message_id}</w:MessageID>
    <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>
"""


def build_probe(message_id: Optional[str] = None) -> bytes:
    """A WS-Discovery Probe for network video transmitters (ONVIF cameras)."""
    return _PROBE_TEMPLATE.format(
        message_id=message_id or uuid.uuid4()
    ).encode("utf-8")


def _local(tag: str) -> str:
    """Strip the XML namespace: ONVIF responses differ in prefix and sometimes in namespace."""
    return tag.rsplit("}", 1)[-1]


def _name_from_scopes(scopes: Sequence[str]) -> str:
    """Extract a readable name from ONVIF scopes.

    Scopes look like ``onvif://www.onvif.org/name/Entrance``; the last path segment is the name.
    """
    for scope in scopes:
        marker = "/name/"
        if marker in scope:
            candidate = scope.split(marker, 1)[1].strip()
            if candidate:
                return candidate.replace("%20", " ")
    return ""


def parse_ws_discovery_response(payload: bytes) -> list[DiscoveredCamera]:
    """Parse one WS-Discovery response into cameras.

    Returns an empty list for anything that is not a ``ProbeMatch`` - a device may answer with an
    error or a byebye, and neither is a discovery result. Never raises: a malformed packet from the
    network must not take the API down.
    """
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        logger.warning("ignoring an unparsable WS-Discovery response (%d bytes)", len(payload))
        return []

    cameras: list[DiscoveredCamera] = []
    for element in root.iter():
        if _local(element.tag) != "ProbeMatch":
            continue

        addresses: list[str] = []
        scopes: list[str] = []
        types = ""
        for child in element:
            name = _local(child.tag)
            text = (child.text or "").strip()
            if name == "XAddrs":
                addresses = text.split()
            elif name == "Scopes":
                scopes = text.split()
            elif name == "Types":
                types = text

        if not addresses:
            continue

        cameras.append(
            DiscoveredCamera(
                addresses=tuple(addresses),
                name=_name_from_scopes(scopes),
                types=types,
                scopes=tuple(scopes),
            )
        )

    return cameras


def discover_onvif(
    timeout: float = 2.0,
    targets: Optional[Iterable[tuple[str, int]]] = None,
) -> list[DiscoveredCamera]:
    """Broadcast a WS-Discovery probe and collect the answers.

    Args:
        timeout: how long to listen. Two seconds is enough on a quiet plant network; the call is
            blocking, so the API runs it in a worker thread.
        targets: explicit ``(host, port)`` pairs. Defaults to the ONVIF multicast group; the
            parameter exists so tests can answer from a loopback socket and so an operator can
            probe a camera whose subnet drops multicast.

    Returns:
        One entry per distinct stream address. Never raises.
    """
    destinations = list(targets) if targets is not None else [
        (ONVIF_MULTICAST_ADDRESS, ONVIF_PORT)
    ]
    if not destinations:
        return []

    probe = build_probe()
    found: dict[str, DiscoveredCamera] = {}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.2)
        # Only meaningful for the multicast group; harmless for unicast targets.
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        except OSError:  # pragma: no cover - platform dependent
            pass

        for host, port in destinations:
            try:
                sock.sendto(probe, (host, port))
            except OSError as exc:
                logger.info("WS-Discovery probe to %s:%s failed: %s", host, port, exc)

        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            try:
                payload, sender = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:  # pragma: no cover - socket closed
                logger.info("WS-Discovery receive failed: %s", exc)
                break

            for camera in parse_ws_discovery_response(payload):
                key = camera.stream_address or f"{sender[0]}:{len(found)}"
                found.setdefault(key, camera)
    finally:
        sock.close()

    logger.info("WS-Discovery found %d camera(s)", len(found))
    return list(found.values())
