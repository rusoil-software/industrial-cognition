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
"""Discovery tests.

The device probe talks to real hardware and is exercised against the laptop camera separately; the
ONVIF path is tested against a loopback responder that speaks the wire format, because the thing
worth protecting is the parsing and the socket handling, not a vendor's firmware.
"""

from __future__ import annotations

import socket
import threading

import pytest

from src.camera.discovery import (
    ONVIF_PORT,
    build_probe,
    discover_onvif,
    list_local_devices,
    parse_ws_discovery_response,
)

#: A real ONVIF ProbeMatch, trimmed to the fields the parser reads. Prefixes differ between
#: vendors, which is why the parser matches local names instead of namespace-qualified tags.
PROBE_MATCH = b"""<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
                   xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
                   xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <SOAP-ENV:Body>
    <wsdd:ProbeMatches>
      <wsdd:ProbeMatch>
        <wsdd:EndpointReference>
          <wsdd:Address>urn:uuid:5f8a2b1c-1111-2222-3333-444455556666</wsdd:Address>
        </wsdd:EndpointReference>
        <wsdd:Types>dn:NetworkVideoTransmitter</wsdd:Types>
        <wsdd:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/Entrance%20Gate onvif://www.onvif.org/hardware/DS-2CD2143G0-I</wsdd:Scopes>
        <wsdd:XAddrs>http://192.168.1.64/onvif/device_service rtsp://192.168.1.64:554/Streaming/Channels/101</wsdd:XAddrs>
        <wsdd:MetadataVersion>1</wsdd:MetadataVersion>
      </wsdd:ProbeMatch>
    </wsdd:ProbeMatches>
  </SOAP-ENV:Body>
</SOAP-ENV:Envelope>
"""


def test_parse_extracts_addresses_and_name() -> None:
    cameras = parse_ws_discovery_response(PROBE_MATCH)

    assert len(cameras) == 1
    camera = cameras[0]
    assert camera.addresses == (
        "http://192.168.1.64/onvif/device_service",
        "rtsp://192.168.1.64:554/Streaming/Channels/101",
    )
    assert camera.name == "Entrance Gate"
    assert "NetworkVideoTransmitter" in camera.types


def test_the_stream_address_prefers_rtsp() -> None:
    """The ONVIF device service URL is not a stream; the RTSP one is what the pipeline opens."""
    camera = parse_ws_discovery_response(PROBE_MATCH)[0]
    assert camera.stream_address == "rtsp://192.168.1.64:554/Streaming/Channels/101"


def test_parse_keeps_the_hardware_scope() -> None:
    """Operators recognise the model number, not a UUID."""
    camera = parse_ws_discovery_response(PROBE_MATCH)[0]
    assert any("DS-2CD2143G0-I" in scope for scope in camera.scopes)


def test_a_response_without_addresses_is_ignored() -> None:
    payload = b"""<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope"><Body>
        <ProbeMatch xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">
          <Scopes>onvif://www.onvif.org/name/Nothing</Scopes>
        </ProbeMatch></Body></Envelope>"""
    assert parse_ws_discovery_response(payload) == []


def test_garbage_is_ignored_not_raised() -> None:
    """A malformed packet arrives from the network; it must not take the API down."""
    assert parse_ws_discovery_response(b"not xml at all") == []


def test_a_non_probe_message_is_ignored() -> None:
    """Hello/Bye messages are also WS-Discovery traffic and are not discovery results."""
    bye = b"""<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope"><Body>
        <Bye xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery"/></Body></Envelope>"""
    assert parse_ws_discovery_response(bye) == []


def test_the_probe_asks_for_network_video_transmitters() -> None:
    probe = build_probe("00000000-0000-0000-0000-000000000000").decode("utf-8")

    assert "Probe" in probe
    assert "NetworkVideoTransmitter" in probe
    assert "00000000-0000-0000-0000-000000000000" in probe
    # The action URI is what distinguishes a probe from a hello.
    assert "discovery/Probe" in probe


def test_discovery_port_is_the_onvif_standard() -> None:
    assert ONVIF_PORT == 3702


def test_discover_finds_a_camera_that_answers_on_loopback() -> None:
    """End-to-end through the socket layer, with a responder instead of camera firmware."""
    responder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    responder.bind(("127.0.0.1", 0))
    port = responder.getsockname()[1]
    received: list[bytes] = []
    ready = threading.Event()

    def serve() -> None:
        try:
            responder.settimeout(3.0)
            ready.set()
            payload, sender = responder.recvfrom(65535)
            received.append(payload)
            responder.sendto(PROBE_MATCH, sender)
        except OSError:  # pragma: no cover - the test simply fails below
            pass
        finally:
            responder.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    ready.wait(2.0)

    cameras = discover_onvif(timeout=5.0, targets=[("127.0.0.1", port)])
    thread.join(2.0)

    assert cameras, "the responder answered but discovery returned nothing"
    assert cameras[0].stream_address == "rtsp://192.168.1.64:554/Streaming/Channels/101"
    assert received and b"Probe" in received[0], "the probe was never sent"


def test_discovery_with_no_answer_returns_empty_and_does_not_hang() -> None:
    """A quiet network is the normal case, not an error."""
    quiet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    quiet.bind(("127.0.0.1", 0))
    port = quiet.getsockname()[1]
    try:
        assert discover_onvif(timeout=0.3, targets=[("127.0.0.1", port)]) == []
    finally:
        quiet.close()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cv2"),
    reason="OpenCV is not installed, so there is nothing to probe",
)
def test_local_probe_is_bounded_and_never_raises() -> None:
    """The probe must return within a sensible time on a machine with no cameras.

    Zero indices is the cheap case, but it still proves the import path and the return type on a
    machine whose OpenCV may be built without any capture backend.
    """
    assert list_local_devices(max_index=0) == []
