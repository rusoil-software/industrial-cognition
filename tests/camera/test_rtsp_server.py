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
"""The local RTSP server: the RFC 2435 payload, SDP, and a real handshake."""

from __future__ import annotations

import shutil
import socket
import struct
import subprocess

import pytest

from src.camera.rtsp_server import (
    JPEG_PAYLOAD_TYPE,
    MjpegRtspServer,
    build_sdp,
    candidate_addresses,
    packetize_frame,
    parse_jpeg,
    parse_transport,
)

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 64, 48


@pytest.fixture(scope="module")
def jpeg() -> bytes:
    import numpy as np

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, : WIDTH // 2] = (30, 120, 220)
    image[:, WIDTH // 2 :] = (200, 80, 40)
    # A flat picture compresses to a handful of bytes, which cannot exercise fragmentation;
    # a deterministic pattern keeps the fixture reproducible and the scan data long.
    noise = (np.indices((HEIGHT, WIDTH)).sum(axis=0) * 7 % 256).astype(np.uint8)
    image[:] = (image.astype(np.uint16) + noise[:, :, None] * 3).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return buffer.tobytes()


@pytest.fixture(scope="module")
def frame(jpeg):
    return parse_jpeg(jpeg)


def test_parse_jpeg_extracts_scan_tables_and_geometry(frame):
    assert (frame.width, frame.height) == (WIDTH, HEIGHT)
    assert len(frame.qtables) in (64, 128)
    assert frame.scan
    # The scan data is what follows SOS: it must be far shorter than the whole file
    # because SOI, JFIF, DQT, DHT, SOF and SOS stay behind.
    assert len(frame.scan) < len(frame.qtables) + 4096
    assert frame.sampling in (0, 1)


def test_parse_jpeg_rejects_a_file_that_is_not_jpeg():
    with pytest.raises(ValueError, match="not a JPEG"):
        parse_jpeg(b"PNG or something")


def test_parse_jpeg_rejects_progressive_jpeg(jpeg):
    # SOF0 (baseline) is replaced with SOF2 (progressive) in place; same length.
    broken = jpeg.replace(b"\xff\xc0", b"\xff\xc2", 1)

    with pytest.raises(ValueError, match="progressive"):
        parse_jpeg(broken)


def test_packetize_survives_a_round_trip(frame):
    payloads = packetize_frame(frame, max_payload=400)

    assert len(payloads) > 1
    first = payloads[0]
    qtable_length = struct.unpack(">H", first[8 + 2 : 8 + 4])[0]
    assert qtable_length == len(frame.qtables)
    # The first payload carries the tables, so its scan fragment is smaller.
    first_scan = first[12 + qtable_length :]
    assert first_scan == frame.scan[: len(first_scan)]

    offsets = [int.from_bytes(payload[1:4], "big") for payload in payloads]
    assert offsets[0] == 0
    assert offsets[1] == len(first_scan)
    rebuilt = first_scan + b"".join(payload[8:] for payload in payloads[1:])
    assert rebuilt == frame.scan


def test_packetize_encodes_the_jpeg_header(frame):
    payload = packetize_frame(frame)[0]

    assert payload[0] == 0  # type-specific: 0 (no restart markers)
    assert payload[4] == frame.sampling
    assert payload[5] == 255  # Q: the payload brings its own quantisation tables
    assert payload[6] == (WIDTH + 7) // 8
    assert payload[7] == (HEIGHT + 7) // 8


def test_packetize_rejects_a_payload_too_small_for_a_header(frame):
    with pytest.raises(ValueError):
        packetize_frame(frame, max_payload=8)


def test_sdp_describes_an_mjpeg_track():
    sdp = build_sdp("192.168.1.10", fps=15.0, session=42, version=7)

    assert "m=video 0 RTP/AVP 26" in sdp
    assert "a=rtpmap:26 JPEG/90000" in sdp
    assert "a=framerate:15.000" in sdp
    assert "a=control:trackID=0" in sdp
    assert "o=- 42 7 IN IP4 192.168.1.10" in sdp


@pytest.mark.parametrize(
    ("header", "protocol", "ports", "interleaved"),
    [
        ("RTP/AVP;unicast;client_port=5000-5001", "udp", (5000, 5001), None),
        ("RTP/AVP;client_port=6000", "udp", (6000, 6001), None),
        ("RTP/AVP/TCP;unicast;interleaved=0-1", "tcp", None, (0, 1)),
        ("RTP/AVP/TCP;interleaved=2", "tcp", None, (2, 3)),
    ],
)
def test_parse_transport(header, protocol, ports, interleaved):
    parsed = parse_transport(header)

    assert parsed["protocol"] == protocol
    assert parsed["client_port"] == ports
    assert parsed["interleaved"] == interleaved


def test_candidate_addresses_always_include_loopback():
    addresses = candidate_addresses()

    assert "127.0.0.1" in addresses
    assert len(addresses) == len(set(addresses))


def _request(sock: socket.socket, payload: str) -> str:
    sock.sendall(payload.encode())
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(4096)
        assert chunk, "the server closed the connection"
        buffer += chunk
    head, _, rest = buffer.partition(b"\r\n\r\n")
    headers = head.decode().split("\r\n")
    length = 0
    for line in headers[1:]:
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            length = int(value.strip())
    while len(rest) < length:
        rest += sock.recv(4096)
    return head.decode() + "\r\n\r\n" + rest.decode(errors="replace")


def test_a_client_can_negotiate_play_and_receive_jpeg_frames(jpeg, frame):
    server = MjpegRtspServer(
        lambda: jpeg, fps=25.0, width=WIDTH, height=HEIGHT, host="127.0.0.1", port=0
    )
    server.start()
    rtsp = socket.create_connection(("127.0.0.1", server.snapshot()["port"]), timeout=5)
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("127.0.0.1", 0))
    rtp.settimeout(5.0)
    client_port = rtp.getsockname()[1]
    base = f"rtsp://127.0.0.1:{server.snapshot()['port']}/cam"
    session = None
    try:
        options = _request(rtsp, f"OPTIONS {base} RTSP/1.0\r\nCSeq: 1\r\n\r\n")
        assert options.startswith("RTSP/1.0 200")
        assert "DESCRIBE" in options

        describe = _request(
            rtsp, f"DESCRIBE {base} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n"
        )
        assert describe.startswith("RTSP/1.0 200")
        assert "a=rtpmap:26 JPEG/90000" in describe

        setup = _request(
            rtsp,
            f"SETUP {base}/trackID=0 RTSP/1.0\r\nCSeq: 3\r\n"
            f"Transport: RTP/AVP;unicast;client_port={client_port}-{client_port + 1}\r\n\r\n",
        )
        assert setup.startswith("RTSP/1.0 200")
        assert "server_port=" in setup
        for line in setup.split("\r\n"):
            if line.lower().startswith("session:"):
                session = line.split(":", 1)[1].strip()

        play = _request(
            rtsp,
            f"PLAY {base} RTSP/1.0\r\nCSeq: 4\r\nSession: {session}\r\nRange: npt=0.000-\r\n\r\n",
        )
        assert play.startswith("RTSP/1.0 200")
        assert "RTP-Info" in play

        payloads = []
        sequences = []
        while True:
            packet, _ = rtp.recvfrom(2048)
            marker = packet[1] & 0x80
            assert packet[1] & 0x7F == JPEG_PAYLOAD_TYPE
            sequences.append(struct.unpack(">H", packet[2:4])[0])
            payloads.append(packet[12:])
            if marker:
                break
        # Every fragment carries a distinct sequence number.
        assert len(sequences) == len(set(sequences))
        first = payloads[0]
        qtable_length = struct.unpack(">H", first[10:12])[0]
        assert qtable_length == len(frame.qtables)
        rebuilt = first[12 + qtable_length :] + b"".join(
            payload[8:] for payload in payloads[1:]
        )
        assert rebuilt == frame.scan

        teardown = _request(
            rtsp, f"TEARDOWN {base} RTSP/1.0\r\nCSeq: 5\r\nSession: {session}\r\n\r\n"
        )
        assert teardown.startswith("RTSP/1.0 200")
    finally:
        rtp.close()
        rtsp.close()
        server.stop()


def test_play_without_a_session_is_refused(jpeg):
    server = MjpegRtspServer(
        lambda: jpeg, fps=10.0, width=WIDTH, height=HEIGHT, host="127.0.0.1", port=0
    )
    server.start()
    rtsp = socket.create_connection(("127.0.0.1", server.snapshot()["port"]), timeout=5)
    try:
        base = f"rtsp://127.0.0.1:{server.snapshot()['port']}/cam"
        response = _request(rtsp, f"PLAY {base} RTSP/1.0\r\nCSeq: 9\r\nSession: NOPE\r\n\r\n")
        assert response.startswith("RTSP/1.0 454")
    finally:
        rtsp.close()
        server.stop()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_ffmpeg_decodes_the_stream(jpeg, tmp_path):
    """The client that matters: OpenCV uses FFmpeg, so FFmpeg must accept the stream."""
    server = MjpegRtspServer(
        lambda: jpeg, fps=25.0, width=WIDTH, height=HEIGHT, host="127.0.0.1", port=0
    )
    server.start()
    target = tmp_path / "frame.jpg"
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "udp",
                "-i",
                f"rtsp://127.0.0.1:{server.snapshot()['port']}/cam",
                "-frames:v",
                "1",
                "-y",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
    finally:
        server.stop()

    decoded = cv2.imread(str(target))
    assert decoded is not None
    assert decoded.shape[:2] == (HEIGHT, WIDTH)
