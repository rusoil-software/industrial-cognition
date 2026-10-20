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
"""A small RTSP server that publishes JPEG frames as an MJPEG stream.

Why this exists: Docker Desktop and WSL cannot see a USB camera, so a laptop camera
cannot be handed to the cluster directly. Publishing it as RTSP puts it on the same
code path as a real IP camera - the service opens `rtsp://host:8554/cam` exactly as it
would open a camera on the shop floor.

Why not ffmpeg or MediaMTX: those need a binary that is not in the image, and the point
is a five-minute test stream, not a production relay.

The payload format is RFC 2435 RTP/JPEG, and it is not simply "a JPEG file in a UDP
packet": receivers such as FFmpeg (which is what OpenCV uses) rebuild the JPEG headers
themselves from the RTP/JPEG header and expect only the entropy-coded scan data in the
payload. Sending a whole JPEG file makes the header parser read image bytes as a
quantisation table and drop every frame. So `parse_jpeg` splits an encoded JPEG into the
pieces the protocol wants, and `packetize_frame` puts them back on the wire.
"""

from __future__ import annotations

import logging
import random
import re
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "JPEG_PAYLOAD_TYPE",
    "JpegFrame",
    "MjpegRtspServer",
    "RtpSession",
    "build_sdp",
    "candidate_addresses",
    "packetize_frame",
    "parse_jpeg",
    "parse_transport",
]

#: RTP payload type for MJPEG (RFC 2435).
JPEG_PAYLOAD_TYPE = 26

#: One RTP timestamp tick per second; JPEG has no frame rate of its own.
RTP_CLOCK_HZ = 90000

#: Ethernet MTU minus IP, UDP and RTP headers - the usual safe RTP payload.
MAX_PAYLOAD = 1400

#: JPEG RTP header is 8 bytes: type-specific, 3-byte offset, type, Q, width/8, height/8.
JPEG_HEADER_SIZE = 8

#: Q value above 127 tells the receiver the payload carries its own quantisation tables.
Q_VALUE_WITH_QTABLES = 255

#: Sampling factors in the SOF segment: 4:2:0 halves chroma both ways, 4:2:2 only across.
SAMPLING_420 = 1
SAMPLING_422 = 0

_SDP = """v=0
o=- {session} {version} IN IP4 {host}
s=industrial-cognition local camera
c=IN IP4 0.0.0.0
t=0 0
m=video 0 RTP/AVP {payload_type}
a=rtpmap:{payload_type} JPEG/{clock}
a=framerate:{fps:.3f}
a=control:trackID=0
"""


@dataclass(frozen=True)
class JpegFrame:
    """The parts of an encoded JPEG that RFC 2435 sends separately from the headers."""

    scan: bytes
    qtables: bytes
    width: int
    height: int
    sampling: int = SAMPLING_420

    def __post_init__(self) -> None:
        if not self.scan:
            raise ValueError("a JPEG frame needs scan data")
        if len(self.qtables) not in (64, 128):
            raise ValueError("expected one or two 8-bit quantisation tables")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")


def build_sdp(
    host: str,
    fps: float,
    payload_type: int = JPEG_PAYLOAD_TYPE,
    session: Optional[int] = None,
    version: int = 1,
) -> str:
    """Describe the single video track.

    `m=video 0` with a port of 0 is what every RTSP server sends: the port only matters
    after SETUP, where it is negotiated per client.
    """
    return _SDP.format(
        session=session if session is not None else random.randrange(1 << 31),
        version=version,
        host=host,
        payload_type=payload_type,
        clock=RTP_CLOCK_HZ,
        fps=fps,
    )


def parse_jpeg(jpeg: bytes) -> JpegFrame:
    """Split an encoded JPEG into scan data, quantisation tables and geometry.

    Only baseline JPEG is accepted: progressive scans cannot be expressed as one
    contiguous entropy-coded segment, and the camera encoder emits baseline anyway.
    """
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG: no start-of-image marker")

    qtables: Dict[int, bytes] = {}
    width = height = 0
    sampling = SAMPLING_420
    position = 2
    scan = b""

    while position + 4 <= len(jpeg):
        if jpeg[position] != 0xFF:
            raise ValueError(f"malformed JPEG: expected a marker at offset {position}")
        marker = jpeg[position + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        if marker == 0xD9:  # end of image before any scan data
            break
        length = int.from_bytes(jpeg[position + 2 : position + 4], "big")
        if length < 2:
            raise ValueError("malformed JPEG: a segment claims a length below 2")
        segment = jpeg[position + 4 : position + 2 + length]

        if marker == 0xDB:  # DQT: one or more 8-bit tables, in zig-zag order
            offset = 0
            while offset < len(segment):
                precision_and_id = segment[offset]
                precision = precision_and_id >> 4
                table_id = precision_and_id & 0x0F
                offset += 1
                if precision:
                    raise ValueError("16-bit quantisation tables are not supported")
                table = segment[offset : offset + 64]
                if len(table) != 64:
                    raise ValueError("truncated quantisation table")
                qtables[table_id] = table
                offset += 64
        elif marker in (0xC0, 0xC1):  # SOF0/SOF1: baseline, one scan
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            components = segment[5] if len(segment) > 5 else 0
            for index in range(components):
                component = segment[6 + index * 3 : 9 + index * 3]
                if len(component) < 3:
                    raise ValueError("truncated frame header")
                sampling = SAMPLING_420 if component[1] == 0x22 else SAMPLING_422
                break
        elif marker == 0xC2:
            raise ValueError("progressive JPEG cannot be published as RTP/JPEG; use baseline")
        elif marker == 0xDA:  # SOS: everything after this header is entropy-coded data
            start = position + 2 + length
            end = jpeg.rfind(b"\xff\xd9")
            if end < start:
                raise ValueError("JPEG has no end-of-image marker after the scan")
            scan = jpeg[start:end]
            break
        position += 2 + length

    if not scan:
        raise ValueError("JPEG has no scan data")
    if not qtables:
        raise ValueError("JPEG has no quantisation tables")
    if width <= 0 or height <= 0:
        raise ValueError("JPEG has no frame size")

    tables = qtables.get(0) or b""
    chroma = qtables.get(1)
    if chroma:
        tables += chroma
    if not tables:
        raise ValueError("JPEG has no luma quantisation table")
    return JpegFrame(
        scan=scan, qtables=tables, width=width, height=height, sampling=sampling
    )


def packetize_frame(frame: JpegFrame, max_payload: int = MAX_PAYLOAD) -> List[bytes]:
    """Split a frame into RTP payloads (RFC 2435).

    Each payload starts with the 8-byte JPEG RTP header. The first fragment also carries
    the quantisation tables, because Q=255 announces that they travel in-band; later
    fragments only carry scan data, and the offset counts scan bytes alone - which is
    exactly what a receiver compares against what it has buffered so far.
    """
    if max_payload <= JPEG_HEADER_SIZE + 4:
        raise ValueError("max_payload leaves no room for a fragment")

    qtable_prefix = (
        b"\x00\x00"  # reserved, then precision 0 (8-bit coefficients)
        + struct.pack(">H", len(frame.qtables))
        + frame.qtables
    )
    blocks = (
        min(255, (frame.width + 7) // 8),
        min(255, (frame.height + 7) // 8),
    )

    def header(offset: int) -> bytes:
        return (
            struct.pack(">B", 0)  # type-specific: 0
            + struct.pack(">I", offset & 0xFFFFFF)[1:]
            + struct.pack(
                ">BBBB",
                frame.sampling,
                Q_VALUE_WITH_QTABLES,
                blocks[0],
                blocks[1],
            )
        )

    payloads: List[bytes] = []
    offset = 0
    while offset < len(frame.scan):
        prefix = header(offset)
        extra = qtable_prefix if offset == 0 else b""
        room = max_payload - len(prefix) - len(extra)
        if room <= 0:
            raise ValueError(
                f"max_payload {max_payload} cannot hold the {len(extra)}-byte table header"
            )
        fragment = frame.scan[offset : offset + room]
        payloads.append(prefix + extra + fragment)
        offset += len(fragment)
    return payloads


def parse_transport(header: str) -> Dict[str, object]:
    """Extract the parts of an RTSP `Transport` header this server understands.

    Returns ``{"protocol": "udp"|"tcp", "client_port": (rtp, rtcp)|None,
    "interleaved": (rtp, rtcp)|None}``. Unknown parameters are ignored rather than
    rejected: clients send more than they need to.
    """
    lowered = header.lower()
    protocol = "tcp" if "rtp/avp/tcp" in lowered else "udp"
    ports: Optional[Tuple[int, int]] = None
    match = re.search(r"client_port=(\d+)(?:-(\d+))?", header)
    if match:
        first = int(match.group(1))
        ports = (first, int(match.group(2)) if match.group(2) else first + 1)
    interleaved: Optional[Tuple[int, int]] = None
    match = re.search(r"interleaved=(\d+)(?:-(\d+))?", header)
    if match:
        first = int(match.group(1))
        interleaved = (first, int(match.group(2)) if match.group(2) else first + 1)
    return {"protocol": protocol, "client_port": ports, "interleaved": interleaved}


def candidate_addresses() -> List[str]:
    """IPv4 addresses a client might reach this host on, best effort.

    Docker and WSL add their own adapters, so this returns everything it can find and
    lets the operator pick - guessing wrong would print an address that does not work.
    """
    found: List[str] = []

    def add(address: str) -> None:
        if address and address not in found and not address.startswith("169.254."):
            found.append(address)

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    for probe in ("8.8.8.8", "1.1.1.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 80))
                add(sock.getsockname()[0])
        except OSError:
            continue
    # Loopback last: it always works locally, but it is useless to another machine.
    add("127.0.0.1")
    return found


@dataclass
class RtpSession:
    """One client that has asked to play the stream."""

    session_id: str
    protocol: str
    client_rtp_port: Optional[int] = None
    connection: Optional[socket.socket] = None
    interleaved: Optional[int] = None
    socket: Optional[socket.socket] = None
    ssrc: int = field(default_factory=lambda: random.randrange(1 << 32))
    playing: bool = False
    address: Optional[Tuple[str, int]] = None

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None


class MjpegRtspServer:
    """Serve one MJPEG stream over RTSP.

    ``frames`` is a zero-argument callable returning the next JPEG bytes, or ``None``
    when no new frame is available. Keeping it a callable (rather than an OpenCV capture)
    is what lets tests drive the protocol without a camera or a display.
    """

    def __init__(
        self,
        frames: Callable[[], Optional[bytes]],
        *,
        fps: float = 15.0,
        width: int = 1280,
        height: int = 720,
        host: str = "0.0.0.0",
        port: int = 8554,
        path: str = "cam",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._frames = frames
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)
        self._host = host
        self._port = int(port)
        self._path = path.strip("/") or "cam"
        self._sessions: Dict[str, RtpSession] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        self._sequence = random.randrange(1 << 16)
        self._timestamp = random.randrange(1 << 31)
        self._clients_served = 0
        self._frames_sent = 0

    # -- lifecycle --------------------------------------------------------- #

    @property
    def url(self) -> str:
        return f"rtsp://{self._host if self._host != '0.0.0.0' else '127.0.0.1'}:{self._port}/{self._path}"

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(8)
        listener.settimeout(0.5)
        self._listener = listener
        self._port = listener.getsockname()[1]
        self._spawn(self._accept_loop, "rtsp-accept")
        self._spawn(self._send_loop, "rtsp-send")
        logger.info("RTSP server listening on %s", self.url)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        for thread in self._threads:
            thread.join(timeout=2.0)

    def snapshot(self) -> dict:
        with self._lock:
            playing = sum(1 for session in self._sessions.values() if session.playing)
        return {
            "url": self.url,
            "port": self._port,
            "clients": playing,
            "frames_sent": self._frames_sent,
            "fps": self._fps,
        }

    def _spawn(self, target, name: str, *args) -> None:  # noqa: ANN001
        thread = threading.Thread(target=target, name=name, args=args, daemon=True)
        thread.start()
        self._threads.append(thread)

    # -- protocol ---------------------------------------------------------- #

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                connection, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            logger.info("RTSP client connected from %s", address[0])
            connection.settimeout(30.0)
            self._spawn(self._serve_client, f"rtsp-client-{address[0]}", connection, address)

    def _serve_client(self, connection: socket.socket, address: Tuple[str, int]) -> None:
        try:
            while not self._stop.is_set():
                request = self._read_request(connection)
                if request is None:
                    return
                method, url, headers = request
                session_header = headers.get("session", "").split(";")[0].strip()
                session = self._sessions.get(session_header) if session_header else None

                if method == "OPTIONS":
                    self._respond(
                        connection,
                        headers,
                        200,
                        "OK",
                        {
                            "Public": (
                                "OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN, GET_PARAMETER"
                            )
                        },
                    )
                elif method == "DESCRIBE":
                    body = build_sdp(self._advertised_host(), self._fps)
                    self._respond(
                        connection,
                        headers,
                        200,
                        "OK",
                        {
                            "Content-Type": "application/sdp",
                            "Content-Base": f"{url}/",
                            "Content-Length": str(len(body.encode())),
                        },
                        body,
                    )
                elif method == "SETUP":
                    session = self._setup(connection, address, headers)
                    if session is None:
                        self._respond(connection, headers, 461, "Unsupported Transport", {})
                    else:
                        self._respond(
                            connection,
                            headers,
                            200,
                            "OK",
                            {"Session": session.session_id, "Transport": self._transport_reply(session)},
                        )
                elif method == "PLAY":
                    if session is None:
                        self._respond(connection, headers, 454, "Session Not Found", {})
                    else:
                        session.playing = True
                        self._respond(
                            connection,
                            headers,
                            200,
                            "OK",
                            {
                                "Session": session.session_id,
                                "Range": "npt=0.000-",
                                "RTP-Info": (
                                    f"url={url}/trackID=0;seq={self._sequence & 0xFFFF};"
                                    f"rtptime={self._timestamp & 0xFFFFFFFF}"
                                ),
                            },
                        )
                elif method == "TEARDOWN":
                    if session is not None:
                        self._drop(session.session_id)
                    self._respond(connection, headers, 200, "OK", {})
                elif method == "GET_PARAMETER":
                    self._respond(connection, headers, 200, "OK", {})
                else:
                    self._respond(connection, headers, 501, "Not Implemented", {})
        except (OSError, ValueError) as exc:
            logger.debug("RTSP client %s dropped: %s", address[0], exc)
        finally:
            with self._lock:
                stale = [key for key, item in self._sessions.items() if item.connection is connection]
                sessions = [self._sessions.pop(key) for key in stale]
            for session in sessions:
                session.close()
            try:
                connection.close()
            except OSError:
                pass

    def _read_request(
        self, connection: socket.socket
    ) -> Optional[Tuple[str, str, Dict[str, str]]]:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = connection.recv(4096)
            if not chunk:
                return None
            buffer += chunk
            if len(buffer) > 64 * 1024:
                raise ValueError("RTSP request too large")
        head, _, rest = buffer.partition(b"\r\n\r\n")
        lines = head.decode("iso-8859-1").split("\r\n")
        try:
            method, url, _protocol = lines[0].split(" ", 2)
        except ValueError as exc:
            raise ValueError(f"malformed RTSP request line: {lines[0]!r}") from exc
        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", 0) or 0)
        while len(rest) < length:
            rest += connection.recv(4096)
        return method.upper(), url, headers

    def _setup(
        self, connection: socket.socket, address: Tuple[str, int], headers: Dict[str, str]
    ) -> Optional[RtpSession]:
        transport = parse_transport(headers.get("transport", ""))
        session_id = f"{random.randrange(1 << 32):08X}"
        session = RtpSession(
            session_id=session_id,
            protocol=str(transport["protocol"]),
            connection=connection,
            address=address,
        )
        if session.protocol == "tcp":
            interleaved = transport["interleaved"]
            if interleaved is None:
                return None
            session.interleaved = int(interleaved[0])
        else:
            ports = transport["client_port"]
            if ports is None:
                return None
            session.client_rtp_port = int(ports[0])
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp.bind((self._host if self._host != "0.0.0.0" else "", 0))
            session.socket = udp
            session.address = (address[0], session.client_rtp_port)
        with self._lock:
            self._sessions[session_id] = session
            self._clients_served += 1
        logger.info(
            "RTSP session %s set up over %s for %s", session_id, session.protocol, address[0]
        )
        return session

    def _drop(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def _transport_reply(self, session: RtpSession) -> str:
        if session.protocol == "tcp":
            return f"RTP/AVP/TCP;unicast;interleaved={session.interleaved}-{session.interleaved + 1}"
        server_port = session.socket.getsockname()[1] if session.socket else 0
        return (
            f"RTP/AVP;unicast;client_port={session.client_rtp_port}-{session.client_rtp_port + 1};"
            f"server_port={server_port}-{server_port + 1};ssrc={session.ssrc:08X}"
        )

    @staticmethod
    def _respond(
        connection: socket.socket,
        request_headers: Dict[str, str],
        status: int,
        reason: str,
        headers: Dict[str, str],
        body: str = "",
    ) -> None:
        lines = [f"RTSP/1.0 {status} {reason}"]
        lines.append(f"CSeq: {request_headers.get('cseq', '1')}")
        lines.append("Server: industrial-cognition-rtsp/1.0")
        lines.extend(f"{key}: {value}" for key, value in headers.items())
        message = "\r\n".join(lines) + "\r\n\r\n" + body
        connection.sendall(message.encode("iso-8859-1"))

    def _advertised_host(self) -> str:
        for address in candidate_addresses():
            if address != "127.0.0.1":
                return address
        return "127.0.0.1"

    # -- streaming --------------------------------------------------------- #

    def _send_loop(self) -> None:
        interval = 1.0 / self._fps
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                jpeg = self._frames()
            except Exception:  # noqa: BLE001 - a bad frame must not stop the stream
                logger.exception("frame source failed")
                jpeg = None
            if jpeg:
                self._broadcast(jpeg)
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, interval - elapsed))

    def _broadcast(self, jpeg: bytes) -> None:
        try:
            frame = parse_jpeg(jpeg)
        except ValueError as exc:
            logger.warning("skipping a frame that is not baseline JPEG: %s", exc)
            return
        self._timestamp = (
            self._timestamp + int(RTP_CLOCK_HZ / self._fps)
        ) & 0xFFFFFFFF
        payloads = packetize_frame(frame)
        with self._lock:
            sessions = [session for session in self._sessions.values() if session.playing]
        if not sessions:
            return
        # Every RTP packet needs its own sequence number: receivers use them for ordering
        # and treat a repeated number as a duplicate, which drops every fragment after the
        # first ("received packet without a start chunk" in FFmpeg's log).
        sequence = []
        for index in range(len(payloads)):
            self._sequence = (self._sequence + 1) & 0xFFFF
            sequence.append((self._sequence, index == len(payloads) - 1))
        for session in sessions:
            for (seq, marker), payload in zip(sequence, payloads):
                packet = struct.pack(
                    ">BBHII",
                    0x80,
                    (0x80 if marker else 0x00) | JPEG_PAYLOAD_TYPE,
                    seq,
                    self._timestamp,
                    session.ssrc,
                ) + payload
                self._send_packet(session, packet)
        self._frames_sent += 1

    def _send_packet(self, session: RtpSession, packet: bytes) -> None:
        try:
            if session.protocol == "tcp":
                connection = session.connection
                channel = session.interleaved or 0
                if connection is None:
                    return
                connection.sendall(bytes([0x24, channel]) + struct.pack(">H", len(packet)) + packet)
            elif session.socket is not None and session.address is not None:
                session.socket.sendto(packet, session.address)
        except OSError as exc:
            # A client that walked away is normal; drop it instead of logging per packet.
            logger.debug("RTSP session %s send failed: %s", session.session_id, exc)
            self._drop(session.session_id)
