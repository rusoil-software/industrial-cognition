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


def _enable_keepalive(sock: socket.socket) -> None:
    """Ask TCP to notice a dead peer.

    Without it a camera that loses power leaves a half-open connection that only a
    write failure (after minutes of retransmission) reveals. Best effort: Windows
    exposes SO_KEEPALIVE but not the tuning knobs.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
            option = getattr(socket, name, None)
            if option is not None:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
    except OSError:  # pragma: no cover - platform dependent, never fatal
        pass


@dataclass
class _Peer:
    """One client TCP connection, with the lock its writers share.

    RTSP replies and interleaved RTP packets travel over the same socket and are
    written from two threads (the request handler and the per-session sender). Two
    concurrent ``sendall`` calls would interleave and corrupt the framing, so every
    write takes this lock.
    """

    sock: socket.socket
    address: Tuple[str, int]
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False

    def send(self, data: bytes) -> None:
        with self.lock:
            if self.closed:
                raise OSError("the client connection is closed")
            self.sock.sendall(data)

    def shutdown(self) -> None:
        """Wake a thread blocked in ``recv`` so it can observe the stop flag."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


@dataclass
class RtpSession:
    """One client that has asked to play the stream."""

    session_id: str
    protocol: str
    client_rtp_port: Optional[int] = None
    connection: Optional[_Peer] = None
    interleaved: Optional[int] = None
    socket: Optional[socket.socket] = None
    ssrc: int = field(default_factory=lambda: random.randrange(1 << 32))
    playing: bool = False
    address: Optional[Tuple[str, int]] = None
    #: Per-session RTP sequence: a receiver tolerates gaps but not repeats.
    sequence: int = field(default_factory=lambda: random.randrange(1 << 16))
    #: Newest frame waiting for this client's socket, ``(rtp timestamp, payloads)``.
    #: A single slot, not a queue: a client that cannot keep up should skip frames,
    #: never build a backlog of stale ones.
    pending: Optional[Tuple[int, List[bytes]]] = None
    wake: threading.Condition = field(default_factory=threading.Condition)
    frames_sent: int = 0
    frames_dropped: int = 0
    error: str = ""
    closed: bool = False

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
        self._timestamp = random.randrange(1 << 31)
        self._clients_served = 0
        self._frames_sent = 0
        self._frames_dropped = 0
        self._last_frame_at: Optional[float] = None

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
            # Shut the control socket down before closing it: the client thread is
            # blocked in recv() on a connection that is idle by design, and a close
            # alone does not wake it on every platform.
            if session.connection is not None:
                session.connection.shutdown()
            with session.wake:
                session.closed = True
                session.pending = None
                session.wake.notify_all()
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
            sessions = list(self._sessions.values())
        playing = [session for session in sessions if session.playing]
        age = (
            time.monotonic() - self._last_frame_at
            if self._last_frame_at is not None
            else None
        )
        return {
            "url": self.url,
            "port": self._port,
            "clients": len(playing),
            "frames_sent": self._frames_sent,
            "frames_dropped": self._frames_dropped,
            "fps": self._fps,
            "frame_age_seconds": age,
            "sessions": [
                {
                    "id": session.session_id,
                    "protocol": session.protocol,
                    "address": session.address[0] if session.address else None,
                    "playing": session.playing,
                    "frames_sent": session.frames_sent,
                    "frames_dropped": session.frames_dropped,
                    "error": session.error or None,
                }
                for session in sessions
            ],
        }

    def _spawn(self, target, name: str, *args):  # noqa: ANN001
        thread = threading.Thread(target=target, name=name, args=args, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

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
            # No read timeout on purpose. An RTSP control connection is idle between
            # requests by design (the client sends at most a keepalive every minute),
            # and a timed-out recv used to tear down a healthy session after 30 s -
            # the camera kept publishing while the server counted zero clients.
            connection.settimeout(None)
            _enable_keepalive(connection)
            peer = _Peer(sock=connection, address=address)
            self._spawn(self._serve_client, f"rtsp-client-{address[0]}", peer)

    def _serve_client(self, peer: _Peer) -> None:
        try:
            while not self._stop.is_set():
                request = self._read_request(peer.sock)
                if request is None:
                    logger.info("RTSP client %s closed the connection", peer.address[0])
                    return
                method, url, headers = request
                session_header = headers.get("session", "").split(";")[0].strip()
                session = self._sessions.get(session_header) if session_header else None

                if method == "OPTIONS":
                    self._respond(
                        peer,
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
                        peer,
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
                    session = self._setup(peer, headers)
                    if session is None:
                        self._respond(peer, headers, 461, "Unsupported Transport", {})
                    else:
                        self._respond(
                            peer,
                            headers,
                            200,
                            "OK",
                            {
                                "Session": session.session_id,
                                "Transport": self._transport_reply(session),
                            },
                        )
                elif method == "PLAY":
                    if session is None:
                        self._respond(peer, headers, 454, "Session Not Found", {})
                    else:
                        session.playing = True
                        self._respond(
                            peer,
                            headers,
                            200,
                            "OK",
                            {
                                "Session": session.session_id,
                                "Range": "npt=0.000-",
                                "RTP-Info": (
                                    f"url={url}/trackID=0;seq={session.sequence & 0xFFFF};"
                                    f"rtptime={self._timestamp & 0xFFFFFFFF}"
                                ),
                            },
                        )
                elif method == "TEARDOWN":
                    if session is not None:
                        logger.info("RTSP session %s torn down by the client", session.session_id)
                        self._drop(session.session_id)
                    self._respond(peer, headers, 200, "OK", {})
                elif method == "GET_PARAMETER":
                    # This is the keepalive most clients use; answering it is the whole
                    # point. The session must stay alive while it waits.
                    self._respond(peer, headers, 200, "OK", {})
                else:
                    self._respond(peer, headers, 501, "Not Implemented", {})
        except (OSError, ValueError) as exc:
            logger.info("RTSP client %s dropped: %s", peer.address[0], exc)
        finally:
            with self._lock:
                stale = [
                    key for key, item in self._sessions.items() if item.connection is peer
                ]
                sessions = [self._sessions.pop(key) for key in stale]
            for session in sessions:
                with session.wake:
                    session.closed = True
                    session.pending = None
                    session.wake.notify_all()
                session.close()
            peer.close()

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

    def _setup(self, peer: _Peer, headers: Dict[str, str]) -> Optional[RtpSession]:
        address = peer.address
        transport = parse_transport(headers.get("transport", ""))
        session_id = f"{random.randrange(1 << 32):08X}"
        session = RtpSession(
            session_id=session_id,
            protocol=str(transport["protocol"]),
            connection=peer,
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
        # One sender per client: a client whose socket is not draining must not
        # delay the shared send loop, the other clients, or its own keepalives.
        session.writer = self._spawn(
            self._write_frames, f"rtsp-send-{session_id}", session
        )
        logger.info(
            "RTSP session %s set up over %s for %s", session_id, session.protocol, address[0]
        )
        return session

    def _drop(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            with session.wake:
                session.closed = True
                session.pending = None
                session.wake.notify_all()
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
        peer: _Peer,
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
        peer.send(message.encode("iso-8859-1"))

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
        try:
            payloads = packetize_frame(frame)
        except ValueError as exc:
            logger.warning("cannot packetise the frame: %s", exc)
            return
        with self._lock:
            sessions = [
                session
                for session in self._sessions.values()
                if session.playing and not session.closed
            ]
        self._last_frame_at = time.monotonic()
        self._frames_sent += 1
        for session in sessions:
            with session.wake:
                if session.pending is not None:
                    # The client has not drained the previous frame. Skipping it keeps
                    # the viewer on the newest frame instead of feeding it a backlog;
                    # a queue here would grow without bound on a slow client.
                    session.frames_dropped += 1
                    self._frames_dropped += 1
                session.pending = (self._timestamp, payloads)
                session.wake.notify()

    def _write_frames(self, session: RtpSession) -> None:
        """Send this client's frames, one at a time, on its own thread."""
        while not self._stop.is_set():
            with session.wake:
                while (
                    session.pending is None
                    and not session.closed
                    and not self._stop.is_set()
                ):
                    session.wake.wait(0.5)
                if session.closed or self._stop.is_set():
                    return
                timestamp, payloads = session.pending
                session.pending = None
            try:
                self._send_frame(session, timestamp, payloads)
            except OSError as exc:
                session.error = str(exc) or exc.__class__.__name__
                logger.info(
                    "RTSP session %s dropped: %s", session.session_id, session.error
                )
                self._drop(session.session_id)
                return
            session.frames_sent += 1
            if session.frames_sent == 1:
                logger.info("RTSP session %s is receiving frames", session.session_id)

    def _send_frame(
        self, session: RtpSession, timestamp: int, payloads: List[bytes]
    ) -> None:
        last = len(payloads) - 1
        if session.protocol == "tcp":
            peer = session.connection
            if peer is None:
                raise OSError("the client connection is gone")
            channel = session.interleaved or 0
            # One write per frame (chunked to keep a single send short): fewer syscalls,
            # and the frame cannot be interleaved with an RTSP reply.
            buffer = bytearray()
            for index, payload in enumerate(payloads):
                packet = self._rtp_packet(session, timestamp, payload, marker=index == last)
                buffer += bytes([0x24, channel]) + struct.pack(">H", len(packet)) + packet
                if len(buffer) >= 256 * 1024:
                    peer.send(bytes(buffer))
                    buffer.clear()
            if buffer:
                peer.send(bytes(buffer))
            return
        if session.socket is None or session.address is None:
            raise OSError("the client RTP socket is gone")
        for index, payload in enumerate(payloads):
            session.socket.sendto(
                self._rtp_packet(session, timestamp, payload, marker=index == last),
                session.address,
            )

    @staticmethod
    def _rtp_packet(
        session: RtpSession, timestamp: int, payload: bytes, *, marker: bool
    ) -> bytes:
        # Every RTP packet needs its own sequence number: receivers use them for
        # ordering and treat a repeated number as a duplicate, which drops every
        # fragment after the first ("received packet without a start chunk" in
        # FFmpeg's log).
        session.sequence = (session.sequence + 1) & 0xFFFF
        return (
            struct.pack(
                ">BBHII",
                0x80,
                (0x80 if marker else 0x00) | JPEG_PAYLOAD_TYPE,
                session.sequence,
                timestamp,
                session.ssrc,
            )
            + payload
        )
