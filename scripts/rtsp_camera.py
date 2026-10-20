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
"""Publish a local camera device as an RTSP stream, then print where to find it.

A laptop camera is not visible inside Docker Desktop or WSL, so the cluster cannot open
it as a device. This script publishes it as `rtsp://<address>:8554/cam`, which the service
opens exactly like a shop-floor IP camera:

    python scripts/rtsp_camera.py 0
    # then, in the console: Sources -> register rtsp://192.168.1.20:8554/cam

It is a test aid, not a production relay: MJPEG only, no authentication, no recording.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera.discovery import _backend_candidates, list_local_devices  # noqa: E402
from src.camera.rtsp_server import MjpegRtspServer, candidate_addresses  # noqa: E402

LOGGER = logging.getLogger("rtsp-camera")

DEFAULT_PORT = 8554
DEFAULT_PATH = "cam"
DEFAULT_FPS = 15.0
DEFAULT_QUALITY = 80

_BACKEND_NAMES = {
    "CAP_DSHOW": "DirectShow",
    "CAP_MSMF": "Media Foundation",
    "CAP_V4L2": "V4L2",
    "CAP_AVFOUNDATION": "AVFoundation",
    "CAP_ANY": "any backend",
}


def build_parser() -> argparse.ArgumentParser:
    """Command line: a device number, how to encode it, where to serve it."""
    parser = argparse.ArgumentParser(
        prog="rtsp_camera.py",
        description=(
            "Publish a local camera as an RTSP MJPEG stream and print its address. "
            "Useful for testing the camera registry without IP camera hardware."
        ),
    )
    parser.add_argument(
        "device",
        nargs="?",
        type=int,
        default=0,
        help="OpenCV device index (default: 0; try 1, 2, ... for other cameras)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the camera devices this machine can open, then exit",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="RTSP port (0 picks a free one)")
    parser.add_argument("--path", default=DEFAULT_PATH, help="stream path in the URL")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="frames per second to publish")
    parser.add_argument(
        "--width", type=int, default=0, help="resize width (0 keeps the camera's own size)"
    )
    parser.add_argument(
        "--height", type=int, default=0, help="resize height (0 keeps the camera's own size)"
    )
    parser.add_argument(
        "--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality 1-100 (default: 80)"
    )
    parser.add_argument(
        "--backend",
        default="",
        help="OpenCV capture backend to force, e.g. CAP_DSHOW, CAP_MSMF, CAP_V4L2",
    )
    parser.add_argument(
        "--max-index", type=int, default=4, help="how many device indices --list probes"
    )
    return parser


def format_urls(port: int, path: str, addresses: Optional[Sequence[str]] = None) -> List[str]:
    """Every `rtsp://` URL a client could use, loopback last.

    A single guess would be wrong on a machine with Docker or WSL adapters, so the script
    prints the candidates and says which one is local-only.
    """
    normalized = path.strip().strip("/") or DEFAULT_PATH
    rendered = []
    for address in addresses if addresses is not None else candidate_addresses():
        rendered.append(f"rtsp://{address}:{port}/{normalized}")
    return rendered


def print_urls(port: int, path: str) -> None:
    print("\nRTSP addresses for this stream:\n")
    for url in format_urls(port, path):
        local_only = "  (this machine only)" if "127.0.0.1" in url else ""
        print(f"  {url}{local_only}")
    print(
        "\nRegister one of the non-local addresses in the console: "
        "Sources -> source = that address -> Register and activate.\n"
        "Docker Desktop keeps the host behind NAT: a cluster pod can usually reach the "
        "host at 'host.docker.internal' but not always by name, and 172.x.y.1 adapters are "
        "Docker/WSL networks, not your LAN. The address that works is the one on the "
        "interface your camera and cluster share.\n"
    )


def list_devices(max_index: int) -> int:
    """Print the devices OpenCV can open here. Returns a process exit code."""
    devices = list_local_devices(max_index=max_index)
    if not devices:
        print(
            "No camera device could be opened.\n"
            "On Windows try another index (0, 1, 2), close any app that already holds the "
            "camera, and allow camera access in Settings -> Privacy -> Camera.\n"
            "Inside Docker or WSL no USB camera is visible: run this script on the host."
        )
        return 1
    print(f"{len(devices)} device(s) available:")
    for device in devices:
        print(
            f"  index {device.index}: {device.label} "
            f"({device.width}x{device.height} @ {device.fps:.1f} fps, {device.backend})"
        )
    print("\nPublish one of them with: python scripts/rtsp_camera.py <index>")
    return 0


def open_capture(device: int, backend: str):  # noqa: ANN201 - cv2.VideoCapture
    """Open the device, forcing a backend when one is named."""
    import cv2

    if backend:
        code = getattr(cv2, backend, None)
        if code is None:
            raise SystemExit(f"unknown backend {backend!r}; try one of CAP_DSHOW, CAP_MSMF, CAP_V4L2")
        capture = cv2.VideoCapture(device, code)
        if not capture.isOpened():
            capture.release()
            raise SystemExit(f"device {device} did not open with {backend}")
        return capture

    for name, code in _backend_candidates():
        capture = cv2.VideoCapture(device, code)
        if capture.isOpened():
            LOGGER.info("opened device %s with %s", device, _BACKEND_NAMES.get(name, name))
            return capture
        capture.release()
    raise SystemExit(
        f"device {device} did not open with any backend. Run --list to see what is available."
    )


def build_frame_reader(capture, width: int, height: int, quality: int):  # noqa: ANN001, ANN201
    """Return a callable that encodes the next camera frame as JPEG.

    The RTSP server pulls one frame per tick, so the capture is driven by the publish rate
    instead of running a second loop that would drift and double the latency.
    """
    import cv2

    lock = threading.Lock()
    state = {"size": (width, height) if width and height else None}

    def read_jpeg() -> Optional[bytes]:
        with lock:
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            if state["size"] is None:
                # The camera decides its own size; report it back through the closure so
                # the RTP header carries the truth rather than the requested values.
                height_px, width_px = frame.shape[:2]
                state["size"] = (width_px, height_px)
                LOGGER.info("camera delivers %sx%s", width_px, height_px)
            elif width and height:
                frame = cv2.resize(frame, state["size"], interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None
            return buffer.tobytes()

    return read_jpeg, state


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.list:
        return list_devices(args.max_index)
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    capture = open_capture(args.device, args.backend)
    try:
        read_jpeg, size = build_frame_reader(capture, args.width, args.height, args.quality)
        first = read_jpeg()
        if first is None:
            raise SystemExit(
                f"device {args.device} opened but produced no frame; it may be in use by "
                "another application or disabled in the OS privacy settings"
            )
        width, height = size["size"] or (640, 480)
        server = MjpegRtspServer(
            read_jpeg,
            fps=args.fps,
            width=width,
            height=height,
            port=args.port,
            path=args.path,
        )
        server.start()
        print(f"\nPublishing device {args.device} ({width}x{height} @ {args.fps:g} fps)")
        print_urls(server.snapshot()["port"], args.path)
        print("Press Ctrl+C to stop.\n")

        stop = threading.Event()
        try:
            while not stop.wait(10.0):
                stats = server.snapshot()
                LOGGER.info(
                    "serving %s - %s client(s), %s frame(s) sent",
                    stats["url"].rsplit("/", 1)[-1],
                    stats["clients"],
                    stats["frames_sent"],
                )
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            server.stop()
        return 0
    finally:
        capture.release()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
