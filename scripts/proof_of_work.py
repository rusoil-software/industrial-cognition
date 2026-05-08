#!/usr/bin/env python3

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
# date   : 2026-05-07 (optimised v5)
# ==============================================================================

"""
Real-time object detection: OWLv2 (ONNX Runtime) + OpenCV tracking pipeline.
FIXES:
  - Joint label tokenization (all labels in one forward pass).
  - Robust postprocessing that works with dynamic‑batch ONNX exports.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

Detection = Tuple[List[int], str, float]  # ([x1,y1,x2,y2], label, conf)
TimedFrame = Tuple[float, np.ndarray]  # (perf_counter timestamp, bgr frame)


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def parse_labels(label_str: str) -> List[str]:
    return [s.strip() for s in label_str.split(",") if s.strip()]


def box_iou_single(a: List[int], boxes: np.ndarray) -> np.ndarray:
    """IoU of one box against an (N,4) array."""
    if not len(boxes):
        return np.array([], dtype=np.float32)
    ax1, ay1, ax2, ay2 = a
    ix1 = np.maximum(ax1, boxes[:, 0])
    iy1 = np.maximum(ay1, boxes[:, 1])
    ix2 = np.minimum(ax2, boxes[:, 2])
    iy2 = np.minimum(ay2, boxes[:, 3])
    inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    area_b = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clip(1)
    return inter / (area_a + area_b - inter + 1e-6)


# ══════════════════════════════════════════════════════════════
#  Main‑thread camera capture with ring buffer
# ══════════════════════════════════════════════════════════════

class CameraCapture:
    """Main‑loop frame capture – NO separate thread."""

    _RING_MAXLEN = 120

    def __init__(self, camera_id: int, width: int, height: int) -> None:
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._ring: Deque[TimedFrame] = deque(maxlen=self._RING_MAXLEN)

    def grab(self) -> Optional[Tuple[np.ndarray, float]]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        ts = time.perf_counter()
        frame_copy = frame.copy()
        self._ring.append((ts, frame_copy))
        return frame_copy, ts

    def frames_since(self, ts: float) -> List[TimedFrame]:
        return [(t, f.copy()) for t, f in self._ring if t > ts]

    def release(self) -> None:
        self._cap.release()


# ══════════════════════════════════════════════════════════════
#  Tracker pool
# ══════════════════════════════════════════════════════════════

class TrackerPool:
    """One lightweight OpenCV tracker per active detection."""

    def __init__(self, tracker_type: str = "CSRT") -> None:
        self._type = tracker_type.upper()
        self._trackers: List[cv2.Tracker] = []
        self._meta: List[Tuple[str, float]] = []

    def reset(self, seed_frame: np.ndarray, detections: List[Detection]) -> None:
        self._trackers, self._meta = [], []
        for (x1, y1, x2, y2), label, conf in detections:
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            tr = self._make()
            tr.init(seed_frame, (x1, y1, w, h))
            self._trackers.append(tr)
            self._meta.append((label, conf))
        logger.debug("TrackerPool seeded: %d trackers", len(self._trackers))

    def replay(self, frames: List[TimedFrame]) -> None:
        if not frames or not self._trackers:
            return
        survival = [True] * len(self._trackers)
        for _, frame in frames:
            for idx, (tr, (label, _)) in enumerate(zip(self._trackers, self._meta)):
                if not survival[idx]:
                    continue
                ok, _ = tr.update(frame)
                if not ok:
                    survival[idx] = False
                    logger.debug("Tracker lost during replay: %s", label)
        alive_tr, alive_meta = [], []
        for idx, (tr, meta) in enumerate(zip(self._trackers, self._meta)):
            if survival[idx]:
                alive_tr.append(tr)
                alive_meta.append(meta)
        self._trackers, self._meta = alive_tr, alive_meta
        logger.debug("Replay complete over %d frames; %d trackers alive",
                     len(frames), len(self._trackers))

    def update(self, frame: np.ndarray) -> Tuple[List[Detection], bool]:
        alive_tr, alive_meta, results = [], [], []
        any_lost = False
        for tr, (label, conf) in zip(self._trackers, self._meta):
            ok, rect = tr.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in rect)
                results.append(([x, y, x + w, y + h], label, conf))
                alive_tr.append(tr)
                alive_meta.append((label, conf))
            else:
                any_lost = True
                logger.debug("Tracker lost: %s", label)
        self._trackers, self._meta = alive_tr, alive_meta
        return results, any_lost

    def _make(self) -> cv2.Tracker:
        if self._type == "CSRT":
            return cv2.TrackerCSRT.create()
        elif self._type == "KCF":
            return cv2.TrackerKCF.create()
        elif self._type == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
        raise ValueError(f"Unknown tracker: {self._type}")


# ══════════════════════════════════════════════════════════════
#  Scene watcher
# ══════════════════════════════════════════════════════════════

class SceneWatcher:
    def __init__(
            self,
            redetect_interval: float = 8.0,
            fg_blob_min_area: int = 2000,
            fg_overlap_thr: float = 0.15,
            mog2_history: int = 60,
    ) -> None:
        self._interval = redetect_interval
        self._blob_min_area = fg_blob_min_area
        self._overlap_thr = fg_overlap_thr
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history, varThreshold=50, detectShadows=False,
        )
        self._last_ts = 0.0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def check(
            self, frame: np.ndarray, tracker_lost: bool, live_boxes: List[List[int]],
    ) -> Tuple[bool, str]:
        if tracker_lost:
            self._last_ts = time.perf_counter()
            return True, "tracker_loss"
        now = time.perf_counter()
        if now - self._last_ts >= self._interval:
            self._last_ts = now
            return True, "interval"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self._bg_sub.apply(gray)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, self._kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes_arr = np.array(live_boxes, dtype=np.float32) if live_boxes else np.empty((0, 4))
        for cnt in contours:
            if cv2.contourArea(cnt) < self._blob_min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            blob = [bx, by, bx + bw, by + bh]
            if len(boxes_arr) == 0 or box_iou_single(blob, boxes_arr).max() < self._overlap_thr:
                self._last_ts = now
                return True, "new_foreground"
        return False, ""


# ══════════════════════════════════════════════════════════════
#  OWLv2 detector – joint label inference with robust postprocessing
# ══════════════════════════════════════════════════════════════

class OWLv2ONNXDetector:
    IMAGE_SIZE = 960
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        nc = os.cpu_count() or 4
        opts.intra_op_num_threads = nc
        opts.inter_op_num_threads = max(1, nc // 2)
        opts.enable_mem_pattern = False

        self._use_cuda = False
        providers, popts = [], []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            popts.append({
                "gpu_mem_limit": str(6 * 1024 ** 3),
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": "1",
            })
            self._use_cuda = True
        providers.append("CPUExecutionProvider");
        popts.append({})

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=providers, provider_options=popts,
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info("Inputs : %s", self.input_names)
        logger.info("Outputs: %s", self.output_names)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        S = self.IMAGE_SIZE
        self._pad_buf = np.zeros((S, S, 3), dtype=np.float32)

        self._iob: Optional[ort.IOBinding] = None
        if self._use_cuda:
            self._iob = self.session.io_binding()
            logger.info("CUDA IO-Binding enabled")

    def _tokens(self, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tok = self.tokenizer(
            labels,
            padding="max_length",
            max_length=16,
            truncation=True,
            return_tensors="np",
        )
        return tok["input_ids"], tok["attention_mask"]

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, int, int]]:
        S = self.IMAGE_SIZE
        h, w = image.shape[:2]
        scale = S / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        pl = (S - nw) // 2
        pt = (S - nh) // 2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self._pad_buf[:] = 0.0
        self._pad_buf[pt:pt + nh, pl:pl + nw] = resized.astype(np.float32) * (1.0 / 255.0)
        self._pad_buf -= self.MEAN
        self._pad_buf /= self.STD
        batch = np.ascontiguousarray(self._pad_buf.transpose(2, 0, 1)[np.newaxis])
        return batch, (scale, pl, pt)

    def postprocess(
            self, outputs: List[np.ndarray], conf_thr: float,
            orig_shape: Tuple[int, int], transform: Tuple[float, int, int],
            labels: List[str],
    ) -> List[Detection]:
        # 1. Find logits and pred_boxes from outputs
        logits = pred_boxes = None
        for i, name in enumerate(self.output_names):
            lo = name.lower()
            if "logits" in lo and logits is None:
                logits = outputs[i][0]
            elif "pred_boxes" in lo and pred_boxes is None:
                pred_boxes = outputs[i][0]
        if logits is None or pred_boxes is None:
            logger.warning("Missing logits or pred_boxes in model outputs")
            return []

        # 2. Remove trivial dimensions and any leading batch dim of size 1
        for arr, name in [(logits, "logits"), (pred_boxes, "pred_boxes")]:
            arr = np.squeeze(arr)  # remove all size-1 dims
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]  # remove explicit batch dim
        # After this:
        #   pred_boxes must be 2D: (num_boxes, 4)
        #   logits    must be 2D: (num_boxes, num_queries) or (num_queries, num_boxes)

        if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
            logger.warning(f"Unexpected pred_boxes shape: {pred_boxes.shape}")
            return []
        num_boxes = pred_boxes.shape[0]

        # 3. Ensure logits is (num_boxes, num_queries)
        if logits.ndim != 2:
            # if still 3D, try to take first (batch) dimension
            if logits.ndim == 3 and logits.shape[0] == 1:
                logits = logits[0]
            else:
                logger.warning(f"Cannot reduce logits shape {logits.shape} to 2D")
                return []

        # Determine which axis is the query axis
        if logits.shape[0] == num_boxes:
            # already (num_boxes, num_queries)
            pass
        elif logits.shape[1] == num_boxes:
            logits = logits.T  # transpose to (num_boxes, num_queries)
        else:
            logger.warning(
                f"logits shape {logits.shape} does not match num_boxes {num_boxes}"
            )
            return []

        # 4. Compute per-box best query
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))  # (num_boxes, num_queries)
        max_scores = scores.max(axis=1)  # (num_boxes,)
        best_idx = scores.argmax(axis=1)  # (num_boxes,)
        mask = max_scores >= conf_thr  # (num_boxes,)
        if not mask.any():
            return []

        # 5. Scale boxes back to original image coordinates
        S = self.IMAGE_SIZE
        scale, pl, pt = transform
        oh, ow = orig_shape
        cb = pred_boxes[mask]  # (keep, 4)   cx,cy,w,h in [0,1]
        x1 = ((cb[:, 0] - cb[:, 2] / 2) * S - pl) / scale
        y1 = ((cb[:, 1] - cb[:, 3] / 2) * S - pt) / scale
        x2 = ((cb[:, 0] + cb[:, 2] / 2) * S - pl) / scale
        y2 = ((cb[:, 1] + cb[:, 3] / 2) * S - pt) / scale
        boxes = np.stack([
            np.clip(x1, 0, ow), np.clip(y1, 0, oh),
            np.clip(x2, 0, ow), np.clip(y2, 0, oh),
        ], axis=1)

        vscores = max_scores[mask]
        vlabels = best_idx[mask]

        detections: List[Detection] = []
        for k in range(len(boxes)):
            bx1, by1, bx2, by2 = boxes[k]
            if bx2 > bx1 and by2 > by1:
                label_idx = vlabels[k]
                if label_idx < len(labels):
                    detections.append(
                        ([int(bx1), int(by1), int(bx2), int(by2)],
                         labels[label_idx], float(vscores[k]))
                    )
        detections.sort(key=lambda d: d[2], reverse=True)
        return detections

    def _run_session(self, pixel_values: np.ndarray, ids: np.ndarray, attn_mask: np.ndarray) -> List[np.ndarray]:
        if self._iob is not None:
            iob = self._iob
            iob.clear_binding_inputs()
            iob.clear_binding_outputs()
            if "pixel_values" in self.input_names: iob.bind_cpu_input("pixel_values", pixel_values)
            if "input_ids" in self.input_names: iob.bind_cpu_input("input_ids", ids)
            if "attention_mask" in self.input_names: iob.bind_cpu_input("attention_mask", attn_mask)
            for name in self.output_names:
                iob.bind_output(name, device_type="cuda")
            self.session.run_with_iobinding(iob)
            return [iob.get_outputs()[i].numpy() for i in range(len(self.output_names))]
        else:
            feed: Dict[str, np.ndarray] = {"pixel_values": pixel_values}
            if "input_ids" in self.input_names: feed["input_ids"] = ids
            if "attention_mask" in self.input_names: feed["attention_mask"] = attn_mask
            return self.session.run(self.output_names, feed)

    def predict(
            self, image: np.ndarray, labels: List[str],
            conf_thr: float, nms_iou: float = 0.5,
    ) -> List[Detection]:
        """Single joint inference with all labels, plus NMS."""
        pixel_values, transform = self.preprocess(image)
        ids, attn_mask = self._tokens(labels)
        outputs = self._run_session(pixel_values, ids, attn_mask)
        dets = self.postprocess(outputs, conf_thr, image.shape[:2], transform, labels)

        # simple NMS
        if dets:
            boxes = np.array([d[0] for d in dets], dtype=np.float32)
            scores = np.array([d[2] for d in dets], dtype=np.float32)
            keep = []
            order = scores.argsort()[::-1]
            while order.size:
                i = order[0]
                keep.append(i)
                iou = box_iou_single(dets[i][0], boxes[order[1:]])
                order = order[1:][iou <= nms_iou]
            dets = [dets[k] for k in keep]
            dets.sort(key=lambda d: d[2], reverse=True)
        return dets


# ══════════════════════════════════════════════════════════════
#  Thread B — async detector
# ══════════════════════════════════════════════════════════════

DetectorResult = Tuple[List[Detection], np.ndarray, float]


class DetectorThread:
    def __init__(
            self, detector: OWLv2ONNXDetector, labels: List[str],
            conf_thr: float, nms_iou: float,
    ) -> None:
        self._det = detector
        self._labels = labels
        self._conf_thr = conf_thr
        self._nms_iou = nms_iou

        self._pending: Optional[Tuple[np.ndarray, float]] = None
        self._pending_lock = threading.Lock()
        self._work_event = threading.Event()

        self._result: Optional[DetectorResult] = None
        self._result_lock = threading.Lock()
        self._result_ready = threading.Event()

        self._running = True
        self._thread = threading.Thread(target=self._run, name="Detector", daemon=True)
        self._thread.start()

    def request(self, frame: np.ndarray, capture_ts: float) -> None:
        with self._pending_lock:
            self._pending = (frame.copy(), capture_ts)
        self._work_event.set()

    def get_latest(self) -> Optional[DetectorResult]:
        if self._result_ready.is_set():
            self._result_ready.clear()
            with self._result_lock:
                return self._result
        return None

    def stop(self) -> None:
        self._running = False
        self._work_event.set()

    def _run(self) -> None:
        logger.info("Detector thread started")
        while self._running:
            self._work_event.wait()
            self._work_event.clear()
            if not self._running:
                break
            with self._pending_lock:
                work = self._pending
                self._pending = None
            if work is None:
                continue
            frame, t_capture = work
            t0 = time.perf_counter()
            try:
                dets = self._det.predict(frame, self._labels, self._conf_thr, self._nms_iou)
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                continue
            logger.debug("Inference %.0f ms → %d dets", (time.perf_counter() - t0) * 1000, len(dets))
            with self._result_lock:
                self._result = (dets, frame, t_capture)
            self._result_ready.set()


# ══════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════

_COL_DETECT = (0, 220, 0)
_COL_TRACK = (0, 220, 220)
_COL_TRIGGER = (0, 200, 255)


def draw_detections(
        image: np.ndarray, detections: List[Detection],
        fps: float = 0.0, det_ms: float = 0.0,
        source: str = "detect", trigger_reason: str = "",
) -> np.ndarray:
    colour = {"detect": _COL_DETECT, "track": _COL_TRACK}.get(source, _COL_TRIGGER)
    for bbox, label, conf in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        prefix = "" if source == "detect" else "~"
        tag = f"{prefix}{label}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (255, 255, 255), -1)
        cv2.putText(image, tag, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    status = f"FPS:{fps:.1f}  Det:{det_ms:.0f}ms"
    if trigger_reason:
        status += f"  [{trigger_reason}]"
    cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return image


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="OWLv2 real-time detector – robust postprocessing")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--labels", default="cat")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--tracker", default="CSRT", choices=["CSRT", "KCF", "MOSSE"])
    parser.add_argument("--redetect-interval", type=float, default=8.0)
    parser.add_argument("--fg-blob-area", type=int, default=2000)
    parser.add_argument("--fg-overlap", type=float, default=0.15)
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    logger.info("Labels: %s | device: %s | tracker: %s", labels, args.device, args.tracker)

    model_dir = Path("src/owl/models") / args.device
    model_path = model_dir / "owl2_model.onnx"
    if not model_path.exists():
        alt = model_dir / "owl2_model.onnx"
        if alt.exists():
            model_path = alt
        else:
            logger.error("Model not found")
            return 1

    detector = OWLv2ONNXDetector(model_path, args.device)

    # Warm‑up
    dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    try:
        detector.predict(dummy, labels, 0.5)
        logger.info("Warm-up complete")
    except Exception as exc:
        logger.warning("Warm-up failed (non-fatal): %s", exc)

    camera = CameraCapture(args.camera, args.width, args.height)
    pool = TrackerPool(args.tracker)
    watcher = SceneWatcher(
        redetect_interval=args.redetect_interval,
        fg_blob_min_area=args.fg_blob_area,
        fg_overlap_thr=args.fg_overlap,
    )
    det_thread = DetectorThread(detector, labels, args.confidence, args.nms_iou)

    current_detections: List[Detection] = []
    live_boxes: List[List[int]] = []
    display_source = "detect"
    trigger_reason = "startup"
    fps = 0.0
    last_det_ms = 0.0
    frame_count = 0
    fps_ts = time.perf_counter()
    detection_in_flight = False
    det_t0 = 0.0

    first_result = camera.grab()
    while first_result is None:
        time.sleep(0.01)
        first_result = camera.grab()
    first_frame, first_ts = first_result
    det_thread.request(first_frame, first_ts)
    det_t0 = time.perf_counter()
    detection_in_flight = True

    logger.info("Display loop started — press Q to quit")

    while True:
        grab_result = camera.grab()
        if grab_result is None:
            time.sleep(0.005)
            continue
        frame, frame_ts = grab_result

        result = det_thread.get_latest()
        if result is not None:
            last_det_ms = (time.perf_counter() - det_t0) * 1000
            new_dets, seed_frame, t_capture = result
            pool.reset(seed_frame, new_dets)
            replay_frames = camera.frames_since(t_capture)
            if replay_frames:
                pool.replay(replay_frames)
                logger.debug("Replayed %d frames", len(replay_frames))
            current_detections = new_dets
            live_boxes = [d[0] for d in new_dets]
            display_source = "detect"
            detection_in_flight = False

        if pool._trackers:
            tracked_dets, any_lost = pool.update(frame)
            if tracked_dets:
                current_detections = tracked_dets
                live_boxes = [d[0] for d in tracked_dets]
                if display_source != "detect":
                    display_source = "track"
            elif any_lost:
                current_detections = []
                live_boxes = []
        else:
            any_lost = False

        if not detection_in_flight:
            should, reason = watcher.check(frame, any_lost, live_boxes)
            if should:
                trigger_reason = reason
                display_source = "trigger"
                det_thread.request(frame, frame_ts)
                det_t0 = time.perf_counter()
                detection_in_flight = True
                logger.debug("Re-detect triggered: %s", reason)

        frame_count += 1
        now = time.perf_counter()
        if now - fps_ts >= 1.0:
            fps = frame_count / (now - fps_ts)
            frame_count = 0
            fps_ts = now

        out = draw_detections(
            frame, current_detections, fps=fps, det_ms=last_det_ms,
            source=display_source,
            trigger_reason=trigger_reason if detection_in_flight else "",
        )
        cv2.imshow("OWLv2 Detection", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    det_thread.stop()
    camera.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())