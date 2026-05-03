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
# date   : 2026-05-03 (fixed)
# ==============================================================================

"""
Proof-of-work script for real-time object detection using OWLv2 (ONNX Runtime) and OpenCV.
Fixed preprocessing (letterbox) and coordinate mapping.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_labels(label_str: str) -> List[str]:
    return [l.strip() for l in label_str.split(",") if l.strip()]


class OWLv2ONNXDetector:
    def __init__(self, model_path: Path, device: str = "cpu"):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        providers = []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"Model inputs: {self.input_names}")
        logger.info(f"Model outputs: {self.output_names}")

        # Load tokenizer – same as used during ONNX export (clip-vit-base-patch32)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.image_size = 960
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, int, int]]:
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_left = (self.image_size - new_w) // 2
        pad_top = (self.image_size - new_h) // 2
        padded = np.full((self.image_size, self.image_size, 3), 0.0, dtype=np.float32)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return padded, (scale, pad_left, pad_top)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, int, int]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        padded, transform = self._letterbox(rgb)
        normalized = (padded - self.mean) / self.std
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0).astype(np.float32)
        return batch, transform

    def tokenize_labels(self, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return input_ids and attention_mask of shape (1, seq_len)."""
        # OWLv2 expects a single batch of text, with each label separated by a special token.
        # We'll join all labels with a delimiter and tokenize.
        # For simplicity, tokenize each label separately and stack? Actually
        # the exported model usually expects one text sequence per image.
        # Let's replicate the HF pipeline: tokenize the list of labels directly.
        tokenized = self.tokenizer(
            labels,
            padding="max_length",
            max_length=16,  # as used in the export script
            truncation=True,
            return_tensors="np"
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def postprocess(self, outputs: List[np.ndarray], confidence_threshold: float,
                    original_shape: Tuple[int, int], transform: Tuple[float, int, int],
                    labels: List[str]) -> List[Tuple[List[int], str, float]]:
        # Identify logits and pred_boxes
        logits = None
        pred_boxes = None
        for i, name in enumerate(self.output_names):
            if "logits" in name.lower() and i < len(outputs):
                logits = outputs[i]
            elif "pred_boxes" in name.lower() and i < len(outputs):
                pred_boxes = outputs[i]

        if logits is None or pred_boxes is None:
            logger.warning("Could not find logits or pred_boxes in outputs")
            return []

        # Squeeze extra dims
        if logits.ndim == 3 and logits.shape[2] == 1:
            logits = np.squeeze(logits, axis=2)
        if pred_boxes.ndim == 3 and pred_boxes.shape[0] == 1:
            pred_boxes = np.squeeze(pred_boxes, axis=0)

        # logits shape: (num_queries, num_boxes)  OR (num_boxes, num_queries)
        # Typically OWLv2 ONNX exports logits as (num_queries, num_boxes)
        # Let's detect based on shape
        num_boxes = pred_boxes.shape[0]
        if logits.shape[1] == num_boxes:
            logits = logits.T  # make it (num_boxes, num_queries)
        # Now logits should be (num_boxes, num_queries)
        scores = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))  # sigmoid per box per query

        # For each box, pick the highest scoring label
        max_scores = np.max(scores, axis=1)
        best_label_idx = np.argmax(scores, axis=1)

        valid = np.where(max_scores >= confidence_threshold)[0]
        scale, pad_left, pad_top = transform
        orig_h, orig_w = original_shape
        detections = []

        for idx in valid:
            conf = float(max_scores[idx])
            label = labels[best_label_idx[idx]]

            # Box in normalized [cx, cy, w, h] relative to 960x960
            cx, cy, w, h = pred_boxes[idx]
            x1 = (cx - w / 2) * self.image_size
            y1 = (cy - h / 2) * self.image_size
            x2 = (cx + w / 2) * self.image_size
            y2 = (cy + h / 2) * self.image_size

            # Remove padding
            x1 -= pad_left
            y1 -= pad_top
            x2 -= pad_left
            y2 -= pad_top

            # Scale back to original dimensions
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(orig_w, int(x2))
            y2 = min(orig_h, int(y2))

            if x2 > x1 and y2 > y1:
                detections.append(([x1, y1, x2, y2], label, conf))

        detections.sort(key=lambda x: x[2], reverse=True)
        logger.debug(f"Kept {len(detections)} detections")
        return detections

    def predict(self, image: np.ndarray, labels: List[str], confidence_threshold: float):
        pixel_values, transform = self.preprocess(image)
        input_ids, attention_mask = self.tokenize_labels(labels)

        input_feed = {"pixel_values": pixel_values}
        if "input_ids" in self.input_names:
            input_feed["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            input_feed["attention_mask"] = attention_mask

        outputs = self.session.run(self.output_names, input_feed)
        return self.postprocess(outputs, confidence_threshold, image.shape[:2], transform, labels)


def draw_detections(image: np.ndarray, detections: List, fps: float = None):
    for bbox, label, conf in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (255, 255, 255), -1)
        cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    if fps is not None:
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--labels", type=str, default="cat", help="Comma-separated labels")
    parser.add_argument("--confidence", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    logger.info(f"Detecting: {labels} on {args.device}")

    model_dir = Path("src/owl/models") / args.device
    model_path = model_dir / "owl_model.onnx" / "model.onnx"
    if not model_path.exists():
        alt = model_dir / "model.onnx"
        if alt.exists():
            model_path = alt
        else:
            logger.error(f"Model not found at {model_path} or {alt}")
            return 1

    detector = OWLv2ONNXDetector(model_path, args.device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.predict(frame, labels, args.confidence)

        frame_count += 1
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()

        out_frame = draw_detections(frame.copy(), detections, fps)
        cv2.imshow("OWLv2 Detection", out_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
