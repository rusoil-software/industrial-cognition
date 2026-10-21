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
# date   : 2026-Oct-21
# ==============================================================================
"""One adapter per runtime, behind one interface, so the scoring code cannot tell them apart.

Every adapter returns :class:`~src.benchmark.scoring.Prediction` in normalised top-left
coordinates. That is the whole contract: a model that wants prompts in a sentence, a
batch of labels or a task token hides that inside its adapter, where it cannot leak
into the comparison.

The imports are lazy and the heavy ones are optional. A machine without PyTorch can
still measure the ONNX artifact and report the rest as not measurable, which is more
useful than a script that refuses to start.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from src.benchmark.scoring import Prediction

LOGGER = logging.getLogger(__name__)

#: Boxes below this score are not reported by any adapter. Low on purpose: the station
#: gate applies its own floor, and a model that only finds the part at 0.06 should be
#: seen doing it rather than hidden by the harness.
CONFIDENCE_FLOOR = 0.05

#: The ONNX export the cluster runs.
DEFAULT_ONNX_PATH = Path(os.getenv("OWL2_MODEL_PATH", "src/owl/models/cuda/owl2_model.onnx"))


class Runner(Protocol):
    """What the benchmark needs from a model."""

    key: str
    device: str

    def load(self) -> None:
        """Bring weights into memory. Timed separately from inference."""

    def detect(self, image: np.ndarray, labels: Sequence[str]) -> list[Prediction]:
        """Return every box the model reports for ``labels`` in ``image``."""

    def close(self) -> None:
        """Release memory and devices."""

    def params(self) -> Optional[float]:
        """Parameter count in millions when the runtime can say."""


class UnavailableRunner:
    """Stands in for a model that cannot be measured; keeps the pipeline uniform."""

    def __init__(self, key: str, reason: str) -> None:
        self.key = key
        self.device = ""
        self.reason = reason

    def load(self) -> None:
        raise RuntimeError(self.reason)

    def detect(self, image: np.ndarray, labels: Sequence[str]) -> list[Prediction]:
        raise RuntimeError(self.reason)

    def close(self) -> None:
        return None

    def params(self) -> Optional[float]:
        return None


class Owl2OnnxRunner:
    """The incumbent: the exact ONNX artifact, one forward pass per label.

    Per label, not per batch, because this export takes a single text query per graph
    run. That is also how the cluster spends its time, so the number is the one that
    matters.
    """

    key = "owl2-onnx"

    def __init__(self, model_path: Path = DEFAULT_ONNX_PATH) -> None:
        self._model_path = Path(model_path)
        self._backend = None
        self._tokenizer = None
        self.device = "cpu"

    def load(self) -> None:
        from src.vision.backend import Owl2OnnxBackend, TextQuery  # noqa: PLC0415
        from src.vision.prompt import TransformersPromptTokenizer, build_prompt_text  # noqa: PLC0415

        if not self._model_path.exists():
            raise FileNotFoundError(f"OWLv2 artifact missing: {self._model_path}")
        self._tokenizer = TransformersPromptTokenizer(local_files_only=False)
        # The backend wants a tokenised query at construction time even though every
        # infer() call supplies its own: this one only decides the text length.
        ids, mask = self._tokenizer.encode(build_prompt_text("object"))
        self._backend = Owl2OnnxBackend(
            self._model_path,
            text_query=TextQuery(labels=("object",), input_ids=ids, attention_mask=mask),
        )
        self._backend.warmup()

    def detect(self, image: np.ndarray, labels: Sequence[str]) -> list[Prediction]:
        from src.vision.backend import TextQuery  # noqa: PLC0415
        from src.vision.decode import decode_queries  # noqa: PLC0415
        from src.vision.preprocess import PreprocessConfig, to_model_input  # noqa: PLC0415
        from src.vision.prompt import build_prompt_text  # noqa: PLC0415

        if self._backend is None or self._tokenizer is None:
            raise RuntimeError("load() must be called before detect()")
        pixels = to_model_input(image, PreprocessConfig())
        predictions: list[Prediction] = []
        for label in labels:
            ids, mask = self._tokenizer.encode(build_prompt_text(label))
            out = self._backend.infer(
                pixels,
                text_query=TextQuery(labels=(label,), input_ids=ids, attention_mask=mask),
            )
            detections = decode_queries(
                out.logits,
                out.pred_boxes,
                [label],
                [CONFIDENCE_FLOOR],
                iou_threshold=0.5,
                max_detections=32,
            )
            predictions.extend(
                Prediction(
                    label=detection.object_type,
                    confidence=float(detection.confidence),
                    box=(
                        float(detection.x_norm),
                        float(detection.y_norm),
                        float(detection.width_norm),
                        float(detection.height_norm),
                    ),
                )
                for detection in detections
            )
        return predictions

    def close(self) -> None:
        self._backend = None
        self._tokenizer = None

    def params(self) -> Optional[float]:
        return None


class _TorchRunner:
    """Shared plumbing for the PyTorch adapters: device choice and text handling."""

    key = ""
    device = "cpu"

    def _resolve_device(self) -> str:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def params(self) -> Optional[float]:
        model = getattr(self, "_model", None)
        if model is None:
            return None
        try:
            return float(sum(p.numel() for p in model.parameters())) / 1e6
        except Exception:  # noqa: BLE001
            return None


class GroundingDinoRunner(_TorchRunner):
    """Grounding DINO (Swin-T): one prompt string, every object as a phrase."""

    key = "grounding-dino-base"
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self.device = self._resolve_device()

    def load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415

        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID, dtype=dtype)
        self._model.to(self.device).eval()

    def detect(self, image: np.ndarray, labels: Sequence[str]) -> list[Prediction]:
        import torch  # noqa: PLC0415

        if self._model is None or self._processor is None:
            raise RuntimeError("load() must be called before detect()")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = [f"{label.lower()}." for label in labels]
        inputs = self._processor(images=rgb, text=" ".join(text), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=CONFIDENCE_FLOOR,
            text_threshold=CONFIDENCE_FLOOR,
            target_sizes=[image.shape[:2]],
        )[0]
        height, width = image.shape[:2]
        predictions: list[Prediction] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = (float(value) for value in box)
            predictions.append(
                Prediction(
                    label=_match_label(str(label), labels),
                    confidence=float(score),
                    box=(x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height),
                )
            )
        return predictions

    def close(self) -> None:
        self._model = None
        self._processor = None


class Florence2Runner(_TorchRunner):
    """Florence-2: a seq2seq model that answers with coordinates as tokens.

    One label per call - the prompt template takes a single target phrase - and the
    decoder is not cheap, which the latency column will show.
    """

    key = "florence2-base"
    MODEL_ID = "microsoft/Florence-2-base"
    TASK = "<OPEN_VOCABULARY_DETECTION>"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self.device = self._resolve_device()

    def load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415

        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, dtype=dtype
        )
        self._model.to(self.device).eval()

    def detect(self, image: np.ndarray, labels: Sequence[str]) -> list[Prediction]:
        import torch  # noqa: PLC0415

        if self._model is None or self._processor is None:
            raise RuntimeError("load() must be called before detect()")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        predictions: list[Prediction] = []
        for label in labels:
            prompt = self.TASK + label
            inputs = self._processor(text=prompt, images=rgb, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3,
                    do_sample=False,
                )
            text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                text, task=self.TASK, image_size=(width, height)
            )
            payload = parsed.get(self.TASK) or {}
            for box in payload.get("bboxes", []):
                x1, y1, x2, y2 = (float(value) for value in box[:4])
                predictions.append(
                    Prediction(
                        label=label,
                        confidence=CONFIDENCE_FLOOR,
                        box=(x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height),
                    )
                )
        return predictions

    def close(self) -> None:
        self._model = None
        self._processor = None


class YoloWorldRunner(_TorchRunner):
    """YOLO-World v2 through ``ultralytics``: real-time, AGPL-licensed runtime."""

    def __init__(self, key: str, weights: str) -> None:
        self.key = key
        self.weights = weights
        self._model = None
        self._classes: Optional[tuple[str, ...]] = None
        self.device = self._resolve_device()

    def load(self) -> None:
        from ultralytics import YOLOWorld  # noqa: PLC0415

        self._model = YOLOWorld(self.weights)
        self._model.to(self.device)

    def detect(self, image: np.ndarray, labels: Sequence[str]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("load() must be called before detect()")
        classes = tuple(labels)
        # set_classes() rebuilds the CLIP text embeddings and takes 6-9 s on this hardware,
        # so it must run only when the vocabulary actually changes: the benchmark's sample
        # loop changes labels once, and setting them per frame would measure CLIP, not the
        # detector.
        if classes != self._classes:
            self._model.set_classes([label for label in classes])
            self._classes = classes
        results = self._model.predict(image, conf=CONFIDENCE_FLOOR, verbose=False)
        height, width = image.shape[:2]
        predictions: list[Prediction] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            # The result carries the names for the classes it was just given; the model
            # attribute can be stale after set_classes().
            names = getattr(result, "names", None) or getattr(self._model, "names", {})
            for box, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                x1, y1, x2, y2 = (float(value) for value in box[:4])
                label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
                predictions.append(
                    Prediction(
                        label=label,
                        confidence=float(score),
                        box=(x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height),
                    )
                )
        return predictions

    def close(self) -> None:
        self._model = None


def _match_label(returned: str, labels: Sequence[str]) -> str:
    """Map a model's label text back onto the prompt that produced it."""
    lowered = returned.strip().lower().rstrip(".")
    for label in labels:
        if lowered == label.lower() or lowered.startswith(label.lower()):
            return label
    return labels[0] if labels else returned


def weights_path(name: str, directory: Optional[str] = None) -> str:
    """Where a downloadable ``.pt`` lives.

    ``ultralytics`` saves the file it downloads into the working directory, which
    inside the benchmark container is the mounted repository: a 50-100 MB surprise in
    ``git status``. ``YOLO_WEIGHTS_DIR`` (or an explicit ``directory``) moves it to a
    cache, and it also makes a re-run skip the download.
    """
    raw = directory if directory is not None else os.environ.get("YOLO_WEIGHTS_DIR", "")
    if not raw.strip():
        return name
    return str(Path(raw) / name)


def build_runner(key: str, *, onnx_path: Path = DEFAULT_ONNX_PATH) -> Runner:
    """Construct the adapter for ``key``; heavy imports happen in ``load()``."""
    if key == "owl2-onnx":
        return Owl2OnnxRunner(onnx_path)
    if key == "grounding-dino-base":
        return GroundingDinoRunner()
    if key == "florence2-base":
        return Florence2Runner()
    if key == "yolo-world-m":
        return YoloWorldRunner("yolo-world-m", weights_path("yolov8m-worldv2.pt"))
    if key == "yolo-world-l":
        return YoloWorldRunner("yolo-world-l", weights_path("yolov8l-worldv2.pt"))
    if key == "grounding-dino-15-edge":
        return UnavailableRunner(key, "API-only release: needs IDEA Research credentials")
    if key == "dino-x":
        return UnavailableRunner(key, "closed API: needs IDEA Research credentials")
    raise KeyError(f"no adapter for {key!r}")


__all__ = [
    "CONFIDENCE_FLOOR",
    "DEFAULT_ONNX_PATH",
    "Florence2Runner",
    "GroundingDinoRunner",
    "Owl2OnnxRunner",
    "Runner",
    "UnavailableRunner",
    "YoloWorldRunner",
    "build_runner",
    "weights_path",
]
