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
# date   : 2026-Oct-17
# ==============================================================================
"""Turn compressed camera frames into the tensor the ONNX graph expects.

The exported graph's contract, read from the artifact rather than assumed (see
`scripts/inspect_model.py`):

    pixel_values   (batch, 3, 960, 960)  float32
    input_ids      (batch, 16)           int64
    attention_mask (batch, 16)           int64

so this module owns image decoding, resizing and normalisation. It is
deliberately dependency-light - NumPy and OpenCV only - so it can run in the
Celery worker without `transformers`, `torch` or a HuggingFace processor. That is
not a corner cut: the reference `Owlv2ImageProcessor` is a resize plus a
mean/std normalisation, and running the full processor per frame would add
hundreds of milliseconds of pure NumPy-identical work at 60 FPS.

Normalisation uses CLIP statistics (OWLv2's vision tower is CLIP-initialised):
``(pixel / 255 - mean) / std`` with the mean/std below. These constants are
asserted against the reference processor by
`tests/vision/test_preprocess.py::test_matches_reference_processor` whenever
`transformers` is importable, so the assumption is checked in CI rather than
trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np

__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "MODEL_INPUT_SIZE",
    "PreprocessConfig",
    "decode_image",
    "to_model_input",
]

#: Spatial size the ONNX graph was exported with. Changing this requires
#: re-exporting the model; it is read back from the artifact by
#: `scripts/inspect_model.py` and asserted by the pipeline's startup check.
MODEL_INPUT_SIZE: int = 960

#: CLIP image normalisation statistics, matching `Owlv2ImageProcessor.image_mean`
#: and `.image_std`. RGB order.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

ResizeMode = Literal["stretch", "letterbox"]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """How a captured frame becomes a model input.

    ``resize_mode``:
        * ``"stretch"`` - resize directly to ``height x width``, ignoring aspect
          ratio. This is what the reference OWLv2 processor does, so normalised
          box coordinates map linearly back to the frame, which is why it is the
          default and why :class:`~src.vision.detections.Detection` can stay
          resolution independent.
        * ``"letterbox"`` - preserve aspect ratio and pad. Available because a
          wide workcell seen off-axis may warrant it, but it changes the mapping
          back to frame coordinates, so :meth:`to_model_input` also reports the
          scale/offset via :class:`LetterboxMapping` and callers must apply it.
          Not used by the default pipeline.
    """

    width: int = MODEL_INPUT_SIZE
    height: int = MODEL_INPUT_SIZE
    mean: tuple[float, float, float] = CLIP_MEAN
    std: tuple[float, float, float] = CLIP_STD
    resize_mode: ResizeMode = "stretch"
    #: Interpolation used on resize. OWLv2's processor uses bicubic.
    interpolation: int = cv2.INTER_CUBIC

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"invalid model input size {self.width}x{self.height}"
            )
        if self.resize_mode not in ("stretch", "letterbox"):
            raise ValueError(f"unsupported resize_mode {self.resize_mode!r}")
        if any(s == 0 for s in self.std):
            raise ValueError("standard deviation must be non-zero")


@dataclass(frozen=True, slots=True)
class LetterboxMapping:
    """How a letterboxed input maps back onto the source frame.

    Only produced for ``resize_mode="letterbox"``. ``scale`` is the uniform
    resize factor and ``pad_x``/``pad_y`` the padding in model-input pixels, so a
    box in model space converts back with ``(x - pad_x) / scale / src_width``.
    """

    scale: float
    pad_x: int
    pad_y: int
    src_width: int
    src_height: int


def decode_image(data: bytes) -> np.ndarray:
    """Decode compressed image bytes (JPEG/PNG) into a BGR ``uint8`` array.

    Raises:
        ValueError: when the buffer is empty or not a decodable image. Failing
            here gives a precise error at the trust boundary instead of an
            obscure shape mismatch inside the graph.
    """
    if not data:
        raise ValueError("empty frame buffer")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"could not decode frame ({len(data)} bytes); not a supported image format"
        )
    return image


def _stretch(image_bgr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if image_bgr.shape[1] == config.width and image_bgr.shape[0] == config.height:
        return image_bgr
    return cv2.resize(
        image_bgr,
        (config.width, config.height),
        interpolation=config.interpolation,
    )


def _letterbox(image_bgr: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, LetterboxMapping]:
    src_h, src_w = image_bgr.shape[:2]
    scale = min(config.width / src_w, config.height / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=config.interpolation)

    canvas = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    pad_x = (config.width - new_w) // 2
    pad_y = (config.height - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxMapping(
        scale=scale, pad_x=pad_x, pad_y=pad_y, src_width=src_w, src_height=src_h
    )


def to_model_input(
    image_bgr: np.ndarray,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Convert a BGR frame into a ``(1, 3, H, W)`` float32 model input.

    The returned array is C-contiguous and float32 because the graph declares
    ``tensor(float)``; passing float64 or a non-contiguous view makes ONNX
    Runtime copy the whole 11 MB tensor internally on every call, which at 60 FPS
    is a measurable, avoidable cost.

    Args:
        image_bgr: frame as produced by OpenCV (BGR, ``uint8``).
        config: preprocessing settings; defaults to the exported model's contract.

    Returns:
        ``(1, 3, height, width)`` float32, RGB, CLIP-normalised.
    """
    config = config or PreprocessConfig()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"expected a 3-channel HxWx3 BGR frame, got shape {image_bgr.shape}"
        )

    if config.resize_mode == "letterbox":
        resized, _ = _letterbox(image_bgr, config)
    else:
        resized = _stretch(image_bgr, config)

    # BGR -> RGB before normalisation: the CLIP means are per-channel in RGB
    # order, so skipping this silently swaps the red and blue channels and
    # degrades detection quality without ever raising an error.
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)

    # float32 throughout, and normalise in place: at 960x960x3 an avoidable
    # float64 upcast turns ~11 MB into ~22 MB of temporary per frame.
    array = rgb.astype(np.float32)
    array /= 255.0
    array -= mean
    array /= std

    # HWC -> CHW and add the batch dimension the graph expects.
    chw = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, ...])


def letterbox_mapping(src_shape: Sequence[int], config: PreprocessConfig | None = None) -> LetterboxMapping:
    """Compute the letterbox mapping for a source shape without preprocessing.

    Useful for converting detections back to source coordinates without holding
    on to the intermediate image.
    """
    config = config or PreprocessConfig()
    src_h, src_w = int(src_shape[0]), int(src_shape[1])
    scale = min(config.width / src_w, config.height / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    return LetterboxMapping(
        scale=scale,
        pad_x=(config.width - new_w) // 2,
        pad_y=(config.height - new_h) // 2,
        src_width=src_w,
        src_height=src_h,
    )
