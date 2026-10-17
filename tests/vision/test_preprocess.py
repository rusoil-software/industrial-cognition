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
"""Tests for frame -> model-input preprocessing.

The point of these tests is that an incorrect preprocessing step does not raise:
it silently degrades detection quality. A swapped colour channel or the wrong mean
still produces a (1, 3, 960, 960) float32 array, and the model still returns
boxes - just worse ones. So each property is asserted numerically rather than by
shape alone.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.preprocess import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    PreprocessConfig,
    decode_image,
    letterbox_mapping,
    to_model_input,
)


def solid_bgr(blue: int, green: int, red: int, size: int = 32) -> np.ndarray:
    """A uniform BGR image, which makes the expected normalised values exact."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = blue
    image[:, :, 1] = green
    image[:, :, 2] = red
    return image


def test_output_shape_dtype_and_contiguity() -> None:
    image = solid_bgr(10, 20, 30)
    tensor = to_model_input(image, PreprocessConfig(width=64, height=48))

    assert tensor.shape == (1, 3, 48, 64)
    assert tensor.dtype == np.float32
    # Non-contiguous input makes ONNX Runtime copy the whole tensor internally on
    # every call, which at 60 FPS is avoidable overhead.
    assert tensor.flags["C_CONTIGUOUS"]


def test_channel_order_is_rgb_not_bgr() -> None:
    """A blue frame must not be normalised as if it were red.

    OWLv2's CLIP means are per-channel in RGB order. OpenCV delivers BGR, so a
    missing cvtColor swaps red and blue - invisible in shape and dtype, visible
    only in detection quality.
    """
    blue_frame = solid_bgr(blue=255, green=0, red=0)
    tensor = to_model_input(blue_frame, PreprocessConfig(width=8, height=8))

    # After BGR->RGB, channel 0 is red (0) and channel 2 is blue (255).
    expected_r = (0.0 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0]
    expected_b = (255.0 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2]
    assert tensor[0, 0].mean() == pytest.approx(expected_r, abs=1e-5)
    assert tensor[0, 2].mean() == pytest.approx(expected_b, abs=1e-5)


def test_normalisation_matches_clip_statistics() -> None:
    """`(pixel/255 - mean) / std`, asserted exactly on a uniform image."""
    image = solid_bgr(blue=100, green=150, red=200)
    tensor = to_model_input(image, PreprocessConfig(width=4, height=4))

    # BGR (100, 150, 200) -> RGB (200, 150, 100)
    for channel, value in enumerate((200, 150, 100)):
        expected = (value / 255.0 - CLIP_MEAN[channel]) / CLIP_STD[channel]
        assert tensor[0, channel].mean() == pytest.approx(expected, abs=1e-5)


def test_uniform_image_has_zero_spatial_variance() -> None:
    """A solid frame must produce a spatially constant tensor.

    A non-constant result would mean padding, a stray crop, or an accidental
    transpose - all of which shift every box coordinate.
    """
    tensor = to_model_input(solid_bgr(120, 120, 120), PreprocessConfig(width=32, height=32))
    for channel in range(3):
        plane = tensor[0, channel]
        assert plane.std() == pytest.approx(0.0, abs=1e-6)


def test_stretch_resizes_without_preserving_aspect_ratio() -> None:
    """Stretch is the default because it keeps box coordinates linearly mapped."""
    wide = np.zeros((100, 400, 3), dtype=np.uint8)
    wide[:, :200] = (255, 255, 255)  # left half white
    tensor = to_model_input(wide, PreprocessConfig(width=64, height=64))

    # Stretched to a square, the white region occupies exactly the left half.
    left = tensor[0, :, :, :32].mean()
    right = tensor[0, :, :, 32:].mean()
    assert left > right
    assert (tensor[0, :, :, :32] > -1.0).all()


def test_preprocessing_is_deterministic() -> None:
    image = solid_bgr(11, 22, 33)
    first = to_model_input(image)
    second = to_model_input(image)
    assert np.array_equal(first, second)


def test_no_op_when_frame_already_matches_model_size() -> None:
    """The resize must be skipped for an exact-size frame, not applied again."""
    exact = np.full((960, 960, 3), 128, dtype=np.uint8)
    tensor = to_model_input(exact)
    assert tensor.shape == (1, 3, 960, 960)
    assert tensor[0, 0, 0, 0] == pytest.approx(
        (128 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0], abs=1e-5
    )


def test_rejects_non_three_channel_input() -> None:
    grayscale = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        to_model_input(grayscale)


def test_letterbox_preserves_aspect_ratio_and_pads() -> None:
    wide = np.full((100, 400, 3), 200, dtype=np.uint8)
    config = PreprocessConfig(width=64, height=64, resize_mode="letterbox")
    tensor = to_model_input(wide, config)

    mapping = letterbox_mapping((100, 400), config)
    assert mapping.scale == pytest.approx(64 / 400)
    assert mapping.pad_y > 0
    assert mapping.pad_x == 0
    # Padded rows are zeros in the source, so they normalise to -mean/std, which is
    # lower than any real content.
    assert tensor[0, :, 0, :].mean() < tensor[0, :, 32, :].mean()


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(width=0)
    with pytest.raises(ValueError):
        PreprocessConfig(resize_mode="squash")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PreprocessConfig(std=(0.0, 0.1, 0.1))


def test_decode_image_round_trips_a_jpeg() -> None:
    original = np.zeros((48, 64, 3), dtype=np.uint8)
    original[:, :, 1] = 200
    ok, buffer = cv2.imencode(".jpg", original)
    assert ok

    decoded = decode_image(buffer.tobytes())
    assert decoded.shape == original.shape
    assert decoded.dtype == np.uint8


def test_decode_image_rejects_empty_and_garbage() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode_image(b"")
    with pytest.raises(ValueError, match="decode"):
        decode_image(b"not an image at all")


def test_reference_processor_agrees_when_transformers_is_available() -> None:
    """Cross-check the constants and maths against HuggingFace's processor.

    Skipped when `transformers` is absent (it is an export-time dependency, not a
    runtime one). When present in CI this turns the CLIP mean/std assumption from
    a comment into a test.
    """
    transformers = pytest.importorskip("transformers")
    # `Owlv2ImageProcessor` needs both torch and torchvision at import time, even
    # though it is a pure image processor.
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    processor = transformers.Owlv2ImageProcessor(size={"height": 960, "width": 960})

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(300, 200, 3), dtype=np.uint8)
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    ours = to_model_input(image_bgr)

    reference = processor(images=Image.fromarray(rgb), return_tensors="np")
    theirs = np.asarray(reference["pixel_values"], dtype=np.float32)

    assert ours.shape == theirs.shape
    # The processor resizes with bicubic exactly as PreprocessConfig does, so the
    # arrays should agree to within interpolation rounding.
    assert np.allclose(ours, theirs, atol=2e-3)
