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
# date   : 2026-Apr-23
# ==============================================================================

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2ForObjectDetection

logger = logging.getLogger(__name__)


class OWL2ModelExporter:
    """
    Handles downloading OWL 2 models from HuggingFace and exporting them to ONNX format.
    Supports both vision-language models like google/owlv2-base-patch16-ensemble.
    """

    SUPPORTED_MODELS = {
        "owlv2-base": "google/owlv2-base-patch16-ensemble",
        "owlv2-large": "google/owlv2-large-patch14-ensemble",
    }

    def __init__(
            self,
            model_name: str = "owlv2-base",
            device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the exporter.

        Args:
            model_name: Key from SUPPORTED_MODELS or full HuggingFace model ID
            device: "cpu" or "cuda"
            cache_dir: Where to cache downloaded models (default: ~/.cache/huggingface)
        """
        self.device = device
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve model ID
        self.model_id = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.model_name = model_name

        self.model: Optional[nn.Module] = None
        self.processor = None

    def load_model(self) -> None:
        """Download and load the PyTorch model and processor from HuggingFace."""
        logger.info("Loading OWL 2 model: %s from %s", self.model_name, self.model_id)

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
            )
            # self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model = Owlv2ForObjectDetection.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully on device: %s", self.device)
        except Exception as exc:
            logger.exception("Failed to load model %s", self.model_id)
            raise

    def export_to_onnx(
            self,
            output_path: Path,
            opset_version: int = 18,
            optimize_model: bool = True,
            use_external_data_format: bool = False,
    ) -> None:
        """
        Export the loaded PyTorch model to ONNX format.

        Args:
            output_path: Where to save the .onnx file
            opset_version: ONNX opset version (18 is widely supported)
            optimize_model: Whether to optimize the ONNX model after export
            use_external_data_format: For large models, split weights into external files

        Raises:
            RuntimeError: If model not loaded or export fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting OWL 2 model to ONNX: %s", output_path)

        # Create dummy inputs matching the model's expected shape
        dummy_inputs = self._create_dummy_inputs()

        class _ExportWrapper(torch.nn.Module):
            def __init__(self, model): super().__init__(); self.m = model

            def forward(self, pixel_values, input_ids, attention_mask):
                out = self.m(pixel_values=pixel_values,
                             input_ids=input_ids,
                             attention_mask=attention_mask)
                return out.logits, out.pred_boxes

        wrapped = _ExportWrapper(self.model).eval()

        try:
            # onnx_export_from_model(
            #     self.model,
            #     str(output_path),
            #     monolith=True,
            #     preprocessors=[self.processor],
            #     slim=True,
            #     use_subprocess=True,
            #     opt_level=opset_version,
            #     device=self.device,
            #     no_dynamic_axes=True,
            #     dtype="fp32",            # ← force clean fp32 trace, cast happens after
            # )
            torch.onnx.export(
                wrapped,
                dummy_inputs,
                str(output_path),
                input_names=["pixel_values", "input_ids", "attention_mask"],
                output_names=["logits", "pred_boxes"],
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                    "pred_boxes": {0: "batch_size"},
                },
            )
            logger.info("ONNX export completed: %s", output_path)

        except Exception as exc:
            logger.exception("ONNX export failed")
            raise

        if optimize_model:
            self._optimize_onnx(output_path)

    def _create_dummy_inputs(self) -> tuple:
        """
        Create dummy inputs for ONNX export tracing.
        Dimensions should match what OWL 2 expects.
        """
        batch_size = 1
        image_size = 960  # Standard for OWL v2
        seq_length = 16  # Max query tokens

        # Dummy image: [batch, channels, height, width]
        pixel_values = torch.randn(
            batch_size, 3, image_size, image_size,
            dtype=torch.float32,
            device=self.device,
        )

        # Dummy text tokens: [batch, seq_len]
        input_ids = torch.randint(
            0, 30522,  # BERT vocab size
            (batch_size, seq_length),
            dtype=torch.int64,
            device=self.device,
        )

        # Dummy attention mask: [batch, seq_len]
        attention_mask = torch.ones(
            batch_size, seq_length,
            dtype=torch.int64,
            device=self.device,
        )

        return pixel_values, input_ids, attention_mask

    @staticmethod
    def _optimize_onnx(
            onnx_path: Path,
            opset_version: int = 18
    ) -> None:
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
            from onnxruntime.transformers.float16 import convert_float_to_float16

            logger.info("Converting raw ONNX proto to fp16: %s", onnx_path)
            model_proto = onnx.load(str(onnx_path))
            model_proto = convert_float_to_float16(
                model_proto,
                keep_io_types=True,  # keep pixel_values input as fp32 so the
                # preprocessor doesn't need to change
                disable_shape_infer=False,
            )
            fp16_path = onnx_path.parent / f"{onnx_path.stem}_fp16.onnx"
            onnx.save(model_proto, str(fp16_path))
            logger.info("fp16 proto saved: %s", fp16_path)

            logger.info("Optimising fp16 model...")
            opt_options = BertOptimizationOptions("bert")
            opt_options.enable_all()

            opt_model = optimizer.optimize_model(
                str(fp16_path),
                model_type="bert",
                num_heads=12,
                hidden_size=768,
                use_gpu=True,
                opt_level=opset_version,
                only_onnxruntime=True,
                optimization_options=opt_options,
            )
            optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
            opt_model.save_model_to_file(str(optimized_path))
            logger.info("Optimised fp16 model saved: %s", optimized_path)

        except ImportError:
            logger.warning("onnx or onnxruntime[transformers] not installed. Skipping.")
        except Exception as exc:
            logger.warning("ONNX optimisation failed (non-fatal): %s", exc)


class OWL2ModelDownloader:
    """
    Convenience wrapper for downloading pre-trained OWL 2 models
    without exporting (useful for verification or direct PyTorch use).
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model_name: str = "owlv2-base") -> tuple:
        """
        Download model and processor, return both.

        Returns:
            (model, processor) tuple
        """
        model_id = OWL2ModelExporter.SUPPORTED_MODELS.get(
            model_name, model_name
        )

        logger.info("Downloading %s from HuggingFace...", model_id)

        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(self.cache_dir),
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            cache_dir=str(self.cache_dir),
        )

        logger.info("Download complete. Model cached at: %s", self.cache_dir)
        return model, processor


def export_owl2_to_onnx(
        model_name: str = "owlv2-base",
        output_path: Path = Path("models/owl2_model.onnx"),
        device: str = "cpu",
        optimize: bool = True,
) -> Path:
    """
    Convenience function to download and export an OWL 2 model in one call.

    Args:
        model_name: "owlv2-base", "owlv2-large", or full HuggingFace model ID
        output_path: Where to save the ONNX model
        device: "cpu" or "cuda"
        optimize: Whether to optimize the ONNX model

    Returns:
        Path to the exported ONNX file

    Example:
        >>> export_owl2_to_onnx(
        ...     model_name="owlv2-base",
        ...     output_path=Path("models/owl2_model.onnx"),
        ...     device="cuda",
        ... )
        PosixPath('models/owl2_model.onnx')
    """
    exporter = OWL2ModelExporter(model_name=model_name, device=device)
    exporter.load_model()
    exporter.export_to_onnx(output_path, optimize_model=optimize)

    logger.info("Export complete. Model saved to: %s", output_path)
    return output_path
