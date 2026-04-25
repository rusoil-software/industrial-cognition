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
# date   : 2026-Apr-13
# ==============================================================================

"""
Standalone script to download and export OWL 2 models to ONNX format.

Usage:
    python scripts/export_model.py --model owlv2-base --output models/owl2.onnx --device cuda
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.owl.inference.utils import OWL2ModelExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export OWL 2 models from PyTorch to ONNX format."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="owlv2-base",
        choices=list(OWL2ModelExporter.SUPPORTED_MODELS.keys()),
        help="Model variant to export (default: owlv2-base)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/owl2_model.onnx"),
        help="Output path for ONNX model (default: models/owl2_model.onnx)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export (default: cpu)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Optimize ONNX model after export (default: True)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_false",
        dest="optimize",
        help="Skip ONNX model optimization",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for HuggingFace models",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting OWL 2 model export...")
        logger.info(
            "Config: model=%s, output=%s, device=%s, optimize=%s",
            args.model,
            args.output,
            args.device,
            args.optimize,
        )

        exporter = OWL2ModelExporter(
            model_name=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
        )

        logger.info("Loading PyTorch model...")
        exporter.load_model()

        logger.info("Exporting to ONNX...")
        exporter.export_to_onnx(
            output_path=args.output,
            opset_version=args.opset,
            optimize_model=args.optimize,
        )

        logger.info("✓ Export completed successfully!")
        logger.info("Model saved to: %s", args.output.resolve())
        return 0

    except Exception as exc:
        logger.exception("Export failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
