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
"""The cost side of the comparison: time, memory, and how big the artifact is.

Everything here degrades to ``None`` rather than raising, because a benchmark that
cannot read one counter should still produce the other nine. A missing number is
reported as missing; a zero would be a lie.
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

#: HuggingFace cache root, honouring the environment the way ``transformers`` does.
HF_CACHE = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


@dataclass(slots=True)
class Latency:
    """Wall-clock samples of one model's inference calls."""

    samples_ms: list[float] = field(default_factory=list)

    def add(self, elapsed_ms: float) -> None:
        self.samples_ms.append(elapsed_ms)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    def mean(self) -> float:
        return statistics.fmean(self.samples_ms) if self.samples_ms else 0.0

    def percentile(self, fraction: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    def fps(self) -> float:
        """Frames per second a single process can sustain, from the mean latency."""
        mean = self.mean()
        return 1000.0 / mean if mean > 0 else 0.0


@contextmanager
def timed(latency: Latency) -> Iterator[None]:
    """Record the wall-clock duration of the block in ``latency``."""
    started = time.perf_counter()
    try:
        yield
    finally:
        latency.add((time.perf_counter() - started) * 1000.0)


def peak_rss_mb() -> Optional[float]:
    """High-water mark of this process' resident memory, in MiB."""
    try:
        import resource  # noqa: PLC0415 - POSIX only

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001 - Windows has no resource module
        pass
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def gpu_memory_mb() -> Optional[float]:
    """Peak memory allocated on the GPU by this process, in MiB, when there is one."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def reset_gpu_peak() -> None:
    """Start a fresh peak measurement, so loading one model does not bill the next."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        return


def directory_size_mb(path: Path) -> Optional[float]:
    """Total size of the files under ``path``, in MiB, or None when it does not exist."""
    if not path.exists():
        return None
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total / (1024.0 * 1024.0)


def weights_size_mb(reference: str) -> Optional[float]:
    """How big a model's weights are on disk, read from the caches that hold them.

    Handles the three shapes this benchmark uses: a local path (the ONNX export), a
    HuggingFace repository id (``org/name``), and a single file that ``ultralytics``
    downloaded next to itself.
    """
    path = Path(reference)
    if path.exists():
        return directory_size_mb(path) if path.is_dir() else path.stat().st_size / (1024.0 * 1024.0)
    if "/" in reference and reference.count("/") == 1:
        folder = HF_CACHE / ("models--" + reference.replace("/", "--"))
        if folder.exists():
            return directory_size_mb(folder)
    for candidate in (Path.cwd() / reference, Path.home() / reference, Path("/models") / reference):
        if candidate.exists():
            return candidate.stat().st_size / (1024.0 * 1024.0)
    return None


def parameter_count(model: object) -> Optional[float]:
    """Parameter count in millions, for models that expose one."""
    for attribute in ("num_parameters", "count_parameters"):
        method = getattr(model, attribute, None)
        if callable(method):
            try:
                return float(method()) / 1e6
            except Exception:  # noqa: BLE001
                pass
    try:
        parameters = getattr(model, "parameters")
        return float(sum(p.numel() for p in parameters())) / 1e6
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "HF_CACHE",
    "Latency",
    "directory_size_mb",
    "gpu_memory_mb",
    "parameter_count",
    "peak_rss_mb",
    "reset_gpu_peak",
    "timed",
    "weights_size_mb",
]
