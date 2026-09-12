"""Resolve reproducible training devices without making Torch a Q-learning dependency."""

from __future__ import annotations

import os
from typing import Any

try:
    import torch
except ImportError:  # Q-learning remains usable without PyTorch.
    torch = None


DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_device(algorithm: str, mode: str, requested: str) -> dict[str, Any]:
    """Validate a requested device and return metadata for the actual device."""
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}")
    if algorithm in {"q_learning", "legal_random"}:
        if requested == "cuda":
            label = "Q-learning" if algorithm == "q_learning" else "Legal random"
            raise ValueError(f"{label} always runs on CPU; CUDA is not applicable")
        return _metadata(requested, "cpu")
    if algorithm != "dqn":
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")
    if mode == "evaluate":
        if requested == "cuda":
            raise ValueError("DQN evaluation is forced to CPU for official compatibility")
        return _metadata(requested, "cpu")
    if mode != "train":
        raise ValueError(f"Unsupported mode: {mode!r}")
    if torch is None:
        raise RuntimeError("PyTorch is required for the DQN agent")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is not available")
    actual = "cuda:0" if available and requested in {"auto", "cuda"} else "cpu"
    return _metadata(requested, actual)


def _metadata(requested: str, actual: str) -> dict[str, Any]:
    is_cuda = actual.startswith("cuda")
    return {
        "actual": actual,
        "cuda_version": None if torch is None else torch.version.cuda,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(0) if is_cuda else None,
        "requested": requested,
        "torch_version": None if torch is None else str(torch.__version__),
        "type": "cuda" if is_cuda else "cpu",
    }
