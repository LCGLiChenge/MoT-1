#!/usr/bin/env python3
"""Partial per-image RMS normalization of the frozen E31 combined gain."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from e31_pareto_relative_endpoint import combine_e31_numpy, combine_e31_torch


RMS_EPSILON = 1e-8
ALLOWED_POWERS = (0.25, 0.5, 0.75)
_AMPLITUDE_POWER: float | None = None


def set_amplitude_power(value: float) -> None:
    global _AMPLITUDE_POWER
    value = float(value)
    if value not in ALLOWED_POWERS:
        raise ValueError(f"E42 amplitude power must be one of {ALLOWED_POWERS}")
    _AMPLITUDE_POWER = value


def amplitude_power() -> float:
    if _AMPLITUDE_POWER is None:
        raise RuntimeError("set_amplitude_power must be called before E42 combination")
    return _AMPLITUDE_POWER


def combine_e42_numpy(
    components: Mapping[str, np.ndarray], scales: Mapping[str, float]
) -> np.ndarray:
    values = combine_e31_numpy(components, scales)
    if values.ndim != 2 or values.shape[1] != 256:
        raise ValueError(f"E42 NumPy gains must be [N,256], got {values.shape}")
    rms = np.sqrt(np.mean(np.square(values), axis=1, keepdims=True, dtype=np.float32))
    divisor = np.power(np.maximum(rms, np.float32(RMS_EPSILON)), amplitude_power())
    result = values / divisor
    if not np.isfinite(result).all():
        raise FloatingPointError("E42 NumPy gain contains NaN/Inf")
    return result.astype(np.float32, copy=False)


def combine_e42_torch(
    components: Mapping[str, torch.Tensor], scales: Mapping[str, float]
) -> torch.Tensor:
    values = combine_e31_torch(components, scales)
    if values.ndim != 4 or values.shape[1:] != (1, 16, 16):
        raise ValueError(f"E42 Torch gains must be [B,1,16,16], got {values.shape}")
    rms = values.square().flatten(1).mean(1).sqrt().clamp_min(RMS_EPSILON)
    divisor = rms.pow(amplitude_power())
    result = values / divisor[:, None, None, None]
    if not torch.isfinite(result).all():
        raise FloatingPointError("E42 Torch gain contains NaN/Inf")
    return result
