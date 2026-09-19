#!/usr/bin/env python3
"""Frozen E31 1/2/6 Pareto component combination."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from e24_endpoint_composite_gain import COMPONENT_NAMES


E31_COMPONENT_WEIGHTS = {"inception": 1.0, "lpips": 2.0, "pixel_mse": 6.0}


def combine_e31_numpy(
    components: Mapping[str, np.ndarray], scales: Mapping[str, float]
) -> np.ndarray:
    result = None
    for name in COMPONENT_NAMES:
        value = np.asarray(components[name], dtype=np.float32) * np.float32(
            E31_COMPONENT_WEIGHTS[name] / float(scales[name])
        )
        result = value if result is None else result + value
    if result is None or not np.isfinite(result).all():
        raise FloatingPointError("invalid E31 NumPy gain")
    return result


def combine_e31_torch(
    components: Mapping[str, torch.Tensor], scales: Mapping[str, float]
) -> torch.Tensor:
    result = None
    for name in COMPONENT_NAMES:
        value = components[name].float() * (
            E31_COMPONENT_WEIGHTS[name] / float(scales[name])
        )
        result = value if result is None else result + value
    if result is None or not torch.isfinite(result).all():
        raise FloatingPointError("invalid E31 Torch gain")
    return result

