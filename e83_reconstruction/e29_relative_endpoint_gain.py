#!/usr/bin/env python3
"""Per-image scale-invariant endpoint gain maps for E29."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from e24_endpoint_composite_gain import (
    COMPONENT_NAMES,
    endpoint_component_gain_maps,
)


RELATIVE_DENOMINATOR_EPSILON = 1e-8


def normalize_endpoint_gains_by_base_loss(
    gains: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, Mapping[str, torch.Tensor]],
    epsilon: float = RELATIVE_DENOMINATOR_EPSILON,
) -> dict[str, torch.Tensor]:
    """Convert signed endpoint gains into per-image relative error reductions."""
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("epsilon must be in (0,1)")
    if set(gains) != set(COMPONENT_NAMES):
        raise ValueError(f"gain components must be exactly {COMPONENT_NAMES}")
    if set(diagnostics) != set(COMPONENT_NAMES):
        raise ValueError(f"diagnostic components must be exactly {COMPONENT_NAMES}")

    result: dict[str, torch.Tensor] = {}
    batch = None
    for name in COMPONENT_NAMES:
        value = gains[name].float()
        if value.ndim != 4 or value.shape[1:] != (1, 16, 16):
            raise ValueError(f"{name} gain must be [B,1,16,16], got {value.shape}")
        if batch is None:
            batch = value.shape[0]
        elif value.shape[0] != batch:
            raise ValueError("component batch sizes differ")
        base_loss = diagnostics[name].get("base_loss")
        if not isinstance(base_loss, torch.Tensor) or base_loss.shape != (batch,):
            raise ValueError(f"{name} base_loss must be [B]")
        base_loss = base_loss.float()
        if not torch.isfinite(value).all() or not torch.isfinite(base_loss).all():
            raise FloatingPointError(f"non-finite {name} gain or base loss")
        if (base_loss < 0).any():
            raise ValueError(f"negative {name} base loss")
        relative = value / base_loss.clamp_min(float(epsilon))[:, None, None, None]
        if not torch.isfinite(relative).all():
            raise FloatingPointError(f"non-finite relative {name} gain")
        result[name] = relative
    return result


@torch.no_grad()
def relative_endpoint_component_gain_maps(
    target: torch.Tensor,
    x_base: torch.Tensor,
    x_native: torch.Tensor,
    inception: torch.nn.Module,
    lpips_metric: torch.nn.Module,
    image_range: str,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Run E24 endpoint attribution, then normalize each image by its base loss."""
    gains, diagnostics = endpoint_component_gain_maps(
        target, x_base, x_native, inception, lpips_metric, image_range
    )
    relative = normalize_endpoint_gains_by_base_loss(gains, diagnostics)
    return relative, diagnostics

