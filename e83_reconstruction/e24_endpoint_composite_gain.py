#!/usr/bin/env python3
"""Mass-preserving all-1D/all-2D endpoint gain maps for E24."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import interpolate_bilinear_2d_like_tensorflow1x

from eval_oracle_dynamic_budget import to_lpips_range, to_zero_one


GRID_SIDE = 16
COMPONENT_NAMES = ("inception", "lpips", "pixel_mse")
COMPONENT_WEIGHTS = {"inception": 1.0, "lpips": 2.0, "pixel_mse": 4.0}


def _mean_preserving_resize(values: torch.Tensor, side: int = GRID_SIDE) -> torch.Tensor:
    """Resize a nonnegative map while preserving each sample's spatial mean."""
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(f"values must be [B,1,H,W], got {values.shape}")
    values = values.float()
    source_mean = values.mean(dim=(2, 3), keepdim=True)
    if values.shape[-2:] == (side, side):
        resized = values
    else:
        resized = F.interpolate(
            values, size=(side, side), mode="bilinear", align_corners=False
        )
    resized_mean = resized.mean(dim=(2, 3), keepdim=True)
    scale = torch.where(
        resized_mean.abs() > 1e-20,
        source_mean / resized_mean,
        torch.zeros_like(resized_mean),
    )
    return resized * scale


def _inception_mixed7c(
    inception: torch.nn.Module, image_255: torch.Tensor
) -> torch.Tensor:
    """TorchMetrics FID Inception forward through the final spatial block."""
    x = image_255.float()
    if inception.use_antialias:
        x = F.interpolate(
            x,
            size=(inception.INPUT_IMAGE_SIZE, inception.INPUT_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    else:
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(inception.INPUT_IMAGE_SIZE, inception.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
    x = (x - 128.0) / 128.0
    x = inception.Conv2d_1a_3x3(x)
    x = inception.Conv2d_2a_3x3(x)
    x = inception.Conv2d_2b_3x3(x)
    x = inception.MaxPool_1(x)
    x = inception.Conv2d_3b_1x1(x)
    x = inception.Conv2d_4a_3x3(x)
    x = inception.MaxPool_2(x)
    x = inception.Mixed_5b(x)
    x = inception.Mixed_5c(x)
    x = inception.Mixed_5d(x)
    x = inception.Mixed_6a(x)
    x = inception.Mixed_6b(x)
    x = inception.Mixed_6c(x)
    x = inception.Mixed_6d(x)
    x = inception.Mixed_6e(x)
    x = inception.Mixed_7a(x)
    x = inception.Mixed_7b(x)
    return inception.Mixed_7c(x)


def _mass_preserving_inception_error_map(
    target_map: torch.Tensor, prediction_map: torch.Tensor, inception: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    local_error = (prediction_map.float() - target_map.float()).square().mean(
        dim=1, keepdim=True
    )
    target_feature = torch.flatten(inception.AvgPool(target_map.float()), 1)
    prediction_feature = torch.flatten(inception.AvgPool(prediction_map.float()), 1)
    scalar_error = (prediction_feature - target_feature).square().mean(dim=1)
    resized = _mean_preserving_resize(local_error)
    local_mean = resized.mean(dim=(2, 3), keepdim=True)
    scale = torch.where(
        local_mean > 1e-20,
        scalar_error[:, None, None, None] / local_mean,
        torch.zeros_like(local_mean),
    )
    return resized * scale, scalar_error


def _lpips_error_maps(
    metric: torch.nn.Module,
    target: torch.Tensor,
    base: torch.Tensor,
    native: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target/base and target/native learned-LPIPS spatial energies."""
    required = ("net", "scaling_layer", "lins", "L")
    if any(not hasattr(metric, name) for name in required):
        raise TypeError("E24 requires the standard learned lpips.LPIPS implementation")
    batch = target.shape[0]
    tripled = torch.cat([target.float(), base.float(), native.float()], dim=0)
    if getattr(metric, "version", None) == "0.1":
        tripled = metric.scaling_layer(tripled)
    layers = metric.net.forward(tripled)
    if len(layers) != int(metric.L):
        raise ValueError("LPIPS feature layer count mismatch")
    base_map = target.new_zeros((batch, 1, GRID_SIDE, GRID_SIDE), dtype=torch.float32)
    native_map = torch.zeros_like(base_map)
    base_scalar = target.new_zeros(batch, dtype=torch.float32)
    native_scalar = torch.zeros_like(base_scalar)
    for index, layer in enumerate(layers):
        target_layer, base_layer, native_layer = layer.float().split(batch, dim=0)
        target_layer = target_layer / torch.sqrt(
            target_layer.square().sum(dim=1, keepdim=True) + 1e-10
        )
        base_layer = base_layer / torch.sqrt(
            base_layer.square().sum(dim=1, keepdim=True) + 1e-10
        )
        native_layer = native_layer / torch.sqrt(
            native_layer.square().sum(dim=1, keepdim=True) + 1e-10
        )
        base_energy = metric.lins[index]((base_layer - target_layer).square()).float()
        native_energy = metric.lins[index]((native_layer - target_layer).square()).float()
        base_scalar = base_scalar + base_energy.mean(dim=(1, 2, 3))
        native_scalar = native_scalar + native_energy.mean(dim=(1, 2, 3))
        base_map = base_map + _mean_preserving_resize(base_energy)
        native_map = native_map + _mean_preserving_resize(native_energy)
    return base_map, native_map, base_scalar, native_scalar


@torch.no_grad()
def endpoint_component_gain_maps(
    target: torch.Tensor,
    x_base: torch.Tensor,
    x_native: torch.Tensor,
    inception: torch.nn.Module,
    lpips_metric: torch.nn.Module,
    image_range: str,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Compute signed all-1D minus all-2D endpoint gains on a 16x16 grid."""
    if target.shape != x_base.shape or target.shape != x_native.shape:
        raise ValueError("target, x_base and x_native must have identical shapes")
    batch = target.shape[0]
    target_01 = to_zero_one(target, image_range).clamp(0.0, 1.0)
    base_01 = to_zero_one(x_base, image_range).clamp(0.0, 1.0)
    native_01 = to_zero_one(x_native, image_range).clamp(0.0, 1.0)

    base_pixel = F.adaptive_avg_pool2d(
        (base_01 - target_01).square().mean(dim=1, keepdim=True),
        (GRID_SIDE, GRID_SIDE),
    )
    native_pixel = F.adaptive_avg_pool2d(
        (native_01 - target_01).square().mean(dim=1, keepdim=True),
        (GRID_SIDE, GRID_SIDE),
    )

    target_lpips = to_lpips_range(target, image_range)
    base_lpips = to_lpips_range(x_base, image_range)
    native_lpips = to_lpips_range(x_native, image_range)
    base_lpips_map, native_lpips_map, base_lpips_scalar, native_lpips_scalar = (
        _lpips_error_maps(
            lpips_metric, target_lpips, base_lpips, native_lpips
        )
    )

    inception_triplet = _inception_mixed7c(
        inception, torch.cat([target_01, base_01, native_01], dim=0) * 255.0
    )
    target_inc, base_inc, native_inc = inception_triplet.split(batch, dim=0)
    base_inc_map, base_inc_scalar = _mass_preserving_inception_error_map(
        target_inc, base_inc, inception
    )
    native_inc_map, native_inc_scalar = _mass_preserving_inception_error_map(
        target_inc, native_inc, inception
    )

    base_pixel_scalar = base_pixel.mean(dim=(1, 2, 3))
    native_pixel_scalar = native_pixel.mean(dim=(1, 2, 3))
    base_maps = {
        "inception": base_inc_map,
        "lpips": base_lpips_map,
        "pixel_mse": base_pixel,
    }
    native_maps = {
        "inception": native_inc_map,
        "lpips": native_lpips_map,
        "pixel_mse": native_pixel,
    }
    base_scalars = {
        "inception": base_inc_scalar,
        "lpips": base_lpips_scalar,
        "pixel_mse": base_pixel_scalar,
    }
    native_scalars = {
        "inception": native_inc_scalar,
        "lpips": native_lpips_scalar,
        "pixel_mse": native_pixel_scalar,
    }
    gains = {name: base_maps[name] - native_maps[name] for name in COMPONENT_NAMES}
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for name in COMPONENT_NAMES:
        if gains[name].shape != (batch, 1, GRID_SIDE, GRID_SIDE):
            raise AssertionError(f"unexpected {name} gain shape {gains[name].shape}")
        if not torch.isfinite(gains[name]).all():
            raise FloatingPointError(f"non-finite {name} endpoint gain")
        diagnostics[name] = {
            "base_loss": base_scalars[name],
            "native_loss": native_scalars[name],
            "base_map_mean_error": (
                base_maps[name].mean(dim=(1, 2, 3)) - base_scalars[name]
            ).abs(),
            "native_map_mean_error": (
                native_maps[name].mean(dim=(1, 2, 3)) - native_scalars[name]
            ).abs(),
        }
    return gains, diagnostics


def fit_component_scales(
    components: Mapping[str, np.ndarray], epsilon: float = 1e-12
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in COMPONENT_NAMES:
        values = np.asarray(components[name], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != GRID_SIDE * GRID_SIDE:
            raise ValueError(f"{name} must have shape [N,256], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        std = float(values.std())
        if std <= epsilon:
            raise ValueError(f"{name} has degenerate standard deviation {std}")
        result[name] = std
    return result


def combine_components_numpy(
    components: Mapping[str, np.ndarray], scales: Mapping[str, float]
) -> np.ndarray:
    result = None
    for name in COMPONENT_NAMES:
        value = (
            np.asarray(components[name], dtype=np.float32)
            * np.float32(COMPONENT_WEIGHTS[name] / float(scales[name]))
        )
        result = value if result is None else result + value
    if result is None:
        raise AssertionError("no endpoint components were combined")
    return result


def combine_components_torch(
    components: Mapping[str, torch.Tensor], scales: Mapping[str, float]
) -> torch.Tensor:
    result = None
    for name in COMPONENT_NAMES:
        value = components[name].float() * (
            float(COMPONENT_WEIGHTS[name]) / float(scales[name])
        )
        result = value if result is None else result + value
    if result is None:
        raise AssertionError("no endpoint components were combined")
    return result

