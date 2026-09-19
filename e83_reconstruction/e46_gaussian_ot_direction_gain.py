#!/usr/bin/env python3
"""Gaussian-OT direction-aware endpoint gains for independent K selection.

The population statistics are frozen on ImageNet-train at fixed K=96.  At
inference, each source image is handled independently: the frozen Gaussian
transport map converts its real Inception feature into a fake-side partner,
and only reconstruction progress along that transport displacement is scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from e24_endpoint_composite_gain import (
    _inception_mixed7c,
    _mass_preserving_inception_error_map,
)
from e29_relative_endpoint_gain import relative_endpoint_component_gain_maps
from eval_oracle_dynamic_budget import to_zero_one


EXPECTED_DIRECTION_FORMAT = "e21_frozen_train_fid_direction_v1"
EXPECTED_FEATURE_DIM = 2048
EXPECTED_TARGET_TOKENS = 96

_STATE_CPU: dict[str, torch.Tensor] | None = None
_STATE_METADATA: dict[str, object] | None = None
_STATE_BY_DEVICE: dict[str, dict[str, torch.Tensor]] = {}


def configure_gaussian_ot_direction(path: str | Path) -> dict[str, object]:
    """Load and validate the frozen train-only fixed96 Gaussian transport."""
    global _STATE_CPU, _STATE_METADATA, _STATE_BY_DEVICE
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "fake_mean",
            "mean_delta",
            "covariance_gradient",
            "metadata_json",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"direction archive is missing {missing}")
        fake_mean = np.asarray(archive["fake_mean"], dtype=np.float64)
        mean_delta = np.asarray(archive["mean_delta"], dtype=np.float64)
        gradient = np.asarray(archive["covariance_gradient"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))

    if metadata.get("format") != EXPECTED_DIRECTION_FORMAT:
        raise ValueError("E46 requires the audited E21 direction format")
    if int(metadata.get("fid_feature", -1)) != EXPECTED_FEATURE_DIM:
        raise ValueError("E46 requires FID-2048 population statistics")
    if int(metadata.get("target_tokens", -1)) != EXPECTED_TARGET_TOKENS:
        raise ValueError("E46 requires population statistics at fixed K=96")
    if metadata.get("validation_statistics_used") is not False:
        raise ValueError("E46 direction must be prepared from train-only statistics")
    if fake_mean.shape != (EXPECTED_FEATURE_DIM,):
        raise ValueError(f"unexpected fake mean shape {fake_mean.shape}")
    if mean_delta.shape != fake_mean.shape:
        raise ValueError(f"unexpected mean delta shape {mean_delta.shape}")
    if gradient.shape != (EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM):
        raise ValueError(f"unexpected covariance gradient shape {gradient.shape}")
    if not all(np.isfinite(value).all() for value in (fake_mean, mean_delta, gradient)):
        raise FloatingPointError("non-finite E46 direction statistics")

    # For the Gaussian FID derivative, G = I - A where A is the symmetric
    # optimal affine transport from fixed96 fake features to real features.
    transport = np.eye(EXPECTED_FEATURE_DIM, dtype=np.float64) - 0.5 * (
        gradient + gradient.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(transport)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("frozen Gaussian transport is not positive definite")
    inverse = (eigenvectors * (1.0 / eigenvalues)[None, :]) @ eigenvectors.T
    real_mean = fake_mean - mean_delta
    _STATE_CPU = {
        "fake_mean": torch.from_numpy(fake_mean.astype(np.float32)),
        "real_mean": torch.from_numpy(real_mean.astype(np.float32)),
        "inverse_transport": torch.from_numpy(inverse.astype(np.float32)),
    }
    _STATE_BY_DEVICE = {}
    _STATE_METADATA = {
        **metadata,
        "transport_eigenvalue_min": float(eigenvalues[0]),
        "transport_eigenvalue_max": float(eigenvalues[-1]),
        "transport_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "transport_nonpositive_eigenvalues": int((eigenvalues <= 0.0).sum()),
        "direction_rule": "unit(real_feature - inverse_OT(real_feature))",
    }
    return dict(_STATE_METADATA)


def configured_metadata() -> dict[str, object]:
    if _STATE_METADATA is None:
        raise RuntimeError("configure_gaussian_ot_direction must be called first")
    return dict(_STATE_METADATA)


def _state_for(device: torch.device) -> dict[str, torch.Tensor]:
    if _STATE_CPU is None:
        raise RuntimeError("configure_gaussian_ot_direction must be called first")
    key = str(device)
    if key not in _STATE_BY_DEVICE:
        _STATE_BY_DEVICE[key] = {
            name: value.to(device=device, dtype=torch.float32)
            for name, value in _STATE_CPU.items()
        }
    return _STATE_BY_DEVICE[key]


def gaussian_ot_unit_direction(real_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fake-to-real OT unit directions and fake-side paired features."""
    if real_features.ndim != 2 or real_features.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"real_features must be [B,{EXPECTED_FEATURE_DIM}], got {real_features.shape}"
        )
    state = _state_for(real_features.device)
    centered = real_features.float() - state["real_mean"][None, :]
    fake_partner = state["fake_mean"][None, :] + centered @ state[
        "inverse_transport"
    ].T
    displacement = real_features.float() - fake_partner
    norm = displacement.norm(dim=1, keepdim=True)
    unit = displacement / norm.clamp_min(1e-12)
    unit = torch.where(norm > 1e-12, unit, torch.zeros_like(unit))
    if not torch.isfinite(unit).all() or not torch.isfinite(fake_partner).all():
        raise FloatingPointError("non-finite E46 transport direction")
    return unit, fake_partner


def _rescale_map_mean(values: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or values.shape[1:] != (1, 16, 16):
        raise ValueError(f"values must be [B,1,16,16], got {values.shape}")
    if scalar.shape != (values.shape[0],):
        raise ValueError(f"scalar must be [B], got {scalar.shape}")
    mean = values.float().mean(dim=(2, 3), keepdim=True)
    scale = torch.where(
        mean > 1e-20,
        scalar.float()[:, None, None, None] / mean,
        torch.zeros_like(mean),
    )
    return values.float() * scale


@torch.no_grad()
def gaussian_ot_direction_relative_endpoint_component_gain_maps(
    target: torch.Tensor,
    x_base: torch.Tensor,
    x_native: torch.Tensor,
    inception: torch.nn.Module,
    lpips_metric: torch.nn.Module,
    image_range: str,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Replace E31's paired Inception gain with Gaussian-OT directional gain."""
    components, diagnostics = relative_endpoint_component_gain_maps(
        target, x_base, x_native, inception, lpips_metric, image_range
    )
    batch = int(target.shape[0])
    target_01 = to_zero_one(target, image_range).clamp(0.0, 1.0)
    base_01 = to_zero_one(x_base, image_range).clamp(0.0, 1.0)
    native_01 = to_zero_one(x_native, image_range).clamp(0.0, 1.0)
    triplet = _inception_mixed7c(
        inception, torch.cat([target_01, base_01, native_01], dim=0) * 255.0
    )
    target_map, base_map, native_map = triplet.split(batch, dim=0)
    target_features = torch.flatten(inception.AvgPool(target_map.float()), 1)
    base_features = torch.flatten(inception.AvgPool(base_map.float()), 1)
    native_features = torch.flatten(inception.AvgPool(native_map.float()), 1)
    unit, _fake_partner = gaussian_ot_unit_direction(target_features)

    displacement = target_features - _fake_partner
    displacement_energy = displacement.square().sum(dim=1).clamp_min(1e-12)
    base_progress = (
        (base_features - _fake_partner) * displacement
    ).sum(dim=1) / displacement_energy
    native_progress = (
        (native_features - _fake_partner) * displacement
    ).sum(dim=1) / displacement_energy
    # The signed cost is minus fractional progress along the fixed Gaussian OT
    # path.  Therefore base_cost - native_cost is positive exactly when the
    # all-2D endpoint moves farther toward the real-side transport partner.
    base_cost = -base_progress
    native_cost = -native_progress
    base_shape, _ = _mass_preserving_inception_error_map(
        target_map, base_map, inception
    )
    native_shape, _ = _mass_preserving_inception_error_map(
        target_map, native_map, inception
    )
    spatial_energy = base_shape + native_shape
    base_ot_map = _rescale_map_mean(spatial_energy, base_cost)
    native_ot_map = _rescale_map_mean(spatial_energy, native_cost)
    progress_gain = base_ot_map - native_ot_map
    if not torch.isfinite(progress_gain).all():
        raise FloatingPointError("non-finite E46 OT-progress Inception gain")

    components["inception"] = progress_gain
    diagnostics["inception"] = {
        "base_loss": base_cost,
        "native_loss": native_cost,
        "base_map_mean_error": (
            base_ot_map.mean(dim=(1, 2, 3)) - base_cost
        ).abs(),
        "native_map_mean_error": (
            native_ot_map.mean(dim=(1, 2, 3)) - native_cost
        ).abs(),
    }
    return components, diagnostics
