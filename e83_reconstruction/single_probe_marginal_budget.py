#!/usr/bin/env python3
"""Single-probe linearized rate-distortion utilities for independent budgets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from eval_balanced_gradient_spatial_oracle import differentiable_inception_2048
from eval_oracle_dynamic_budget import to_lpips_range, to_zero_one


COMPONENT_NAMES = ("inception", "lpips", "mse")
COMPONENT_WEIGHTS = {"inception": 1.0, "lpips": 2.0, "mse": 4.0}


def marginal_component_maps(
    f_1d: torch.Tensor,
    f_2d: torch.Tensor,
    fixed_mask: torch.Tensor,
    target: torch.Tensor,
    target_inception_features: torch.Tensor,
    decoder: torch.nn.Module,
    inception: torch.nn.Module,
    lpips_metric: torch.nn.Module,
    image_range: str,
    autocast_type: torch.dtype,
    autocast_enabled: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiate one fixed-budget reconstruction loss with respect to its mask."""
    expected_mask = (f_1d.shape[0], 1, f_1d.shape[2], f_1d.shape[3])
    if fixed_mask.shape != expected_mask:
        raise ValueError(f"fixed_mask must have shape {expected_mask}, got {fixed_mask.shape}")
    if f_1d.shape != f_2d.shape:
        raise ValueError("f_1d and f_2d must have identical shapes")
    with torch.enable_grad():
        probe = fixed_mask.detach().float().requires_grad_(True)
        with torch.autocast(
            device_type=f_1d.device.type,
            dtype=autocast_type,
            enabled=autocast_enabled,
        ):
            prediction = decoder(
                (1.0 - probe) * f_1d.detach() + probe * f_2d.detach()
            )
        prediction_01 = to_zero_one(
            prediction.float(), image_range
        ).clamp(0.0, 1.0)
        target_01 = to_zero_one(target.detach().float(), image_range).clamp(0.0, 1.0)
        prediction_features = differentiable_inception_2048(
            inception, prediction_01 * 255.0
        )
        losses = {
            "inception": (
                prediction_features - target_inception_features.detach().float()
            ).square().mean(dim=1),
            "lpips": lpips_metric(
                to_lpips_range(prediction.float(), image_range),
                to_lpips_range(target.detach().float(), image_range),
            ).reshape(f_1d.shape[0], -1).mean(dim=1),
            "mse": (prediction_01 - target_01).square().flatten(1).mean(dim=1),
        }
        gradients: dict[str, torch.Tensor] = {}
        for index, name in enumerate(COMPONENT_NAMES):
            gradients[name] = -torch.autograd.grad(
                losses[name].sum(),
                probe,
                retain_graph=index + 1 < len(COMPONENT_NAMES),
                only_inputs=True,
            )[0].detach().float()
    detached_losses = {name: value.detach().float() for name, value in losses.items()}
    return gradients, prediction.detach(), detached_losses


def fit_component_scales(
    components: Mapping[str, np.ndarray],
    epsilon: float = 1e-12,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in COMPONENT_NAMES:
        values = np.asarray(components[name], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 256:
            raise ValueError(f"{name} must have shape [N,256], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        mean = float(values.mean())
        std = float(values.std())
        if std <= float(epsilon):
            raise ValueError(f"{name} has degenerate standard deviation {std}")
        result[name] = {"mean": mean, "std": std}
    return result


def combine_components_numpy(
    components: Mapping[str, np.ndarray],
    scales: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    result = None
    for name in COMPONENT_NAMES:
        values = np.asarray(components[name], dtype=np.float32)
        normalized = values / np.float32(scales[name]["std"])
        weighted = np.float32(COMPONENT_WEIGHTS[name]) * normalized
        result = weighted if result is None else result + weighted
    if result is None:
        raise AssertionError("no marginal components were combined")
    return result


def combine_components_torch(
    components: Mapping[str, torch.Tensor],
    scales: Mapping[str, Mapping[str, float]],
) -> torch.Tensor:
    result = None
    for name in COMPONENT_NAMES:
        normalized = components[name].float() / float(scales[name]["std"])
        weighted = float(COMPONENT_WEIGHTS[name]) * normalized
        result = weighted if result is None else result + weighted
    if result is None:
        raise AssertionError("no marginal components were combined")
    return result


def router_ranked_numpy(gains: np.ndarray, router_scores: np.ndarray) -> np.ndarray:
    gains = np.asarray(gains, dtype=np.float32)
    router_scores = np.asarray(router_scores, dtype=np.float32)
    if gains.shape != router_scores.shape or gains.ndim != 2 or gains.shape[1] != 256:
        raise ValueError("gains and router_scores must both have shape [N,256]")
    order = np.argsort(-router_scores, axis=1, kind="stable")
    return np.take_along_axis(gains, order, axis=1)


def router_ranked_torch(
    gains: torch.Tensor, router_scores: torch.Tensor
) -> torch.Tensor:
    if gains.shape != router_scores.shape or gains.ndim != 4:
        raise ValueError("gains and router_scores must have identical [B,1,16,16] shapes")
    flat_gain = gains.float().flatten(1)
    order = torch.argsort(router_scores.float().flatten(1), dim=1, descending=True)
    return torch.gather(flat_gain, 1, order)


def prefix_choices_numpy(
    ranked_gains: np.ndarray,
    price: float,
    min_tokens: int,
    max_tokens: int,
    token_step: int,
) -> np.ndarray:
    ranked = np.asarray(ranked_gains, dtype=np.float32)
    if ranked.ndim != 2 or ranked.shape[1] != 256:
        raise ValueError("ranked_gains must have shape [N,256]")
    candidates = np.arange(min_tokens, max_tokens + 1, token_step, dtype=np.int64)
    if candidates.size == 0 or candidates[-1] > 256:
        raise ValueError("invalid token candidate range")
    cumulative = np.cumsum(ranked, axis=1, dtype=np.float32)
    utilities = (
        cumulative[:, candidates - 1]
        - np.float32(price) * candidates.astype(np.float32)[None, :]
    )
    return candidates[np.argmax(utilities, axis=1)]


def prefix_choices_torch(
    ranked_gains: torch.Tensor,
    price: float,
    min_tokens: int,
    max_tokens: int,
    token_step: int,
) -> torch.Tensor:
    if ranked_gains.ndim != 2 or ranked_gains.shape[1] != 256:
        raise ValueError("ranked_gains must have shape [B,256]")
    candidates = torch.arange(
        min_tokens,
        max_tokens + 1,
        token_step,
        device=ranked_gains.device,
        dtype=torch.long,
    )
    cumulative = torch.cumsum(ranked_gains.float(), dim=1)
    utilities = cumulative[:, candidates - 1] - float(price) * candidates.float()[None, :]
    return candidates[utilities.argmax(dim=1)]


def calibrate_price(
    ranked_gains: np.ndarray,
    target_tokens: float,
    min_tokens: int,
    max_tokens: int,
    token_step: int,
    iterations: int = 80,
) -> tuple[float, np.ndarray]:
    values = np.asarray(ranked_gains, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ranked_gains contain non-finite values")
    lo = float(np.nextafter(values.min(), np.float32(-np.inf)))
    hi = float(np.nextafter(values.max(), np.float32(np.inf)))
    candidates: list[tuple[float, np.ndarray]] = []
    for _ in range(max(1, int(iterations))):
        mid = float(np.float32((lo + hi) * 0.5))
        choices = prefix_choices_numpy(
            values, mid, min_tokens, max_tokens, token_step
        )
        candidates.append((mid, choices))
        if float(choices.mean()) > float(target_tokens):
            lo = mid
        else:
            hi = mid
    for value in (lo, hi):
        candidates.append(
            (
                value,
                prefix_choices_numpy(
                    values, value, min_tokens, max_tokens, token_step
                ),
            )
        )
    return min(
        candidates,
        key=lambda item: (abs(float(item[1].mean()) - float(target_tokens)), item[0]),
    )
