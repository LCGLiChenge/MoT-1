#!/usr/bin/env python3
"""Pure utilities for batch-independent, single-pass refinement budgets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


SCORE_NAMES = (
    "base_l1",
    "latent_l1",
    "latent_l2",
    "router_logit",
    "base_latent",
    "base_latent_router",
)


@torch.no_grad()
def single_pass_score_maps(
    x_base: torch.Tensor,
    target: torch.Tensor,
    f_1d: torch.Tensor,
    f_2d: torch.Tensor,
    router_score: torch.Tensor,
    grid_hw: int = 16,
) -> dict[str, torch.Tensor]:
    """Return raw per-grid signals without candidate reconstructions or batch statistics."""
    if x_base.shape != target.shape:
        raise ValueError(
            f"x_base/target shape mismatch: {tuple(x_base.shape)} vs {tuple(target.shape)}"
        )
    if f_1d.shape != f_2d.shape:
        raise ValueError(
            f"f_1d/f_2d shape mismatch: {tuple(f_1d.shape)} vs {tuple(f_2d.shape)}"
        )
    if f_1d.ndim != 4 or f_1d.shape[-2:] != (grid_hw, grid_hw):
        raise ValueError(f"expected latent grid {grid_hw}x{grid_hw}, got {tuple(f_1d.shape)}")
    if router_score.shape != (f_1d.shape[0], 1, grid_hw, grid_hw):
        raise ValueError(
            "router_score must have shape "
            f"({f_1d.shape[0]}, 1, {grid_hw}, {grid_hw}), got {tuple(router_score.shape)}"
        )

    pixel_error = (x_base.detach().float() - target.detach().float()).abs().mean(
        dim=1, keepdim=True
    )
    base_l1 = F.adaptive_avg_pool2d(pixel_error, (grid_hw, grid_hw))
    latent_delta = f_2d.detach().float() - f_1d.detach().float()
    return {
        "base_l1": base_l1,
        "latent_l1": latent_delta.abs().mean(dim=1, keepdim=True),
        "latent_l2": latent_delta.square().mean(dim=1, keepdim=True),
        "router_logit": router_score.detach().float(),
    }


def fit_component_statistics(
    raw_scores: Mapping[str, np.ndarray],
    epsilon: float = 1e-12,
) -> dict[str, dict[str, float]]:
    """Fit global train-only normalization constants for fixed inference use."""
    result: dict[str, dict[str, float]] = {}
    for name in ("base_l1", "latent_l1", "latent_l2", "router_logit"):
        values = np.asarray(raw_scores[name], dtype=np.float64)
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


def compose_score_arrays(
    raw_scores: Mapping[str, np.ndarray],
    component_statistics: Mapping[str, Mapping[str, float]],
) -> dict[str, np.ndarray]:
    """Compose fixed score rules using only train-fitted scalar constants."""
    normalized: dict[str, np.ndarray] = {}
    for name in ("base_l1", "latent_l1", "latent_l2", "router_logit"):
        values = np.asarray(raw_scores[name], dtype=np.float64)
        stats = component_statistics[name]
        normalized[name] = (values - float(stats["mean"])) / float(stats["std"])
    return {
        "base_l1": np.asarray(raw_scores["base_l1"], dtype=np.float64),
        "latent_l1": np.asarray(raw_scores["latent_l1"], dtype=np.float64),
        "latent_l2": np.asarray(raw_scores["latent_l2"], dtype=np.float64),
        "router_logit": np.asarray(raw_scores["router_logit"], dtype=np.float64),
        "base_latent": normalized["base_l1"] + normalized["latent_l1"],
        "base_latent_router": (
            normalized["base_l1"]
            + normalized["latent_l1"]
            + 0.25 * normalized["router_logit"]
        ),
    }


def compose_score_tensors(
    raw_scores: Mapping[str, torch.Tensor],
    component_statistics: Mapping[str, Mapping[str, float]],
) -> dict[str, torch.Tensor]:
    """Torch inference counterpart of :func:`compose_score_arrays`."""
    normalized: dict[str, torch.Tensor] = {}
    for name in ("base_l1", "latent_l1", "latent_l2", "router_logit"):
        values = raw_scores[name].float()
        stats = component_statistics[name]
        normalized[name] = (values - float(stats["mean"])) / float(stats["std"])
    return {
        "base_l1": raw_scores["base_l1"].float(),
        "latent_l1": raw_scores["latent_l1"].float(),
        "latent_l2": raw_scores["latent_l2"].float(),
        "router_logit": raw_scores["router_logit"].float(),
        "base_latent": normalized["base_l1"] + normalized["latent_l1"],
        "base_latent_router": (
            normalized["base_l1"]
            + normalized["latent_l1"]
            + 0.25 * normalized["router_logit"]
        ),
    }


def token_counts_from_scores_numpy(
    scores: np.ndarray,
    threshold: float,
    min_tokens: int,
    max_tokens: int,
    quantize_step: int = 1,
) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != 256:
        raise ValueError(f"scores must have shape [N,256], got {values.shape}")
    counts = (values > float(threshold)).sum(axis=1).astype(np.int64)
    if quantize_step > 1:
        counts = np.rint(counts / float(quantize_step)).astype(np.int64) * quantize_step
    return np.clip(counts, int(min_tokens), int(max_tokens))


def token_counts_from_scores_torch(
    scores: torch.Tensor,
    threshold: float,
    min_tokens: int,
    max_tokens: int,
    quantize_step: int = 1,
) -> torch.Tensor:
    if scores.ndim != 4 or scores.shape[1:] != (1, 16, 16):
        raise ValueError(f"scores must have shape [B,1,16,16], got {tuple(scores.shape)}")
    counts = (scores.float().flatten(1) > float(threshold)).sum(dim=1)
    if quantize_step > 1:
        counts = torch.round(counts.float() / float(quantize_step)).long() * quantize_step
    return counts.clamp(min=int(min_tokens), max=int(max_tokens)).long()


def calibrate_threshold(
    scores: np.ndarray,
    target_tokens: float,
    min_tokens: int,
    max_tokens: int,
    quantize_step: int = 1,
    iterations: int = 80,
) -> tuple[float, np.ndarray]:
    """Freeze one scalar threshold using train data only."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 256:
        raise ValueError(f"scores must have shape [N,256], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    if not (min_tokens <= target_tokens <= max_tokens):
        raise ValueError("target_tokens must lie inside [min_tokens,max_tokens]")

    lo = float(np.nextafter(values.min(), -np.inf))
    hi = float(np.nextafter(values.max(), np.inf))
    candidates: list[tuple[float, np.ndarray]] = []
    for _ in range(max(1, int(iterations))):
        mid = (lo + hi) * 0.5
        counts = token_counts_from_scores_numpy(
            values, mid, min_tokens, max_tokens, quantize_step
        )
        candidates.append((mid, counts))
        if float(counts.mean()) > float(target_tokens):
            lo = mid
        else:
            hi = mid
    for value in (lo, hi):
        candidates.append(
            (
                value,
                token_counts_from_scores_numpy(
                    values, value, min_tokens, max_tokens, quantize_step
                ),
            )
        )
    return min(
        candidates,
        key=lambda item: (abs(float(item[1].mean()) - float(target_tokens)), item[0]),
    )


def token_summary(tokens: np.ndarray, target_tokens: float) -> dict[str, object]:
    values = np.asarray(tokens, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "mean_delta_from_target": float(values.mean() - float(target_tokens)),
        "std": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
        "histogram": {
            str(int(token)): int(count) for token, count in zip(unique, counts, strict=True)
        },
    }
