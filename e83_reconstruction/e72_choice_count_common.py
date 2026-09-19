#!/usr/bin/env python3
"""Frozen candidate sets and scalar-price utilities for E72."""

from __future__ import annotations

import numpy as np


CHOICE_SETS: dict[str, tuple[int, ...]] = {
    "choice2_gap64": (64, 128),
    "choice3_gap32": (64, 96, 128),
    "choice5_gap16": (64, 80, 96, 112, 128),
    "choice9_gap8": tuple(range(64, 129, 8)),
}
ALL_TOKENS = CHOICE_SETS["choice9_gap8"]
TARGET_TOKENS = 96.0


def validate_choice_sets() -> None:
    if tuple(CHOICE_SETS) != (
        "choice2_gap64",
        "choice3_gap32",
        "choice5_gap16",
        "choice9_gap8",
    ):
        raise AssertionError("E72 choice-set order changed")
    for name, candidates in CHOICE_SETS.items():
        if candidates[0] != 64 or candidates[-1] != 128:
            raise AssertionError(f"{name} changed its fixed K range")
        gaps = np.diff(np.asarray(candidates, dtype=np.int64))
        if gaps.size and (np.any(gaps != gaps[0]) or int(gaps[0]) % 8 != 0):
            raise AssertionError(f"{name} is not equally spaced by a multiple of 8")
        if not set(candidates).issubset(ALL_TOKENS):
            raise AssertionError(f"{name} is not nested in the nine-choice set")


def prefix_choices_numpy(
    prefixes: np.ndarray,
    candidates: tuple[int, ...],
    price: float,
) -> np.ndarray:
    values = np.asarray(prefixes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(candidates):
        raise ValueError(f"prefixes must have shape [N,{len(candidates)}]")
    candidate_array = np.asarray(candidates, dtype=np.float32)
    utility = values - np.float32(price) * candidate_array[None, :]
    return np.asarray(candidates, dtype=np.int64)[np.argmax(utility, axis=1)]


def calibrate_prefix_price(
    prefixes: np.ndarray,
    candidates: tuple[int, ...],
    target_tokens: float = TARGET_TOKENS,
    iterations: int = 80,
) -> tuple[float, np.ndarray]:
    values = np.asarray(prefixes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(candidates):
        raise ValueError(f"prefixes must have shape [N,{len(candidates)}]")
    if not np.isfinite(values).all():
        raise ValueError("prefixes contain NaN/Inf")
    candidate_array = np.asarray(candidates, dtype=np.float32)
    thresholds = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            thresholds.append(
                (values[:, right] - values[:, left])
                / (candidate_array[right] - candidate_array[left])
            )
    boundaries = np.concatenate(thresholds)
    span = max(float(boundaries.max() - boundaries.min()), 1.0)
    lo = float(boundaries.min() - span)
    hi = float(boundaries.max() + span)
    best: tuple[float, np.ndarray] | None = None

    def consider(price: float, choices: np.ndarray) -> None:
        nonlocal best
        key = (abs(float(choices.mean()) - target_tokens), float(price))
        if best is None:
            best = (float(price), choices.copy())
            return
        best_key = (
            abs(float(best[1].mean()) - target_tokens),
            float(best[0]),
        )
        if key < best_key:
            best = (float(price), choices.copy())

    for price in (lo, hi):
        consider(price, prefix_choices_numpy(values, candidates, price))
    for _ in range(max(1, int(iterations))):
        mid = float(np.float32((lo + hi) * 0.5))
        choices = prefix_choices_numpy(values, candidates, mid)
        consider(mid, choices)
        if float(choices.mean()) > target_tokens:
            lo = mid
        else:
            hi = mid
    if best is None:
        raise AssertionError("E72 price calibration produced no candidate")
    return best


def allocation_summary(tokens: np.ndarray) -> dict[str, object]:
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("allocation must be a non-empty vector")
    unique, counts = np.unique(values, return_counts=True)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "mean_delta_from_target": float(values.mean() - TARGET_TOKENS),
        "std": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
        "histogram": {
            str(int(token)): int(count) for token, count in zip(unique, counts)
        },
    }


validate_choice_sets()
