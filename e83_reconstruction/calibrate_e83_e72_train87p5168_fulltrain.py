#!/usr/bin/env python3
"""Freeze E72's binary price at mean K=87.5168 on complete ImageNet-train."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from calibrate_e82_e72_train90_fulltrain import (
    REFERENCE_PRICE,
    REFERENCE_TARGET_TOKENS,
    allocation,
    load_scores,
)
from generate_e74_binary_teacher_labels import (
    CANDIDATES,
    FORMAL_TRAIN_IMAGES,
    calibrate_binary_score_price,
)


FORMAT = "e83_e72_train87p5168_fulltrain_price_v1"
TARGET_TOKENS = 87.5168
E82_TRAIN_TARGET = 90.0
E82_VAL_ACTUAL_MEAN = 98.4832
DESIRED_VAL_MEAN = 96.0


def main(args: argparse.Namespace) -> None:
    output = Path(args.output_json)
    if not output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("E83 output JSON must remain under dynamic/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if float(args.target_tokens) != TARGET_TOKENS:
        raise ValueError(f"formal E83 freezes train target at {TARGET_TOKENS}")

    started = time.time()
    scores, sources = load_scores([Path(value) for value in args.label_npz])
    reference_price, reference_tokens = calibrate_binary_score_price(
        scores, target_tokens=REFERENCE_TARGET_TOKENS
    )
    if reference_price != REFERENCE_PRICE:
        raise AssertionError("E83 loader failed to reproduce the train-96 price")
    price, tokens = calibrate_binary_score_price(
        scores, target_tokens=TARGET_TOKENS
    )
    result = {
        "format": FORMAT,
        "status": "completed",
        "experiment": "E83 E72 second-pass rate-drift compensation",
        "algorithm": "E72_binary_average_marginal_fixed_fulltrain_price",
        "candidate_tokens": list(CANDIDATES),
        "score": "float32((G128-G64)/64)",
        "decision_rule": "K128 iff score > price; otherwise K64",
        "target_train_mean_tokens": TARGET_TOKENS,
        "price": float(price),
        "allocation": allocation(tokens, TARGET_TOKENS),
        "target_derivation": {
            "formula": "e82_train_target - (e82_val_mean - desired_val_mean)",
            "e82_train_target": E82_TRAIN_TARGET,
            "e82_val_actual_mean": E82_VAL_ACTUAL_MEAN,
            "desired_val_mean": DESIRED_VAL_MEAN,
            "result": TARGET_TOKENS,
            "used_only_aggregate_validation_token_mean": True,
            "used_validation_fid": False,
            "used_validation_distortion_metrics": False,
            "validation_guided_hyperparameter_tuning": True,
        },
        "reference_reproduction": {
            "target_tokens": REFERENCE_TARGET_TOKENS,
            "price": float(reference_price),
            "allocation": allocation(reference_tokens, REFERENCE_TARGET_TOKENS),
            "expected_price": REFERENCE_PRICE,
            "exact_price_reproduced": True,
        },
        "score_summary": {
            "count": int(scores.size),
            "mean": float(scores.mean(dtype=np.float64)),
            "std": float(scores.std(dtype=np.float64)),
            "min": float(scores.min()),
            "max": float(scores.max()),
        },
        "price_sources": sources,
        "complete_imagenet_train_coverage": True,
        "train_images": FORMAL_TRAIN_IMAGES,
        "validation_images_used_for_direct_price_fit": 0,
        "validation_rate_statistic_used_to_choose_train_target": True,
        "reported_reconstruction_metrics_used_to_choose_price": False,
        "selection_is_per_image_independent_after_price_frozen": True,
        "selection_requires_batch_after_price_frozen": False,
        "features_saved": False,
        "reconstructions_saved": False,
        "npz_written": False,
        "stats_pt_written": False,
        "output_json_only": True,
        "runtime_seconds": time.time() - started,
        "runtime_args": vars(args),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(
        "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/dynamic/"
        "e74/teacher_fulltrain_v1"
    )
    parser.add_argument("--label-npz", action="append", default=None)
    parser.add_argument("--target-tokens", type=float, default=TARGET_TOKENS)
    parser.add_argument(
        "--output-json",
        default="results/e83_e72_train87p5168_v1/fulltrain_price.json",
    )
    args = parser.parse_args()
    if args.label_npz is None:
        args.label_npz = [
            str(root / "teacher_rank00.npz"),
            str(root / "teacher_rank01.npz"),
        ]
    return args


if __name__ == "__main__":
    main(build_parser())
