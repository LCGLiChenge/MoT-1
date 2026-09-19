#!/usr/bin/env python3
"""Evaluate E83 by reusing E82's frozen per-image reconstruction core."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import eval_e82_e72_train90_val50k as core
from calibrate_e82_e72_train90_fulltrain import REFERENCE_PRICE
from calibrate_e83_e72_train87p5168_fulltrain import (
    FORMAT as PRICE_FORMAT,
    TARGET_TOKENS,
)
from e04_distilled_router import file_sha256


FORMAT = "e83_e72_train87p5168_val50k_eval_v1"
EXPERIMENT = "E83 E72 second-pass rate-drift compensation"
METRIC_KEY = "e83_train87p5168_dynamic"


def load_price(path: Path, expected_sha256: str) -> tuple[dict, float]:
    if len(expected_sha256) != 64 or file_sha256(path) != expected_sha256:
        raise ValueError("E83 price artifact SHA-256 changed")
    artifact = json.loads(path.read_text())
    allocation = artifact.get("allocation", {})
    reference = artifact.get("reference_reproduction", {})
    derivation = artifact.get("target_derivation", {})
    if (
        artifact.get("format") != PRICE_FORMAT
        or artifact.get("status") != "completed"
        or artifact.get("algorithm")
        != "E72_binary_average_marginal_fixed_fulltrain_price"
        or artifact.get("candidate_tokens") != list(core.CANDIDATES)
        or float(artifact.get("target_train_mean_tokens", np.nan))
        != TARGET_TOKENS
        or int(allocation.get("count", -1)) != 1_281_167
        or abs(float(allocation.get("mean", np.nan)) - TARGET_TOKENS) > 1e-3
        or artifact.get("complete_imagenet_train_coverage") is not True
        or int(artifact.get("train_images", -1)) != 1_281_167
        or artifact.get("validation_images_used_for_direct_price_fit") != 0
        or artifact.get("validation_rate_statistic_used_to_choose_train_target")
        is not True
        or artifact.get("reported_reconstruction_metrics_used_to_choose_price")
        is not False
        or artifact.get("selection_is_per_image_independent_after_price_frozen")
        is not True
        or artifact.get("selection_requires_batch_after_price_frozen") is not False
        or derivation.get("used_only_aggregate_validation_token_mean") is not True
        or derivation.get("used_validation_fid") is not False
        or derivation.get("used_validation_distortion_metrics") is not False
        or reference.get("exact_price_reproduced") is not True
        or float(reference.get("price", np.nan)) != REFERENCE_PRICE
        or artifact.get("npz_written") is not False
        or artifact.get("stats_pt_written") is not False
    ):
        raise ValueError("E83 price artifact contract failed")
    price = float(artifact.get("price", np.nan))
    if not np.isfinite(price):
        raise ValueError("E83 price is not finite")
    return artifact, price


def transform_result(path: Path) -> dict:
    result = json.loads(path.read_text())
    dynamic = result["metrics"].pop("e82_train90_dynamic")
    result["metrics"][METRIC_KEY] = dynamic
    result["format"] = FORMAT
    result["experiment"] = EXPERIMENT
    result["train_target_tokens"] = TARGET_TOKENS
    result.pop("target_90_motivated_by_previously_observed_val_rate_drift", None)
    result["target_87p5168_derived_from_e82_val_rate_error"] = True
    result["validation_rate_statistic_used_to_choose_train_target"] = True
    result["validation_fid_used_to_choose_price"] = False
    result["validation_distortion_metrics_used_to_choose_price"] = False
    result["validation_used_to_fit_price_directly"] = False
    result["validation_guided_hyperparameter_tuning"] = True
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_runtime_args():
    return core.build_parser().parse_args()


def main() -> None:
    args = parse_runtime_args()
    core.load_price = load_price
    core.FORMAT = FORMAT
    core.TRAIN_TARGET_TOKENS = TARGET_TOKENS
    core.main(args)
    if int(os.environ.get("RANK", "0")) == 0:
        result = transform_result(Path(args.output_json))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stage": result["stage"],
                    "allocation": result["allocation"],
                    "metrics": result["metrics"],
                    "primary_success_gate": result["primary_success_gate"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
