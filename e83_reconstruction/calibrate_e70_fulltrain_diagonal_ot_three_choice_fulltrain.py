#!/usr/bin/env python3
"""Calibrate the frozen E70 three-choice price on complete ImageNet-train."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import calibrate_e66_e47_three_choice_fulltrain as legacy
from e04_distilled_router import add_base_model_arguments, file_sha256
from e31_pareto_relative_endpoint import E31_COMPONENT_WEIGHTS
from screen_e70_fulltrain_diagonal_ot_three_choice import (
    ALGORITHM,
    ALPHA,
    BASE_CHECKPOINT_SHA256,
    CANDIDATE_TOKENS,
    DIRECTION_FORMAT,
    DIRECTION_SHA256,
)


FORMAT = "e70_fulltrain_diagonal_ot_three_choice_fulltrain_price_v1"
CONFIRMATION_FORMAT = "e70_fulltrain_diagonal_ot_three_choice_screen_v1"
CONFIRMATION_SHA256 = (
    "5195365d35884b02ea4b7639176b2e42b1df42ae10dc28c103b19f490ad44b74"
)
FORMAL_TRAIN_IMAGES = 1_281_167
FORMAL_WORLD_SIZE = 4
FORMAL_BATCH_SIZE = 8


def load_e70_confirmation(
    path: Path, smoke: bool
) -> tuple[dict, dict[str, float]]:
    if not smoke and file_sha256(path) != CONFIRMATION_SHA256:
        raise ValueError("E70 confirmation artifact changed")
    artifact = json.loads(path.read_text())
    if artifact.get("format") != CONFIRMATION_FORMAT:
        raise ValueError("full-train calibration requires an E70 screen artifact")
    if not smoke and (
        artifact.get("e70_stage") != "fresh_confirmation"
        or artifact.get("screen_gates", {}).get("all_pass") is not True
    ):
        raise ValueError("full-train calibration requires passing E70 confirmation")
    if artifact.get("algorithm") != ALGORITHM:
        raise ValueError("E70 algorithm changed")
    if artifact.get("candidate_tokens") != list(CANDIDATE_TOKENS):
        raise ValueError("E70 candidate tokens changed")
    if float(artifact.get("amplitude_power", -1.0)) != ALPHA:
        raise ValueError("E70 amplitude power changed")
    if artifact.get("component_weights") != E31_COMPONENT_WEIGHTS:
        raise ValueError("E70 component weights changed")
    direction = artifact.get("diagonal_gaussian_ot_direction")
    if (
        not isinstance(direction, dict)
        or direction.get("sha256") != DIRECTION_SHA256
        or direction.get("population_statistics_frozen_complete_train") is not True
        or int(direction.get("population_images", -1)) != FORMAL_TRAIN_IMAGES
    ):
        raise ValueError("E70 confirmation direction changed")
    scales = artifact.get("component_scales_train_only")
    if not isinstance(scales, dict) or set(scales) != {
        "inception",
        "lpips",
        "pixel_mse",
    }:
        raise ValueError("E70 confirmation has invalid component scales")
    scales = {name: float(value) for name, value in scales.items()}
    if not all(np.isfinite(value) and value > 0.0 for value in scales.values()):
        raise ValueError("E70 scales must be finite and positive")
    return artifact, scales


def configure_reused_core() -> None:
    """Freeze E70 constants in the already-audited E66 full-train core."""
    legacy.FORMAT = FORMAT
    legacy.CONFIRMATION_FORMAT = CONFIRMATION_FORMAT
    legacy.CONFIRMATION_SHA256 = CONFIRMATION_SHA256
    legacy.FORMAL_TRAIN_IMAGES = FORMAL_TRAIN_IMAGES
    legacy.FORMAL_WORLD_SIZE = FORMAL_WORLD_SIZE
    legacy.FORMAL_BATCH_SIZE = FORMAL_BATCH_SIZE
    legacy.ALGORITHM = ALGORITHM
    legacy.ALPHA = ALPHA
    legacy.BASE_CHECKPOINT_SHA256 = BASE_CHECKPOINT_SHA256
    legacy.CANDIDATE_TOKENS = CANDIDATE_TOKENS
    legacy.DIRECTION_FORMAT = DIRECTION_FORMAT
    legacy.DIRECTION_SHA256 = DIRECTION_SHA256
    legacy.load_confirmation = load_e70_confirmation


def rewrite_e70_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("format") != FORMAT or artifact.get("status") != "completed":
        raise ValueError("reused full-train core returned an unexpected artifact")
    runtime_args = artifact.get("runtime_args")
    if not isinstance(runtime_args, dict) or "e66_direction_npz" not in runtime_args:
        raise ValueError("E70 full-train artifact lacks direction runtime metadata")
    runtime_args["e70_direction_npz"] = runtime_args.pop("e66_direction_npz")
    validation_flag = artifact.pop("validation_used_to_fit_or_select_e66", None)
    if validation_flag is not False:
        raise ValueError("reused core lost its no-validation assertion")
    direction = artifact.get("diagonal_gaussian_ot_direction")
    if not isinstance(direction, dict):
        raise ValueError("E70 full-train artifact lacks direction metadata")
    metadata = direction.get("metadata")
    if (
        direction.get("sha256") != DIRECTION_SHA256
        or not isinstance(metadata, dict)
        or metadata.get("format") != DIRECTION_FORMAT
        or metadata.get("full_train_coverage_exact") is not True
        or int(metadata.get("num_images", -1)) != FORMAL_TRAIN_IMAGES
        or metadata.get("validation_statistics_used") is not False
    ):
        raise ValueError("E70 full-train artifact direction contract failed")
    direction.update(
        {
            "population_statistics_frozen_complete_train": True,
            "population_images": FORMAL_TRAIN_IMAGES,
            "covariance_off_diagonal_parameters": 0,
        }
    )
    artifact.update(
        {
            "experiment": "E70-fulltrain-direction-three-choice-fulltrain-price",
            "validation_used_to_fit_or_select_e70": False,
            "fulltrain_direction_fitted_on_all_1_281_167_train_images": True,
            "implementation_core_reused": (
                "calibrate_e66_e47_three_choice_fulltrain.py"
            ),
            "implementation_core_behavior_changed": False,
        }
    )
    required_false = (
        "test_batch_statistics_used_for_selection",
        "test_set_budget_rebalancing_used",
        "validation_statistics_used",
        "validation_used_to_fit_or_select_e70",
        "prefix_values_saved_to_disk",
        "features_saved_to_disk",
        "npz_written",
        "stats_pt_written",
    )
    if any(artifact.get(name) is not False for name in required_false):
        raise AssertionError("E70 full-train price violates its frozen contract")
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-json", required=True)
    parser.add_argument(
        "--e70-direction-npz", dest="e66_direction_npz", required=True
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--ckpt",
        default=(
            "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/dynamic/"
            "e54/checkpoints/base_latest.pt"
        ),
    )
    parser.add_argument("--data-path", default="/var/tmp/heyefei_ImageNet/train")
    parser.add_argument("--num-images", type=int, default=FORMAL_TRAIN_IMAGES)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--price-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips-net", choices=("alex",), default="alex")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    add_base_model_arguments(parser)
    return parser


def main() -> None:
    configure_reused_core()
    args = build_parser().parse_args()
    legacy.main(args)
    if int(os.environ.get("RANK", "0")) == 0:
        artifact = rewrite_e70_artifact(Path(args.output_json))
        print(
            json.dumps(
                {
                    "status": artifact["status"],
                    "price": artifact["price"],
                    "allocation": artifact["allocation"],
                    "full_train_coverage_exact": artifact[
                        "full_train_coverage_exact"
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
