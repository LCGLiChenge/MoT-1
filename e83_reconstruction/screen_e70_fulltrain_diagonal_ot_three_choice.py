#!/usr/bin/env python3
"""Staged E70 screen of E47 gain using a full-train direction and three K choices."""

from __future__ import annotations

import json
from pathlib import Path

import e46_gaussian_ot_direction_gain as ot
import screen_e24_endpoint_composite_gain as shared
from e04_distilled_router import file_sha256
from e31_pareto_relative_endpoint import E31_COMPONENT_WEIGHTS
from e42_amplitude_power_gain import (
    combine_e42_numpy,
    combine_e42_torch,
    set_amplitude_power,
)


ALPHA = 0.25
SHRINK = 1.0
CANDIDATE_TOKENS = (64, 96, 128)
ALGORITHM = "E47_three_choice_with_fulltrain_diagonal_OT_direction"
DIRECTION_FORMAT = "e70_fulltrain_diagonal_gaussian_ot_direction_v1"
DIRECTION_SHA256 = (
    "4573d32687d3f087b5384817043a1b893a4c144d7b2748530795aff4ba75ace0"
)
CALIBRATION_SHA256 = (
    "6c25e665e6dba7f348d9b6b6f47ae75382f4eddd6e7f2ade1412f9d6535ec050"
)
BASE_CHECKPOINT_SHA256 = (
    "86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446"
)
STAGES: dict[str, tuple[str, str]] = {
    "development": (
        "e70_fulltrain_diagonal_ot_three_choice_development",
        "51467104932c203cc1b43cc5cd209c7706a16932c0078049ed0258c5bddd0d66",
    ),
    "fresh_confirmation": (
        "e70_fulltrain_diagonal_ot_three_choice_confirmation",
        "1b2f23e567b09a7a60261def1405483d8f7887993d7dcbcf08032a86627597d6",
    ),
}


def validate_args(args) -> dict | None:
    candidate_contract = {
        "target_tokens": 96,
        "min_tokens": CANDIDATE_TOKENS[0],
        "max_tokens": CANDIDATE_TOKENS[-1],
        "quantize_step": 32,
        "allocation_shrink": SHRINK,
        "fid_feature": 2048,
        "mixed_precision": "bf16",
        "seed": 0,
    }
    for name, expected in candidate_contract.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"E70 requires --{name.replace('_', '-')}={expected}"
            )
    if file_sha256(args.e70_direction_npz) != DIRECTION_SHA256:
        raise ValueError("E70 full-train diagonal direction changed")
    if file_sha256(args.ckpt) != BASE_CHECKPOINT_SHA256:
        raise ValueError("E70 base checkpoint changed")

    if args.e70_smoke:
        if args.num_calibration_images > 8 or args.num_evaluation_images > 8:
            raise ValueError("E70 smoke permits at most eight images per split")
        return None

    exact = {
        "calibration_expected_split": "calibration",
        "num_calibration_images": 10000,
        "num_evaluation_images": 2000,
        "evaluation_offset": 0,
        "calibration_batch_size": 2,
        "batch_size": 2,
        "price_iterations": 80,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"formal E70 requires --{name.replace('_', '-')}={expected}"
            )
    if file_sha256(args.calibration_index_manifest) != CALIBRATION_SHA256:
        raise ValueError("E70 calibration manifest changed")
    if args.e70_stage not in STAGES:
        raise ValueError("formal E70 requires development or fresh_confirmation")
    expected_split, expected_hash = STAGES[args.e70_stage]
    if args.evaluation_expected_split != expected_split:
        raise ValueError(
            f"E70 {args.e70_stage} requires split {expected_split}"
        )
    if file_sha256(args.evaluation_index_manifest) != expected_hash:
        raise ValueError(f"E70 {args.e70_stage} manifest changed")

    prior = None
    if args.e70_stage == "development":
        if args.e70_development_json:
            raise ValueError("development cannot receive its own result")
    else:
        if not args.e70_development_json:
            raise ValueError("fresh confirmation requires development JSON")
        prior_path = Path(args.e70_development_json)
        prior = json.loads(prior_path.read_text())
        if (
            prior.get("format") != "e70_fulltrain_diagonal_ot_three_choice_screen_v1"
            or prior.get("e70_stage") != "development"
            or prior.get("screen_gates", {}).get("all_pass") is not True
        ):
            raise ValueError("fresh E70 confirmation requires passing development")
    return prior


def rewrite(
    path: Path,
    args,
    prior: dict | None,
    direction_metadata: dict,
) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("format") != "e24_endpoint_composite_gain_screen_v1":
        raise ValueError("shared E70 evaluator returned an unexpected format")
    artifact["metrics"]["e70_three_choice_dynamic"] = artifact["metrics"].pop(
        "endpoint_composite_dynamic"
    )
    allocation = artifact["evaluation_allocation"]
    changes = artifact["relative_changes_percent"]
    independent_contract = (
        artifact.get("selection_is_per_image_independent_after_train_calibration")
        is True
        and artifact.get("probe_reconstructions_per_image") == 1
        and artifact.get("candidate_k_search_reconstructions_per_image") == 0
        and artifact.get("candidate_k_search_loss_evaluations_per_image") == 0
        and artifact.get("test_batch_statistics_used_for_selection") is False
        and artifact.get("test_set_budget_rebalancing_used") is False
    )
    gates = {
        "mean_k_in_94p5_97p5": 94.5 <= float(allocation["mean"]) <= 97.5,
        "std_k_at_least_8": float(allocation["std"]) >= 8.0,
        "fid_reduction_at_least_0p5_percent": (
            float(changes["fid_reduction"]) >= 0.5
        ),
        "lpips_not_worse": float(changes["lpips_reduction"]) >= 0.0,
        "mse_not_worse": float(changes["mse_reduction"]) >= 0.0,
        "single_probe_independent_contract": independent_contract,
    }
    gates["all_pass"] = all(gates.values())
    artifact.update(
        {
            "format": "e70_fulltrain_diagonal_ot_three_choice_screen_v1",
            "experiment": "E70-fulltrain-direction-three-choice",
            "e70_stage": args.e70_stage,
            "algorithm": ALGORITHM,
            "candidate_tokens": list(CANDIDATE_TOKENS),
            "decision_rule": "argmax_K cumulative_E47_gain(K)-price*K",
            "amplitude_power": ALPHA,
            "allocation_shrink": SHRINK,
            "component_weights": E31_COMPONENT_WEIGHTS,
            "inception_attribution": (
                "mixed7c_energy_rescaled_to_final2048_signed_diagonal_OT_progress"
            ),
            "diagonal_gaussian_ot_direction": {
                "path": str(Path(args.e70_direction_npz).resolve()),
                "sha256": file_sha256(args.e70_direction_npz),
                "metadata": direction_metadata,
                "population_statistics_frozen_complete_train": True,
                "population_images": 1_281_167,
                "covariance_off_diagonal_parameters": 0,
            },
            "screen_gates": gates,
            "smoke": bool(args.e70_smoke),
            "development_artifact": None
            if prior is None
            else {
                "path": str(Path(args.e70_development_json).resolve()),
                "sha256": file_sha256(args.e70_development_json),
            },
            "train_only_sanity_not_out_of_training_generalization": True,
            "validation_used_to_fit_or_select_e70": False,
            "evaluation_manifest_was_fresh_before_e70": not args.e70_smoke,
        }
    )
    artifact["evaluation_source"]["fresh_confirmation"] = (
        args.e70_stage == "fresh_confirmation" and not args.e70_smoke
    )
    artifact["evaluation_source"]["already_consumed_development_evidence"] = False
    artifact["runtime_args"]["e70_stage"] = args.e70_stage
    artifact["runtime_args"]["e70_smoke"] = bool(args.e70_smoke)
    required_false = (
        "test_batch_statistics_used_for_selection",
        "test_set_budget_rebalancing_used",
        "validation_statistics_used",
        "validation_used_to_fit_or_select_e70",
        "features_saved_to_disk",
        "npz_written",
        "reconstructions_saved",
        "stats_pt_written",
    )
    if any(artifact.get(name) is not False for name in required_false):
        raise AssertionError("E70 screen violates the JSON-only independent contract")
    observed = set(int(value) for value in allocation["histogram"])
    if not observed.issubset(CANDIDATE_TOKENS):
        raise AssertionError(f"E70 emitted invalid K values: {sorted(observed)}")
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> None:
    parser = shared.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--e70-stage",
        choices=("smoke", "development", "fresh_confirmation"),
        required=True,
    )
    parser.add_argument("--e70-direction-npz", required=True)
    parser.add_argument("--e70-development-json")
    parser.add_argument("--e70-smoke", action="store_true")
    args = parser.parse_args()
    prior = validate_args(args)

    ot.EXPECTED_DIRECTION_FORMAT = DIRECTION_FORMAT
    direction_metadata = ot.configure_gaussian_ot_direction(args.e70_direction_npz)
    set_amplitude_power(ALPHA)
    shared.endpoint_component_gain_maps = (
        ot.gaussian_ot_direction_relative_endpoint_component_gain_maps
    )
    shared.combine_components_numpy = combine_e42_numpy
    shared.combine_components_torch = combine_e42_torch
    shared.COMPONENT_WEIGHTS = E31_COMPONENT_WEIGHTS
    shared.main(args)
    artifact = rewrite(Path(args.output_json), args, prior, direction_metadata)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "stage": artifact["e70_stage"],
                "allocation": artifact["evaluation_allocation"],
                "changes": artifact["relative_changes_percent"],
                "gates": artifact["screen_gates"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
