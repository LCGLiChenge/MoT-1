#!/usr/bin/env python3
"""Staged E66 screen of frozen E47 gain restricted to K=64/96/128."""

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
ALGORITHM = "E47_gain_restricted_to_64_96_128_by_train_price"
DIRECTION_FORMAT = "e47_diagonal_gaussian_ot_direction_v1"
DIRECTION_SHA256 = (
    "43a101bb7fb35fb7f4962adb4af6485fb5df283d106156c92d77672289499e9a"
)
CALIBRATION_SHA256 = (
    "6c25e665e6dba7f348d9b6b6f47ae75382f4eddd6e7f2ade1412f9d6535ec050"
)
BASE_CHECKPOINT_SHA256 = (
    "86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446"
)
STAGES: dict[str, tuple[str, str]] = {
    "development": (
        "e66_e47_three_choice_development",
        "f83ac5c7d6dca33d35dd408c57d8b0eaedb9f6f95ee209e1cb3fe2aa0ad2d700",
    ),
    "fresh_confirmation": (
        "e66_e47_three_choice_confirmation",
        "946e9bc67d799eef89e09b5f1a7ee4f8f7e3656ca12268bf45f29aa58d4a883b",
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
                f"E66 requires --{name.replace('_', '-')}={expected}"
            )
    if file_sha256(args.e66_direction_npz) != DIRECTION_SHA256:
        raise ValueError("E66 diagonal direction changed")
    if file_sha256(args.ckpt) != BASE_CHECKPOINT_SHA256:
        raise ValueError("E66 base checkpoint changed")

    if args.e66_smoke:
        if args.num_calibration_images > 8 or args.num_evaluation_images > 8:
            raise ValueError("E66 smoke permits at most eight images per split")
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
                f"formal E66 requires --{name.replace('_', '-')}={expected}"
            )
    if file_sha256(args.calibration_index_manifest) != CALIBRATION_SHA256:
        raise ValueError("E66 calibration manifest changed")
    if args.e66_stage not in STAGES:
        raise ValueError("formal E66 requires development or fresh_confirmation")
    expected_split, expected_hash = STAGES[args.e66_stage]
    if args.evaluation_expected_split != expected_split:
        raise ValueError(
            f"E66 {args.e66_stage} requires split {expected_split}"
        )
    if file_sha256(args.evaluation_index_manifest) != expected_hash:
        raise ValueError(f"E66 {args.e66_stage} manifest changed")

    prior = None
    if args.e66_stage == "development":
        if args.e66_development_json:
            raise ValueError("development cannot receive its own result")
    else:
        if not args.e66_development_json:
            raise ValueError("fresh confirmation requires development JSON")
        prior_path = Path(args.e66_development_json)
        prior = json.loads(prior_path.read_text())
        if (
            prior.get("format") != "e66_e47_three_choice_screen_v1"
            or prior.get("e66_stage") != "development"
            or prior.get("screen_gates", {}).get("all_pass") is not True
        ):
            raise ValueError("fresh E66 confirmation requires passing development")
    return prior


def rewrite(
    path: Path,
    args,
    prior: dict | None,
    direction_metadata: dict,
) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("format") != "e24_endpoint_composite_gain_screen_v1":
        raise ValueError("shared E66 evaluator returned an unexpected format")
    artifact["metrics"]["e66_e47_three_choice_dynamic"] = artifact["metrics"].pop(
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
            "format": "e66_e47_three_choice_screen_v1",
            "experiment": "E66-E47-three-choice",
            "e66_stage": args.e66_stage,
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
                "path": str(Path(args.e66_direction_npz).resolve()),
                "sha256": file_sha256(args.e66_direction_npz),
                "metadata": direction_metadata,
                "population_statistics_frozen_train_only": True,
                "covariance_off_diagonal_parameters": 0,
            },
            "screen_gates": gates,
            "smoke": bool(args.e66_smoke),
            "development_artifact": None
            if prior is None
            else {
                "path": str(Path(args.e66_development_json).resolve()),
                "sha256": file_sha256(args.e66_development_json),
            },
            "validation_used_to_fit_or_select_e66": False,
            "evaluation_manifest_was_fresh_before_e66": not args.e66_smoke,
        }
    )
    artifact["evaluation_source"]["fresh_confirmation"] = (
        args.e66_stage == "fresh_confirmation" and not args.e66_smoke
    )
    artifact["evaluation_source"]["already_consumed_development_evidence"] = False
    artifact["runtime_args"]["e66_stage"] = args.e66_stage
    artifact["runtime_args"]["e66_smoke"] = bool(args.e66_smoke)
    required_false = (
        "test_batch_statistics_used_for_selection",
        "test_set_budget_rebalancing_used",
        "validation_statistics_used",
        "validation_used_to_fit_or_select_e66",
        "features_saved_to_disk",
        "npz_written",
        "reconstructions_saved",
        "stats_pt_written",
    )
    if any(artifact.get(name) is not False for name in required_false):
        raise AssertionError("E66 screen violates the JSON-only independent contract")
    observed = set(int(value) for value in allocation["histogram"])
    if not observed.issubset(CANDIDATE_TOKENS):
        raise AssertionError(f"E66 emitted invalid K values: {sorted(observed)}")
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> None:
    parser = shared.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--e66-stage",
        choices=("smoke", "development", "fresh_confirmation"),
        required=True,
    )
    parser.add_argument("--e66-direction-npz", required=True)
    parser.add_argument("--e66-development-json")
    parser.add_argument("--e66-smoke", action="store_true")
    args = parser.parse_args()
    prior = validate_args(args)

    ot.EXPECTED_DIRECTION_FORMAT = DIRECTION_FORMAT
    direction_metadata = ot.configure_gaussian_ot_direction(args.e66_direction_npz)
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
                "stage": artifact["e66_stage"],
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
