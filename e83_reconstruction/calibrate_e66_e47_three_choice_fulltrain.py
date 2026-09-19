#!/usr/bin/env python3
"""Calibrate the frozen E66 three-choice price on all ImageNet-train images."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import e46_gaussian_ot_direction_gain as ot
from e04_distilled_router import add_base_model_arguments, file_sha256
from e31_pareto_relative_endpoint import E31_COMPONENT_WEIGHTS
from e42_amplitude_power_gain import combine_e42_torch, set_amplitude_power
from eval_oracle_dynamic_budget import (
    EvalImageDataset,
    build_lpips_metric,
    build_model,
)
from eval_single_pass_budget import distributed_setup
from screen_native_gain_budget import model_batch
from screen_e66_e47_three_choice import (
    ALGORITHM,
    ALPHA,
    BASE_CHECKPOINT_SHA256,
    CANDIDATE_TOKENS,
    DIRECTION_FORMAT,
    DIRECTION_SHA256,
)
from single_probe_marginal_budget import router_ranked_torch


FORMAT = "e66_e47_three_choice_fulltrain_price_v1"
CONFIRMATION_FORMAT = "e66_e47_three_choice_screen_v1"
CONFIRMATION_SHA256 = (
    "a5c9981058a6f4537facde425a75082e3bf2359fbdd95cfee3e6fae4f6abada3"
)
FORMAL_TRAIN_IMAGES = 1_281_167
FORMAL_WORLD_SIZE = 4
FORMAL_BATCH_SIZE = 8


def prefix_choices_numpy(prefixes: np.ndarray, price: float) -> np.ndarray:
    values = np.asarray(prefixes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(CANDIDATE_TOKENS):
        raise ValueError("E66 prefixes must have shape [N,3]")
    candidates = np.asarray(CANDIDATE_TOKENS, dtype=np.float32)
    utility = values - np.float32(price) * candidates[None, :]
    return np.asarray(CANDIDATE_TOKENS, dtype=np.int64)[np.argmax(utility, axis=1)]


def calibrate_prefix_price(
    prefixes: np.ndarray,
    target_tokens: float = 96.0,
    iterations: int = 80,
) -> tuple[float, np.ndarray]:
    values = np.asarray(prefixes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(CANDIDATE_TOKENS):
        raise ValueError("E66 prefixes must have shape [N,3]")
    if not np.isfinite(values).all():
        raise ValueError("E66 prefixes contain NaN/Inf")
    candidates = np.asarray(CANDIDATE_TOKENS, dtype=np.float32)
    thresholds = []
    for left in range(len(CANDIDATE_TOKENS)):
        for right in range(left + 1, len(CANDIDATE_TOKENS)):
            thresholds.append(
                (values[:, right] - values[:, left])
                / (candidates[right] - candidates[left])
            )
    boundaries = np.concatenate(thresholds)
    span = max(float(boundaries.max() - boundaries.min()), 1.0)
    lo = float(boundaries.min() - span)
    hi = float(boundaries.max() + span)
    best: tuple[float, np.ndarray] | None = None

    def consider(price: float, choices: np.ndarray) -> None:
        nonlocal best
        key = (abs(float(choices.mean()) - float(target_tokens)), float(price))
        if best is None:
            best = (float(price), choices.copy())
            return
        best_key = (
            abs(float(best[1].mean()) - float(target_tokens)),
            float(best[0]),
        )
        if key < best_key:
            best = (float(price), choices.copy())

    for price in (lo, hi):
        consider(price, prefix_choices_numpy(values, price))
    for _ in range(max(1, int(iterations))):
        mid = float(np.float32((lo + hi) * 0.5))
        choices = prefix_choices_numpy(values, mid)
        consider(mid, choices)
        if float(choices.mean()) > float(target_tokens):
            lo = mid
        else:
            hi = mid
    if best is None:
        raise AssertionError("E66 price calibration produced no candidate")
    return best


def allocation_summary(tokens: np.ndarray) -> dict[str, object]:
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("allocation must be a non-empty vector")
    unique, counts = np.unique(values, return_counts=True)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "mean_delta_from_target": float(values.mean() - 96.0),
        "std": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
        "histogram": {
            str(int(token)): int(count) for token, count in zip(unique, counts)
        },
    }


def load_confirmation(path: Path, smoke: bool) -> tuple[dict, dict[str, float]]:
    if not smoke:
        if CONFIRMATION_SHA256 == "PENDING":
            raise ValueError("freeze E66 confirmation hash before full-train calibration")
        if file_sha256(path) != CONFIRMATION_SHA256:
            raise ValueError("E66 confirmation artifact changed")
    artifact = json.loads(path.read_text())
    if artifact.get("format") != CONFIRMATION_FORMAT:
        raise ValueError("full-train calibration requires an E66 screen artifact")
    if not smoke and (
        artifact.get("e66_stage") != "fresh_confirmation"
        or artifact.get("screen_gates", {}).get("all_pass") is not True
    ):
        raise ValueError("full-train calibration requires passing fresh confirmation")
    if artifact.get("algorithm") != ALGORITHM:
        raise ValueError("E66 algorithm changed")
    if artifact.get("candidate_tokens") != list(CANDIDATE_TOKENS):
        raise ValueError("E66 candidate tokens changed")
    if float(artifact.get("amplitude_power", -1.0)) != ALPHA:
        raise ValueError("E66 amplitude power changed")
    if artifact.get("component_weights") != E31_COMPONENT_WEIGHTS:
        raise ValueError("E66 component weights changed")
    scales = artifact.get("component_scales_train_only")
    if not isinstance(scales, dict) or set(scales) != {
        "inception",
        "lpips",
        "pixel_mse",
    }:
        raise ValueError("E66 confirmation has invalid component scales")
    scales = {name: float(value) for name, value in scales.items()}
    if not all(np.isfinite(value) and value > 0.0 for value in scales.values()):
        raise ValueError("E66 scales must be finite and positive")
    return artifact, scales


def validate_formal_args(args, world_size: int, dataset_size: int) -> None:
    if args.smoke:
        if args.num_images > 8:
            raise ValueError("E66 full-train smoke permits at most eight images")
        return
    exact = {
        "num_images": FORMAL_TRAIN_IMAGES,
        "batch_size": FORMAL_BATCH_SIZE,
        "mixed_precision": "bf16",
        "lpips_net": "alex",
        "seed": 0,
        "use_model_ema": True,
        "price_iterations": 80,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"formal E66 full-train requires --{name.replace('_', '-')}={expected}"
            )
    if world_size != FORMAL_WORLD_SIZE:
        raise ValueError(f"formal E66 full-train requires {FORMAL_WORLD_SIZE} GPUs")
    if dataset_size != FORMAL_TRAIN_IMAGES:
        raise ValueError(
            f"ImageNet-train has {dataset_size} images, expected {FORMAL_TRAIN_IMAGES}"
        )
    if file_sha256(args.ckpt) != BASE_CHECKPOINT_SHA256:
        raise ValueError("E66 full-train checkpoint changed")
    if file_sha256(args.e66_direction_npz) != DIRECTION_SHA256:
        raise ValueError("E66 full-train direction changed")


def main(args: argparse.Namespace) -> None:
    output = Path(args.output_json)
    distributed, rank, _local_rank, world_size, device = distributed_setup()
    is_main = rank == 0
    exists = torch.tensor(
        int(is_main and output.exists() and not args.overwrite),
        device=device,
        dtype=torch.long,
    )
    if distributed:
        dist.broadcast(exists, src=0)
    if int(exists.item()):
        raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")

    confirmation_path = Path(args.confirmation_json)
    confirmation, component_scales = load_confirmation(confirmation_path, args.smoke)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    started = time.time()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, model_metadata = build_model(args, device, checkpoint)
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)

    mot_root = str(Path(args.mot_root).resolve())
    if mot_root not in sys.path:
        sys.path.insert(0, mot_root)
    from train_titok_llamagen_recon import autocast_dtype

    llamagen_root = model_metadata["llamagen_root"]
    if llamagen_root not in sys.path:
        sys.path.insert(0, llamagen_root)
    from dataset.augmentation import center_crop_arr

    dataset = EvalImageDataset(args.data_path, args.image_size, center_crop_arr)
    validate_formal_args(args, world_size, len(dataset))
    count = min(args.num_images, len(dataset))
    shard_start = count * rank // world_size
    shard_end = count * (rank + 1) // world_size
    loader = DataLoader(
        Subset(dataset, range(shard_start, shard_end)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    lpips_metric = build_lpips_metric(args, device)
    fid_extractor = (
        FrechetInceptionDistance(feature=2048, normalize=False)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    ot.EXPECTED_DIRECTION_FORMAT = DIRECTION_FORMAT
    direction_metadata = ot.configure_gaussian_ot_direction(args.e66_direction_npz)
    set_amplitude_power(ALPHA)
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = args.mixed_precision != "none"
    candidate_indices = torch.tensor(
        [token - 1 for token in CANDIDATE_TOKENS],
        device=device,
        dtype=torch.long,
    )
    local_prefixes = np.empty(
        (shard_end - shard_start, len(CANDIDATE_TOKENS)),
        dtype=np.float32,
    )
    observed = 0
    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"e66_fulltrain_price rank={rank}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for images_01, _indices in progress:
            images_01 = images_01.to(device, non_blocking=True)
            target, x_base, _f_1d, _f_2d, router_score, x_native = model_batch(
                images_01,
                model,
                model_metadata,
                args,
                autocast_type,
                autocast_enabled,
            )
            components, _diagnostics = (
                ot.gaussian_ot_direction_relative_endpoint_component_gain_maps(
                    target,
                    x_base,
                    x_native,
                    fid_extractor.inception,
                    lpips_metric,
                    args.llamagen_input_range,
                )
            )
            combined = combine_e42_torch(components, component_scales)
            ranked = router_ranked_torch(combined, router_score)
            prefixes = torch.cumsum(ranked.float(), dim=1).index_select(
                1, candidate_indices
            )
            batch_prefixes = prefixes.cpu().numpy().astype(np.float32)
            local_prefixes[
                observed : observed + images_01.shape[0]
            ] = batch_prefixes
            observed += images_01.shape[0]
            if is_main:
                progress.set_postfix(images=observed)
    if observed != shard_end - shard_start:
        raise AssertionError("E66 full-train local image count mismatch")
    del model, lpips_metric, fid_extractor
    gc.collect()
    torch.cuda.empty_cache()

    gathered = [None for _ in range(world_size)] if is_main else None
    if distributed:
        dist.gather_object(local_prefixes, gathered, dst=0)
    else:
        gathered = [local_prefixes]

    if is_main:
        if gathered is None:
            raise AssertionError("E66 gather failed")
        all_prefixes = np.concatenate(gathered, axis=0)
        if all_prefixes.shape != (count, len(CANDIDATE_TOKENS)):
            raise AssertionError(
                f"E66 gathered prefix shape {all_prefixes.shape} != {(count, 3)}"
            )
        price, tokens = calibrate_prefix_price(
            all_prefixes,
            target_tokens=96.0,
            iterations=args.price_iterations,
        )
        allocation = allocation_summary(tokens)
        if not set(int(value) for value in allocation["histogram"]).issubset(
            CANDIDATE_TOKENS
        ):
            raise AssertionError("E66 full-train calibration emitted an invalid K")
        result = {
            "format": FORMAT,
            "status": "completed",
            "experiment": "E66-E47-three-choice-fulltrain-price",
            "algorithm": ALGORITHM,
            "candidate_tokens": list(CANDIDATE_TOKENS),
            "decision_rule": "argmax_K cumulative_E47_gain(K)-price*K",
            "price": float(price),
            "price_iterations": int(args.price_iterations),
            "allocation": allocation,
            "component_scales_frozen_train10k": component_scales,
            "component_weights": E31_COMPONENT_WEIGHTS,
            "amplitude_power": ALPHA,
            "full_train_images_used_for_price": count,
            "full_train_dataset_size": len(dataset),
            "full_train_coverage_exact": count == len(dataset) == FORMAL_TRAIN_IMAGES,
            "train_data_path": str(Path(args.data_path).resolve()),
            "calibration_source": {
                "split": "complete_imagenet_train",
                "data_path": str(Path(args.data_path).resolve()),
                "num_images": count,
                "dataset_size": len(dataset),
                "coverage_exact": count == len(dataset) == FORMAL_TRAIN_IMAGES,
                "validation_statistics_used": False,
            },
            "confirmation_artifact": {
                "path": str(confirmation_path.resolve()),
                "sha256": file_sha256(confirmation_path),
                "screen_gates": confirmation.get("screen_gates"),
            },
            "diagonal_gaussian_ot_direction": {
                "path": str(Path(args.e66_direction_npz).resolve()),
                "sha256": file_sha256(args.e66_direction_npz),
                "metadata": direction_metadata,
            },
            "checkpoint": str(Path(args.ckpt).resolve()),
            "checkpoint_sha256": file_sha256(args.ckpt),
            "checkpoint_state": model_metadata["checkpoint_state"],
            "selection_is_per_image_independent_after_train_calibration": True,
            "test_batch_statistics_used_for_selection": False,
            "test_set_budget_rebalancing_used": False,
            "validation_statistics_used": False,
            "validation_used_to_fit_or_select_e66": False,
            "probe_reconstructions_per_image": 1,
            "candidate_k_search_reconstructions_per_image": 0,
            "candidate_k_search_loss_evaluations_per_image": 0,
            "prefix_values_saved_to_disk": False,
            "features_saved_to_disk": False,
            "npz_written": False,
            "stats_pt_written": False,
            "output_json_only": True,
            "world_size": world_size,
            "runtime_seconds": time.time() - started,
            "runtime_args": vars(args),
            "model": model_metadata,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "price": result["price"],
                    "allocation": result["allocation"],
                    "runtime_seconds": result["runtime_seconds"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-json", required=True)
    parser.add_argument("--e66-direction-npz", required=True)
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


if __name__ == "__main__":
    main(build_parser().parse_args())
