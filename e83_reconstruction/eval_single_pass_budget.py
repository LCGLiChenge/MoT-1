#!/usr/bin/env python3
"""Paired fixed-96 evaluation of one frozen single-pass dynamic-budget rule."""

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
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from tqdm import tqdm

from e04_distilled_router import add_base_model_arguments, file_sha256
from e04_stratified_manifest import load_index_manifest
from eval_fid_from_dynamic_choices import ssim_batch_sum, to_uint8, variable_topk_mask
from eval_independent_fixed_price_encoder import (
    feature_sufficient_statistics,
    moments_from_sufficient_statistics,
)
from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    EvalImageDataset,
    build_lpips_metric,
    build_model,
    per_image_metrics,
)
from single_pass_budget_score import (
    SCORE_NAMES,
    compose_score_tensors,
    single_pass_score_maps,
    token_counts_from_scores_torch,
)


METRIC_NAMES = ("lpips", "l1_01", "mse_01", "psnr")
PATH_NAMES = ("fixed_router", "single_pass_dynamic")


def distributed_setup() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def load_frozen_rule(
    calibration_path: Path,
    score_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("format") != "single_pass_budget_calibration_v1":
        raise ValueError(f"unsupported calibration format: {calibration.get('format')}")
    if calibration.get("status") != "completed":
        raise ValueError("calibration did not complete")
    if calibration.get("validation_statistics_used") is not False:
        raise ValueError("calibration must explicitly record validation_statistics_used=false")
    manifest = calibration.get("index_manifest")
    if not isinstance(manifest, dict) or not str(manifest.get("split", "")).startswith(
        "teacher_shard_"
    ):
        raise ValueError("calibration must be linked to a train teacher-shard manifest")
    if calibration.get("test_batch_statistics_required") is not False:
        raise ValueError("rule must not require test batch statistics")
    if calibration.get("test_set_budget_rebalancing_required") is not False:
        raise ValueError("rule must not require test-set budget rebalancing")
    if score_name not in SCORE_NAMES:
        raise ValueError(f"unknown --score-name {score_name}")
    rules = calibration.get("rules")
    if not isinstance(rules, dict) or score_name not in rules:
        raise ValueError(f"calibration is missing score rule {score_name}")
    rule = rules[score_name]
    if not isinstance(rule, dict) or "threshold" not in rule:
        raise ValueError(f"calibration rule {score_name} is malformed")
    return calibration, rule


def allocation_summary(histogram: np.ndarray, target_tokens: int) -> dict[str, object]:
    histogram = np.asarray(histogram, dtype=np.int64)
    count = int(histogram.sum())
    if histogram.shape != (257,) or count <= 0:
        raise ValueError("allocation histogram must have shape [257] and nonzero count")
    values = np.arange(257, dtype=np.float64)
    mean = float(np.dot(histogram, values) / count)
    second = float(np.dot(histogram, np.square(values)) / count)
    active = np.flatnonzero(histogram)
    return {
        "count": count,
        "mean": mean,
        "mean_delta_from_target": mean - float(target_tokens),
        "std": float(np.sqrt(max(0.0, second - mean * mean))),
        "min": int(active[0]),
        "max": int(active[-1]),
        "histogram": {
            str(int(index)): int(histogram[index]) for index in active.tolist()
        },
    }


def select_evaluation_indices(dataset, args, calibration):
    if args.index_manifest:
        manifest = load_index_manifest(args.index_manifest, dataset.root, dataset.paths)
        split = str(manifest.metadata.get("split", ""))
        if not split.startswith("teacher_shard_"):
            raise ValueError("--index-manifest must be a train teacher shard")
        if split == calibration["index_manifest"]["split"]:
            raise ValueError("evaluation and calibration must use different train shards")
        available = np.asarray(manifest.dataset_indices, dtype=np.int64)
    else:
        split = "dataset_prefix"
        available = np.arange(len(dataset), dtype=np.int64)
    count = (
        available.size
        if args.num_images <= 0
        else min(args.num_images, available.size)
    )
    return available[:count], split


def main(args: argparse.Namespace) -> None:
    from skimage.metrics import structural_similarity

    output = Path(args.output_json)
    calibration_path = Path(args.calibration_json)
    calibration, rule = load_frozen_rule(calibration_path, args.score_name)
    if args.target_tokens != int(calibration["target_tokens"]):
        raise ValueError("--target-tokens differs from the frozen calibration")
    if args.min_tokens != int(calibration["min_tokens"]):
        raise ValueError("--min-tokens differs from the frozen calibration")
    if args.max_tokens != int(calibration["max_tokens"]):
        raise ValueError("--max-tokens differs from the frozen calibration")
    if args.quantize_step != int(calibration["quantize_step"]):
        raise ValueError("--quantize-step differs from the frozen calibration")

    distributed, rank, _local_rank, world_size, device = distributed_setup()
    is_main = rank == 0
    if is_main and output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")
    if distributed:
        exists = torch.tensor(
            int(output.exists() and not args.overwrite), device=device, dtype=torch.long
        )
        dist.broadcast(exists, src=0)
        if int(exists.item()):
            raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")

    seed = int(args.seed) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import (
        native_llamagen_feature,
    )
    from train_titok_llamagen_recon import autocast_dtype, convert_image_range

    llamagen_root = model_metadata["llamagen_root"]
    if llamagen_root not in sys.path:
        sys.path.insert(0, llamagen_root)
    from dataset.augmentation import center_crop_arr

    dataset = EvalImageDataset(args.data_path, args.image_size, center_crop_arr)
    selected_indices, evaluation_split = select_evaluation_indices(dataset, args, calibration)
    count = int(selected_indices.size)
    if count <= 1:
        raise ValueError("FID evaluation requires at least two images")
    shard_start = count * rank // world_size
    shard_end = count * (rank + 1) // world_size
    loader = DataLoader(
        Subset(dataset, selected_indices[shard_start:shard_end].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    lpips_metric = build_lpips_metric(args, device)
    fid_extractor = FrechetInceptionDistance(
        feature=args.fid_feature, normalize=False
    ).to(device).eval().requires_grad_(False)
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"
    component_statistics = calibration["component_statistics"]
    threshold = float(rule["threshold"])

    real_feature_chunks: list[np.ndarray] = []
    fixed_feature_chunks: list[np.ndarray] = []
    dynamic_feature_chunks: list[np.ndarray] = []
    metric_sums = torch.zeros(
        len(PATH_NAMES), len(METRIC_NAMES) + 1, device=device, dtype=torch.float64
    )
    allocation_histogram = torch.zeros(257, device=device, dtype=torch.long)
    local_images = 0

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"single_pass_{args.score_name} rank={rank}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for images_01, _global_indices in progress:
            batch = images_01.shape[0]
            images_01 = images_01.to(device, non_blocking=True)
            target = convert_image_range(images_01, args.llamagen_input_range)
            titok_input = convert_image_range(images_01, args.titok_input_range)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_type,
                enabled=autocast_enabled,
            ):
                x_base, extra = model(titok_input)
                f_1d = extra["f_1d_lg"]
                f_2d, _ = native_llamagen_feature(
                    model.llamagen_vq,
                    target,
                    model_metadata["codebook_embed_dim"],
                    allow_encoder_grad=False,
                )
                router_score, _ = model.router(f_1d, x_base, f_2d)
            raw_maps = single_pass_score_maps(
                x_base, target, f_1d, f_2d, router_score
            )
            score = compose_score_tensors(raw_maps, component_statistics)[args.score_name]
            dynamic_tokens = token_counts_from_scores_torch(
                score,
                threshold,
                args.min_tokens,
                args.max_tokens,
                args.quantize_step,
            )
            allocation_histogram += torch.bincount(
                dynamic_tokens, minlength=257
            )[:257]

            target_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            real_feature_chunks.append(target_features.cpu().numpy())
            path_tokens = (
                torch.full_like(dynamic_tokens, args.target_tokens),
                dynamic_tokens,
            )
            path_feature_chunks = (fixed_feature_chunks, dynamic_feature_chunks)
            for path_index, tokens in enumerate(path_tokens):
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_type,
                    enabled=autocast_enabled,
                ):
                    mask = variable_topk_mask(router_score, tokens, f_1d.dtype)
                    prediction = model.llamagen_vq.decoder(
                        (1.0 - mask) * f_1d + mask * f_2d
                    )
                features = fid_extractor.inception(
                    to_uint8(prediction, args.llamagen_input_range)
                ).reshape(batch, -1).float()
                metrics = per_image_metrics(
                    prediction.float(),
                    target.float(),
                    args.llamagen_input_range,
                    lpips_metric,
                )
                path_feature_chunks[path_index].append(features.cpu().numpy())
                for metric_index, metric_name in enumerate(METRIC_NAMES):
                    metric_sums[path_index, metric_index] += float(
                        np.asarray(metrics[metric_name], dtype=np.float64).sum()
                    )
                metric_sums[path_index, -1] += ssim_batch_sum(
                    prediction.float(),
                    target.float(),
                    args.llamagen_input_range,
                    structural_similarity,
                )
            local_images += batch
            if is_main:
                progress.set_postfix(images=local_images)
        if is_main and isinstance(progress, tqdm):
            progress.close()

    expected_local = shard_end - shard_start
    if local_images != expected_local:
        raise AssertionError(f"rank {rank} evaluated {local_images}, expected {expected_local}")
    feature_arrays = [
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for chunks in (
            real_feature_chunks,
            fixed_feature_chunks,
            dynamic_feature_chunks,
        )
    ]
    del model, lpips_metric, fid_extractor
    gc.collect()
    torch.cuda.empty_cache()
    sufficient = [
        feature_sufficient_statistics(values, device, args.fid_moment_chunk_size)
        for values in feature_arrays
    ]
    del feature_arrays
    gc.collect()

    if distributed:
        for feature_sum, feature_cross, feature_count in sufficient:
            dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_cross, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(allocation_histogram, op=dist.ReduceOp.SUM)

    if is_main:
        counts = [int(item[2].item()) for item in sufficient]
        if counts != [count, count, count]:
            raise AssertionError(f"global FID counts disagree: {counts} vs {count}")
        moments = [
            moments_from_sufficient_statistics(item[0], item[1], count)
            for item in sufficient
        ]
        real_mean, real_cov = moments[0]
        fid_values = [
            float(
                _compute_fid(real_mean, real_cov, fake_mean, fake_cov).item()
            )
            for fake_mean, fake_cov in moments[1:]
        ]
        metrics: dict[str, dict[str, float]] = {}
        for path_index, path_name in enumerate(PATH_NAMES):
            record = {
                metric_name: float(
                    metric_sums[path_index, metric_index].item() / count
                )
                for metric_index, metric_name in enumerate(METRIC_NAMES)
            }
            record["ssim"] = float(metric_sums[path_index, -1].item() / count)
            record["fid"] = fid_values[path_index]
            metrics[path_name] = record
        fixed = metrics["fixed_router"]
        dynamic = metrics["single_pass_dynamic"]
        result = {
            "format": "single_pass_budget_eval_v1",
            "status": "completed",
            "algorithm": "count_per_grid_scores_above_frozen_train_threshold",
            "score_name": args.score_name,
            "score_threshold": threshold,
            "selection_is_per_image_independent": True,
            "candidate_reconstructions_evaluated_per_image": 0,
            "encoding_loss_evaluations_per_image": 0,
            "final_dynamic_reconstructions_per_image": 1,
            "test_batch_statistics_used_for_selection": False,
            "test_set_budget_rebalancing_used": False,
            "source_image_is_available_to_reconstruction_encoder": True,
            "spatial_ranking": "frozen_original_router",
            "calibration_json": str(calibration_path.resolve()),
            "calibration_file_sha256": file_sha256(calibration_path),
            "calibration_split": calibration["index_manifest"]["split"],
            "calibration_num_images": int(calibration["num_images"]),
            "validation_statistics_used_for_calibration": False,
            "checkpoint": str(Path(args.ckpt).resolve()),
            "checkpoint_state": model_metadata["checkpoint_state"],
            "data_path": str(Path(args.data_path).resolve()),
            "evaluation_split": evaluation_split,
            "evaluation_index_manifest": (
                str(Path(args.index_manifest).resolve()) if args.index_manifest else ""
            ),
            "num_images": count,
            "target_tokens": int(args.target_tokens),
            "allocation": allocation_summary(
                allocation_histogram.cpu().numpy(), args.target_tokens
            ),
            "metrics": metrics,
            "relative_changes_percent": {
                "fid_reduction": 100.0 * (fixed["fid"] - dynamic["fid"]) / fixed["fid"],
                "lpips_reduction": 100.0
                * (fixed["lpips"] - dynamic["lpips"])
                / fixed["lpips"],
                "l1_reduction": 100.0
                * (fixed["l1_01"] - dynamic["l1_01"])
                / fixed["l1_01"],
                "mse_reduction": 100.0
                * (fixed["mse_01"] - dynamic["mse_01"])
                / fixed["mse_01"],
                "psnr_increase": dynamic["psnr"] - fixed["psnr"],
                "ssim_increase": dynamic["ssim"] - fixed["ssim"],
            },
            "fid_feature": int(args.fid_feature),
            "fid_input": "uint8",
            "features_saved_to_disk": False,
            "npz_written": False,
            "stats_pt_written": False,
            "output_json_only": True,
            "world_size": world_size,
            "runtime_seconds": float(time.time() - started),
            "runtime_args": vars(args),
            "model": model_metadata,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(f"saved {output}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--score-name", required=True, choices=SCORE_NAMES)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--index-manifest", default="")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--quantize-step", type=int, default=1)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--fid-moment-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze", "llamagen_vgg"],
        default="alex",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
