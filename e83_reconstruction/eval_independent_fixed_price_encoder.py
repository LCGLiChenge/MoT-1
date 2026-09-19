#!/usr/bin/env python3
"""Evaluate a deterministic per-image fixed-price dynamic refinement encoder."""

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
from eval_fid_from_dynamic_choices import ssim_batch_sum, to_uint8, variable_topk_mask
from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    EvalImageDataset,
    build_lpips_metric,
    build_model,
    per_image_metrics,
)
from train_independent_budget_router import calibrate_teacher_price


METRIC_NAMES = ("lpips", "l1_01", "mse_01", "psnr")
PATH_NAMES = ("fixed_router", "independent_fixed_price")


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


def load_train_price(
    label_paths: list[str],
    target_tokens: int,
    iterations: int,
) -> tuple[np.ndarray, float, dict[str, object], list[dict[str, object]]]:
    cost_chunks = []
    candidate_tokens: np.ndarray | None = None
    sources = []
    seen_splits: set[str] = set()
    split_plan_hashes: set[str] = set()
    for raw_path in label_paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as archive:
            required = ("budget_costs", "candidate_tokens", "metadata_json")
            missing = [name for name in required if name not in archive]
            if missing:
                raise KeyError(f"{path} is missing {missing}")
            current_tokens = np.asarray(archive["candidate_tokens"], dtype=np.int64)
            if candidate_tokens is None:
                candidate_tokens = current_tokens
            elif not np.array_equal(candidate_tokens, current_tokens):
                raise ValueError("teacher label candidate tokens disagree")
            costs = np.asarray(archive["budget_costs"], dtype=np.float32)
            if costs.shape[1] != current_tokens.size:
                raise ValueError(f"budget cost shape mismatch in {path}")
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        source_manifest = metadata.get("source_index_manifest")
        if not isinstance(source_manifest, dict):
            raise ValueError("price calibration requires manifest-linked train labels")
        split = str(source_manifest.get("split", ""))
        if not split.startswith("teacher_shard_"):
            raise ValueError(f"price source is not a train teacher shard: {split}")
        if split in seen_splits:
            raise ValueError(f"duplicate price source split: {split}")
        seen_splits.add(split)
        split_plan_hashes.add(str(source_manifest.get("split_plan_sha256", "")))
        cost_chunks.append(costs)
        sources.append(
            {
                "path": str(path.resolve()),
                "file_sha256": file_sha256(path),
                "split": split,
                "num_images": int(costs.shape[0]),
                "data_root": metadata.get("data_root"),
                "split_plan_sha256": source_manifest.get("split_plan_sha256"),
            }
        )
    if candidate_tokens is None or not cost_chunks:
        raise ValueError("no teacher labels were supplied")
    if len(split_plan_hashes) != 1 or "" in split_plan_hashes:
        raise ValueError("price sources have different or missing split plans")
    if target_tokens not in candidate_tokens:
        raise ValueError("target tokens are absent from teacher candidates")
    costs = np.concatenate(cost_chunks, axis=0)
    price, record = calibrate_teacher_price(
        costs, candidate_tokens, target_tokens, iterations
    )
    record["source_split_plan_sha256"] = next(iter(split_plan_hashes))
    record["validation_statistics_used"] = False
    return candidate_tokens, price, record, sources


def standardize(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    values = values.float()
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False)
    return (values - mean) / std.clamp_min(float(epsilon))


def feature_sufficient_statistics(
    features: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[0] <= 0:
        raise ValueError("feature matrix must contain at least one row")
    dimension = features.shape[1]
    feature_sum = torch.zeros(dimension, device=device, dtype=torch.float64)
    feature_cross = torch.zeros(
        dimension, dimension, device=device, dtype=torch.float64
    )
    for start in range(0, features.shape[0], chunk_size):
        values = torch.from_numpy(features[start : start + chunk_size]).to(
            device=device, dtype=torch.float64
        )
        feature_sum += values.sum(dim=0)
        feature_cross += values.T @ values
    count = torch.tensor(features.shape[0], device=device, dtype=torch.long)
    return feature_sum, feature_cross, count


def moments_from_sufficient_statistics(
    feature_sum: torch.Tensor,
    feature_cross: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 1:
        raise ValueError("FID requires at least two samples")
    mean = feature_sum / float(count)
    covariance = (
        feature_cross - float(count) * mean[:, None] @ mean[None, :]
    ) / float(count - 1)
    return mean, covariance


def allocation_summary(histogram: np.ndarray, candidate_tokens: np.ndarray) -> dict[str, object]:
    count = int(histogram.sum())
    if count <= 0:
        raise ValueError("allocation histogram is empty")
    mean = float(np.dot(histogram, candidate_tokens) / count)
    second = float(np.dot(histogram, candidate_tokens.astype(np.float64) ** 2) / count)
    active = np.flatnonzero(histogram)
    return {
        "mean": mean,
        "mean_delta_from_target": mean - 96.0,
        "std": float(np.sqrt(max(0.0, second - mean**2))),
        "min": int(candidate_tokens[active[0]]),
        "max": int(candidate_tokens[active[-1]]),
        "histogram": {
            str(int(candidate_tokens[index])): int(histogram[index])
            for index in active
        },
    }


def main(args: argparse.Namespace) -> None:
    from skimage.metrics import structural_similarity

    output = Path(args.output_json)
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

    candidate_tokens, teacher_price, price_record, price_sources = load_train_price(
        args.teacher_label_npz,
        args.target_tokens,
        args.price_search_iterations,
    )
    fixed_index = int(np.where(candidate_tokens == args.target_tokens)[0][0])
    token_units = torch.tensor(
        (candidate_tokens - args.target_tokens) / 16.0,
        device=device,
        dtype=torch.float32,
    )

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, model_metadata = build_model(args, device, checkpoint)
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)

    mot_root = str(Path(args.mot_root).resolve())
    if mot_root not in sys.path:
        sys.path.insert(0, mot_root)
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import native_llamagen_feature
    from train_titok_llamagen_recon import autocast_dtype, convert_image_range

    llamagen_root = model_metadata["llamagen_root"]
    if llamagen_root not in sys.path:
        sys.path.insert(0, llamagen_root)
    from dataset.augmentation import center_crop_arr

    dataset = EvalImageDataset(args.data_path, args.image_size, center_crop_arr)
    count = len(dataset) if args.num_images <= 0 else min(args.num_images, len(dataset))
    if count <= 1:
        raise ValueError("FID evaluation requires at least two images")
    shard_start = count * rank // world_size
    shard_end = count * (rank + 1) // world_size
    indices = range(shard_start, shard_end)
    loader = DataLoader(
        Subset(dataset, indices),
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
    real_feature_chunks: list[np.ndarray] = []
    fixed_feature_chunks: list[np.ndarray] = []
    dynamic_feature_chunks: list[np.ndarray] = []
    metric_sums = torch.zeros(
        len(PATH_NAMES), len(METRIC_NAMES) + 1, device=device, dtype=torch.float64
    )
    allocation_histogram = torch.zeros(
        candidate_tokens.size, device=device, dtype=torch.long
    )
    local_images = 0

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"independent_price rank={rank}",
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
            target_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            real_feature_chunks.append(target_features.cpu().numpy())

            inception_losses = []
            lpips_losses = []
            mse_losses = []
            fixed_prediction = None
            fixed_features = None
            fixed_metrics = None
            for candidate_index, token_count in enumerate(candidate_tokens.tolist()):
                tokens = torch.full(
                    (batch,), token_count, device=device, dtype=torch.long
                )
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
                inception_losses.append(
                    (features - target_features).square().mean(dim=1)
                )
                lpips_losses.append(
                    torch.from_numpy(metrics["lpips"]).to(device=device)
                )
                mse_losses.append(
                    torch.from_numpy(metrics["mse_01"]).to(device=device)
                )
                if candidate_index == fixed_index:
                    fixed_prediction = prediction.detach()
                    fixed_features = features.detach()
                    fixed_metrics = metrics
                del prediction, features, metrics, mask

            inception_curve = torch.stack(inception_losses, dim=1)
            lpips_curve = torch.stack(lpips_losses, dim=1)
            mse_curve = torch.stack(mse_losses, dim=1)
            adjusted_cost = (
                standardize(inception_curve)
                + 2.0 * standardize(lpips_curve)
                + 4.0 * standardize(mse_curve)
                + float(teacher_price) * token_units[None, :]
            )
            dynamic_choices = adjusted_cost.argmin(dim=1)
            dynamic_tokens = torch.as_tensor(
                candidate_tokens, device=device, dtype=torch.long
            )[dynamic_choices]
            allocation_histogram += torch.bincount(
                dynamic_choices, minlength=candidate_tokens.size
            )

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_type,
                enabled=autocast_enabled,
            ):
                dynamic_mask = variable_topk_mask(
                    router_score, dynamic_tokens, f_1d.dtype
                )
                dynamic_prediction = model.llamagen_vq.decoder(
                    (1.0 - dynamic_mask) * f_1d + dynamic_mask * f_2d
                )
            dynamic_features = fid_extractor.inception(
                to_uint8(dynamic_prediction, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            dynamic_metrics = per_image_metrics(
                dynamic_prediction.float(),
                target.float(),
                args.llamagen_input_range,
                lpips_metric,
            )
            if fixed_prediction is None or fixed_features is None or fixed_metrics is None:
                raise AssertionError("fixed candidate was not captured")
            fixed_feature_chunks.append(fixed_features.cpu().numpy())
            dynamic_feature_chunks.append(dynamic_features.cpu().numpy())
            for path_index, metrics in enumerate((fixed_metrics, dynamic_metrics)):
                for metric_index, metric_name in enumerate(METRIC_NAMES):
                    metric_sums[path_index, metric_index] += float(
                        np.asarray(metrics[metric_name], dtype=np.float64).sum()
                    )
            metric_sums[0, -1] += ssim_batch_sum(
                fixed_prediction.float(),
                target.float(),
                args.llamagen_input_range,
                structural_similarity,
            )
            metric_sums[1, -1] += ssim_batch_sum(
                dynamic_prediction.float(),
                target.float(),
                args.llamagen_input_range,
                structural_similarity,
            )
            local_images += batch
            if is_main:
                progress.set_postfix(images=local_images)
            del (
                x_base,
                extra,
                f_1d,
                f_2d,
                router_score,
                target_features,
                inception_curve,
                lpips_curve,
                mse_curve,
                adjusted_cost,
                dynamic_mask,
                dynamic_prediction,
                dynamic_features,
                fixed_prediction,
                fixed_features,
            )
        if is_main and isinstance(progress, tqdm):
            progress.close()

    if local_images != shard_end - shard_start:
        raise AssertionError(
            f"rank {rank} evaluated {local_images}, expected {shard_end - shard_start}"
        )
    real_features_np = np.concatenate(real_feature_chunks, axis=0).astype(
        np.float32, copy=False
    )
    fixed_features_np = np.concatenate(fixed_feature_chunks, axis=0).astype(
        np.float32, copy=False
    )
    dynamic_features_np = np.concatenate(dynamic_feature_chunks, axis=0).astype(
        np.float32, copy=False
    )
    del model, lpips_metric, fid_extractor
    gc.collect()
    torch.cuda.empty_cache()

    sufficient = []
    for features in (real_features_np, fixed_features_np, dynamic_features_np):
        sufficient.append(
            feature_sufficient_statistics(
                features, device, args.fid_moment_chunk_size
            )
        )
    del real_features_np, fixed_features_np, dynamic_features_np
    gc.collect()

    if distributed:
        for feature_sum, feature_cross, feature_count in sufficient:
            dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_cross, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(allocation_histogram, op=dist.ReduceOp.SUM)

    if is_main:
        real_sum, real_cross, real_count_tensor = sufficient[0]
        fixed_sum, fixed_cross, fixed_count_tensor = sufficient[1]
        dynamic_sum, dynamic_cross, dynamic_count_tensor = sufficient[2]
        counts = [
            int(real_count_tensor.item()),
            int(fixed_count_tensor.item()),
            int(dynamic_count_tensor.item()),
        ]
        if counts != [count, count, count]:
            raise AssertionError(f"global FID counts disagree: {counts} vs {count}")
        real_mean, real_cov = moments_from_sufficient_statistics(
            real_sum, real_cross, count
        )
        fixed_mean, fixed_cov = moments_from_sufficient_statistics(
            fixed_sum, fixed_cross, count
        )
        dynamic_mean, dynamic_cov = moments_from_sufficient_statistics(
            dynamic_sum, dynamic_cross, count
        )
        metrics = {}
        fid_values = (
            float(_compute_fid(real_mean, real_cov, fixed_mean, fixed_cov).item()),
            float(_compute_fid(real_mean, real_cov, dynamic_mean, dynamic_cov).item()),
        )
        for path_index, path_name in enumerate(PATH_NAMES):
            record = {
                metric_name: float(metric_sums[path_index, metric_index].item() / count)
                for metric_index, metric_name in enumerate(METRIC_NAMES)
            }
            record["ssim"] = float(metric_sums[path_index, -1].item() / count)
            record["fid"] = fid_values[path_index]
            metrics[path_name] = record
        histogram_np = allocation_histogram.cpu().numpy().astype(np.int64)
        allocation = allocation_summary(histogram_np, candidate_tokens)
        allocation["mean_delta_from_target"] = float(
            allocation["mean"] - args.target_tokens
        )
        fixed = metrics["fixed_router"]
        dynamic = metrics["independent_fixed_price"]
        result = {
            "format": "independent_fixed_price_encoder_eval_v1",
            "status": "completed",
            "algorithm": "per_image_argmin_z_inception_plus_2z_lpips_plus_4z_mse_plus_fixed_rate_price",
            "evaluation_is_per_image_independent": True,
            "test_batch_statistics_used_for_selection": False,
            "test_set_budget_rebalancing_used": False,
            "source_image_is_available_to_reconstruction_encoder": True,
            "encoding_candidate_decodes_per_image": int(candidate_tokens.size + 1),
            "spatial_ranking": "frozen_original_router",
            "teacher_price": teacher_price,
            "teacher_price_record": price_record,
            "teacher_price_sources": price_sources,
            "checkpoint": str(Path(args.ckpt).resolve()),
            "checkpoint_state": model_metadata["checkpoint_state"],
            "data_path": str(Path(args.data_path).resolve()),
            "num_images": count,
            "candidate_tokens": candidate_tokens.tolist(),
            "target_tokens": int(args.target_tokens),
            "allocation": allocation,
            "metrics": metrics,
            "relative_changes_percent": {
                "fid_reduction": 100.0 * (fixed["fid"] - dynamic["fid"]) / fixed["fid"],
                "lpips_reduction": 100.0 * (fixed["lpips"] - dynamic["lpips"]) / fixed["lpips"],
                "l1_reduction": 100.0 * (fixed["l1_01"] - dynamic["l1_01"]) / fixed["l1_01"],
                "mse_reduction": 100.0 * (fixed["mse_01"] - dynamic["mse_01"]) / fixed["mse_01"],
                "psnr_increase": dynamic["psnr"] - fixed["psnr"],
                "ssim_increase": dynamic["ssim"] - fixed["ssim"],
            },
            "fid_feature": int(args.fid_feature),
            "fid_input": "uint8",
            "features_saved_to_disk": False,
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
    parser.add_argument("--teacher-label-npz", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--price-search-iterations", type=int, default=100)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--fid-moment-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze", "llamagen_vgg"],
        default="alex",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=0)
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
