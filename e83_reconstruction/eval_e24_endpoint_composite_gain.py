#!/usr/bin/env python3
"""Paired fixed-96 validation of a frozen E24 endpoint-composite budget rule."""

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
from e24_endpoint_composite_gain import (
    COMPONENT_NAMES,
    COMPONENT_WEIGHTS,
    combine_components_torch,
    endpoint_component_gain_maps,
)
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
from eval_single_pass_budget import allocation_summary, distributed_setup
from screen_native_gain_budget import model_batch, shrink_token_counts
from single_probe_marginal_budget import prefix_choices_torch, router_ranked_torch


METRIC_NAMES = ("lpips", "l1_01", "mse_01", "psnr")
PATH_NAMES = ("fixed_router", "endpoint_composite_dynamic")


def load_frozen_e24(path: Path, args) -> tuple[dict, dict[str, float], float, float]:
    artifact = json.loads(path.read_text())
    if artifact.get("format") != "e24_endpoint_composite_gain_screen_v1":
        raise ValueError("unsupported E24 calibration artifact format")
    if artifact.get("status") != "completed":
        raise ValueError("E24 calibration artifact did not complete")
    if artifact.get("algorithm") != (
        "router_prefix_signed_mass_preserving_endpoint_gain_minus_price"
    ):
        raise ValueError("unexpected E24 algorithm")
    required_false = (
        "test_batch_statistics_used_for_selection",
        "test_set_budget_rebalancing_used",
        "validation_statistics_used",
    )
    if any(artifact.get(name) is not False for name in required_false):
        raise ValueError("E24 artifact violates the no-validation/no-batch protocol")
    if artifact.get("selection_is_per_image_independent_after_train_calibration") is not True:
        raise ValueError("E24 artifact does not assert per-image independence")

    source = artifact.get("calibration_source")
    runtime = artifact.get("runtime_args")
    if not isinstance(source, dict) or source.get("split") != "calibration":
        raise ValueError("E24 calibration source is not the frozen train calibration split")
    if not isinstance(runtime, dict):
        raise ValueError("E24 artifact is missing runtime arguments")
    calibration_data = Path(str(runtime.get("data_path", ""))).resolve()
    evaluation_data = Path(args.data_path).resolve()
    if calibration_data == evaluation_data or calibration_data.name != "train":
        raise ValueError("E24 calibration must come from a separate train dataset")

    recorded_weights = artifact.get("component_weights")
    if recorded_weights != COMPONENT_WEIGHTS:
        raise ValueError("E24 component weights differ from the implementation")
    scales = artifact.get("component_scales_train_only")
    if not isinstance(scales, dict) or set(scales) != set(COMPONENT_NAMES):
        raise ValueError("E24 artifact has invalid component scales")
    scales = {name: float(scales[name]) for name in COMPONENT_NAMES}
    if not all(np.isfinite(value) and value > 0.0 for value in scales.values()):
        raise ValueError("E24 component scales must be finite and positive")

    for name in ("target_tokens", "min_tokens", "max_tokens", "quantize_step"):
        if int(runtime.get(name, -1)) != int(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} differs from frozen E24 calibration")
    if int(args.fid_feature) != 2048:
        raise ValueError("E24 endpoint attribution requires --fid-feature 2048")
    price = float(artifact["price"])
    shrink = float(artifact["allocation_shrink"])
    if not np.isfinite(price) or not 0.0 <= shrink <= 1.0:
        raise ValueError("invalid frozen E24 price or shrink")
    return artifact, scales, price, shrink


def update_diagnostics(
    sums: torch.Tensor, maxima: torch.Tensor, diagnostics: dict[str, dict[str, torch.Tensor]]
) -> None:
    batch = next(iter(diagnostics.values()))["base_loss"].shape[0]
    sums[0] += batch
    for index, name in enumerate(COMPONENT_NAMES):
        record = diagnostics[name]
        sums[1 + 2 * index] += record["base_loss"].double().sum()
        sums[2 + 2 * index] += record["native_loss"].double().sum()
        maxima[2 * index] = torch.maximum(
            maxima[2 * index], record["base_map_mean_error"].double().max()
        )
        maxima[2 * index + 1] = torch.maximum(
            maxima[2 * index + 1], record["native_map_mean_error"].double().max()
        )


def finalize_diagnostics(sums: torch.Tensor, maxima: torch.Tensor) -> dict:
    count = int(sums[0].item())
    if count <= 0:
        raise AssertionError("empty E24 endpoint diagnostics")
    result: dict[str, object] = {"count": count}
    for index, name in enumerate(COMPONENT_NAMES):
        base_sum = float(sums[1 + 2 * index].item())
        native_sum = float(sums[2 + 2 * index].item())
        result[name] = {
            "base_loss_mean": base_sum / count,
            "native_loss_mean": native_sum / count,
            "endpoint_loss_reduction_mean": (base_sum - native_sum) / count,
            "base_map_mean_max_abs_error": float(maxima[2 * index].item()),
            "native_map_mean_max_abs_error": float(maxima[2 * index + 1].item()),
        }
    return result


def main(args) -> None:
    from skimage.metrics import structural_similarity

    output = Path(args.output_json)
    calibration_path = Path(args.calibration_json)
    calibration, component_scales, price, allocation_shrink = load_frozen_e24(
        calibration_path, args
    )
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

    torch.manual_seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    started = time.time()

    recorded_ckpt = Path(str(calibration.get("checkpoint", ""))).resolve()
    requested_ckpt = Path(args.ckpt).resolve()
    if recorded_ckpt != requested_ckpt:
        raise ValueError(
            f"checkpoint differs from frozen E24 artifact: {requested_ckpt} != {recorded_ckpt}"
        )
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
    count = len(dataset) if args.num_images <= 0 else min(args.num_images, len(dataset))
    if count <= 1:
        raise ValueError("FID evaluation requires at least two images")
    shard_start = count * rank // world_size
    shard_end = count * (rank + 1) // world_size
    loader = DataLoader(
        Subset(dataset, range(shard_start, shard_end)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    lpips_metric = build_lpips_metric(args, device)
    fid_extractor = (
        FrechetInceptionDistance(feature=args.fid_feature, normalize=False)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"
    feature_chunks: list[list[np.ndarray]] = [[], [], []]
    metric_sums = torch.zeros(
        len(PATH_NAMES), len(METRIC_NAMES) + 1, device=device, dtype=torch.float64
    )
    allocation_histogram = torch.zeros(257, device=device, dtype=torch.long)
    raw_allocation_histogram = torch.zeros(257, device=device, dtype=torch.long)
    diagnostic_sums = torch.zeros(1 + 2 * len(COMPONENT_NAMES), device=device, dtype=torch.float64)
    diagnostic_maxima = torch.zeros(2 * len(COMPONENT_NAMES), device=device, dtype=torch.float64)
    local_images = 0

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"e24_val rank={rank}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for images_01, _indices in progress:
            batch = images_01.shape[0]
            images_01 = images_01.to(device, non_blocking=True)
            target, x_base, f_1d, f_2d, router_score, x_native = model_batch(
                images_01, model, model_metadata, args, autocast_type, autocast_enabled
            )
            components, diagnostics = endpoint_component_gain_maps(
                target,
                x_base,
                x_native,
                fid_extractor.inception,
                lpips_metric,
                args.llamagen_input_range,
            )
            update_diagnostics(diagnostic_sums, diagnostic_maxima, diagnostics)
            combined = combine_components_torch(components, component_scales)
            ranked = router_ranked_torch(combined, router_score)
            raw_dynamic_tokens = prefix_choices_torch(
                ranked, price, args.min_tokens, args.max_tokens, args.quantize_step
            )
            dynamic_tokens = shrink_token_counts(
                raw_dynamic_tokens,
                args.target_tokens,
                args.min_tokens,
                args.max_tokens,
                args.quantize_step,
                allocation_shrink,
            )
            raw_allocation_histogram += torch.bincount(
                raw_dynamic_tokens, minlength=257
            )[:257]
            allocation_histogram += torch.bincount(dynamic_tokens, minlength=257)[:257]
            fixed_tokens = torch.full_like(dynamic_tokens, args.target_tokens)

            target_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            feature_chunks[0].append(target_features.cpu().numpy())
            for path_index, tokens in enumerate((fixed_tokens, dynamic_tokens)):
                with torch.autocast(
                    device_type=device.type, dtype=autocast_type, enabled=autocast_enabled
                ):
                    mask = variable_topk_mask(router_score, tokens, f_1d.dtype)
                    prediction = model.llamagen_vq.decoder(
                        (1.0 - mask) * f_1d + mask * f_2d
                    )
                features = fid_extractor.inception(
                    to_uint8(prediction, args.llamagen_input_range)
                ).reshape(batch, -1).float()
                feature_chunks[path_index + 1].append(features.cpu().numpy())
                metrics = per_image_metrics(
                    prediction.float(), target.float(), args.llamagen_input_range, lpips_metric
                )
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

    expected_local = shard_end - shard_start
    if local_images != expected_local:
        raise AssertionError(f"rank {rank} evaluated {local_images}, expected {expected_local}")
    feature_arrays = [
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for chunks in feature_chunks
    ]
    del model, lpips_metric, fid_extractor, feature_chunks
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
        for tensor in (
            metric_sums,
            allocation_histogram,
            raw_allocation_histogram,
            diagnostic_sums,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(diagnostic_maxima, op=dist.ReduceOp.MAX)

    if is_main:
        counts = [int(item[2].item()) for item in sufficient]
        if counts != [count, count, count]:
            raise AssertionError(f"global FID counts disagree: {counts} vs {count}")
        moments = [
            moments_from_sufficient_statistics(item[0], item[1], count)
            for item in sufficient
        ]
        real_mean, real_covariance = moments[0]
        fid_values = [
            float(_compute_fid(real_mean, real_covariance, mean, covariance).item())
            for mean, covariance in moments[1:]
        ]
        metrics_out: dict[str, dict[str, float]] = {}
        for path_index, path_name in enumerate(PATH_NAMES):
            record = {
                metric_name: float(metric_sums[path_index, metric_index].item() / count)
                for metric_index, metric_name in enumerate(METRIC_NAMES)
            }
            record["ssim"] = float(metric_sums[path_index, -1].item() / count)
            record["fid"] = fid_values[path_index]
            metrics_out[path_name] = record
        fixed = metrics_out["fixed_router"]
        dynamic = metrics_out["endpoint_composite_dynamic"]
        result = {
            "format": "e24_endpoint_composite_gain_eval_v1",
            "status": "completed",
            "experiment": "E24-endpoint-composite-gain",
            "algorithm": calibration["algorithm"],
            "selection_is_per_image_independent_after_train_calibration": True,
            "test_batch_statistics_used_for_selection": False,
            "test_set_budget_rebalancing_used": False,
            "validation_statistics_used_for_calibration": False,
            "source_image_is_available_to_reconstruction_encoder": True,
            "probe_reconstructions_per_image": 1,
            "distinct_probe_k_values": [256],
            "candidate_k_search_reconstructions_per_image": 0,
            "candidate_k_search_loss_evaluations_per_image": 0,
            "final_dynamic_reconstructions_per_image": 1,
            "spatial_ranking": "frozen_original_router",
            "component_weights": COMPONENT_WEIGHTS,
            "component_scales_train_only": component_scales,
            "price_train_only": price,
            "allocation_shrink_train_only": allocation_shrink,
            "calibration_json": str(calibration_path.resolve()),
            "calibration_sha256": file_sha256(calibration_path),
            "calibration_source": calibration["calibration_source"],
            "checkpoint": str(requested_ckpt),
            "checkpoint_state": model_metadata["checkpoint_state"],
            "data_path": str(Path(args.data_path).resolve()),
            "evaluation_split": "imagenet_validation",
            "num_images": count,
            "target_tokens": int(args.target_tokens),
            "raw_allocation": allocation_summary(
                raw_allocation_histogram.cpu().numpy(), args.target_tokens
            ),
            "allocation": allocation_summary(
                allocation_histogram.cpu().numpy(), args.target_tokens
            ),
            "endpoint_diagnostics": finalize_diagnostics(
                diagnostic_sums, diagnostic_maxima
            ),
            "metrics": metrics_out,
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
            "reconstructions_saved": False,
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
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--quantize-step", type=int, default=1)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--fid-moment-chunk-size", type=int, default=1024)
    parser.add_argument("--lpips-net", choices=["alex"], default="alex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
