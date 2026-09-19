#!/usr/bin/env python3
"""Calibrate and paired-screen the independent E24 endpoint-gain budget."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from tqdm import tqdm

from e04_distilled_router import add_base_model_arguments, file_sha256
from e04_stratified_manifest import load_index_manifest
from e24_endpoint_composite_gain import (
    COMPONENT_NAMES,
    COMPONENT_WEIGHTS,
    combine_components_numpy,
    combine_components_torch,
    endpoint_component_gain_maps,
    fit_component_scales,
)
from eval_fid_from_dynamic_choices import ssim_batch_sum, to_uint8, variable_topk_mask
from eval_independent_fixed_price_encoder import (
    feature_sufficient_statistics,
    moments_from_sufficient_statistics,
)
from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    EvalImageDataset,
    build_lpips_metric,
    build_model,
    per_image_metrics,
)
from eval_single_pass_budget import allocation_summary
from screen_native_gain_budget import make_loader, model_batch, shrink_token_counts
from single_probe_marginal_budget import (
    calibrate_price,
    prefix_choices_torch,
    router_ranked_numpy,
    router_ranked_torch,
)
from single_pass_budget_score import token_summary


METRIC_NAMES = ("lpips", "l1_01", "mse_01", "psnr")
PATH_NAMES = ("fixed_router", "endpoint_composite_dynamic")


def load_exact_manifest(path, dataset, expected_split):
    manifest = load_index_manifest(path, dataset.root, dataset.paths)
    split = str(manifest.metadata.get("split", ""))
    if split != expected_split:
        raise ValueError(
            f"manifest {path} has split {split!r}, expected {expected_split!r}"
        )
    indices = np.asarray(manifest.dataset_indices, dtype=np.int64)
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError(f"manifest {path} has invalid or duplicate indices")
    return manifest, indices


def update_endpoint_diagnostics(accumulator, diagnostics):
    batch = next(iter(diagnostics.values()))["base_loss"].shape[0]
    accumulator["count"] += int(batch)
    for name in COMPONENT_NAMES:
        record = accumulator[name]
        record["base_loss_sum"] += float(diagnostics[name]["base_loss"].sum().item())
        record["native_loss_sum"] += float(
            diagnostics[name]["native_loss"].sum().item()
        )
        record["base_map_mean_max_abs_error"] = max(
            record["base_map_mean_max_abs_error"],
            float(diagnostics[name]["base_map_mean_error"].max().item()),
        )
        record["native_map_mean_max_abs_error"] = max(
            record["native_map_mean_max_abs_error"],
            float(diagnostics[name]["native_map_mean_error"].max().item()),
        )


def empty_endpoint_diagnostics():
    result = {"count": 0}
    for name in COMPONENT_NAMES:
        result[name] = {
            "base_loss_sum": 0.0,
            "native_loss_sum": 0.0,
            "base_map_mean_max_abs_error": 0.0,
            "native_map_mean_max_abs_error": 0.0,
        }
    return result


def finalize_endpoint_diagnostics(accumulator):
    count = int(accumulator["count"])
    if count <= 0:
        raise AssertionError("endpoint diagnostic accumulator is empty")
    result = {"count": count}
    for name in COMPONENT_NAMES:
        record = accumulator[name]
        result[name] = {
            "base_loss_mean": record["base_loss_sum"] / count,
            "native_loss_mean": record["native_loss_sum"] / count,
            "endpoint_loss_reduction_mean": (
                record["base_loss_sum"] - record["native_loss_sum"]
            )
            / count,
            "base_map_mean_max_abs_error": record[
                "base_map_mean_max_abs_error"
            ],
            "native_map_mean_max_abs_error": record[
                "native_map_mean_max_abs_error"
            ],
        }
    return result


def main(args):
    from skimage.metrics import structural_similarity

    mot_root = str(Path(args.mot_root).resolve())
    if mot_root not in sys.path:
        sys.path.insert(0, mot_root)
    from train_titok_llamagen_recon import autocast_dtype

    output = Path(args.output_json)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")
    if args.fid_feature != 2048:
        raise ValueError("E24 scoring and paired FID require --fid-feature 2048")
    if not 0 <= args.min_tokens <= args.target_tokens <= args.max_tokens <= 256:
        raise ValueError("invalid token bounds")
    if not 0.0 <= args.allocation_shrink <= 1.0:
        raise ValueError("allocation shrink must be in [0,1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E24 endpoint screen requires CUDA")
    started = time.time()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, model_metadata = build_model(args, device, checkpoint)
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)
    llamagen_root = model_metadata["llamagen_root"]
    if llamagen_root not in sys.path:
        sys.path.insert(0, llamagen_root)
    from dataset.augmentation import center_crop_arr

    dataset = EvalImageDataset(args.data_path, args.image_size, center_crop_arr)
    calibration_manifest, calibration_available = load_exact_manifest(
        args.calibration_index_manifest, dataset, args.calibration_expected_split
    )
    evaluation_manifest, evaluation_available = load_exact_manifest(
        args.evaluation_index_manifest, dataset, args.evaluation_expected_split
    )
    overlap = np.intersect1d(calibration_available, evaluation_available)
    if overlap.size:
        raise ValueError(f"calibration/evaluation overlap at {overlap.size} indices")
    calibration_indices = calibration_available[: args.num_calibration_images]
    evaluation_end = args.evaluation_offset + args.num_evaluation_images
    evaluation_indices = evaluation_available[args.evaluation_offset : evaluation_end]
    if calibration_indices.size != args.num_calibration_images:
        raise ValueError("calibration manifest does not contain requested images")
    if evaluation_indices.size != args.num_evaluation_images:
        raise ValueError("evaluation manifest does not contain requested range")

    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = args.mixed_precision != "none"
    lpips_metric = build_lpips_metric(args, device)
    fid_extractor = (
        FrechetInceptionDistance(feature=args.fid_feature, normalize=False)
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    calibration_loader = make_loader(
        dataset,
        calibration_indices,
        args.calibration_batch_size,
        args.num_workers,
        device,
    )
    component_chunks = {name: [] for name in COMPONENT_NAMES}
    router_chunks = []
    calibration_diagnostics = empty_endpoint_diagnostics()
    with torch.inference_mode():
        for images_01, _ in tqdm(
            calibration_loader, desc="e24_endpoint_calibration", dynamic_ncols=True
        ):
            images_01 = images_01.to(device, non_blocking=True)
            target, x_base, _f_1d, _f_2d, router_score, x_native = model_batch(
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
            update_endpoint_diagnostics(calibration_diagnostics, diagnostics)
            for name in COMPONENT_NAMES:
                component_chunks[name].append(
                    components[name].flatten(1).cpu().numpy().astype(np.float32)
                )
            router_chunks.append(
                router_score.float().flatten(1).cpu().numpy().astype(np.float32)
            )
    calibration_components = {
        name: np.concatenate(component_chunks[name], axis=0)
        for name in COMPONENT_NAMES
    }
    calibration_router = np.concatenate(router_chunks, axis=0)
    component_scales = fit_component_scales(calibration_components)
    calibration_combined = combine_components_numpy(
        calibration_components, component_scales
    )
    calibration_ranked = router_ranked_numpy(calibration_combined, calibration_router)
    price, calibration_raw_tokens = calibrate_price(
        calibration_ranked,
        args.target_tokens,
        args.min_tokens,
        args.max_tokens,
        args.quantize_step,
        args.price_iterations,
    )
    calibration_tokens = shrink_token_counts(
        calibration_raw_tokens,
        args.target_tokens,
        args.min_tokens,
        args.max_tokens,
        args.quantize_step,
        args.allocation_shrink,
    )
    del component_chunks, router_chunks, calibration_components
    del calibration_router, calibration_combined, calibration_ranked
    gc.collect()

    evaluation_loader = make_loader(
        dataset,
        evaluation_indices,
        args.batch_size,
        args.num_workers,
        device,
    )
    feature_chunks = [[], [], []]
    metric_sums = np.zeros((len(PATH_NAMES), len(METRIC_NAMES) + 1), dtype=np.float64)
    allocation_histogram = np.zeros(257, dtype=np.int64)
    raw_allocation_histogram = np.zeros(257, dtype=np.int64)
    evaluation_diagnostics = empty_endpoint_diagnostics()
    observed = 0
    with torch.inference_mode():
        for images_01, _ in tqdm(
            evaluation_loader, desc="e24_endpoint_screen", dynamic_ncols=True
        ):
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
            update_endpoint_diagnostics(evaluation_diagnostics, diagnostics)
            combined = combine_components_torch(components, component_scales)
            ranked = router_ranked_torch(combined, router_score)
            raw_dynamic_tokens = prefix_choices_torch(
                ranked,
                price,
                args.min_tokens,
                args.max_tokens,
                args.quantize_step,
            )
            dynamic_tokens = shrink_token_counts(
                raw_dynamic_tokens,
                args.target_tokens,
                args.min_tokens,
                args.max_tokens,
                args.quantize_step,
                args.allocation_shrink,
            )
            raw_allocation_histogram += np.bincount(
                raw_dynamic_tokens.cpu().numpy(), minlength=257
            )[:257]
            allocation_histogram += np.bincount(
                dynamic_tokens.cpu().numpy(), minlength=257
            )[:257]
            fixed_tokens = torch.full_like(dynamic_tokens, args.target_tokens)
            predictions = []
            for tokens in (fixed_tokens, dynamic_tokens):
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_type,
                    enabled=autocast_enabled,
                ):
                    mask = variable_topk_mask(router_score, tokens, f_1d.dtype)
                    predictions.append(
                        model.llamagen_vq.decoder(
                            (1.0 - mask) * f_1d + mask * f_2d
                        )
                    )

            target_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            feature_chunks[0].append(target_features.cpu().numpy())
            for path_index, prediction in enumerate(predictions):
                features = fid_extractor.inception(
                    to_uint8(prediction, args.llamagen_input_range)
                ).reshape(batch, -1).float()
                feature_chunks[path_index + 1].append(features.cpu().numpy())
                values = per_image_metrics(
                    prediction.float(),
                    target.float(),
                    args.llamagen_input_range,
                    lpips_metric,
                )
                for metric_index, name in enumerate(METRIC_NAMES):
                    metric_sums[path_index, metric_index] += np.asarray(
                        values[name], dtype=np.float64
                    ).sum()
                metric_sums[path_index, -1] += ssim_batch_sum(
                    prediction.float(),
                    target.float(),
                    args.llamagen_input_range,
                    structural_similarity,
                )
            observed += batch
    if observed != evaluation_indices.size:
        raise AssertionError("E24 evaluation count mismatch")

    feature_arrays = [
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for chunks in feature_chunks
    ]
    sufficient = [
        feature_sufficient_statistics(values, device, args.fid_moment_chunk_size)
        for values in feature_arrays
    ]
    moments = [
        moments_from_sufficient_statistics(item[0], item[1], observed)
        for item in sufficient
    ]
    real_mean, real_covariance = moments[0]
    fid_values = [
        float(_compute_fid(real_mean, real_covariance, mean, covariance).item())
        for mean, covariance in moments[1:]
    ]
    metrics = {}
    for path_index, name in enumerate(PATH_NAMES):
        record = {
            metric: float(metric_sums[path_index, index] / observed)
            for index, metric in enumerate(METRIC_NAMES)
        }
        record["ssim"] = float(metric_sums[path_index, -1] / observed)
        record["fid"] = fid_values[path_index]
        metrics[name] = record
    fixed = metrics["fixed_router"]
    dynamic = metrics["endpoint_composite_dynamic"]
    changes = {
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
    }
    allocation = allocation_summary(allocation_histogram, args.target_tokens)
    screen_gates = {
        "mean_k_in_94p5_97p5": 94.5 <= allocation["mean"] <= 97.5,
        "std_k_at_least_8": allocation["std"] >= 8.0,
        "fid_reduction_at_least_0p5_percent": changes["fid_reduction"] >= 0.5,
        "lpips_not_worse_than_0p1_percent": changes["lpips_reduction"] >= -0.1,
    }
    screen_gates["all_pass"] = all(screen_gates.values())

    result = {
        "format": "e24_endpoint_composite_gain_screen_v1",
        "status": "completed",
        "experiment": "E24-endpoint-composite-gain",
        "algorithm": "router_prefix_signed_mass_preserving_endpoint_gain_minus_price",
        "component_weights": COMPONENT_WEIGHTS,
        "component_scales_train_only": component_scales,
        "inception_attribution": "mixed7c_mass_preserving_to_final_2048_feature_mse",
        "lpips_attribution": "learned_linear_heads_mean_preserving_resize",
        "pixel_attribution": "exact_channel_mse_average_pool_16x16",
        "price": float(price),
        "allocation_shrink": float(args.allocation_shrink),
        "calibration_raw_allocation": token_summary(
            calibration_raw_tokens, args.target_tokens
        ),
        "calibration_allocation": token_summary(
            calibration_tokens, args.target_tokens
        ),
        "evaluation_raw_allocation": allocation_summary(
            raw_allocation_histogram, args.target_tokens
        ),
        "evaluation_allocation": allocation,
        "calibration_endpoint_diagnostics": finalize_endpoint_diagnostics(
            calibration_diagnostics
        ),
        "evaluation_endpoint_diagnostics": finalize_endpoint_diagnostics(
            evaluation_diagnostics
        ),
        "metrics": metrics,
        "relative_changes_percent": changes,
        "screen_gates": screen_gates,
        "selection_is_per_image_independent_after_train_calibration": True,
        "test_batch_statistics_used_for_selection": False,
        "test_set_budget_rebalancing_used": False,
        "validation_statistics_used": False,
        "probe_reconstructions_per_image": 1,
        "distinct_probe_k_values": [256],
        "candidate_k_search_reconstructions_per_image": 0,
        "candidate_k_search_loss_evaluations_per_image": 0,
        "final_dynamic_reconstructions_per_image": 1,
        "spatial_ranking": "frozen_original_router",
        "source_image_is_available_to_reconstruction_encoder": True,
        "calibration_source": {
            "split": args.calibration_expected_split,
            "index_manifest": str(Path(args.calibration_index_manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.calibration_index_manifest)),
            "entries_sha256": calibration_manifest.metadata.get("entries_sha256"),
            "num_images": int(calibration_indices.size),
        },
        "evaluation_source": {
            "split": args.evaluation_expected_split,
            "index_manifest": str(Path(args.evaluation_index_manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.evaluation_index_manifest)),
            "entries_sha256": evaluation_manifest.metadata.get("entries_sha256"),
            "num_images": int(evaluation_indices.size),
            "manifest_position_range": [int(args.evaluation_offset), int(evaluation_end)],
            "already_consumed_development_evidence": (
                args.evaluation_expected_split == "e09_screen"
            ),
            "fresh_confirmation": args.evaluation_expected_split == "e09_confirmation",
        },
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_state": model_metadata["checkpoint_state"],
        "data_path": str(Path(args.data_path).resolve()),
        "features_saved_to_disk": False,
        "npz_written": False,
        "reconstructions_saved": False,
        "stats_pt_written": False,
        "output_json_only": True,
        "runtime_seconds": time.time() - started,
        "runtime_args": vars(args),
        "model": model_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"saved {output}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-index-manifest", required=True)
    parser.add_argument("--calibration-expected-split", default="calibration")
    parser.add_argument("--evaluation-index-manifest", required=True)
    parser.add_argument("--evaluation-expected-split", default="e09_screen")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default="/var/tmp/heyefei_ImageNet/train")
    parser.add_argument("--num-calibration-images", type=int, default=10000)
    parser.add_argument("--num-evaluation-images", type=int, default=2000)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--quantize-step", type=int, default=1)
    parser.add_argument("--allocation-shrink", type=float, default=0.4)
    parser.add_argument("--price-iterations", type=int, default=80)
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

