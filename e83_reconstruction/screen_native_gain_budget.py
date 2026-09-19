#!/usr/bin/env python3
"""Train-only screen of a batch-independent one-native-probe budget rule."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    EvalImageDataset,
    build_lpips_metric,
    build_model,
    per_image_metrics,
)
from eval_single_pass_budget import allocation_summary
from single_pass_budget_score import (
    calibrate_threshold,
    token_counts_from_scores_torch,
    token_summary,
)


METRIC_NAMES = ("lpips", "l1_01", "mse_01", "psnr")
PATH_NAMES = ("fixed_router", "native_gain_dynamic")


@torch.no_grad()
def native_mse_gain_score(x_base, x_native, target):
    if x_base.shape != x_native.shape or x_base.shape != target.shape:
        raise ValueError("base, native, and target images must have equal shapes")
    base_error = (x_base.float() - target.float()).square().mean(1, keepdim=True)
    native_error = (x_native.float() - target.float()).square().mean(1, keepdim=True)
    return F.adaptive_avg_pool2d(base_error - native_error, (16, 16)).clamp_min(0.0)


def shrink_token_counts(tokens, target, minimum, maximum, step, shrink):
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("allocation shrink must be in [0, 1]")
    if torch.is_tensor(tokens):
        values = target + shrink * (tokens.float() - target)
        values = minimum + torch.round((values - minimum) / step) * step
        return values.clamp(minimum, maximum).long()
    values = target + shrink * (np.asarray(tokens, dtype=np.float64) - target)
    values = minimum + np.rint((values - minimum) / step) * step
    return np.clip(values, minimum, maximum).astype(np.int64)


def router_ordered_gain(score, router_score):
    if score.shape != router_score.shape or score.shape[-2:] != (16, 16):
        raise ValueError("gain and router score must have equal [B,1,16,16] shapes")
    order = router_score.float().flatten(1).argsort(dim=1, descending=True)
    return torch.gather(score.float().flatten(1), 1, order)


def prefix_tokens_numpy(ordered_gain, price, minimum, maximum, step):
    values = np.asarray(ordered_gain, dtype=np.float32)
    candidates = np.arange(minimum, maximum + 1, step, dtype=np.int64)
    prefixes = np.cumsum(values, axis=1, dtype=np.float32)[:, candidates - 1]
    utility = prefixes - np.float32(price) * candidates[None, :]
    return candidates[np.argmax(utility, axis=1)].astype(np.int64)


def prefix_tokens_torch(ordered_gain, price, minimum, maximum, step):
    candidates = torch.arange(
        minimum, maximum + 1, step, device=ordered_gain.device, dtype=torch.long
    )
    prefixes = ordered_gain.float().cumsum(dim=1)[:, candidates - 1]
    utility = prefixes - float(price) * candidates.float()[None, :]
    return candidates[utility.argmax(dim=1)]


def calibrate_prefix_price(ordered_gain, target, minimum, maximum, step, iterations):
    values = np.asarray(ordered_gain, dtype=np.float32)
    scale = max(float(np.max(values)), 1e-12)
    low, high = -scale, 2.0 * scale
    candidates = []
    for price in (low, high):
        tokens = prefix_tokens_numpy(values, price, minimum, maximum, step)
        candidates.append((price, tokens))
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        tokens = prefix_tokens_numpy(values, middle, minimum, maximum, step)
        candidates.append((middle, tokens))
        if float(tokens.mean()) > target:
            low = middle
        else:
            high = middle
    return min(
        candidates,
        key=lambda item: (abs(float(item[1].mean()) - target), abs(item[0])),
    )


def load_manifest_indices(path, dataset, expected_prefix="teacher_shard_"):
    manifest = load_index_manifest(path, dataset.root, dataset.paths)
    split = str(manifest.metadata.get("split", ""))
    if not split.startswith(expected_prefix):
        raise ValueError(f"manifest split {split!r} is not a train teacher shard")
    indices = np.asarray(manifest.dataset_indices, dtype=np.int64)
    if np.unique(indices).size != indices.size:
        raise ValueError("manifest contains duplicate dataset indices")
    return manifest, split, indices


def make_loader(dataset, indices, batch_size, workers, device):
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=workers > 0,
    )


def model_batch(images_01, model, model_metadata, args, autocast_type, enabled):
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import native_llamagen_feature
    from train_titok_llamagen_recon import convert_image_range

    target = convert_image_range(images_01, args.llamagen_input_range)
    titok_input = convert_image_range(images_01, args.titok_input_range)
    with torch.autocast(
        device_type=images_01.device.type, dtype=autocast_type, enabled=enabled
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
        x_native = model.llamagen_vq.decoder(f_2d)
    return target, x_base, f_1d, f_2d, router_score, x_native


def main(args):
    from skimage.metrics import structural_similarity

    output = Path(args.output_json)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")
    if not 0 <= args.min_tokens <= args.target_tokens <= args.max_tokens <= 256:
        raise ValueError("invalid token bounds")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    started = time.time()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, model_metadata = build_model(args, device, checkpoint)
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)
    mot_root = str(Path(args.mot_root).resolve())
    if mot_root not in sys.path:
        sys.path.insert(0, mot_root)
    llamagen_root = model_metadata["llamagen_root"]
    if llamagen_root not in sys.path:
        sys.path.insert(0, llamagen_root)
    from dataset.augmentation import center_crop_arr
    from train_titok_llamagen_recon import autocast_dtype

    dataset = EvalImageDataset(args.data_path, args.image_size, center_crop_arr)
    calibration_manifest, calibration_split, calibration_available = load_manifest_indices(
        args.calibration_index_manifest, dataset
    )
    evaluation_manifest, evaluation_split, evaluation_available = load_manifest_indices(
        args.evaluation_index_manifest, dataset
    )
    if calibration_split == evaluation_split:
        raise ValueError("calibration and evaluation must use different train shards")
    overlap = np.intersect1d(calibration_available, evaluation_available)
    if overlap.size:
        raise ValueError(f"train calibration/evaluation manifests overlap at {overlap.size} indices")
    if args.evaluation_offset < 0:
        raise ValueError("evaluation offset must be non-negative")
    calibration_indices = calibration_available[: args.num_calibration_images]
    evaluation_end = args.evaluation_offset + args.num_evaluation_images
    evaluation_indices = evaluation_available[args.evaluation_offset : evaluation_end]
    if calibration_indices.size != args.num_calibration_images:
        raise ValueError("calibration manifest does not contain the requested images")
    if evaluation_indices.size != args.num_evaluation_images:
        raise ValueError("evaluation manifest does not contain the requested offset/range")
    if calibration_indices.size <= 1 or evaluation_indices.size <= 1:
        raise ValueError("calibration and evaluation each require at least two images")

    autocast_type = autocast_dtype(args.mixed_precision)
    enabled = device.type == "cuda" and args.mixed_precision != "none"
    calibration_loader = make_loader(
        dataset, calibration_indices, args.calibration_batch_size,
        args.num_workers, device,
    )
    score_chunks = []
    with torch.inference_mode():
        for images_01, _ in tqdm(
            calibration_loader, desc="native_gain_calibration", dynamic_ncols=True
        ):
            images_01 = images_01.to(device, non_blocking=True)
            target, x_base, _f_1d, _f_2d, router_score, x_native = model_batch(
                images_01, model, model_metadata, args, autocast_type, enabled
            )
            score = native_mse_gain_score(x_base, x_native, target)
            signal = (
                score.flatten(1)
                if args.budget_rule == "threshold_count"
                else router_ordered_gain(score, router_score)
            )
            score_chunks.append(signal.cpu().numpy().astype(np.float32))
    calibration_scores = np.concatenate(score_chunks, axis=0)
    if args.budget_rule == "threshold_count":
        selection_parameter, calibration_raw_tokens = calibrate_threshold(
            calibration_scores, args.target_tokens, args.min_tokens,
            args.max_tokens, args.quantize_step, args.threshold_iterations,
        )
    else:
        selection_parameter, calibration_raw_tokens = calibrate_prefix_price(
            calibration_scores, args.target_tokens, args.min_tokens,
            args.max_tokens, args.quantize_step, args.threshold_iterations,
        )
    calibration_tokens = shrink_token_counts(
        calibration_raw_tokens, args.target_tokens, args.min_tokens,
        args.max_tokens, args.quantize_step, args.allocation_shrink,
    )

    evaluation_loader = make_loader(
        dataset, evaluation_indices, args.batch_size, args.num_workers, device
    )
    lpips_metric = build_lpips_metric(args, device)
    fid_extractor = FrechetInceptionDistance(
        feature=args.fid_feature, normalize=False
    ).to(device).eval().requires_grad_(False)
    feature_chunks = [[], [], []]
    metric_sums = np.zeros((len(PATH_NAMES), len(METRIC_NAMES) + 1), dtype=np.float64)
    allocation_histogram = np.zeros(257, dtype=np.int64)
    raw_allocation_histogram = np.zeros(257, dtype=np.int64)
    observed = 0
    with torch.inference_mode():
        for images_01, _ in tqdm(
            evaluation_loader, desc="native_gain_train_screen", dynamic_ncols=True
        ):
            batch = images_01.shape[0]
            images_01 = images_01.to(device, non_blocking=True)
            target, x_base, f_1d, f_2d, router_score, x_native = model_batch(
                images_01, model, model_metadata, args, autocast_type, enabled
            )
            score = native_mse_gain_score(x_base, x_native, target)
            if args.budget_rule == "threshold_count":
                raw_dynamic_tokens = token_counts_from_scores_torch(
                    score, selection_parameter, args.min_tokens, args.max_tokens,
                    args.quantize_step,
                )
            else:
                raw_dynamic_tokens = prefix_tokens_torch(
                    router_ordered_gain(score, router_score), selection_parameter,
                    args.min_tokens, args.max_tokens, args.quantize_step,
                )
            dynamic_tokens = shrink_token_counts(
                raw_dynamic_tokens, args.target_tokens, args.min_tokens,
                args.max_tokens, args.quantize_step, args.allocation_shrink,
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
                    device_type=device.type, dtype=autocast_type, enabled=enabled
                ):
                    mask = variable_topk_mask(router_score, tokens, f_1d.dtype)
                    predictions.append(
                        model.llamagen_vq.decoder((1.0 - mask) * f_1d + mask * f_2d)
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
                    prediction.float(), target.float(), args.llamagen_input_range, lpips_metric
                )
                for metric_index, name in enumerate(METRIC_NAMES):
                    metric_sums[path_index, metric_index] += np.asarray(
                        values[name], dtype=np.float64
                    ).sum()
                metric_sums[path_index, -1] += ssim_batch_sum(
                    prediction.float(), target.float(), args.llamagen_input_range,
                    structural_similarity,
                )
            observed += batch
    if observed != evaluation_indices.size:
        raise AssertionError("native-gain evaluation count mismatch")

    feature_arrays = [
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for chunks in feature_chunks
    ]
    del model, lpips_metric, fid_extractor
    gc.collect()
    torch.cuda.empty_cache()
    sufficient = [
        feature_sufficient_statistics(values, device, args.fid_moment_chunk_size)
        for values in feature_arrays
    ]
    moments = [
        moments_from_sufficient_statistics(item[0], item[1], observed)
        for item in sufficient
    ]
    real_mean, real_cov = moments[0]
    fid_values = [
        float(_compute_fid(real_mean, real_cov, mean, covariance).item())
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
    dynamic = metrics["native_gain_dynamic"]
    result = {
        "format": "native_gain_budget_train_screen_v1",
        "status": "completed",
        "algorithm": f"shrunk_{args.budget_rule}_native_mse_gain",
        "budget_rule": args.budget_rule,
        "selection_is_per_image_independent": True,
        "probe_reconstructions_per_image": 1,
        "distinct_probe_k_values": [256],
        "candidate_k_search_reconstructions_per_image": 0,
        "candidate_k_search_loss_evaluations_per_image": 0,
        "probe_pixel_loss_maps_per_image": 1,
        "test_batch_statistics_used_for_selection": False,
        "test_set_budget_rebalancing_used": False,
        "validation_statistics_used": False,
        "source_image_is_available_to_reconstruction_encoder": True,
        "spatial_ranking": "frozen_original_router",
        "selection_parameter_name": (
            "threshold" if args.budget_rule == "threshold_count" else "price"
        ),
        "selection_parameter": float(selection_parameter),
        "threshold": (
            float(selection_parameter) if args.budget_rule == "threshold_count" else None
        ),
        "price": (
            float(selection_parameter) if args.budget_rule == "router_prefix" else None
        ),
        "allocation_shrink": float(args.allocation_shrink),
        "calibration_raw_allocation": token_summary(
            calibration_raw_tokens, args.target_tokens
        ),
        "calibration_allocation": token_summary(calibration_tokens, args.target_tokens),
        "evaluation_raw_allocation": allocation_summary(
            raw_allocation_histogram, args.target_tokens
        ),
        "evaluation_allocation": allocation_summary(allocation_histogram, args.target_tokens),
        "metrics": metrics,
        "relative_changes_percent": {
            "fid_reduction": 100.0 * (fixed["fid"] - dynamic["fid"]) / fixed["fid"],
            "lpips_reduction": 100.0 * (fixed["lpips"] - dynamic["lpips"]) / fixed["lpips"],
            "l1_reduction": 100.0 * (fixed["l1_01"] - dynamic["l1_01"]) / fixed["l1_01"],
            "mse_reduction": 100.0 * (fixed["mse_01"] - dynamic["mse_01"]) / fixed["mse_01"],
            "psnr_increase": dynamic["psnr"] - fixed["psnr"],
            "ssim_increase": dynamic["ssim"] - fixed["ssim"],
        },
        "calibration_source": {
            "split": calibration_split,
            "index_manifest": str(Path(args.calibration_index_manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.calibration_index_manifest)),
            "num_images": int(calibration_indices.size),
        },
        "evaluation_source": {
            "split": evaluation_split,
            "index_manifest": str(Path(args.evaluation_index_manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.evaluation_index_manifest)),
            "num_images": int(evaluation_indices.size),
            "manifest_position_range": [
                int(args.evaluation_offset), int(evaluation_end)
            ],
        },
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_state": model_metadata["checkpoint_state"],
        "data_path": str(Path(args.data_path).resolve()),
        "features_saved_to_disk": False,
        "npz_written": False,
        "stats_pt_written": False,
        "output_json_only": True,
        "runtime_seconds": float(time.time() - started),
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
    parser.add_argument("--evaluation-index-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default="/var/tmp/heyefei_ImageNet/train")
    parser.add_argument("--num-calibration-images", type=int, default=10000)
    parser.add_argument("--num-evaluation-images", type=int, default=2000)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--quantize-step", type=int, default=1)
    parser.add_argument(
        "--budget-rule", choices=["threshold_count", "router_prefix"],
        default="threshold_count",
    )
    parser.add_argument("--allocation-shrink", type=float, default=1.0)
    parser.add_argument("--threshold-iterations", type=int, default=80)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--fid-moment-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--lpips-net", choices=["alex", "vgg", "squeeze", "llamagen_vgg"], default="alex"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
