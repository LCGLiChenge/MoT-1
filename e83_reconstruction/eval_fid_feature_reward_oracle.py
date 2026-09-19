#!/usr/bin/env python3
"""Evaluate FID-aware deterministic-group budget rewards under an exact mean budget.

This is a target-aware upper-bound analysis, not an inference-time policy.  It
decodes every candidate budget, keeps Inception features in memory only, and
compares raw and per-image group-normalized reward allocations at equal average
token count.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from tqdm import tqdm

from eval_fid_from_dynamic_choices import to_uint8, variable_topk_mask
from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    DEFAULT_MOT_ROOT,
    EvalImageDataset,
    build_model,
    grouped_exact_budget_dp,
)


def group_standardize(losses: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    losses = np.asarray(losses, dtype=np.float64)
    mean = losses.mean(axis=1, keepdims=True)
    std = losses.std(axis=1, keepdims=True)
    return (losses - mean) / np.maximum(std, float(epsilon))


def feature_moments(features: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(np.asarray(features)).to(device=device, dtype=torch.float64)
    count = values.shape[0]
    if count <= 1:
        raise ValueError("FID requires at least two samples")
    mean = values.mean(dim=0)
    covariance = (values.T @ values - float(count) * mean[:, None] @ mean[None, :]) / float(count - 1)
    return mean, covariance


def token_summary(candidate_tokens: np.ndarray, choices: np.ndarray) -> dict[str, object]:
    selected = candidate_tokens[choices]
    unique, counts = np.unique(selected, return_counts=True)
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std()),
        "min": int(selected.min()),
        "max": int(selected.max()),
        "histogram": {str(int(token)): int(count) for token, count in zip(unique, counts)},
    }


def selected_curve_metrics(
    curve_data: dict[str, np.ndarray], choices: np.ndarray
) -> dict[str, float]:
    rows = np.arange(choices.size)
    return {
        output_name: float(curve_data[array_name][rows, choices].mean())
        for output_name, array_name in (
            ("lpips", "lpips_curves"),
            ("l1_01", "l1_01_curves"),
            ("mse_01", "mse_01_curves"),
            ("psnr", "psnr_curves"),
        )
    }


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    candidate_tokens = np.asarray(sorted(set(args.candidate_tokens)), dtype=np.int64)
    if candidate_tokens[0] < 0 or candidate_tokens[-1] > 256:
        raise ValueError("candidate tokens must lie in [0, 256]")
    if args.target_tokens not in candidate_tokens:
        raise ValueError("target tokens must be one of the candidates")

    curve_archive = np.load(args.metric_curves_npz)
    curve_tokens = np.asarray(curve_archive["candidate_tokens"], dtype=np.int64)
    if not np.array_equal(candidate_tokens, curve_tokens):
        raise ValueError("candidate tokens disagree with the cached metric curves")
    required_curves = ("lpips_curves", "l1_01_curves", "mse_01_curves", "psnr_curves")
    curve_data = {name: np.asarray(curve_archive[name]) for name in required_curves}
    available_images = min(values.shape[0] for values in curve_data.values())
    count = available_images if args.num_images <= 0 else min(args.num_images, available_images)
    curve_data = {name: values[:count] for name, values in curve_data.items()}

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
    if len(dataset) < count:
        raise ValueError(f"dataset has {len(dataset)} images but {count} are required")
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    fid_extractor = FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)
    real_chunks: list[np.ndarray] = []
    candidate_chunks: list[list[np.ndarray]] = [[] for _ in candidate_tokens]
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"

    with torch.inference_mode():
        progress = tqdm(loader, desc="candidate_inception_features", dynamic_ncols=True)
        seen = 0
        for images_01, _indices in progress:
            images_01 = images_01.to(device, non_blocking=True)
            target = convert_image_range(images_01, args.llamagen_input_range)
            titok_input = convert_image_range(images_01, args.titok_input_range)
            with torch.autocast(device_type=device.type, dtype=autocast_type, enabled=autocast_enabled):
                x_base, extra = model(titok_input)
                f_1d = extra["f_1d_lg"]
                f_2d, _ = native_llamagen_feature(
                    model.llamagen_vq,
                    target,
                    model_metadata["codebook_embed_dim"],
                    allow_encoder_grad=False,
                )
                score_logits, _ = model.router(f_1d, x_base, f_2d)

            real_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(images_01.shape[0], -1)
            real_chunks.append(real_features.float().cpu().numpy())

            for candidate_index, token_count in enumerate(candidate_tokens.tolist()):
                tokens = torch.full(
                    (images_01.shape[0],), token_count, device=device, dtype=torch.long
                )
                with torch.autocast(
                    device_type=device.type, dtype=autocast_type, enabled=autocast_enabled
                ):
                    mask = variable_topk_mask(score_logits, tokens, f_1d.dtype)
                    mixed = (1.0 - mask) * f_1d + mask * f_2d
                    prediction = model.llamagen_vq.decoder(mixed)
                features = fid_extractor.inception(
                    to_uint8(prediction, args.llamagen_input_range)
                ).reshape(images_01.shape[0], -1)
                candidate_chunks[candidate_index].append(features.float().cpu().numpy())
            seen += images_01.shape[0]
            progress.set_postfix(images=seen)

    real_features = np.concatenate(real_chunks, axis=0).astype(np.float32, copy=False)
    candidate_features = np.stack(
        [np.concatenate(chunks, axis=0) for chunks in candidate_chunks], axis=1
    ).astype(np.float32, copy=False)
    if real_features.shape[0] != count or candidate_features.shape[:2] != (count, len(candidate_tokens)):
        raise AssertionError("feature collection shape mismatch")

    raw_feature_losses = np.empty((count, len(candidate_tokens)), dtype=np.float64)
    white_feature_losses = np.empty_like(raw_feature_losses)
    real_std = real_features.std(axis=0, dtype=np.float64)
    std_floor = max(float(np.median(real_std) * args.whiten_std_floor_ratio), 1e-6)
    white_scale = np.maximum(real_std, std_floor).astype(np.float32)
    for candidate_index in range(len(candidate_tokens)):
        difference = candidate_features[:, candidate_index] - real_features
        raw_feature_losses[:, candidate_index] = np.square(difference, dtype=np.float64).mean(axis=1)
        white_difference = difference / white_scale
        white_feature_losses[:, candidate_index] = np.square(
            white_difference, dtype=np.float64
        ).mean(axis=1)

    fixed_index = int(np.where(candidate_tokens == args.target_tokens)[0][0])
    mse_group_losses = group_standardize(curve_data["mse_01_curves"])
    lpips_group_losses = group_standardize(curve_data["lpips_curves"])
    inception_group_losses = group_standardize(raw_feature_losses)
    choices: dict[str, np.ndarray] = {
        "fixed": np.full(count, fixed_index, dtype=np.int64),
        "mse_raw": grouped_exact_budget_dp(
            curve_data["mse_01_curves"], candidate_tokens, args.target_tokens, args.allocation_group_size
        ),
        "mse_group_z": grouped_exact_budget_dp(
            mse_group_losses,
            candidate_tokens,
            args.target_tokens,
            args.allocation_group_size,
        ),
        "lpips_raw": grouped_exact_budget_dp(
            curve_data["lpips_curves"], candidate_tokens, args.target_tokens, args.allocation_group_size
        ),
        "lpips_group_z": grouped_exact_budget_dp(
            lpips_group_losses,
            candidate_tokens,
            args.target_tokens,
            args.allocation_group_size,
        ),
        "inception_raw": grouped_exact_budget_dp(
            raw_feature_losses, candidate_tokens, args.target_tokens, args.allocation_group_size
        ),
        "inception_group_z": grouped_exact_budget_dp(
            inception_group_losses,
            candidate_tokens,
            args.target_tokens,
            args.allocation_group_size,
        ),
        "inception_white_raw": grouped_exact_budget_dp(
            white_feature_losses, candidate_tokens, args.target_tokens, args.allocation_group_size
        ),
        "inception_white_group_z": grouped_exact_budget_dp(
            group_standardize(white_feature_losses),
            candidate_tokens,
            args.target_tokens,
            args.allocation_group_size,
        ),
    }
    for weight in args.lpips_aux_weights:
        weight_name = format(float(weight), "g").replace(".", "p")
        choices[f"inception_lpips_group_w{weight_name}"] = grouped_exact_budget_dp(
            inception_group_losses + float(weight) * lpips_group_losses,
            candidate_tokens,
            args.target_tokens,
            args.allocation_group_size,
        )
    for weight in args.mse_aux_weights:
        weight_name = format(float(weight), "g").replace(".", "p")
        choices[f"inception_lpips_mse_group_w{weight_name}"] = grouped_exact_budget_dp(
            inception_group_losses
            + float(args.combo_lpips_weight) * lpips_group_losses
            + float(weight) * mse_group_losses,
            candidate_tokens,
            args.target_tokens,
            args.allocation_group_size,
        )

    real_mean, real_cov = feature_moments(real_features, device)
    rows = np.arange(count)
    metrics: dict[str, dict[str, object]] = {}
    for name, path_choices in choices.items():
        selected_features = candidate_features[rows, path_choices]
        fake_mean, fake_cov = feature_moments(selected_features, device)
        fid = float(_compute_fid(real_mean, real_cov, fake_mean, fake_cov).item())
        path_metrics: dict[str, object] = {
            "fid": fid,
            "allocation": token_summary(candidate_tokens, path_choices),
        }
        path_metrics.update(selected_curve_metrics(curve_data, path_choices))
        path_metrics["paired_inception_mse"] = float(
            raw_feature_losses[rows, path_choices].mean()
        )
        path_metrics["paired_inception_white_mse"] = float(
            white_feature_losses[rows, path_choices].mean()
        )
        metrics[name] = path_metrics

    fixed_fid = float(metrics["fixed"]["fid"])
    result = {
        "analysis_type": "fid_feature_reward_dynamic_budget_oracle",
        "warning": "Every non-fixed allocation uses target reconstruction information and is not deployable.",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "metric_curves_npz": str(Path(args.metric_curves_npz).resolve()),
        "num_images": count,
        "candidate_tokens": candidate_tokens.tolist(),
        "target_tokens": int(args.target_tokens),
        "fid_feature": int(args.fid_feature),
        "fid_input": "uint8",
        "metrics": metrics,
        "fid_change_vs_fixed_percent": {
            name: 100.0 * (float(values["fid"]) - fixed_fid) / fixed_fid
            for name, values in metrics.items()
            if name != "fixed"
        },
        "settings": {
            "batch_size": int(args.batch_size),
            "mixed_precision": args.mixed_precision,
            "allocation_group_size": int(args.allocation_group_size),
            "whiten_std_floor_ratio": float(args.whiten_std_floor_ratio),
            "whiten_std_floor": std_floor,
            "lpips_aux_weights": [float(weight) for weight in args.lpips_aux_weights],
            "combo_lpips_weight": float(args.combo_lpips_weight),
            "mse_aux_weights": [float(weight) for weight in args.mse_aux_weights],
            "seed": int(args.seed),
            "features_saved_to_disk": False,
        },
        "model": model_metadata,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"saved {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--metric-curves-npz",
        default="results/oracle_width_sweep/mid_32_160_2k.npz",
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--mot-root", default=str(DEFAULT_MOT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", choices=["zero_1", "minus1_1"], default="zero_1")
    parser.add_argument("--llamagen-input-range", choices=["zero_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument(
        "--candidate-tokens",
        type=int,
        nargs="+",
        default=[32, 48, 64, 80, 96, 112, 128, 144, 160],
    )
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--allocation-group-size", type=int, default=2000)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--whiten-std-floor-ratio", type=float, default=0.05)
    parser.add_argument(
        "--lpips-aux-weights",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Weights for z(Inception-MSE) + weight * z(LPIPS) reward sweeps.",
    )
    parser.add_argument("--combo-lpips-weight", type=float, default=2.0)
    parser.add_argument(
        "--mse-aux-weights",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="MSE weights added to z(Inception) + combo_lpips_weight * z(LPIPS).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-model-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--titok-root", default=None)
    parser.add_argument("--titok-config", default=None)
    parser.add_argument("--titok-ckpt", default=None)
    parser.add_argument("--llamagen-root", default=None)
    parser.add_argument("--llamagen-ckpt", default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--codebook-embed-dim", type=int, default=None)
    parser.add_argument("--lg-latent-channels", type=int, default=None)
    parser.add_argument("--lg-head-channels", type=int, default=None)
    parser.add_argument("--codebook-temperature", type=float, default=None)
    parser.add_argument("--router-hidden-dim", type=int, default=None)
    parser.add_argument("--router-depth", type=int, default=None)
    parser.add_argument("--router-target-mean-ratio", type=float, default=None)
    parser.add_argument("--router-min-ratio", type=float, default=None)
    parser.add_argument("--router-max-ratio", type=float, default=None)
    parser.add_argument("--router-detach-inputs", action=argparse.BooleanOptionalAction, default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
