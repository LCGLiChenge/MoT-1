#!/usr/bin/env python3
"""Evaluate target-aware spatial rankings with the balanced dynamic-budget oracle.

This is an evaluation-only upper-bound analysis.  It compares a fixed 96-token
Router ranking, a fixed 96-token target-aware ranking, and the same target-aware
ranking with an exact-mean dynamic budget selected by the validated balanced
reward: z(paired Inception MSE) + 2 z(LPIPS) + 4 z(pixel MSE).
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

from eval_fid_feature_reward_oracle import (
    feature_moments,
    group_standardize,
    selected_curve_metrics,
    token_summary,
)
from eval_fid_from_dynamic_choices import to_uint8, variable_topk_mask
from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    DEFAULT_MOT_ROOT,
    EvalImageDataset,
    build_model,
    grouped_exact_budget_dp,
)
from eval_spatial_ranking_dynamic_oracle import spatial_score


RANKINGS = ("base_l1", "native_mse_gain")


def load_metric_curves(
    path: str,
    candidate_tokens: np.ndarray,
    count: int,
) -> dict[str, np.ndarray]:
    archive = np.load(path)
    archive_tokens = np.asarray(archive["candidate_tokens"], dtype=np.int64)
    if not np.array_equal(candidate_tokens, archive_tokens):
        raise ValueError(f"candidate tokens disagree with {path}")
    mapping = {
        "lpips_curves": "lpips_curves",
        "l1_01_curves": "l1_01_curves",
        "mse_01_curves": "mse_01_curves",
        "psnr_curves": "psnr_curves",
    }
    curves = {output: np.asarray(archive[source])[:count] for output, source in mapping.items()}
    if any(values.shape != (count, candidate_tokens.size) for values in curves.values()):
        raise ValueError(f"metric-curve shape mismatch in {path}")
    return curves


def fid_from_features(
    real_mean: torch.Tensor,
    real_cov: torch.Tensor,
    fake_features: np.ndarray,
    device: torch.device,
) -> float:
    fake_mean, fake_cov = feature_moments(fake_features, device)
    return float(_compute_fid(real_mean, real_cov, fake_mean, fake_cov).item())


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

    ranking_archive = np.load(args.ranking_curves_npz)
    available_images = int(ranking_archive["lpips_curves"].shape[0])
    ranking_archive.close()
    count = available_images if args.num_images <= 0 else min(args.num_images, available_images)
    ranking_curves = load_metric_curves(args.ranking_curves_npz, candidate_tokens, count)
    router_curves = load_metric_curves(args.router_curves_npz, candidate_tokens, count)

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
    router_fixed_chunks: list[np.ndarray] = []
    ranking_chunks: list[list[np.ndarray]] = [[] for _ in candidate_tokens]
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"balanced_spatial_{args.ranking}", dynamic_ncols=True)
        seen = 0
        for images_01, _indices in progress:
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
                x_native = (
                    model.llamagen_vq.decoder(f_2d)
                    if args.ranking == "native_mse_gain"
                    else None
                )
            ranking_score = spatial_score(
                args.ranking,
                router_score,
                f_1d,
                f_2d,
                x_base,
                target,
                x_native,
            )

            real_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(images_01.shape[0], -1)
            real_chunks.append(real_features.float().cpu().numpy())

            fixed_tokens = torch.full(
                (images_01.shape[0],),
                args.target_tokens,
                device=device,
                dtype=torch.long,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_type,
                enabled=autocast_enabled,
            ):
                router_mask = variable_topk_mask(router_score, fixed_tokens, f_1d.dtype)
                router_prediction = model.llamagen_vq.decoder(
                    (1.0 - router_mask) * f_1d + router_mask * f_2d
                )
            router_features = fid_extractor.inception(
                to_uint8(router_prediction, args.llamagen_input_range)
            ).reshape(images_01.shape[0], -1)
            router_fixed_chunks.append(router_features.float().cpu().numpy())

            for candidate_index, token_count in enumerate(candidate_tokens.tolist()):
                tokens = torch.full(
                    (images_01.shape[0],), token_count, device=device, dtype=torch.long
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_type,
                    enabled=autocast_enabled,
                ):
                    mask = variable_topk_mask(ranking_score, tokens, f_1d.dtype)
                    prediction = model.llamagen_vq.decoder(
                        (1.0 - mask) * f_1d + mask * f_2d
                    )
                features = fid_extractor.inception(
                    to_uint8(prediction, args.llamagen_input_range)
                ).reshape(images_01.shape[0], -1)
                ranking_chunks[candidate_index].append(features.float().cpu().numpy())
            seen += images_01.shape[0]
            progress.set_postfix(images=seen)

    real_features = np.concatenate(real_chunks, axis=0).astype(np.float32, copy=False)
    router_fixed_features = np.concatenate(router_fixed_chunks, axis=0).astype(
        np.float32, copy=False
    )
    ranking_features = np.stack(
        [np.concatenate(chunks, axis=0) for chunks in ranking_chunks], axis=1
    ).astype(np.float32, copy=False)
    expected_shape = (count, candidate_tokens.size)
    if ranking_features.shape[:2] != expected_shape:
        raise AssertionError("feature collection shape mismatch")

    paired_inception_losses = np.empty(expected_shape, dtype=np.float64)
    for candidate_index in range(candidate_tokens.size):
        difference = ranking_features[:, candidate_index] - real_features
        paired_inception_losses[:, candidate_index] = np.square(
            difference, dtype=np.float64
        ).mean(axis=1)

    balanced_losses = (
        group_standardize(paired_inception_losses)
        + float(args.lpips_weight) * group_standardize(ranking_curves["lpips_curves"])
        + float(args.mse_weight) * group_standardize(ranking_curves["mse_01_curves"])
    )
    balanced_choices = grouped_exact_budget_dp(
        balanced_losses,
        candidate_tokens,
        args.target_tokens,
        args.allocation_group_size,
    )
    fixed_index = int(np.where(candidate_tokens == args.target_tokens)[0][0])
    fixed_choices = np.full(count, fixed_index, dtype=np.int64)
    rows = np.arange(count)

    real_mean, real_cov = feature_moments(real_features, device)
    selected_features = {
        "router_fixed": router_fixed_features,
        "ranking_fixed": ranking_features[rows, fixed_choices],
        "ranking_balanced": ranking_features[rows, balanced_choices],
    }
    metrics: dict[str, dict[str, object]] = {}
    for name, features in selected_features.items():
        choices = fixed_choices if name != "ranking_balanced" else balanced_choices
        curves = router_curves if name == "router_fixed" else ranking_curves
        path_metrics: dict[str, object] = {
            "fid": fid_from_features(real_mean, real_cov, features, device),
            "allocation": token_summary(candidate_tokens, choices),
        }
        path_metrics.update(selected_curve_metrics(curves, choices))
        if name != "router_fixed":
            path_metrics["paired_inception_mse"] = float(
                paired_inception_losses[rows, choices].mean()
            )
        metrics[name] = path_metrics

    router_balanced_reference = None
    if args.router_balanced_json:
        reference_path = Path(args.router_balanced_json)
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())
            reference_metrics = reference.get("metrics", {}).get(
                "inception_lpips_mse_group_w4"
            )
            if int(reference.get("num_images", -1)) == count and reference_metrics:
                router_balanced_reference = {
                    "source": str(reference_path.resolve()),
                    "metrics": reference_metrics,
                }

    router_fixed_fid = float(metrics["router_fixed"]["fid"])
    result = {
        "analysis_type": "balanced_target_aware_spatial_ranking_oracle",
        "warning": "The spatial ranking and dynamic budget both use target information and are not deployable.",
        "ranking": args.ranking,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "ranking_curves_npz": str(Path(args.ranking_curves_npz).resolve()),
        "router_curves_npz": str(Path(args.router_curves_npz).resolve()),
        "num_images": count,
        "candidate_tokens": candidate_tokens.tolist(),
        "target_tokens": int(args.target_tokens),
        "fid_feature": int(args.fid_feature),
        "fid_input": "uint8",
        "balanced_reward": {
            "formula": "z(paired_inception_mse) + lpips_weight*z(lpips) + mse_weight*z(mse_01)",
            "lpips_weight": float(args.lpips_weight),
            "mse_weight": float(args.mse_weight),
        },
        "metrics": metrics,
        "fid_change_vs_router_fixed_percent": {
            name: 100.0 * (float(values["fid"]) - router_fixed_fid) / router_fixed_fid
            for name, values in metrics.items()
            if name != "router_fixed"
        },
        "router_balanced_reference": router_balanced_reference,
        "settings": {
            "allocation_group_size": int(args.allocation_group_size),
            "batch_size": int(args.batch_size),
            "mixed_precision": args.mixed_precision,
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
    parser.add_argument("--ranking", choices=RANKINGS, required=True)
    parser.add_argument("--ranking-curves-npz", required=True)
    parser.add_argument(
        "--router-curves-npz",
        default="results/oracle_width_sweep/mid_32_160_2k.npz",
    )
    parser.add_argument(
        "--router-balanced-json",
        default="results/rl_reward_gates/fid_feature_lpips2_mse_sweep_32_160_2k.json",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--mot-root", default=str(DEFAULT_MOT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--lpips-weight", type=float, default=2.0)
    parser.add_argument("--mse-weight", type=float, default=4.0)
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
