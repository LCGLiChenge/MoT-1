#!/usr/bin/env python3
"""Evaluate a first-order FID-aware target grid ranking at an exact mean budget."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import (
    FrechetInceptionDistance,
    _compute_fid,
    interpolate_bilinear_2d_like_tensorflow1x,
)
from tqdm import tqdm

from eval_balanced_spatial_oracle import fid_from_features
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
    build_lpips_metric,
    build_model,
    grouped_exact_budget_dp,
    per_image_metrics,
    to_lpips_range,
    to_zero_one,
)


def differentiable_inception_2048(inception: torch.nn.Module, image_255: torch.Tensor) -> torch.Tensor:
    """TorchMetrics FID Inception forward without the uint8-only input assertion."""
    x = image_255.to(torch.float32)
    if inception.use_antialias:
        x = F.interpolate(
            x,
            size=(inception.INPUT_IMAGE_SIZE, inception.INPUT_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    else:
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(inception.INPUT_IMAGE_SIZE, inception.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
    x = (x - 128.0) / 128.0
    x = inception.Conv2d_1a_3x3(x)
    x = inception.Conv2d_2a_3x3(x)
    x = inception.Conv2d_2b_3x3(x)
    x = inception.MaxPool_1(x)
    x = inception.Conv2d_3b_1x1(x)
    x = inception.Conv2d_4a_3x3(x)
    x = inception.MaxPool_2(x)
    x = inception.Mixed_5b(x)
    x = inception.Mixed_5c(x)
    x = inception.Mixed_5d(x)
    x = inception.Mixed_6a(x)
    x = inception.Mixed_6b(x)
    x = inception.Mixed_6c(x)
    x = inception.Mixed_6d(x)
    x = inception.Mixed_6e(x)
    x = inception.Mixed_7a(x)
    x = inception.Mixed_7b(x)
    x = inception.Mixed_7c(x)
    return torch.flatten(inception.AvgPool(x), 1)


def spatial_standardize(score: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    mean = score.mean(dim=(2, 3), keepdim=True)
    std = score.std(dim=(2, 3), keepdim=True, unbiased=False)
    return (score - mean) / (std + float(epsilon))


def balanced_gradient_score(
    f_1d: torch.Tensor,
    f_2d: torch.Tensor,
    target: torch.Tensor,
    target_inception_features: torch.Tensor,
    decoder: torch.nn.Module,
    inception: torch.nn.Module,
    lpips_metric: torch.nn.Module,
    image_range: str,
    probe_ratio: float,
    lpips_weight: float,
    mse_weight: float,
    autocast_type: torch.dtype,
    autocast_enabled: bool,
) -> torch.Tensor:
    """First-order per-grid loss reduction for the balanced reconstruction reward."""
    with torch.enable_grad():
        probe = torch.full(
            (f_1d.shape[0], 1, f_1d.shape[2], f_1d.shape[3]),
            float(probe_ratio),
            device=f_1d.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.autocast(
            device_type=f_1d.device.type,
            dtype=autocast_type,
            enabled=autocast_enabled,
        ):
            prediction = decoder((1.0 - probe) * f_1d.detach() + probe * f_2d.detach())
        prediction_01 = to_zero_one(prediction.float(), image_range).clamp(0.0, 1.0)
        target_01 = to_zero_one(target.detach().float(), image_range).clamp(0.0, 1.0)
        prediction_features = differentiable_inception_2048(inception, prediction_01 * 255.0)
        inception_loss = (
            prediction_features - target_inception_features.detach().float()
        ).square().mean(dim=1)
        lpips_loss = lpips_metric(
            to_lpips_range(prediction.float(), image_range),
            to_lpips_range(target.detach().float(), image_range),
        ).reshape(f_1d.shape[0], -1).mean(dim=1)
        mse_loss = (prediction_01 - target_01).square().flatten(1).mean(dim=1)
        inception_gradient = torch.autograd.grad(
            inception_loss.sum(), probe, retain_graph=True, only_inputs=True
        )[0]
        lpips_gradient = torch.autograd.grad(
            lpips_loss.sum(), probe, retain_graph=True, only_inputs=True
        )[0]
        mse_gradient = torch.autograd.grad(mse_loss.sum(), probe, only_inputs=True)[0]
    return (
        spatial_standardize(-inception_gradient.detach().float())
        + float(lpips_weight) * spatial_standardize(-lpips_gradient.detach().float())
        + float(mse_weight) * spatial_standardize(-mse_gradient.detach().float())
    )


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
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    lpips_metric = build_lpips_metric(args, device)
    fid_extractor = FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)
    if args.fid_feature != 2048:
        raise ValueError("balanced gradient scoring requires --fid-feature 2048")
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"

    real_chunks: list[np.ndarray] = []
    router_fixed_chunks: list[np.ndarray] = []
    ranking_chunks: list[list[np.ndarray]] = [[] for _ in candidate_tokens]
    curve_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in ("lpips_curves", "l1_01_curves", "mse_01_curves", "psnr_curves")
    }

    progress = tqdm(loader, desc="balanced_gradient_spatial", dynamic_ncols=True)
    seen = 0
    for images_01, _indices in progress:
        images_01 = images_01.to(device, non_blocking=True)
        target = convert_image_range(images_01, args.llamagen_input_range)
        titok_input = convert_image_range(images_01, args.titok_input_range)
        with torch.no_grad(), torch.autocast(
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
        with torch.no_grad():
            target_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(images_01.shape[0], -1)
        score = balanced_gradient_score(
            f_1d,
            f_2d,
            target,
            target_features,
            model.llamagen_vq.decoder,
            fid_extractor.inception,
            lpips_metric,
            args.llamagen_input_range,
            args.probe_ratio,
            args.lpips_weight,
            args.mse_weight,
            autocast_type,
            autocast_enabled,
        )
        real_chunks.append(target_features.float().cpu().numpy())

        fixed_tokens = torch.full(
            (images_01.shape[0],), args.target_tokens, device=device, dtype=torch.long
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=autocast_type,
            enabled=autocast_enabled,
        ):
            router_mask = variable_topk_mask(router_score, fixed_tokens, f_1d.dtype)
            router_prediction = model.llamagen_vq.decoder(
                (1.0 - router_mask) * f_1d + router_mask * f_2d
            )
        with torch.no_grad():
            router_features = fid_extractor.inception(
                to_uint8(router_prediction, args.llamagen_input_range)
            ).reshape(images_01.shape[0], -1)
        router_fixed_chunks.append(router_features.float().cpu().numpy())

        batch_curves = {name: [] for name in curve_chunks}
        for candidate_index, token_count in enumerate(candidate_tokens.tolist()):
            tokens = torch.full(
                (images_01.shape[0],), token_count, device=device, dtype=torch.long
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=autocast_type,
                enabled=autocast_enabled,
            ):
                mask = variable_topk_mask(score, tokens, f_1d.dtype)
                prediction = model.llamagen_vq.decoder((1.0 - mask) * f_1d + mask * f_2d)
            with torch.no_grad():
                features = fid_extractor.inception(
                    to_uint8(prediction, args.llamagen_input_range)
                ).reshape(images_01.shape[0], -1)
                values = per_image_metrics(
                    prediction.float(), target.float(), args.llamagen_input_range, lpips_metric
                )
            ranking_chunks[candidate_index].append(features.float().cpu().numpy())
            for name, values_array in values.items():
                batch_curves[f"{name}_curves"].append(values_array)
        for name in curve_chunks:
            curve_chunks[name].append(np.stack(batch_curves[name], axis=1))
        seen += images_01.shape[0]
        progress.set_postfix(images=seen)

    real_features = np.concatenate(real_chunks, axis=0).astype(np.float32, copy=False)
    router_fixed_features = np.concatenate(router_fixed_chunks, axis=0).astype(
        np.float32, copy=False
    )
    ranking_features = np.stack(
        [np.concatenate(chunks, axis=0) for chunks in ranking_chunks], axis=1
    ).astype(np.float32, copy=False)
    curves = {
        name: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for name, chunks in curve_chunks.items()
    }
    paired_inception_losses = np.square(
        ranking_features - real_features[:, None, :], dtype=np.float64
    ).mean(axis=2)
    balanced_losses = (
        group_standardize(paired_inception_losses)
        + float(args.lpips_weight) * group_standardize(curves["lpips_curves"])
        + float(args.mse_weight) * group_standardize(curves["mse_01_curves"])
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
        "gradient_fixed": ranking_features[rows, fixed_choices],
        "gradient_balanced": ranking_features[rows, balanced_choices],
    }
    metrics: dict[str, dict[str, object]] = {}
    for name, features in selected_features.items():
        choices = fixed_choices if name != "gradient_balanced" else balanced_choices
        path: dict[str, object] = {
            "fid": fid_from_features(real_mean, real_cov, features, device),
            "allocation": token_summary(candidate_tokens, choices),
        }
        if name == "router_fixed":
            path["metrics_source"] = "router reference is FID-only in this combined pass"
        else:
            path.update(selected_curve_metrics(curves, choices))
            path["paired_inception_mse"] = float(
                paired_inception_losses[rows, choices].mean()
            )
        metrics[name] = path

    reference = None
    if args.router_balanced_json and Path(args.router_balanced_json).exists():
        source = json.loads(Path(args.router_balanced_json).read_text())
        if int(source.get("num_images", -1)) == count:
            reference = {
                "source": str(Path(args.router_balanced_json).resolve()),
                "metrics": source.get("metrics", {}).get("inception_lpips_mse_group_w4"),
            }
    router_fixed_fid = float(metrics["router_fixed"]["fid"])
    result = {
        "analysis_type": "balanced_first_order_gradient_spatial_oracle",
        "warning": "Grid scores and dynamic budgets use target information and are not deployable.",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "num_images": count,
        "candidate_tokens": candidate_tokens.tolist(),
        "target_tokens": int(args.target_tokens),
        "balanced_reward": {
            "formula": "spatial_z(-grad_inception) + lpips_weight*spatial_z(-grad_lpips) + mse_weight*spatial_z(-grad_mse)",
            "probe_ratio": float(args.probe_ratio),
            "lpips_weight": float(args.lpips_weight),
            "mse_weight": float(args.mse_weight),
        },
        "fid_feature": int(args.fid_feature),
        "fid_input": "uint8",
        "metrics": metrics,
        "fid_change_vs_router_fixed_percent": {
            name: 100.0 * (float(values["fid"]) - router_fixed_fid) / router_fixed_fid
            for name, values in metrics.items()
            if name != "router_fixed"
        },
        "router_balanced_reference": reference,
        "settings": {
            "allocation_group_size": int(args.allocation_group_size),
            "batch_size": int(args.batch_size),
            "mixed_precision": args.mixed_precision,
            "lpips_net": args.lpips_net,
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
    parser.add_argument("--router-balanced-json", default="results/rl_reward_gates/fid_feature_lpips2_mse_sweep_32_160_2k.json")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--mot-root", default=str(DEFAULT_MOT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", choices=["zero_1", "minus1_1"], default="zero_1")
    parser.add_argument("--llamagen-input-range", choices=["zero_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--candidate-tokens", type=int, nargs="+", default=[32, 48, 64, 80, 96, 112, 128, 144, 160])
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--allocation-group-size", type=int, default=2000)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--probe-ratio", type=float, default=0.375)
    parser.add_argument("--lpips-weight", type=float, default=2.0)
    parser.add_argument("--mse-weight", type=float, default=4.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze", "llamagen_vgg"], default="alex")
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
