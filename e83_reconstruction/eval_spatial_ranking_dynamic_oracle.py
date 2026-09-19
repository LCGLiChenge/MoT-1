#!/usr/bin/env python3
"""Compare spatial rankings and their per-image dynamic-budget upper bounds.

This is an evaluation-only analysis.  A ranking fixes the nested spatial token
prefix for each image; target LPIPS then chooses one candidate budget per image
under an exact group-average token constraint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    DEFAULT_MOT_ROOT,
    EvalImageDataset,
    build_lpips_metric,
    build_model,
    grouped_exact_budget_dp,
    metric_summary,
    per_image_metrics,
    to_lpips_range,
    token_histogram,
    topk_mask,
)


RANKINGS = ("router", "base_l1", "latent_l1", "native_mse_gain", "lpips_grad")


def lpips_gradient_score(
    f_1d: torch.Tensor,
    f_2d: torch.Tensor,
    target: torch.Tensor,
    decoder: torch.nn.Module,
    lpips_metric: torch.nn.Module,
    image_range: str,
    probe_ratio: float,
    autocast_type: torch.dtype,
    autocast_enabled: bool,
) -> torch.Tensor:
    """First-order LPIPS reduction from increasing each continuous mask value."""
    with torch.inference_mode(False), torch.enable_grad():
        f_1d_probe = f_1d.detach().clone()
        f_2d_probe = f_2d.detach().clone()
        target_probe = target.detach().clone()
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
            mixed = (1.0 - probe) * f_1d_probe + probe * f_2d_probe
            prediction = decoder(mixed)
        loss = lpips_metric(
            to_lpips_range(prediction.float(), image_range),
            to_lpips_range(target_probe.float(), image_range),
        ).reshape(f_1d.shape[0], -1).mean(dim=1).sum()
        gradient = torch.autograd.grad(loss, probe, only_inputs=True)[0]
    return -gradient.detach().float()


@torch.no_grad()
def spatial_score(
    ranking: str,
    router_score: torch.Tensor,
    f_1d: torch.Tensor,
    f_2d: torch.Tensor,
    x_base: torch.Tensor,
    target: torch.Tensor,
    x_native: torch.Tensor | None,
) -> torch.Tensor:
    if ranking == "router":
        return router_score.detach().float()
    if ranking == "base_l1":
        error = (x_base.detach().float() - target.detach().float()).abs().mean(dim=1, keepdim=True)
        return F.adaptive_avg_pool2d(error, (16, 16))
    if ranking == "latent_l1":
        return (f_2d.detach().float() - f_1d.detach().float()).abs().mean(dim=1, keepdim=True)
    if ranking == "native_mse_gain":
        if x_native is None:
            raise ValueError("native_mse_gain requires x_native")
        base_error = (x_base.detach().float() - target.detach().float()).square().mean(dim=1, keepdim=True)
        native_error = (x_native.detach().float() - target.detach().float()).square().mean(dim=1, keepdim=True)
        return F.adaptive_avg_pool2d(base_error - native_error, (16, 16))
    raise ValueError(f"unknown ranking: {ranking}")


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and not args.device.startswith("cpu"):
        raise RuntimeError("CUDA is unavailable; pass --device cpu only for debugging")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    candidate_tokens = np.asarray(sorted(set(args.candidate_tokens)), dtype=np.int64)
    if candidate_tokens[0] < 0 or candidate_tokens[-1] > 256:
        raise ValueError("candidate tokens must lie in [0, 256]")
    if args.target_tokens not in candidate_tokens:
        raise ValueError("--target-tokens must be present in --candidate-tokens")

    device = torch.device(args.device)
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
    if args.num_images > 0:
        dataset = Subset(dataset, range(min(args.num_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    lpips_metric = build_lpips_metric(args, device)
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"

    curve_chunks = {name: [] for name in ("lpips", "l1_01", "mse_01", "psnr")}
    with torch.inference_mode():
        for images_01, _indices in tqdm(loader, desc=f"spatial_{args.ranking}", dynamic_ncols=True):
            images_01 = images_01.to(device, non_blocking=True)
            target = convert_image_range(images_01, args.llamagen_input_range)
            titok_input = convert_image_range(images_01, args.titok_input_range)
            batch_metrics = {name: [] for name in curve_chunks}
            with torch.autocast(device_type=device.type, dtype=autocast_type, enabled=autocast_enabled):
                x_base, extra = model(titok_input)
                f_1d = extra["f_1d_lg"]
                f_2d, _ = native_llamagen_feature(
                    model.llamagen_vq,
                    target,
                    model_metadata["codebook_embed_dim"],
                    allow_encoder_grad=False,
                )
                router_score, _ = model.router(f_1d, x_base, f_2d)
                x_native = model.llamagen_vq.decoder(f_2d) if args.ranking == "native_mse_gain" else None
            if args.ranking == "lpips_grad":
                score = lpips_gradient_score(
                    f_1d,
                    f_2d,
                    target,
                    model.llamagen_vq.decoder,
                    lpips_metric,
                    args.llamagen_input_range,
                    args.gradient_probe_ratio,
                    autocast_type,
                    autocast_enabled,
                )
            else:
                score = spatial_score(args.ranking, router_score, f_1d, f_2d, x_base, target, x_native)

            for token_count in candidate_tokens.tolist():
                with torch.autocast(device_type=device.type, dtype=autocast_type, enabled=autocast_enabled):
                    mask = topk_mask(score, token_count, f_1d.dtype)
                    prediction = model.llamagen_vq.decoder((1.0 - mask) * f_1d + mask * f_2d)
                values = per_image_metrics(prediction.float(), target.float(), args.llamagen_input_range, lpips_metric)
                for name, value in values.items():
                    batch_metrics[name].append(value)
            for name in curve_chunks:
                curve_chunks[name].append(np.stack(batch_metrics[name], axis=1))

    metric_curves = {
        name: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for name, chunks in curve_chunks.items()
    }
    objective = metric_curves["lpips"].astype(np.float64)
    oracle_choices = grouped_exact_budget_dp(
        objective, candidate_tokens, args.target_tokens, args.allocation_group_size
    )
    fixed_index = int(np.where(candidate_tokens == args.target_tokens)[0][0])
    fixed_choices = np.full(objective.shape[0], fixed_index, dtype=np.int64)
    rows = np.arange(objective.shape[0])
    fixed_objective = objective[rows, fixed_choices]
    oracle_objective = objective[rows, oracle_choices]
    delta = fixed_objective - oracle_objective
    oracle_tokens = candidate_tokens[oracle_choices]
    expected_total = int(args.target_tokens) * objective.shape[0]
    if int(oracle_tokens.sum()) != expected_total:
        raise AssertionError("oracle allocation violated the exact total budget")

    result = {
        "analysis_type": "spatial_ranking_dynamic_budget_upper_bound",
        "warning": "Budget allocation uses target LPIPS; native_mse_gain also uses a full native decode.",
        "ranking": args.ranking,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "num_images": int(objective.shape[0]),
        "candidate_tokens": candidate_tokens.tolist(),
        "target_tokens": int(args.target_tokens),
        "allocation": {
            "group_size": int(args.allocation_group_size),
            "oracle_token_mean": float(oracle_tokens.mean()),
            "oracle_token_std": float(oracle_tokens.std()),
            "oracle_token_histogram": token_histogram(oracle_tokens),
        },
        "objective": {
            "fixed_mean": float(fixed_objective.mean()),
            "oracle_mean": float(oracle_objective.mean()),
            "fixed_minus_oracle_mean": float(delta.mean()),
            "fixed_minus_oracle_standard_error": float(delta.std(ddof=1) / math.sqrt(delta.size)),
            "relative_improvement_percent": float(100.0 * delta.mean() / fixed_objective.mean()),
            "images_improved_fraction": float((delta > 0.0).mean()),
            "images_worsened_fraction": float((delta < 0.0).mean()),
        },
        "metrics": {
            "fixed": metric_summary(metric_curves, fixed_choices),
            "oracle_dynamic": metric_summary(metric_curves, oracle_choices),
            "candidate_fixed_budgets": {
                str(int(token)): {
                    name: float(values[:, index].mean()) for name, values in metric_curves.items()
                }
                for index, token in enumerate(candidate_tokens)
            },
        },
        "diagnostics": {
            "images_with_any_lpips_monotonicity_violation_fraction": float(
                (np.diff(objective, axis=1) > args.monotonic_tolerance).any(axis=1).mean()
            ),
        },
        "model": model_metadata,
        "settings": {
            "batch_size": int(args.batch_size),
            "gradient_probe_ratio": float(args.gradient_probe_ratio),
            "mixed_precision": args.mixed_precision,
            "lpips_net": args.lpips_net,
            "seed": int(args.seed),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output_npz = Path(args.output_npz) if args.output_npz else output_json.with_suffix(".npz")
    np.savez_compressed(
        output_npz,
        candidate_tokens=candidate_tokens,
        lpips_curves=metric_curves["lpips"],
        l1_01_curves=metric_curves["l1_01"],
        mse_01_curves=metric_curves["mse_01"],
        psnr_curves=metric_curves["psnr"],
        fixed_choices=fixed_choices,
        oracle_choices=oracle_choices,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"saved {output_json}", flush=True)
    print(f"saved {output_npz}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", choices=RANKINGS, required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--mot-root", default=str(DEFAULT_MOT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", choices=["zero_1", "minus1_1"], default="zero_1")
    parser.add_argument("--llamagen-input-range", choices=["zero_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--candidate-tokens", type=int, nargs="+", default=[32, 48, 64, 80, 96, 112, 128, 144, 160])
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--allocation-group-size", type=int, default=2000)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze", "llamagen_vgg"], default="alex")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--gradient-probe-ratio", type=float, default=0.375)
    parser.add_argument("--monotonic-tolerance", type=float, default=1e-7)
    parser.add_argument("--use-model-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
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
