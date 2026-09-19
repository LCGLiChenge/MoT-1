#!/usr/bin/env python3
"""Second-pass FID/SSIM evaluation for fixed, predicted, and oracle choices."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eval_oracle_dynamic_budget import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    DEFAULT_MOT_ROOT,
    EvalImageDataset,
    build_model,
    to_zero_one,
)


def variable_topk_mask(
    score_logits: torch.Tensor,
    token_counts: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat_score = score_logits.detach().float().flatten(1)
    token_counts = token_counts.to(device=flat_score.device, dtype=torch.long)
    token_counts = token_counts.clamp(min=0, max=flat_score.shape[1])
    order = torch.argsort(flat_score, dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(flat_score.shape[1], device=flat_score.device).view(1, -1)
    ranks.scatter_(1, order, rank_values.expand_as(order))
    mask = ranks < token_counts.view(-1, 1)
    return mask.view(score_logits.shape[0], 1, score_logits.shape[2], score_logits.shape[3]).to(dtype=dtype)


def to_uint8(image: torch.Tensor, image_range: str) -> torch.Tensor:
    return (to_zero_one(image, image_range).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def ssim_batch_sum(prediction: torch.Tensor, target: torch.Tensor, image_range: str, structural_similarity) -> float:
    prediction_np = (
        to_zero_one(prediction, image_range)
        .clamp(0.0, 1.0)
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    target_np = (
        to_zero_one(target, image_range)
        .clamp(0.0, 1.0)
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return sum(
        float(structural_similarity(real, fake, data_range=1.0, channel_axis=2))
        for fake, real in zip(prediction_np, target_np)
    )


def copy_real_fid_statistics(source, destination) -> None:
    for name in ("real_features_sum", "real_features_cov_sum", "real_features_num_samples"):
        getattr(destination, name).copy_(getattr(source, name))


def main(args) -> None:
    from skimage.metrics import structural_similarity
    from torchmetrics.image.fid import FrechetInceptionDistance

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    choice_data = np.load(args.choices_npz)
    candidate_tokens = np.asarray(choice_data["candidate_tokens"], dtype=np.int64)
    available_paths = {
        "fixed": np.asarray(choice_data["fixed_choices"], dtype=np.int64),
        "predicted": np.asarray(choice_data["predicted_choices"], dtype=np.int64),
        "oracle": np.asarray(choice_data["oracle_choices"], dtype=np.int64),
    }
    eval_paths = list(dict.fromkeys(args.eval_paths))
    full_count = len(available_paths["fixed"])
    if any(len(available_paths[name]) != full_count for name in eval_paths):
        raise ValueError("choice arrays have inconsistent lengths")
    count = full_count if args.num_images <= 0 else min(args.num_images, full_count)
    choices = {name: available_paths[name][:count] for name in eval_paths}
    if any(np.any((value < 0) | (value >= len(candidate_tokens))) for value in choices.values()):
        raise ValueError("choice array contains an invalid candidate index")

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
        raise ValueError(f"dataset has {len(dataset)} images but choices require {count}")
    dataset = Subset(dataset, range(count))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    fids = {
        name: FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)
        for name in eval_paths
    }
    real_stats_path = eval_paths[0]
    ssim_sums = {name: 0.0 for name in eval_paths}
    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"
    offset = 0

    with torch.inference_mode():
        progress = tqdm(loader, desc="fid_second_pass", dynamic_ncols=True)
        for images_01, _ in progress:
            batch_size = images_01.shape[0]
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

            real_uint8 = to_uint8(target, args.llamagen_input_range)
            fids[real_stats_path].update(real_uint8, real=True)
            for name in eval_paths:
                batch_choices = choices[name][offset : offset + batch_size]
                token_counts = torch.as_tensor(candidate_tokens[batch_choices], device=device)
                with torch.autocast(device_type=device.type, dtype=autocast_type, enabled=autocast_enabled):
                    mask = variable_topk_mask(score_logits, token_counts, f_1d.dtype)
                    mixed_feature = (1.0 - mask) * f_1d + mask * f_2d
                    prediction = model.llamagen_vq.decoder(mixed_feature)
                fids[name].update(to_uint8(prediction, args.llamagen_input_range), real=False)
                ssim_sums[name] += ssim_batch_sum(
                    prediction.float(), target.float(), args.llamagen_input_range, structural_similarity
                )
            offset += batch_size
            progress.set_postfix(images=offset)

    if offset != count:
        raise AssertionError(f"evaluated {offset} images, expected {count}")
    for name in eval_paths[1:]:
        copy_real_fid_statistics(fids[real_stats_path], fids[name])

    first_pass_summary_path = Path(args.first_pass_json) if args.first_pass_json else Path(args.choices_npz).with_suffix(".json")
    first_pass_summary = None
    if first_pass_summary_path.exists():
        first_pass_summary = json.loads(first_pass_summary_path.read_text())
        if int(first_pass_summary.get("num_images", -1)) != count:
            first_pass_summary = None

    metrics = {}
    for name in eval_paths:
        token_values = candidate_tokens[choices[name]]
        path_metrics = {
            "fid": float(fids[name].compute().item()),
            "ssim": float(ssim_sums[name] / count),
            "tokens_mean": float(token_values.mean()),
            "tokens_std": float(token_values.std()),
            "tokens_min": int(token_values.min()),
            "tokens_max": int(token_values.max()),
        }
        summary_name = {"fixed": "fixed", "predicted": "predicted_dynamic", "oracle": "oracle_dynamic"}[name]
        if first_pass_summary is not None:
            path_metrics.update(first_pass_summary.get("metrics", {}).get(summary_name, {}))
        metrics[name] = path_metrics

    result = {
        "analysis_type": "full_dynamic_reconstruction_metrics_second_pass",
        "warning": "Oracle choices use target LPIPS and are a target-aware upper bound.",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "choices_npz": str(Path(args.choices_npz).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "num_images": count,
        "candidate_tokens": candidate_tokens.tolist(),
        "fid_feature": int(args.fid_feature),
        "fid_input": "uint8",
        "ssim_implementation": "skimage.metrics.structural_similarity",
        "eval_paths": eval_paths,
        "metrics": metrics,
        "model": model_metadata,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"saved {output_json}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choices-npz", required=True)
    parser.add_argument("--first-pass-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--mot-root", default=str(DEFAULT_MOT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", choices=["zero_1", "minus1_1"], default="zero_1")
    parser.add_argument("--llamagen-input-range", choices=["zero_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--eval-paths", nargs="+", choices=["fixed", "predicted", "oracle"], default=["fixed", "predicted", "oracle"])
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
