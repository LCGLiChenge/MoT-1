#!/usr/bin/env python3
"""Evaluate the oracle upper bound of per-image dynamic 2D-token budgets.

The spatial ranking is kept fixed: it comes from the score head of an existing
MoT checkpoint.  For each image, this script decodes several top-k prefixes of
that ranking, measures a per-image quality curve, and then solves a grouped
multiple-choice knapsack problem.  Every allocation group has exactly the same
mean token count as the fixed baseline.

This is an oracle analysis because allocation uses reconstruction losses that
require the target image.  It is not an inference-time Router.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


DEFAULT_MOT_ROOT = Path("/home/heyefei/lichenge/MoT")
DEFAULT_CKPT = Path("/tmp/mot_hf_latest_eval_20260807/latest.pt")
DEFAULT_DATA = Path("/home/heyefei/ImageNet/validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EvalImageDataset(Dataset):
    def __init__(self, root: str, image_size: int, center_crop_arr):
        self.root = Path(root)
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found under {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), index


def candidate_token_units(candidate_tokens: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert token counts to the smallest common integer units."""
    differences = np.diff(candidate_tokens)
    if differences.size == 0:
        return np.zeros(1, dtype=np.int64), 1
    unit = reduce(gcd, (int(value) for value in differences))
    if unit <= 0:
        raise ValueError(f"candidate tokens must be strictly increasing: {candidate_tokens.tolist()}")
    units = (candidate_tokens - candidate_tokens[0]) // unit
    return units.astype(np.int64), unit


def exact_budget_dp(
    losses: np.ndarray,
    candidate_tokens: np.ndarray,
    target_tokens: int,
) -> np.ndarray:
    """Solve one exact-budget multiple-choice knapsack group.

    Complexity is O(num_images * target_units * num_candidates).  The intended
    use is an analysis group of at most a few thousand images and 3--7 budgets.
    """
    losses = np.asarray(losses, dtype=np.float64)
    candidate_tokens = np.asarray(candidate_tokens, dtype=np.int64)
    if losses.ndim != 2 or losses.shape[1] != candidate_tokens.size:
        raise ValueError("losses must have shape [num_images, num_candidates]")
    if losses.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(losses).all():
        raise ValueError("loss matrix contains NaN or infinity")
    if np.any(np.diff(candidate_tokens) <= 0):
        raise ValueError("candidate tokens must be unique and sorted")

    units, token_unit = candidate_token_units(candidate_tokens)
    num_images = losses.shape[0]
    target_delta = int(target_tokens) * num_images - int(candidate_tokens[0]) * num_images
    if target_delta < 0 or target_delta > int(candidate_tokens[-1] - candidate_tokens[0]) * num_images:
        raise ValueError("target token count is outside the candidate range")
    if target_delta % token_unit != 0:
        raise ValueError(
            f"exact target is unreachable: delta={target_delta}, common token unit={token_unit}"
        )
    target_units = target_delta // token_unit

    infinity = np.inf
    dp = np.full(target_units + 1, infinity, dtype=np.float64)
    dp[0] = 0.0
    sentinel = np.iinfo(np.uint8).max
    if candidate_tokens.size >= sentinel:
        raise ValueError("too many candidate budgets for uint8 backpointers")
    parents = np.full((num_images, target_units + 1), sentinel, dtype=np.uint8)

    for image_index in range(num_images):
        next_dp = np.full_like(dp, infinity)
        for candidate_index, token_units in enumerate(units.tolist()):
            if token_units > target_units:
                continue
            values = dp[: target_units + 1 - token_units] + losses[image_index, candidate_index]
            destination = next_dp[token_units:]
            better = values < destination
            destination[better] = values[better]
            parent_destination = parents[image_index, token_units:]
            parent_destination[better] = candidate_index
        dp = next_dp

    if not np.isfinite(dp[target_units]):
        raise RuntimeError("no exact allocation found; candidate budgets may not span the target")

    choices = np.empty(num_images, dtype=np.int64)
    remaining_units = int(target_units)
    for image_index in range(num_images - 1, -1, -1):
        candidate_index = int(parents[image_index, remaining_units])
        if candidate_index == sentinel:
            raise RuntimeError("invalid DP backpointer while reconstructing allocation")
        choices[image_index] = candidate_index
        remaining_units -= int(units[candidate_index])
    if remaining_units != 0:
        raise RuntimeError(f"allocation reconstruction ended with {remaining_units} token units")
    return choices


def grouped_exact_budget_dp(
    losses: np.ndarray,
    candidate_tokens: np.ndarray,
    target_tokens: int,
    group_size: int,
) -> np.ndarray:
    num_images = losses.shape[0]
    if group_size <= 0:
        group_size = num_images
    choices = np.empty(num_images, dtype=np.int64)
    for start in tqdm(range(0, num_images, group_size), desc="exact_budget_dp", dynamic_ncols=True):
        end = min(start + group_size, num_images)
        choices[start:end] = exact_budget_dp(losses[start:end], candidate_tokens, target_tokens)
    return choices


def topk_mask(score_logits: torch.Tensor, token_count: int, dtype: torch.dtype) -> torch.Tensor:
    flat_score = score_logits.detach().float().flatten(1)
    token_count = max(0, min(int(token_count), flat_score.shape[1]))
    mask = torch.zeros_like(flat_score)
    if token_count > 0:
        indices = torch.topk(flat_score, token_count, dim=1, largest=True).indices
        mask.scatter_(1, indices, 1.0)
    return mask.view(score_logits.shape[0], 1, score_logits.shape[2], score_logits.shape[3]).to(dtype=dtype)


def to_zero_one(image: torch.Tensor, image_range: str) -> torch.Tensor:
    if image_range == "minus1_1":
        return (image.float() + 1.0) * 0.5
    if image_range == "zero_1":
        return image.float()
    raise ValueError(f"unsupported image range: {image_range}")


def to_lpips_range(image: torch.Tensor, image_range: str) -> torch.Tensor:
    if image_range == "minus1_1":
        return image.float()
    if image_range == "zero_1":
        return image.float() * 2.0 - 1.0
    raise ValueError(f"unsupported image range: {image_range}")


def per_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_range: str,
    lpips_metric,
) -> dict[str, np.ndarray]:
    prediction_01 = to_zero_one(prediction, image_range).clamp(0.0, 1.0)
    target_01 = to_zero_one(target, image_range).clamp(0.0, 1.0)
    difference = prediction_01 - target_01
    l1 = difference.abs().flatten(1).mean(dim=1)
    mse = difference.square().flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    lpips_value = lpips_metric(
        to_lpips_range(prediction, image_range),
        to_lpips_range(target, image_range),
    ).reshape(prediction.shape[0], -1).mean(dim=1)
    return {
        "lpips": lpips_value.detach().float().cpu().numpy(),
        "l1_01": l1.detach().float().cpu().numpy(),
        "mse_01": mse.detach().float().cpu().numpy(),
        "psnr": psnr.detach().float().cpu().numpy(),
    }


def load_checkpoint_parameters(model, checkpoint: dict, use_model_ema: bool) -> tuple[str, int, int]:
    state_name = "model_ema" if use_model_ema else "model"
    if state_name not in checkpoint:
        if use_model_ema:
            raise KeyError("checkpoint has no model_ema; rerun with --no-use-model-ema")
        state = checkpoint.get("model", checkpoint)
    else:
        state = checkpoint[state_name]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint field {state_name} is not a state dictionary")

    parameters = dict(model.named_parameters())
    copied = 0
    unexpected = []
    mismatched = []
    with torch.no_grad():
        for name, value in state.items():
            if name not in parameters:
                unexpected.append(name)
                continue
            if tuple(value.shape) != tuple(parameters[name].shape):
                mismatched.append((name, tuple(value.shape), tuple(parameters[name].shape)))
                continue
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
            copied += 1
    if unexpected or mismatched:
        raise RuntimeError(
            f"checkpoint mismatch: unexpected={unexpected[:8]}, shape_mismatch={mismatched[:8]}"
        )
    required_prefixes = ("latent_decoder.", "router.")
    missing_required = [
        name
        for name in parameters
        if name.startswith(required_prefixes) and name not in state
    ]
    if missing_required:
        raise RuntimeError(f"checkpoint is missing required parameters: {missing_required[:8]}")
    return state_name, int(checkpoint.get("step", -1)), copied


def build_model(args, device: torch.device, checkpoint: dict):
    mot_root = Path(args.mot_root).resolve()
    if str(mot_root) not in sys.path:
        sys.path.insert(0, str(mot_root))

    from models import TiTokLlamaGenStage2
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import DynamicBudgetRouter
    from train_titok_llamagen_recon import load_llamagen_vq, load_titok

    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = vars(checkpoint_args)

    def value(name: str, fallback):
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        return checkpoint_args.get(name, fallback)

    def project_path(raw_path) -> str:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = mot_root / path
        return str(path.resolve())

    titok_root = project_path(value("titok_root", "/home/heyefei/lichenge/1d-tokenizer"))
    titok_config = project_path(value(
        "titok_config", "/home/heyefei/lichenge/1d-tokenizer/configs/infer/TiTok/titok_l32.yaml"
    ))
    titok_ckpt = project_path(value("titok_ckpt", "/home/heyefei/lichenge/1d-tokenizer/tokenizer_titok_l32.bin"))
    llamagen_root = project_path(value("llamagen_root", "/home/heyefei/lichenge/LlamaGen"))
    llamagen_ckpt = project_path(value(
        "llamagen_ckpt", "/home/heyefei/lichenge/LlamaGen/pretrained_models/vq_ds16_c2i.pt"
    ))
    codebook_size = int(value("codebook_size", 16384))
    codebook_embed_dim = int(value("codebook_embed_dim", 8))
    lg_latent_channels = int(value("lg_latent_channels", 256))
    lg_head_channels = int(value("lg_head_channels", 256))

    for required_path in (titok_root, titok_config, titok_ckpt, llamagen_root, llamagen_ckpt):
        if not Path(required_path).exists():
            raise FileNotFoundError(f"required model dependency does not exist: {required_path}")

    titok = load_titok(titok_root, titok_config, titok_ckpt, device)
    vq_model = load_llamagen_vq(
        llamagen_root,
        llamagen_ckpt,
        device,
        codebook_size,
        codebook_embed_dim,
    )
    model = TiTokLlamaGenStage2(
        titok,
        vq_model,
        lg_latent_channels=lg_latent_channels,
        head_channels=lg_head_channels,
        head_mode="feature",
        codebook_size=codebook_size,
        codebook_temperature=float(value("codebook_temperature", 1.0)),
    ).to(device)
    model.router = DynamicBudgetRouter(
        latent_channels=lg_latent_channels,
        hidden_dim=int(value("router_hidden_dim", 128)),
        depth=int(value("router_depth", 3)),
        target_ratio=float(value("router_target_mean_ratio", 0.375)),
        min_ratio=float(value("router_min_ratio", 0.375)),
        max_ratio=float(value("router_max_ratio", 0.375)),
        detach_inputs=bool(value("router_detach_inputs", True)),
    ).to(device)
    state_name, step, copied = load_checkpoint_parameters(model, checkpoint, args.use_model_ema)
    model.eval().requires_grad_(False)
    metadata = {
        "checkpoint_state": state_name,
        "checkpoint_step": step,
        "copied_parameters": copied,
        "titok_root": str(titok_root),
        "titok_config": str(titok_config),
        "titok_ckpt": str(titok_ckpt),
        "llamagen_root": str(llamagen_root),
        "llamagen_ckpt": str(llamagen_ckpt),
        "codebook_size": codebook_size,
        "codebook_embed_dim": codebook_embed_dim,
    }
    return model, metadata


def build_lpips_metric(args, device: torch.device):
    if args.lpips_net == "llamagen_vgg":
        llamagen_root = args.llamagen_root or "/home/heyefei/lichenge/LlamaGen"
        if llamagen_root not in sys.path:
            sys.path.insert(0, llamagen_root)
        from tokenizer.tokenizer_image.lpips import LPIPS

        return LPIPS().to(device).eval().requires_grad_(False)
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("install lpips or use --lpips-net llamagen_vgg") from exc
    return lpips.LPIPS(net=args.lpips_net).to(device).eval().requires_grad_(False)


def selected_mean(matrix: np.ndarray, choices: np.ndarray) -> float:
    return float(matrix[np.arange(matrix.shape[0]), choices].mean())


def metric_summary(metric_curves: dict[str, np.ndarray], choices: np.ndarray) -> dict[str, float]:
    return {name: selected_mean(values, choices) for name, values in metric_curves.items()}


def token_histogram(tokens: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(tokens, return_counts=True)
    return {str(int(token)): int(count) for token, count in zip(unique, counts)}


def main(args) -> None:
    if not torch.cuda.is_available() and not args.device.startswith("cpu"):
        raise RuntimeError("CUDA is unavailable; pass --device cpu only for code-level debugging")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    candidate_tokens = np.asarray(sorted(set(args.candidate_tokens)), dtype=np.int64)
    if candidate_tokens[0] < 0 or candidate_tokens[-1] > 256:
        raise ValueError("candidate tokens must lie in [0, 256]")
    if args.target_tokens not in candidate_tokens:
        raise ValueError("fixed target must be one of --candidate-tokens")

    device = torch.device(args.device)
    print(f"loading checkpoint metadata and weights: {args.ckpt}", flush=True)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("expected a dictionary checkpoint")
    model, model_metadata = build_model(args, device, checkpoint)
    del checkpoint
    gc.collect()

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

    curve_chunks: dict[str, list[np.ndarray]] = {
        "lpips": [],
        "l1_01": [],
        "mse_01": [],
        "psnr": [],
    }
    dataset_indices: list[np.ndarray] = []

    with torch.inference_mode():
        for images_01, indices in tqdm(loader, desc="quality_curves", dynamic_ncols=True):
            images_01 = images_01.to(device, non_blocking=True)
            target = convert_image_range(images_01, args.llamagen_input_range)
            titok_input = convert_image_range(images_01, args.titok_input_range)
            batch_metrics = {name: [] for name in curve_chunks}
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
                score_logits, _ = model.router(f_1d, x_base, f_2d)

            for token_count in candidate_tokens.tolist():
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_type,
                    enabled=autocast_enabled,
                ):
                    mask = topk_mask(score_logits, token_count, f_1d.dtype)
                    mixed_feature = (1.0 - mask) * f_1d + mask * f_2d
                    prediction = model.llamagen_vq.decoder(mixed_feature)
                values = per_image_metrics(
                    prediction.float(),
                    target.float(),
                    args.llamagen_input_range,
                    lpips_metric,
                )
                for name, value in values.items():
                    batch_metrics[name].append(value)
                del prediction, mixed_feature, mask

            for name in curve_chunks:
                curve_chunks[name].append(np.stack(batch_metrics[name], axis=1))
            dataset_indices.append(indices.numpy())

    metric_curves = {
        name: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for name, chunks in curve_chunks.items()
    }
    indices = np.concatenate(dataset_indices, axis=0)
    objective_curves = (
        args.objective_lpips_weight * metric_curves["lpips"]
        + args.objective_l1_weight * metric_curves["l1_01"]
    ).astype(np.float64)
    oracle_choices = grouped_exact_budget_dp(
        objective_curves,
        candidate_tokens,
        args.target_tokens,
        args.allocation_group_size,
    )
    fixed_index = int(np.where(candidate_tokens == args.target_tokens)[0][0])
    fixed_choices = np.full(objective_curves.shape[0], fixed_index, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    random_choices = oracle_choices[rng.permutation(oracle_choices.size)]

    oracle_tokens = candidate_tokens[oracle_choices]
    fixed_tokens = candidate_tokens[fixed_choices]
    random_tokens = candidate_tokens[random_choices]
    expected_total = int(args.target_tokens) * objective_curves.shape[0]
    if int(oracle_tokens.sum()) != expected_total:
        raise AssertionError("oracle allocation violated the exact total budget")
    if int(random_tokens.sum()) != expected_total:
        raise AssertionError("random control violated the exact total budget")

    candidate_means = {
        str(int(token)): {name: float(values[:, candidate_index].mean()) for name, values in metric_curves.items()}
        for candidate_index, token in enumerate(candidate_tokens)
    }
    fixed_summary = metric_summary(metric_curves, fixed_choices)
    oracle_summary = metric_summary(metric_curves, oracle_choices)
    random_summary = metric_summary(metric_curves, random_choices)
    fixed_objective = objective_curves[:, fixed_index]
    oracle_objective = objective_curves[np.arange(objective_curves.shape[0]), oracle_choices]
    random_objective = objective_curves[np.arange(objective_curves.shape[0]), random_choices]
    objective_delta = fixed_objective - oracle_objective

    monotonic_violations = np.diff(objective_curves, axis=1) > args.monotonic_tolerance
    token_difficulty_corr = 0.0
    if np.std(fixed_objective) > 0.0 and np.std(oracle_tokens) > 0.0:
        token_difficulty_corr = float(np.corrcoef(fixed_objective, oracle_tokens)[0, 1])

    order = np.argsort(fixed_objective)
    difficulty_buckets = {}
    num_difficulty_buckets = min(5, max(1, order.size))
    for bucket_index, bucket_indices in enumerate(np.array_split(order, num_difficulty_buckets)):
        difficulty_buckets[str(bucket_index + 1)] = {
            "count": int(bucket_indices.size),
            "fixed_objective_mean": float(fixed_objective[bucket_indices].mean()),
            "oracle_tokens_mean": float(oracle_tokens[bucket_indices].mean()),
            "oracle_gain_mean": float(objective_delta[bucket_indices].mean()),
        }

    result = {
        "analysis_type": "oracle_dynamic_budget_upper_bound",
        "warning": "Allocation uses target reconstruction losses and is not an inference-time result.",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "num_images": int(objective_curves.shape[0]),
        "candidate_tokens": candidate_tokens.tolist(),
        "target_tokens": int(args.target_tokens),
        "target_ratio": float(args.target_tokens / 256.0),
        "allocation": {
            "solver": "grouped_exact_multiple_choice_dp",
            "group_size": int(args.allocation_group_size),
            "oracle_token_mean": float(oracle_tokens.mean()),
            "oracle_token_std": float(oracle_tokens.std()),
            "oracle_token_min": int(oracle_tokens.min()),
            "oracle_token_max": int(oracle_tokens.max()),
            "oracle_token_histogram": token_histogram(oracle_tokens),
            "random_token_histogram": token_histogram(random_tokens),
        },
        "objective": {
            "lpips_weight": float(args.objective_lpips_weight),
            "l1_01_weight": float(args.objective_l1_weight),
            "fixed_mean": float(fixed_objective.mean()),
            "oracle_mean": float(oracle_objective.mean()),
            "random_same_histogram_mean": float(random_objective.mean()),
            "fixed_minus_oracle_mean": float(objective_delta.mean()),
            "fixed_minus_oracle_standard_error": float(
                objective_delta.std(ddof=1) / math.sqrt(max(objective_delta.size, 1))
            ) if objective_delta.size > 1 else 0.0,
            "relative_improvement_percent": float(
                100.0 * objective_delta.mean() / max(abs(fixed_objective.mean()), 1e-12)
            ),
        },
        "metrics": {
            "fixed": fixed_summary,
            "oracle_dynamic": oracle_summary,
            "random_same_histogram": random_summary,
            "candidate_fixed_budgets": candidate_means,
        },
        "diagnostics": {
            "images_with_any_monotonicity_violation_fraction": float(monotonic_violations.any(axis=1).mean()),
            "adjacent_transitions_with_violation_fraction": float(monotonic_violations.mean()),
            "fixed_difficulty_vs_oracle_tokens_pearson": token_difficulty_corr,
            "difficulty_quintiles_easy_to_hard": difficulty_buckets,
        },
        "model": model_metadata,
        "settings": {
            "mixed_precision": args.mixed_precision,
            "lpips_net": args.lpips_net,
            "use_model_ema": bool(args.use_model_ema),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output_npz = Path(args.output_npz) if args.output_npz else output_json.with_suffix(".npz")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        dataset_indices=indices,
        candidate_tokens=candidate_tokens,
        objective_curves=objective_curves.astype(np.float32),
        lpips_curves=metric_curves["lpips"],
        l1_01_curves=metric_curves["l1_01"],
        mse_01_curves=metric_curves["mse_01"],
        psnr_curves=metric_curves["psnr"],
        fixed_choices=fixed_choices,
        oracle_choices=oracle_choices,
        random_choices=random_choices,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"saved {output_json}", flush=True)
    print(f"saved {output_npz}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--mot-root", default=str(DEFAULT_MOT_ROOT))
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", choices=["zero_1", "minus1_1"], default="zero_1")
    parser.add_argument("--llamagen-input-range", choices=["zero_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--candidate-tokens", type=int, nargs="+", default=[64, 80, 96, 112, 128])
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument(
        "--allocation-group-size",
        type=int,
        default=2000,
        help="0 means one global DP; 2000 bounds DP memory for larger evaluations",
    )
    parser.add_argument("--objective-lpips-weight", type=float, default=1.0)
    parser.add_argument("--objective-l1-weight", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze", "llamagen_vgg"], default="alex")
    parser.add_argument("--monotonic-tolerance", type=float, default=1e-7)
    parser.add_argument("--use-model-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)

    # Optional overrides.  None means: use checkpoint args, then known project defaults.
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
