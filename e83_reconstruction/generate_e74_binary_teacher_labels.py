#!/usr/bin/env python3
"""Generate compact train-only E70/E72 binary teacher labels for E74."""

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
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import e46_gaussian_ot_direction_gain as ot
from calibrate_e70_fulltrain_diagonal_ot_three_choice_fulltrain import (
    load_e70_confirmation,
)
from e04_distilled_router import add_base_model_arguments, file_sha256
from e31_pareto_relative_endpoint import E31_COMPONENT_WEIGHTS
from e42_amplitude_power_gain import combine_e42_torch, set_amplitude_power
from e72_choice_count_common import allocation_summary
from eval_oracle_dynamic_budget import (
    EvalImageDataset,
    build_lpips_metric,
    build_model,
)
from eval_single_pass_budget import distributed_setup
from screen_e70_fulltrain_diagonal_ot_three_choice import (
    ALGORITHM,
    ALPHA,
    BASE_CHECKPOINT_SHA256,
    DIRECTION_FORMAT,
    DIRECTION_SHA256,
)
from screen_native_gain_budget import model_batch
from single_probe_marginal_budget import router_ranked_torch


FORMAT = "e74_binary_teacher_labels_v1"
PARTIAL_FORMAT = "e74_binary_teacher_partial_v1"
SUMMARY_FORMAT = "e74_binary_teacher_fulltrain_summary_v1"
FORMAL_TRAIN_IMAGES = 1_281_167
FORMAL_WORLD_SIZE = 2
FORMAL_BATCH_SIZE = 8
FORMAL_CHECKPOINT_EVERY_BATCHES = 1_000
FORMAL_COMPLETION_TIMEOUT_SECONDS = 86_400
CANDIDATES = (64, 128)


def calibrate_binary_score_price(
    scores: np.ndarray,
    target_tokens: float = 96.0,
) -> tuple[float, np.ndarray]:
    """Calibrate the exact deployed float32 `score > price` decision.

    Candidate prices are observed float32 scores. Equality therefore always
    selects K64 and cannot change under a later scalar cast. If score ties make
    exact mean K impossible, prefer the nearest allocation and then fewer
    tokens.
    """

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("binary teacher scores must be one finite vector")
    unique, counts = np.unique(values, return_counts=True)
    greater_counts = values.size - np.cumsum(counts, dtype=np.int64)
    token_means = 64.0 + 64.0 * greater_counts.astype(np.float64) / values.size
    errors = np.abs(token_means - float(target_tokens))
    best_error = float(errors.min())
    eligible = np.flatnonzero(
        np.isclose(errors, best_error, rtol=0, atol=1e-12)
    )
    # The last eligible threshold has the fewest K128 assignments.
    best_index = int(eligible[-1])
    price = float(unique[best_index])
    tokens = np.where(values > np.float32(price), 128, 64).astype(np.int64)
    if int((tokens == 128).sum()) != int(greater_counts[best_index]):
        raise AssertionError("binary teacher threshold changed after float32 cast")
    return price, tokens


def validate_args(args, world_size: int, dataset_size: int) -> int:
    if args.smoke:
        if not 1 <= args.num_images <= 8:
            raise ValueError("E74 teacher smoke requires 1--8 images")
        if args.num_images < world_size:
            raise ValueError("E74 teacher smoke needs at least one image per rank")
        return min(args.num_images, dataset_size)
    exact = {
        "num_images": FORMAL_TRAIN_IMAGES,
        "batch_size": FORMAL_BATCH_SIZE,
        "mixed_precision": "bf16",
        "lpips_net": "alex",
        "seed": 0,
        "use_model_ema": True,
        "checkpoint_every_batches": FORMAL_CHECKPOINT_EVERY_BATCHES,
        "completion_timeout_seconds": FORMAL_COMPLETION_TIMEOUT_SECONDS,
        "completion_poll_seconds": 5.0,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"formal E74 labels require --{name.replace('_', '-')}={expected}"
            )
    if world_size != FORMAL_WORLD_SIZE:
        raise ValueError(f"formal E74 labels require {FORMAL_WORLD_SIZE} GPUs")
    if dataset_size != FORMAL_TRAIN_IMAGES:
        raise ValueError(
            f"ImageNet-train has {dataset_size} images, expected {FORMAL_TRAIN_IMAGES}"
        )
    if file_sha256(args.ckpt) != BASE_CHECKPOINT_SHA256:
        raise ValueError("E74 base checkpoint changed")
    if file_sha256(args.e70_direction_npz) != DIRECTION_SHA256:
        raise ValueError("E74 complete-train direction changed")
    return FORMAL_TRAIN_IMAGES


def _allowed_output(path: Path) -> bool:
    resolved = path.resolve()
    return resolved.is_relative_to(Path.cwd().resolve()) or resolved.is_relative_to(
        Path("/tmp").resolve()
    )


def _save_rank_shard(
    path: Path,
    relative_paths: np.ndarray,
    indices: np.ndarray,
    prefixes: np.ndarray,
    metadata: dict[str, object],
) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        relative_paths=relative_paths,
        dataset_indices=indices.astype(np.int64, copy=False),
        candidate_tokens=np.asarray(CANDIDATES, dtype=np.int64),
        prefix_gains=prefixes.astype(np.float32, copy=False),
        binary_score=(
            (prefixes[:, 1] - prefixes[:, 0]) / float(CANDIDATES[1] - CANDIDATES[0])
        ).astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, path)


def _save_partial_shard(
    path: Path,
    indices: np.ndarray,
    prefixes: np.ndarray,
    metadata: dict[str, object],
) -> None:
    """Atomically persist only compact, deterministic recovery state."""

    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        dataset_indices=indices.astype(np.int64, copy=False),
        prefix_gains=prefixes.astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, path)


def _expected_recovery_metadata(
    *,
    rank: int,
    world_size: int,
    shard_start: int,
    shard_end: int,
    dataset_size: int,
    component_scales: dict[str, float],
    direction_sha256: str,
    smoke: bool,
) -> dict[str, object]:
    return {
        "format": PARTIAL_FORMAT,
        "rank": rank,
        "world_size": world_size,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "dataset_size": dataset_size,
        "candidate_tokens": list(CANDIDATES),
        "component_scales": component_scales,
        "component_weights": E31_COMPONENT_WEIGHTS,
        "amplitude_power": ALPHA,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "direction_sha256": direction_sha256,
        "validation_images_used": False,
        "smoke": smoke,
    }


def _load_partial_shard(
    path: Path,
    expected_metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "dataset_indices",
            "prefix_gains",
            "metadata_json",
        }:
            raise ValueError(f"E74 partial keys changed: {path}")
        indices = payload["dataset_indices"].astype(np.int64, copy=True)
        prefixes = payload["prefix_gains"].astype(np.float32, copy=True)
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata != expected_metadata:
        raise ValueError(f"E74 partial metadata changed: {path}")
    shard_start = int(expected_metadata["shard_start"])
    shard_end = int(expected_metadata["shard_end"])
    if not 0 <= len(indices) <= shard_end - shard_start:
        raise ValueError(f"E74 partial count is invalid: {path}")
    if prefixes.shape != (len(indices), len(CANDIDATES)):
        raise ValueError(f"E74 partial gain shape changed: {path}")
    expected_indices = np.arange(
        shard_start, shard_start + len(indices), dtype=np.int64
    )
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"E74 partial indices are not a contiguous prefix: {path}")
    if not np.isfinite(prefixes).all():
        raise FloatingPointError(f"E74 partial gains are not finite: {path}")
    return indices, prefixes


def _load_completed_rank_shard(
    path: Path,
    *,
    rank: int,
    world_size: int,
    shard_start: int,
    shard_end: int,
    dataset: EvalImageDataset,
    component_scales: dict[str, float],
    direction_sha256: str,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "relative_paths",
            "dataset_indices",
            "candidate_tokens",
            "prefix_gains",
            "binary_score",
            "metadata_json",
        }
        if set(payload.files) != required:
            raise ValueError(f"E74 completed-shard keys changed: {path}")
        relative_paths = payload["relative_paths"].astype(str, copy=True)
        indices = payload["dataset_indices"].astype(np.int64, copy=True)
        candidate_tokens = payload["candidate_tokens"].astype(np.int64, copy=True)
        prefixes = payload["prefix_gains"].astype(np.float32, copy=True)
        binary_score = payload["binary_score"].astype(np.float32, copy=True)
        metadata = json.loads(str(payload["metadata_json"].item()))
    count = shard_end - shard_start
    expected_metadata = {
        "format": FORMAT,
        "rank": rank,
        "world_size": world_size,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "num_images": count,
        "dataset_size": len(dataset),
        "source_split": "ImageNet-train" if not smoke else "smoke_prefix",
        "candidate_tokens": list(CANDIDATES),
        "teacher_score": "(prefix_gain_128-prefix_gain_64)/64",
        "algorithm": ALGORITHM,
        "component_scales": component_scales,
        "component_weights": E31_COMPONENT_WEIGHTS,
        "amplitude_power": ALPHA,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "direction_sha256": direction_sha256,
        "validation_images_used": False,
        "endpoint_probes_per_image_during_offline_label_generation": 1,
        "candidate_k_reconstructions_per_image": 0,
        "images_saved": False,
        "reconstructions_saved": False,
        "features_saved": False,
        "per_grid_gain_maps_saved": False,
        "smoke": smoke,
    }
    if metadata != expected_metadata:
        raise ValueError(f"E74 completed-shard metadata changed: {path}")
    if relative_paths.shape != (count,) or indices.shape != (count,):
        raise ValueError(f"E74 completed-shard row shape changed: {path}")
    if prefixes.shape != (count, len(CANDIDATES)) or binary_score.shape != (count,):
        raise ValueError(f"E74 completed-shard value shape changed: {path}")
    if not np.array_equal(candidate_tokens, np.asarray(CANDIDATES, dtype=np.int64)):
        raise ValueError(f"E74 completed-shard candidates changed: {path}")
    expected_indices = np.arange(shard_start, shard_end, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"E74 completed-shard indices changed: {path}")
    if not np.isfinite(prefixes).all() or not np.isfinite(binary_score).all():
        raise FloatingPointError(f"E74 completed-shard values are not finite: {path}")
    recomputed = ((prefixes[:, 1] - prefixes[:, 0]) / 64.0).astype(np.float32)
    if not np.array_equal(binary_score, recomputed):
        raise ValueError(f"E74 completed-shard binary score changed: {path}")
    expected_paths = np.asarray(
        [
            item.relative_to(dataset.root).as_posix()
            for item in dataset.paths[shard_start:shard_end]
        ]
    )
    if not np.array_equal(relative_paths, expected_paths):
        raise ValueError(f"E74 completed-shard paths changed: {path}")
    record = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rank": rank,
        "start": shard_start,
        "end": shard_end,
        "count": count,
    }
    return relative_paths, indices, prefixes, record


def _wait_for_paths(
    paths: list[Path],
    *,
    timeout_seconds: int,
    poll_seconds: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {description}: "
                + ", ".join(str(path) for path in missing)
            )
        time.sleep(poll_seconds)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not _allowed_output(output_dir):
        raise ValueError("E74 teacher outputs must remain under dynamic/ or /tmp")
    if args.recover_rank_zero_single_gpu:
        if args.smoke:
            raise ValueError("single-GPU rank-0 recovery is formal-only")
        if args.overwrite:
            raise ValueError("single-GPU rank-0 recovery cannot overwrite artifacts")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("single-GPU rank-0 recovery must not run under torchrun")
        if not torch.cuda.is_available():
            raise RuntimeError("single-GPU rank-0 recovery requires CUDA")
        distributed = False
        rank = 0
        world_size = FORMAL_WORLD_SIZE
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        distributed, rank, _local_rank, world_size, device = distributed_setup()
    is_main = rank == 0
    summary_path = output_dir / "calibration.json"
    rank_path = output_dir / f"teacher_rank{rank:02d}.npz"
    partial_path = output_dir / f"teacher_rank{rank:02d}.partial.npz"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"E74 teacher summary already exists: {summary_path}")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive")
    if args.completion_timeout_seconds <= 0 or args.completion_poll_seconds <= 0:
        raise ValueError("E74 filesystem completion wait must be positive")

    confirmation_path = Path(args.confirmation_json)
    confirmation, component_scales = load_e70_confirmation(
        confirmation_path, smoke=args.smoke
    )
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    started = time.time()

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
    count = validate_args(args, world_size, len(dataset))
    shard_start = count * rank // world_size
    shard_end = count * (rank + 1) // world_size
    shard_size = shard_end - shard_start
    direction_sha256 = file_sha256(args.e70_direction_npz)
    recovery_metadata = _expected_recovery_metadata(
        rank=rank,
        world_size=world_size,
        shard_start=shard_start,
        shard_end=shard_end,
        dataset_size=len(dataset),
        component_scales=component_scales,
        direction_sha256=direction_sha256,
        smoke=bool(args.smoke),
    )
    ot.EXPECTED_DIRECTION_FORMAT = DIRECTION_FORMAT
    direction_metadata = ot.configure_gaussian_ot_direction(
        args.e70_direction_npz
    )
    set_amplitude_power(ALPHA)

    if args.recover_rank_zero_single_gpu:
        other_rank = 1
        other_start = count * other_rank // world_size
        other_end = count * (other_rank + 1) // world_size
        _load_completed_rank_shard(
            output_dir / f"teacher_rank{other_rank:02d}.npz",
            rank=other_rank,
            world_size=world_size,
            shard_start=other_start,
            shard_end=other_end,
            dataset=dataset,
            component_scales=component_scales,
            direction_sha256=direction_sha256,
            smoke=False,
        )
        print(
            "E74 single-GPU recovery verified completed rank=1 shard; "
            "computing logical rank=0 only",
            flush=True,
        )

    reused_completed_shard = rank_path.exists() and not args.overwrite
    resumed_images = 0
    if reused_completed_shard:
        relative_paths, local_indices, local_prefixes, shard_record = (
            _load_completed_rank_shard(
                rank_path,
                rank=rank,
                world_size=world_size,
                shard_start=shard_start,
                shard_end=shard_end,
                dataset=dataset,
                component_scales=component_scales,
                direction_sha256=direction_sha256,
                smoke=bool(args.smoke),
            )
        )
        observed = shard_size
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"E74 rank={rank} reused verified completed shard "
            f"{rank_path} ({observed} images)",
            flush=True,
        )
    else:
        local_prefixes = np.empty(
            (shard_size, len(CANDIDATES)), dtype=np.float32
        )
        local_indices = np.empty(shard_size, dtype=np.int64)
        observed = 0
        if partial_path.exists() and not args.overwrite:
            saved_indices, saved_prefixes = _load_partial_shard(
                partial_path, recovery_metadata
            )
            observed = len(saved_indices)
            local_indices[:observed] = saved_indices
            local_prefixes[:observed] = saved_prefixes
            resumed_images = observed
            print(
                f"E74 rank={rank} resumed {observed}/{shard_size} images "
                f"from {partial_path}",
                flush=True,
            )

        loader = DataLoader(
            Subset(dataset, range(shard_start + observed, shard_end)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        lpips_metric = build_lpips_metric(args, device)
        fid_extractor = (
            FrechetInceptionDistance(feature=2048, normalize=False)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        autocast_type = autocast_dtype(args.mixed_precision)
        autocast_enabled = args.mixed_precision != "none"
        candidate_indices = torch.tensor(
            [token - 1 for token in CANDIDATES], device=device, dtype=torch.long
        )
        initial_observed = observed
        batches_done = (observed + args.batch_size - 1) // args.batch_size
        try:
            with torch.inference_mode():
                progress = tqdm(
                    loader,
                    desc=f"e74_binary_teacher rank={rank}",
                    dynamic_ncols=True,
                    disable=not is_main,
                    initial=batches_done,
                    total=(shard_size + args.batch_size - 1) // args.batch_size,
                )
                for images_01, indices in progress:
                    images_01 = images_01.to(device, non_blocking=True)
                    (
                        target,
                        x_base,
                        _f_1d,
                        _f_2d,
                        router_score,
                        x_native,
                    ) = model_batch(
                        images_01,
                        model,
                        model_metadata,
                        args,
                        autocast_type,
                        autocast_enabled,
                    )
                    components, _diagnostics = (
                        ot.gaussian_ot_direction_relative_endpoint_component_gain_maps(
                            target,
                            x_base,
                            x_native,
                            fid_extractor.inception,
                            lpips_metric,
                            args.llamagen_input_range,
                        )
                    )
                    combined = combine_e42_torch(components, component_scales)
                    ranked = router_ranked_torch(combined, router_score)
                    prefixes = torch.cumsum(ranked.float(), dim=1).index_select(
                        1, candidate_indices
                    )
                    batch = images_01.shape[0]
                    local_prefixes[observed : observed + batch] = (
                        prefixes.cpu().numpy().astype(np.float32)
                    )
                    local_indices[observed : observed + batch] = (
                        indices.numpy().astype(np.int64)
                    )
                    observed += batch
                    batches_done += 1
                    if (
                        batches_done % args.checkpoint_every_batches == 0
                        or observed == shard_size
                    ):
                        output_dir.mkdir(parents=True, exist_ok=True)
                        _save_partial_shard(
                            partial_path,
                            local_indices[:observed],
                            local_prefixes[:observed],
                            recovery_metadata,
                        )
                    if is_main:
                        progress.set_postfix(
                            local_images=observed,
                            resumed_images=resumed_images,
                        )
        except BaseException:
            if observed > initial_observed:
                output_dir.mkdir(parents=True, exist_ok=True)
                _save_partial_shard(
                    partial_path,
                    local_indices[:observed],
                    local_prefixes[:observed],
                    recovery_metadata,
                )
            raise
        if observed != shard_size:
            raise AssertionError("E74 local teacher count mismatch")
        expected_indices = np.arange(shard_start, shard_end, dtype=np.int64)
        if not np.array_equal(local_indices, expected_indices):
            raise AssertionError("E74 contiguous dataset-index contract changed")

        relative_paths = np.asarray(
            [
                path.relative_to(dataset.root).as_posix()
                for path in dataset.paths[shard_start:shard_end]
            ]
        )
        shard_metadata = {
            "format": FORMAT,
            "rank": rank,
            "world_size": world_size,
            "shard_start": shard_start,
            "shard_end": shard_end,
            "num_images": observed,
            "dataset_size": len(dataset),
            "source_split": "ImageNet-train" if not args.smoke else "smoke_prefix",
            "candidate_tokens": list(CANDIDATES),
            "teacher_score": "(prefix_gain_128-prefix_gain_64)/64",
            "algorithm": ALGORITHM,
            "component_scales": component_scales,
            "component_weights": E31_COMPONENT_WEIGHTS,
            "amplitude_power": ALPHA,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "direction_sha256": direction_sha256,
            "validation_images_used": False,
            "endpoint_probes_per_image_during_offline_label_generation": 1,
            "candidate_k_reconstructions_per_image": 0,
            "images_saved": False,
            "reconstructions_saved": False,
            "features_saved": False,
            "per_grid_gain_maps_saved": False,
            "smoke": bool(args.smoke),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_rank_shard(
            rank_path,
            relative_paths,
            local_indices,
            local_prefixes,
            shard_metadata,
        )
        _, _, _, shard_record = _load_completed_rank_shard(
            rank_path,
            rank=rank,
            world_size=world_size,
            shard_start=shard_start,
            shard_end=shard_end,
            dataset=dataset,
            component_scales=component_scales,
            direction_sha256=direction_sha256,
            smoke=bool(args.smoke),
        )
        del model, lpips_metric, fid_extractor
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"E74 rank={rank} wrote verified completed shard "
            f"{rank_path} ({observed} images)",
            flush=True,
        )

    shard_record["reused_existing_at_process_start"] = reused_completed_shard
    shard_record["resumed_images_in_process"] = resumed_images
    rank_paths = [
        output_dir / f"teacher_rank{other_rank:02d}.npz"
        for other_rank in range(world_size)
    ]
    _wait_for_paths(
        rank_paths,
        timeout_seconds=args.completion_timeout_seconds,
        poll_seconds=args.completion_poll_seconds,
        description="all E74 completed rank shards",
    )

    if is_main:
        all_prefixes_parts = []
        shard_records = []
        for other_rank, other_path in enumerate(rank_paths):
            other_start = count * other_rank // world_size
            other_end = count * (other_rank + 1) // world_size
            _, _, other_prefixes, other_record = _load_completed_rank_shard(
                other_path,
                rank=other_rank,
                world_size=world_size,
                shard_start=other_start,
                shard_end=other_end,
                dataset=dataset,
                component_scales=component_scales,
                direction_sha256=direction_sha256,
                smoke=bool(args.smoke),
            )
            if other_rank == rank:
                other_record.update(
                    {
                        "reused_existing_at_process_start": reused_completed_shard,
                        "resumed_images_in_process": resumed_images,
                    }
                )
            all_prefixes_parts.append(other_prefixes)
            shard_records.append(other_record)
        all_prefixes = np.concatenate(all_prefixes_parts, axis=0)
        all_scores = (
            all_prefixes[:, 1] - all_prefixes[:, 0]
        ) / float(CANDIDATES[1] - CANDIDATES[0])
        if all_prefixes.shape != (count, 2) or all_scores.shape != (count,):
            raise AssertionError("E74 disk-gathered teacher shape changed")
        price, tokens = calibrate_binary_score_price(
            all_scores, target_tokens=96.0
        )
        margin = all_scores.astype(np.float64) - float(price)
        margin_scale = float(margin.std(dtype=np.float64))
        if not np.isfinite(margin).all() or not margin_scale > 0:
            raise FloatingPointError("E74 teacher margin is invalid")
        summary = {
            "format": SUMMARY_FORMAT,
            "status": "completed",
            "experiment": "E74-true-one-forward-binary-K-selector-teacher",
            "preregistration": "E74_BINARY_ONE_FORWARD_PREREGISTRATION.md",
            "algorithm": ALGORITHM,
            "candidate_tokens": list(CANDIDATES),
            "decision_rule": "K128 iff (G128-G64)/64 > train_only_price",
            "price": float(price),
            "price_iterations": 0,
            "price_search": (
                "exact observed-float32 score thresholds; ties choose K64; "
                "nearest mean K then fewer tokens"
            ),
            "allocation": allocation_summary(tokens),
            "teacher_score_mean": float(all_scores.mean(dtype=np.float64)),
            "teacher_score_std": float(all_scores.std(dtype=np.float64)),
            "teacher_margin_mean": float(margin.mean(dtype=np.float64)),
            "teacher_margin_scale": margin_scale,
            "teacher_score_quantiles": {
                str(q): float(np.quantile(all_scores, q))
                for q in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
            "num_images": count,
            "dataset_size": len(dataset),
            "complete_imagenet_train_coverage": (
                not args.smoke and count == len(dataset) == FORMAL_TRAIN_IMAGES
            ),
            "component_scales": component_scales,
            "component_weights": E31_COMPONENT_WEIGHTS,
            "amplitude_power": ALPHA,
            "confirmation_artifact": {
                "path": str(confirmation_path.resolve()),
                "sha256": file_sha256(confirmation_path),
                "screen_gates": confirmation.get("screen_gates"),
            },
            "direction_artifact": {
                "path": str(Path(args.e70_direction_npz).resolve()),
                "sha256": direction_sha256,
                "metadata": direction_metadata,
            },
            "base_checkpoint": str(Path(args.ckpt).resolve()),
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "base_checkpoint_state": model_metadata["checkpoint_state"],
            "label_shards": shard_records,
            "validation_images_used": False,
            "teacher_labels_use_other_images_for_per_image_decision": False,
            "selector_inference_endpoint_probes": 0,
            "selector_inference_candidate_reconstructions": 0,
            "selector_inference_perceptual_network_forwards": 0,
            "offline_label_endpoint_probes_per_image": 1,
            "images_saved": False,
            "reconstructions_saved": False,
            "features_saved": False,
            "per_grid_gain_maps_saved": False,
            "stats_pt_written": False,
            "world_size": world_size,
            "cross_rank_completion_protocol": (
                "verified atomic rank shards plus filesystem polling; "
                "no long-wait NCCL collective"
            ),
            "partial_checkpoint_every_batches": args.checkpoint_every_batches,
            "partial_checkpoints_are_recovery_only": True,
            "runtime_seconds": time.time() - started,
            "runtime_args": vars(args),
            "smoke": bool(args.smoke),
        }
        temporary_summary = summary_path.with_suffix(
            summary_path.suffix + f".tmp.{os.getpid()}"
        )
        temporary_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_summary, summary_path)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "price": summary["price"],
                    "allocation": summary["allocation"],
                    "teacher_score_std": summary["teacher_score_std"],
                    "summary": str(summary_path.resolve()),
                    "sha256": file_sha256(summary_path),
                    "runtime_seconds": summary["runtime_seconds"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        _wait_for_paths(
            [summary_path],
            timeout_seconds=args.completion_timeout_seconds,
            poll_seconds=args.completion_poll_seconds,
            description="E74 train-only calibration summary",
        )
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("format") != SUMMARY_FORMAT
            or summary.get("status") != "completed"
            or summary.get("num_images") != count
        ):
            raise ValueError("E74 completed summary failed rank-side validation")
    if distributed:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-json", required=True)
    parser.add_argument("--e70-direction-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ckpt",
        default=(
            "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/dynamic/"
            "e54/checkpoints/base_latest.pt"
        ),
    )
    parser.add_argument("--data-path", default="/var/tmp/heyefei_ImageNet/train")
    parser.add_argument("--num-images", type=int, default=FORMAL_TRAIN_IMAGES)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=FORMAL_CHECKPOINT_EVERY_BATCHES,
    )
    parser.add_argument(
        "--completion-timeout-seconds",
        type=int,
        default=FORMAL_COMPLETION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--completion-poll-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips-net", choices=("alex",), default="alex")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--recover-rank-zero-single-gpu",
        action="store_true",
        help=(
            "resume only logical rank 0 on one GPU after strictly validating "
            "an existing completed rank-1 shard; does not change formal sharding"
        ),
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
