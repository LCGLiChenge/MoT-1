#!/usr/bin/env python3
"""Train a budget-only Router from one consistent per-image rate-distortion target."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from e04_distilled_router import (
    DEFAULT_TARGET_TOKENS,
    DistillationLabelDataset,
    E04DistilledRouter,
    add_base_model_arguments,
    initialize_from_source_router,
    source_router_config,
)
from eval_oracle_dynamic_budget import DEFAULT_CKPT, build_model


def distributed_setup() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def unwrap(module: torch.nn.Module) -> E04DistilledRouter:
    return module.module if isinstance(module, DDP) else module


@torch.no_grad()
def update_ema(ema: E04DistilledRouter, model: torch.nn.Module, decay: float) -> None:
    source = unwrap(model)
    for ema_value, value in zip(ema.parameters(), source.parameters()):
        ema_value.lerp_(value.detach(), 1.0 - float(decay))
    for ema_value, value in zip(ema.buffers(), source.buffers()):
        ema_value.copy_(value.detach())


def reduce_metrics(
    metrics: dict[str, float], device: torch.device, world_size: int
) -> dict[str, float]:
    if world_size == 1:
        return metrics
    names = sorted(metrics)
    values = torch.tensor(
        [metrics[name] for name in names], device=device, dtype=torch.float64
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(world_size)
    return {
        name: float(value) for name, value in zip(names, values.cpu().tolist())
    }


def choices_at_price(
    costs: np.ndarray,
    candidate_tokens: np.ndarray,
    target_tokens: int,
    price: float,
) -> np.ndarray:
    units = (candidate_tokens.astype(np.float64) - float(target_tokens)) / 16.0
    return np.argmin(costs.astype(np.float64) + float(price) * units[None, :], axis=1)


def calibrate_teacher_price(
    costs: np.ndarray,
    candidate_tokens: np.ndarray,
    target_tokens: int,
    iterations: int,
) -> tuple[float, dict[str, object]]:
    lo = -1.0
    hi = 1.0
    lo_choices = choices_at_price(costs, candidate_tokens, target_tokens, lo)
    hi_choices = choices_at_price(costs, candidate_tokens, target_tokens, hi)
    while candidate_tokens[lo_choices].mean() <= target_tokens:
        lo *= 2.0
        lo_choices = choices_at_price(costs, candidate_tokens, target_tokens, lo)
        if abs(lo) > 1e6:
            raise RuntimeError("failed to bracket low teacher price")
    while candidate_tokens[hi_choices].mean() >= target_tokens:
        hi *= 2.0
        hi_choices = choices_at_price(costs, candidate_tokens, target_tokens, hi)
        if abs(hi) > 1e6:
            raise RuntimeError("failed to bracket high teacher price")

    candidates: list[tuple[float, np.ndarray]] = [(lo, lo_choices), (hi, hi_choices)]
    for _ in range(max(1, int(iterations))):
        mid = 0.5 * (lo + hi)
        mid_choices = choices_at_price(
            costs, candidate_tokens, target_tokens, mid
        )
        candidates.append((mid, mid_choices))
        if candidate_tokens[mid_choices].mean() > target_tokens:
            lo = mid
        else:
            hi = mid
    price, choices = min(
        candidates,
        key=lambda item: (
            abs(float(candidate_tokens[item[1]].mean()) - target_tokens),
            abs(item[0]),
        ),
    )
    selected_tokens = candidate_tokens[choices]
    unique, counts = np.unique(selected_tokens, return_counts=True)
    return float(price), {
        "price_units_per_16_tokens": float(price),
        "bracket_low": float(lo),
        "bracket_high": float(hi),
        "num_images": int(costs.shape[0]),
        "achieved_token_mean": float(selected_tokens.mean()),
        "achieved_token_std": float(selected_tokens.std()),
        "token_histogram": {
            str(int(token)): int(count)
            for token, count in zip(unique, counts)
        },
        "selection_is_per_image_independent": True,
    }


def adjusted_teacher_costs(
    budget_costs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    target_tokens: int,
    teacher_price: float,
) -> torch.Tensor:
    units = (
        candidate_tokens.to(device=budget_costs.device, dtype=torch.float32)
        - float(target_tokens)
    ) / 16.0
    return budget_costs.float() + float(teacher_price) * units[None, :]


def budget_losses(
    budget_logits: torch.Tensor,
    budget_costs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    teacher_price: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    costs = adjusted_teacher_costs(
        budget_costs,
        candidate_tokens,
        args.target_tokens,
        teacher_price,
    )
    hard_choices = costs.argmin(dim=1)
    hard_tokens = candidate_tokens.to(budget_logits.device)[hard_choices]
    teacher_probability = torch.softmax(
        -costs / float(args.teacher_temperature), dim=1
    )
    student_log_probability = torch.log_softmax(
        budget_logits.float() / float(args.student_temperature), dim=1
    )
    hard_ce = F.cross_entropy(budget_logits.float(), hard_choices)
    soft_ce = -(
        teacher_probability * student_log_probability
    ).sum(dim=1).mean()
    student_probability = torch.softmax(budget_logits.float(), dim=1)
    token_values = candidate_tokens.to(
        device=budget_logits.device, dtype=torch.float32
    )
    expected_tokens = (student_probability * token_values[None, :]).sum(dim=1)
    token_regression = F.smooth_l1_loss(
        expected_tokens / 16.0, hard_tokens.float() / 16.0
    )
    loss = (
        float(args.hard_ce_weight) * hard_ce
        + float(args.soft_ce_weight) * soft_ce
        + float(args.token_regression_weight) * token_regression
    )
    predicted_choices = budget_logits.float().argmax(dim=1)
    predicted_tokens = candidate_tokens.to(budget_logits.device)[predicted_choices]
    teacher_entropy = -(
        teacher_probability * teacher_probability.clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    return loss, {
        "hard_ce": hard_ce,
        "soft_ce": soft_ce,
        "token_regression": token_regression,
        "teacher_entropy": teacher_entropy,
        "budget_accuracy": (predicted_choices == hard_choices).float().mean(),
        "token_mae": (predicted_tokens - hard_tokens).abs().float().mean(),
        "predicted_token_mean": predicted_tokens.float().mean(),
        "teacher_token_mean": hard_tokens.float().mean(),
    }


def save_checkpoint(
    path: Path,
    student: torch.nn.Module,
    student_ema: E04DistilledRouter,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    candidate_tokens: list[int],
    teacher_price: float,
    teacher_price_record: dict[str, object],
    args: argparse.Namespace,
    model_metadata: dict[str, object],
    label_metadata: list[dict[str, object]],
    initialization_report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = unwrap(student)
    torch.save(
        {
            "format": "independent_budget_router_v1",
            "student": core.state_dict(),
            "student_ema": student_ema.state_dict(),
            "student_config": core.config_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": int(global_step),
            "epoch": int(epoch),
            "candidate_tokens": list(candidate_tokens),
            "target_tokens": int(args.target_tokens),
            "base_checkpoint": str(Path(args.ckpt).resolve()),
            "base_checkpoint_state": model_metadata["checkpoint_state"],
            "label_metadata": label_metadata,
            "initialization_report": initialization_report,
            "teacher_price": float(teacher_price),
            "teacher_price_record": teacher_price_record,
            "spatial_router_frozen": True,
            "test_time_batch_statistics_required": False,
            "args": vars(args),
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    if args.accum_steps <= 0 or args.micro_batch_size <= 0:
        raise ValueError("micro batch size and accumulation steps must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    distributed, rank, _local_rank, world_size, device = distributed_setup()
    is_main = rank == 0
    seed = int(args.seed) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base_model, model_metadata = build_model(args, device, checkpoint)
    del checkpoint
    gc.collect()
    base_model.eval().requires_grad_(False)

    mot_root = str(Path(args.mot_root).resolve())
    if mot_root not in sys.path:
        sys.path.insert(0, mot_root)
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import native_llamagen_feature
    from train_titok_llamagen_recon import autocast_dtype, convert_image_range

    llamagen_root = model_metadata["llamagen_root"]
    if llamagen_root not in sys.path:
        sys.path.insert(0, llamagen_root)
    from dataset.augmentation import center_crop_arr

    dataset = DistillationLabelDataset(
        args.data_path, args.label_npz, args.image_size, center_crop_arr
    )
    candidate_tokens = dataset.candidate_tokens.astype(int).tolist()
    if args.target_tokens not in candidate_tokens:
        raise ValueError("target tokens are absent from label candidates")
    for metadata in dataset.label_metadata:
        teacher_checkpoint = str(metadata.get("base_checkpoint", ""))
        if teacher_checkpoint and Path(teacher_checkpoint).resolve() != Path(args.ckpt).resolve():
            if not args.allow_label_checkpoint_mismatch:
                raise ValueError(
                    f"teacher labels use {teacher_checkpoint}, training uses {args.ckpt}; "
                    "pass --allow-label-checkpoint-mismatch only if intentional"
                )
        source_manifest = metadata.get("source_index_manifest")
        if not isinstance(source_manifest, dict) or not str(
            source_manifest.get("split", "")
        ).startswith("teacher_shard_"):
            raise ValueError("independent budget training requires train teacher shards")

    candidate_array = np.asarray(candidate_tokens, dtype=np.int64)
    if np.isfinite(args.teacher_price):
        teacher_price = float(args.teacher_price)
        selected = choices_at_price(
            dataset.budget_costs,
            candidate_array,
            args.target_tokens,
            teacher_price,
        )
        teacher_price_record = {
            "price_units_per_16_tokens": teacher_price,
            "num_images": len(dataset),
            "achieved_token_mean": float(candidate_array[selected].mean()),
            "achieved_token_std": float(candidate_array[selected].std()),
            "selection_is_per_image_independent": True,
            "provided_by_command_line": True,
        }
    else:
        teacher_price, teacher_price_record = calibrate_teacher_price(
            dataset.budget_costs,
            candidate_array,
            args.target_tokens,
            args.price_search_iterations,
        )
        teacher_price_record["provided_by_command_line"] = False

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("training loader is empty")

    config = source_router_config(base_model.router, len(candidate_tokens))
    config.update(
        {
            "budget_hidden_dim": int(args.budget_hidden_dim),
            "budget_dropout": float(args.budget_dropout),
        }
    )
    student = E04DistilledRouter(**config).to(device)
    initialization_report = initialize_from_source_router(student, base_model.router)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in student.budget_head.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("budget_head.") for name in trainable_names
    ):
        raise AssertionError("only budget_head parameters may be trainable")
    student_ema = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.budget_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if list(resume["candidate_tokens"]) != candidate_tokens:
            raise ValueError("resume candidate tokens disagree with label shards")
        if not np.isclose(float(resume["teacher_price"]), teacher_price):
            raise ValueError("resume teacher price disagrees with current labels")
        student.load_state_dict(resume["student"], strict=True)
        student_ema.load_state_dict(
            resume.get("student_ema", resume["student"]), strict=True
        )
        optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("global_step", 0))
        start_epoch = int(resume.get("epoch", 0))
    if distributed:
        student = DDP(student, device_ids=[device.index], broadcast_buffers=False)
    student.train()

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_record = dict(vars(args))
        config_record["resolved_teacher_price"] = teacher_price
        config_record["teacher_price_record"] = teacher_price_record
        config_record["trainable_parameter_names"] = trainable_names
        (output_dir / "config.json").write_text(
            json.dumps(config_record, indent=2, sort_keys=True) + "\n"
        )
    if distributed:
        dist.barrier()

    autocast_type = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"
    token_tensor = torch.tensor(candidate_tokens, device=device, dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}
    running_count = 0
    log_start = time.time()
    stop_training = False
    final_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs):
        final_epoch = epoch + 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        progress = (
            tqdm(
                loader,
                desc=f"independent_budget epoch={epoch + 1}",
                dynamic_ncols=True,
            )
            if is_main
            else loader
        )
        for micro_step, (images_01, labels) in enumerate(progress):
            images_01 = images_01.to(device, non_blocking=True)
            budget_costs = labels["budget_costs"].to(device, non_blocking=True)
            target = convert_image_range(images_01, args.llamagen_input_range)
            titok_input = convert_image_range(images_01, args.titok_input_range)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=autocast_type,
                enabled=autocast_enabled,
            ):
                x_base, extra = base_model(titok_input)
                f_1d = extra["f_1d_lg"]
                f_2d, _ = native_llamagen_feature(
                    base_model.llamagen_vq,
                    target,
                    model_metadata["codebook_embed_dim"],
                    allow_encoder_grad=False,
                )
            window_start = (micro_step // args.accum_steps) * args.accum_steps
            window_size = min(args.accum_steps, len(loader) - window_start)
            sync_update = micro_step + 1 == window_start + window_size
            sync_context = (
                contextlib.nullcontext()
                if sync_update or not isinstance(student, DDP)
                else student.no_sync()
            )
            with sync_context, torch.autocast(
                device_type=device.type,
                dtype=autocast_type,
                enabled=autocast_enabled,
            ):
                _spatial_logits, budget_logits = student(f_1d, x_base, f_2d)
                loss, components = budget_losses(
                    budget_logits,
                    budget_costs,
                    token_tensor,
                    teacher_price,
                    args,
                )
                scaled_loss = loss / float(window_size)
            scaled_loss.backward()
            current = {"loss": float(loss.detach().item())}
            current.update(
                {name: float(value.detach().item()) for name, value in components.items()}
            )
            for name, value in current.items():
                running[name] = running.get(name, 0.0) + value
            running_count += 1
            if not sync_update:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in unwrap(student).parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(student_ema, student, args.ema_decay)
            global_step += 1
            if is_main and isinstance(progress, tqdm):
                progress.set_postfix(
                    loss=f"{current['loss']:.3f}",
                    acc=f"{current['budget_accuracy']:.3f}",
                    mae=f"{current['token_mae']:.1f}",
                )

            if global_step == 1 or global_step % args.log_every == 0:
                averaged = {
                    name: value / max(running_count, 1)
                    for name, value in running.items()
                }
                averaged["grad_norm"] = float(grad_norm.detach().item())
                averaged["seconds_per_optimizer_step"] = (
                    time.time() - log_start
                ) / max(running_count / args.accum_steps, 1.0)
                averaged["global_step"] = int(global_step)
                averaged["epoch"] = int(epoch + 1)
                averaged = reduce_metrics(averaged, device, world_size)
                if is_main:
                    with (output_dir / "train.jsonl").open("a") as handle:
                        handle.write(json.dumps(averaged, sort_keys=True) + "\n")
                    print(json.dumps(averaged, sort_keys=True), flush=True)
                running = {}
                running_count = 0
                log_start = time.time()

            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    output_dir / "latest.pt",
                    student,
                    student_ema,
                    optimizer,
                    global_step,
                    epoch + 1,
                    candidate_tokens,
                    teacher_price,
                    teacher_price_record,
                    args,
                    model_metadata,
                    dataset.label_metadata,
                    initialization_report,
                )
                if args.keep_step_checkpoints:
                    save_checkpoint(
                        output_dir / f"step_{global_step:08d}.pt",
                        student,
                        student_ema,
                        optimizer,
                        global_step,
                        epoch + 1,
                        candidate_tokens,
                        teacher_price,
                        teacher_price_record,
                        args,
                        model_metadata,
                        dataset.label_metadata,
                        initialization_report,
                    )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break
        if is_main and isinstance(progress, tqdm):
            progress.close()
        if stop_training:
            break

    if is_main:
        save_checkpoint(
            output_dir / "latest.pt",
            student,
            student_ema,
            optimizer,
            global_step,
            final_epoch,
            candidate_tokens,
            teacher_price,
            teacher_price_record,
            args,
            model_metadata,
            dataset.label_metadata,
            initialization_report,
        )
        print(f"saved {output_dir / 'latest.pt'}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-npz", nargs="+", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-path", default="/var/tmp/heyefei_ImageNet/train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--budget-hidden-dim", type=int, default=256)
    parser.add_argument("--budget-dropout", type=float, default=0.0)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--teacher-price", type=float, default=float("nan"))
    parser.add_argument("--price-search-iterations", type=int, default=100)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--hard-ce-weight", type=float, default=1.0)
    parser.add_argument("--soft-ce-weight", type=float, default=1.0)
    parser.add_argument("--token-regression-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument(
        "--keep-step-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-label-checkpoint-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
