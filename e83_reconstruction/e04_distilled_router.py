"""Per-image feed-forward Router distilled from the batch-coupled E04 teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_CANDIDATE_TOKENS = (32, 48, 64, 80, 96, 112, 128, 144, 160)
DEFAULT_TARGET_TOKENS = 96


class E04DistilledRouter(nn.Module):
    """Single-image Router with a spatial ranking head and a discrete budget head."""

    def __init__(
        self,
        latent_channels: int = 256,
        hidden_dim: int = 128,
        depth: int = 3,
        num_budgets: int = 9,
        budget_hidden_dim: int = 256,
        budget_dropout: float = 0.0,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_budgets = int(num_budgets)
        self.budget_hidden_dim = int(budget_hidden_dim)
        self.budget_dropout = float(budget_dropout)
        self.detach_inputs = bool(detach_inputs)

        self.feat_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.f2d_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.delta_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.abs_delta_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.base_proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 16, 16))
        blocks: list[nn.Module] = []
        for _ in range(depth):
            blocks.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        self.trunk = nn.Sequential(*blocks)
        self.score_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        budget_input_dim = 3 * hidden_dim
        self.budget_head = nn.Sequential(
            nn.LayerNorm(budget_input_dim),
            nn.Linear(budget_input_dim, budget_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(budget_dropout),
            nn.Linear(budget_hidden_dim, num_budgets),
        )
        nn.init.normal_(self.score_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.score_head.bias)
        nn.init.zeros_(self.f2d_proj.weight)
        nn.init.zeros_(self.f2d_proj.bias)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.abs_delta_proj.weight)
        nn.init.zeros_(self.abs_delta_proj.bias)
        nn.init.zeros_(self.budget_head[-1].weight)
        nn.init.zeros_(self.budget_head[-1].bias)

    def hidden_features(
        self,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
        f_2d: torch.Tensor,
    ) -> torch.Tensor:
        if f_1d.shape != f_2d.shape or f_1d.shape[-2:] != (16, 16):
            raise ValueError(
                f"expected matching 16x16 f_1d/f_2d, got {tuple(f_1d.shape)} and {tuple(f_2d.shape)}"
            )
        if self.detach_inputs:
            f_1d = f_1d.detach()
            x_base = x_base.detach()
            f_2d = f_2d.detach()
        x_low = F.adaptive_avg_pool2d(x_base.float(), (16, 16)).to(dtype=f_1d.dtype)
        delta = f_2d - f_1d
        hidden = (
            self.feat_proj(f_1d)
            + self.f2d_proj(f_2d)
            + self.delta_proj(delta)
            + self.abs_delta_proj(delta.abs())
            + self.base_proj(x_low)
            + self.pos_embed.to(dtype=f_1d.dtype)
        )
        return self.trunk(hidden)

    @staticmethod
    def pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
        mean = hidden.float().mean(dim=(2, 3))
        std = hidden.float().std(dim=(2, 3), unbiased=False)
        maximum = hidden.float().amax(dim=(2, 3))
        return torch.cat((mean, std, maximum), dim=1)

    def forward(
        self,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
        f_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_features(f_1d, x_base, f_2d)
        return self.score_head(hidden), self.budget_head(self.pool_hidden(hidden))

    def config_dict(self) -> dict[str, object]:
        return {
            "latent_channels": self.latent_channels,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "num_budgets": self.num_budgets,
            "budget_hidden_dim": self.budget_hidden_dim,
            "budget_dropout": self.budget_dropout,
            "detach_inputs": self.detach_inputs,
        }


def source_router_config(source_router: nn.Module, num_budgets: int) -> dict[str, object]:
    required = ("feat_proj", "score_head", "trunk")
    missing = [name for name in required if not hasattr(source_router, name)]
    if missing:
        raise TypeError(f"source Router is missing required modules: {missing}")
    depth = len(source_router.trunk) // 3
    if depth <= 0 or len(source_router.trunk) != 3 * depth:
        raise ValueError("source Router trunk does not have Conv/GroupNorm/SiLU blocks")
    return {
        "latent_channels": int(source_router.feat_proj.in_channels),
        "hidden_dim": int(source_router.score_head.in_channels),
        "depth": int(depth),
        "num_budgets": int(num_budgets),
        "detach_inputs": bool(getattr(source_router, "detach_inputs", True)),
    }


def initialize_from_source_router(
    student: E04DistilledRouter,
    source_router: nn.Module,
) -> dict[str, object]:
    """Copy every shape-compatible pre-existing Router parameter into the student."""
    source = source_router.state_dict()
    target = student.state_dict()
    copied: list[str] = []
    for name, value in source.items():
        if name in target and target[name].shape == value.shape:
            target[name] = value.detach().to(dtype=target[name].dtype, device=target[name].device)
            copied.append(name)
    student.load_state_dict(target, strict=True)
    return {
        "copied_parameter_tensors": len(copied),
        "copied_names": copied,
        "source_parameter_tensors": len(source),
    }


def budget_adjusted_logits(
    logits: torch.Tensor,
    candidate_tokens: torch.Tensor,
    target_tokens: int,
    budget_lambda: float,
) -> torch.Tensor:
    """Apply one calibration-split-frozen rate multiplier to per-image logits."""
    if logits.ndim != 2 or logits.shape[1] != candidate_tokens.numel():
        raise ValueError("budget logits and candidate tokens disagree")
    units = (candidate_tokens.to(device=logits.device, dtype=logits.dtype) - float(target_tokens)) / 16.0
    return logits - float(budget_lambda) * units[None, :]


def independent_budget_choices(
    logits: torch.Tensor,
    candidate_tokens: torch.Tensor,
    target_tokens: int,
    budget_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjusted = budget_adjusted_logits(logits, candidate_tokens, target_tokens, budget_lambda)
    choices = adjusted.argmax(dim=1)
    tokens = candidate_tokens.to(device=choices.device, dtype=torch.long)[choices]
    return choices, tokens


def variable_topk_mask(
    score_logits: torch.Tensor,
    token_counts: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat = score_logits.detach().float().flatten(1)
    counts = token_counts.to(device=flat.device, dtype=torch.long).clamp(0, flat.shape[1])
    order = torch.argsort(flat, dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(flat.shape[1], device=flat.device).view(1, -1)
    ranks.scatter_(1, order, rank_values.expand_as(order))
    mask = ranks < counts[:, None]
    return mask.view(score_logits.shape[0], 1, 16, 16).to(dtype=dtype)


def spatial_ranks(score_logits: torch.Tensor) -> torch.Tensor:
    flat = score_logits.detach().float().flatten(1)
    order = torch.argsort(flat, dim=1, descending=True)
    ranks = torch.empty_like(order)
    values = torch.arange(flat.shape[1], device=flat.device).view(1, -1)
    ranks.scatter_(1, order, values.expand_as(order))
    return ranks.to(torch.uint8)


def dataset_manifest_sha256(relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DistillationLabelDataset(Dataset):
    """Images paired with compact E04 labels from one or more NPZ shards."""

    REQUIRED = (
        "relative_paths",
        "candidate_tokens",
        "teacher_budget_indices",
        "teacher_tokens",
        "teacher_ranking",
        "teacher_spatial_ranks",
        "budget_costs",
        "metadata_json",
    )

    def __init__(
        self,
        data_root: str | Path,
        label_paths: Iterable[str | Path],
        image_size: int,
        center_crop_arr,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
                transforms.ToTensor(),
            ]
        )
        path_chunks: list[np.ndarray] = []
        budget_index_chunks: list[np.ndarray] = []
        token_chunks: list[np.ndarray] = []
        ranking_chunks: list[np.ndarray] = []
        spatial_rank_chunks: list[np.ndarray] = []
        budget_cost_chunks: list[np.ndarray] = []
        self.label_metadata: list[dict[str, object]] = []
        expected_candidates: np.ndarray | None = None
        seen_paths: set[str] = set()
        for label_path in label_paths:
            archive = np.load(label_path, allow_pickle=False)
            missing = [name for name in self.REQUIRED if name not in archive]
            if missing:
                raise KeyError(f"label shard {label_path} is missing {missing}")
            candidates = np.asarray(archive["candidate_tokens"], dtype=np.int64)
            if expected_candidates is None:
                expected_candidates = candidates
            elif not np.array_equal(expected_candidates, candidates):
                raise ValueError(f"candidate tokens disagree in {label_path}")
            relative_paths = np.asarray(archive["relative_paths"]).astype(str)
            duplicates = seen_paths.intersection(relative_paths.tolist())
            if duplicates:
                example = next(iter(duplicates))
                raise ValueError(f"duplicate image label across shards: {example}")
            seen_paths.update(relative_paths.tolist())
            count = relative_paths.shape[0]
            shapes = {
                "teacher_budget_indices": (count,),
                "teacher_tokens": (count,),
                "teacher_ranking": (count,),
                "teacher_spatial_ranks": (count, 256),
                "budget_costs": (count, candidates.size),
            }
            for name, shape in shapes.items():
                if archive[name].shape != shape:
                    raise ValueError(f"{label_path}:{name} has {archive[name].shape}, expected {shape}")
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            metadata["label_file"] = str(Path(label_path).resolve())
            metadata["label_file_sha256"] = file_sha256(label_path)
            self.label_metadata.append(metadata)
            path_chunks.append(relative_paths)
            budget_index_chunks.append(np.asarray(archive["teacher_budget_indices"], dtype=np.int64))
            token_chunks.append(np.asarray(archive["teacher_tokens"], dtype=np.int64))
            ranking_chunks.append(np.asarray(archive["teacher_ranking"], dtype=np.int64))
            spatial_rank_chunks.append(np.asarray(archive["teacher_spatial_ranks"], dtype=np.uint8))
            budget_cost_chunks.append(np.asarray(archive["budget_costs"], dtype=np.float32))
            archive.close()
        if expected_candidates is None or not path_chunks:
            raise ValueError("no E04 label shards were provided")
        self.candidate_tokens = expected_candidates
        self.relative_paths = np.concatenate(path_chunks)
        self.teacher_budget_indices = np.concatenate(budget_index_chunks)
        self.teacher_tokens = np.concatenate(token_chunks)
        self.teacher_ranking = np.concatenate(ranking_chunks)
        self.teacher_spatial_ranks = np.concatenate(spatial_rank_chunks)
        self.budget_costs = np.concatenate(budget_cost_chunks)
        missing_images = [path for path in self.relative_paths if not (self.data_root / path).is_file()]
        if missing_images:
            raise FileNotFoundError(f"missing {len(missing_images)} labeled images; first={missing_images[0]}")

    def __len__(self) -> int:
        return int(self.relative_paths.shape[0])

    def __getitem__(self, index: int):
        relative_path = str(self.relative_paths[index])
        with Image.open(self.data_root / relative_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        labels = {
            "teacher_budget_index": torch.tensor(self.teacher_budget_indices[index], dtype=torch.long),
            "teacher_tokens": torch.tensor(self.teacher_tokens[index], dtype=torch.long),
            "teacher_ranking": torch.tensor(self.teacher_ranking[index], dtype=torch.long),
            "teacher_spatial_ranks": torch.from_numpy(self.teacher_spatial_ranks[index].copy()).long(),
            "budget_costs": torch.from_numpy(self.budget_costs[index].copy()).float(),
        }
        return image_tensor, labels


def load_student_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[E04DistilledRouter, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["student_config"])
    student = E04DistilledRouter(**config).to(device)
    state_name = "student_ema" if use_ema and "student_ema" in checkpoint else "student"
    student.load_state_dict(checkpoint[state_name], strict=True)
    student.eval().requires_grad_(False)
    checkpoint["loaded_student_state"] = state_name
    return student, checkpoint


def add_base_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mot-root", default="/home/heyefei/lichenge/MoT")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", choices=["zero_1", "minus1_1"], default="zero_1")
    parser.add_argument("--llamagen-input-range", choices=["zero_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
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
