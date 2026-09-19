#!/usr/bin/env python3
"""Evaluate E72 `{64,128}` using E82's frozen full-train mean-90 price."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from tqdm import tqdm

import e46_gaussian_ot_direction_gain as ot
import eval_e24_endpoint_composite_gain as shared
from calibrate_e70_fulltrain_diagonal_ot_three_choice_fulltrain import (
    load_e70_confirmation,
)
from calibrate_e82_e72_train90_fulltrain import (
    FORMAT as PRICE_FORMAT,
    REFERENCE_PRICE,
    TARGET_TOKENS as TRAIN_TARGET_TOKENS,
)
from e04_distilled_router import add_base_model_arguments, file_sha256
from e31_pareto_relative_endpoint import E31_COMPONENT_WEIGHTS
from e42_amplitude_power_gain import combine_e42_torch, set_amplitude_power
from eval_fid_from_dynamic_choices import ssim_batch_sum, to_uint8, variable_topk_mask
from eval_independent_fixed_price_encoder import (
    feature_sufficient_statistics,
    moments_from_sufficient_statistics,
)
from eval_oracle_dynamic_budget import (
    EvalImageDataset,
    build_lpips_metric,
    build_model,
    per_image_metrics,
)
from eval_single_pass_budget import allocation_summary, distributed_setup
from screen_native_gain_budget import model_batch
from screen_e70_fulltrain_diagonal_ot_three_choice import (
    ALGORITHM,
    ALPHA,
    BASE_CHECKPOINT_SHA256,
    DIRECTION_FORMAT,
    DIRECTION_SHA256,
)
from single_probe_marginal_budget import router_ranked_torch


FORMAT = "e82_e72_train90_val50k_eval_v1"
CANDIDATES = (64, 128)
FORMAL_VAL_IMAGES = 50_000
FORMAL_MAX_WORLD_SIZE = 4
FORMAL_BATCH_SIZE = 2
EXPECTED_VAL_PATH = Path("/home/heyefei/ImageNet/validation").resolve()
E72_REFERENCE_FORMAT = "e72_choice_count_val_rate_eval_v1"
E72_REFERENCE_SHA256 = (
    "6db992bdd267150f1f852cd458ec84c7fd754f6024b69aa8e194d4ec76536e6f"
)
METRIC_NAMES = ("lpips", "l1_01", "mse_01", "psnr")


def load_price(path: Path, expected_sha256: str) -> tuple[dict, float]:
    if len(expected_sha256) != 64 or file_sha256(path) != expected_sha256:
        raise ValueError("E82 price artifact SHA-256 changed")
    artifact = json.loads(path.read_text())
    allocation = artifact.get("allocation", {})
    reference = artifact.get("reference_reproduction", {})
    if (
        artifact.get("format") != PRICE_FORMAT
        or artifact.get("status") != "completed"
        or artifact.get("algorithm")
        != "E72_binary_average_marginal_fixed_fulltrain_price"
        or artifact.get("candidate_tokens") != list(CANDIDATES)
        or float(artifact.get("target_train_mean_tokens", -1.0))
        != TRAIN_TARGET_TOKENS
        or int(allocation.get("count", -1)) != 1_281_167
        or abs(float(allocation.get("mean", -1.0)) - TRAIN_TARGET_TOKENS) > 0.01
        or artifact.get("complete_imagenet_train_coverage") is not True
        or int(artifact.get("train_images", -1)) != 1_281_167
        or int(artifact.get("validation_images_used_for_price", -1)) != 0
        or artifact.get("validation_statistics_used_for_price") is not False
        or artifact.get("reported_reconstruction_metrics_used_to_choose_price")
        is not False
        or artifact.get("selection_is_per_image_independent_after_price_frozen")
        is not True
        or artifact.get("selection_requires_batch_after_price_frozen") is not False
        or artifact.get("post_hoc_rate_compensation_hypothesis") is not True
        or reference.get("exact_price_reproduced") is not True
        or float(reference.get("price", np.nan)) != REFERENCE_PRICE
        or artifact.get("npz_written") is not False
        or artifact.get("stats_pt_written") is not False
    ):
        raise ValueError("E82 price artifact violates its frozen contract")
    price = float(artifact.get("price", np.nan))
    if not np.isfinite(price):
        raise ValueError("E82 price is not finite")
    return artifact, price


def load_e72_reference(path: Path) -> dict:
    if file_sha256(path) != E72_REFERENCE_SHA256:
        raise ValueError("E72 val50k reference changed")
    artifact = json.loads(path.read_text())
    if (
        artifact.get("format") != E72_REFERENCE_FORMAT
        or artifact.get("status") != "completed"
        or int(artifact.get("num_images", -1)) != FORMAL_VAL_IMAGES
        or "fixed_router" not in artifact.get("metrics", {})
        or "choice2_gap64" not in artifact.get("metrics", {})
        or artifact.get("choice_sets", {}).get("choice2_gap64")
        != list(CANDIDATES)
    ):
        raise ValueError("E72 reference contract changed")
    return artifact


def decision_tokens(score: torch.Tensor, price: float) -> torch.Tensor:
    if score.ndim != 1 or not torch.isfinite(score).all():
        raise ValueError("E82 score must be one finite vector")
    boundary = torch.tensor(
        np.float32(price), device=score.device, dtype=torch.float32
    )
    return torch.where(
        score.float() > boundary,
        torch.full_like(score, CANDIDATES[1], dtype=torch.long),
        torch.full_like(score, CANDIDATES[0], dtype=torch.long),
    )


def formal_world_size_allowed(world_size: int) -> bool:
    """Hardware parallelism changes throughput, not E82's per-image method."""
    return 1 <= world_size <= FORMAL_MAX_WORLD_SIZE


def validate_args(args: argparse.Namespace, world_size: int, dataset_size: int) -> int:
    if Path(args.data_path).resolve() != EXPECTED_VAL_PATH:
        raise ValueError(f"E82 requires the frozen val path {EXPECTED_VAL_PATH}")
    if file_sha256(args.ckpt) != BASE_CHECKPOINT_SHA256:
        raise ValueError("E82 base checkpoint changed")
    if file_sha256(args.e70_direction_npz) != DIRECTION_SHA256:
        raise ValueError("E82 full-train OT direction changed")
    if args.fid_feature != 2048 or args.seed != 0 or args.lpips_net != "alex":
        raise ValueError("E82 freezes FID-2048, seed 0, and LPIPS-Alex")
    if args.mixed_precision != "bf16" or args.use_model_ema is not True:
        raise ValueError("E82 freezes bf16 and the base EMA checkpoint state")
    if args.smoke:
        if not 2 <= args.num_images <= 32:
            raise ValueError("E82 smoke requires 2--32 images")
        if not 1 <= world_size <= 4 or args.num_images < world_size:
            raise ValueError("E82 smoke requires 1--4 ranks with at least one row each")
        return min(args.num_images, dataset_size)
    if (
        args.num_images != FORMAL_VAL_IMAGES
        or dataset_size != FORMAL_VAL_IMAGES
        or not formal_world_size_allowed(world_size)
        or args.batch_size != FORMAL_BATCH_SIZE
    ):
        raise ValueError(
            "formal E82 freezes val50k and batch size two, using one to four GPUs"
        )
    return FORMAL_VAL_IMAGES


def main(args: argparse.Namespace) -> None:
    from skimage.metrics import structural_similarity

    output = Path(args.output_json)
    if not output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("E82 result JSON must remain under dynamic/")
    price_path = Path(args.price_json)
    price_artifact, price = load_price(price_path, args.price_sha256)
    e72_path = Path(args.e72_reference_json)
    e72_reference = load_e72_reference(e72_path)
    _confirmation, component_scales = load_e70_confirmation(
        Path(args.e70_confirmation_json), smoke=False
    )
    if (
        set(component_scales) != set(shared.COMPONENT_NAMES)
        or E31_COMPONENT_WEIGHTS != {"inception": 1, "lpips": 2, "pixel_mse": 6}
    ):
        raise ValueError("E82 frozen component contract changed")

    distributed, rank, _local_rank, world_size, device = distributed_setup()
    is_main = rank == 0
    exists = torch.tensor(
        int(is_main and output.exists()), device=device, dtype=torch.long
    )
    if distributed:
        dist.broadcast(exists, src=0)
    if int(exists.item()):
        raise FileExistsError(f"refusing to overwrite {output}")

    torch.manual_seed(args.seed + rank)
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
    loader = DataLoader(
        Subset(dataset, range(shard_start, shard_end)),
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
    ot.EXPECTED_DIRECTION_FORMAT = DIRECTION_FORMAT
    direction_metadata = ot.configure_gaussian_ot_direction(args.e70_direction_npz)
    set_amplitude_power(ALPHA)
    autocast_type = autocast_dtype(args.mixed_precision)

    real_chunks: list[np.ndarray] = []
    dynamic_chunks: list[np.ndarray] = []
    metric_sums = torch.zeros(
        len(METRIC_NAMES) + 1, device=device, dtype=torch.float64
    )
    allocation_histogram = torch.zeros(257, device=device, dtype=torch.long)
    score_sums = torch.zeros(2, device=device, dtype=torch.float64)
    diagnostic_sums = torch.zeros(
        1 + 2 * len(shared.COMPONENT_NAMES), device=device, dtype=torch.float64
    )
    diagnostic_maxima = torch.zeros(
        2 * len(shared.COMPONENT_NAMES), device=device, dtype=torch.float64
    )
    local_images = 0

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"e82_train90_eval rank={rank}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for images_01, _indices in progress:
            batch = images_01.shape[0]
            images_01 = images_01.to(device, non_blocking=True)
            target, x_base, f_1d, f_2d, router_score, x_native = model_batch(
                images_01,
                model,
                model_metadata,
                args,
                autocast_type,
                True,
            )
            components, diagnostics = (
                ot.gaussian_ot_direction_relative_endpoint_component_gain_maps(
                    target,
                    x_base,
                    x_native,
                    fid_extractor.inception,
                    lpips_metric,
                    args.llamagen_input_range,
                )
            )
            shared.update_diagnostics(
                diagnostic_sums, diagnostic_maxima, diagnostics
            )
            combined = combine_e42_torch(components, component_scales)
            ranked = router_ranked_torch(combined, router_score)
            prefixes = torch.cumsum(ranked.float(), dim=1)
            score = (
                prefixes[:, CANDIDATES[1] - 1]
                - prefixes[:, CANDIDATES[0] - 1]
            ) / float(CANDIDATES[1] - CANDIDATES[0])
            tokens = decision_tokens(score, price)
            allocation_histogram += torch.bincount(tokens, minlength=257)[:257]
            score_sums[0] += score.double().sum()
            score_sums[1] += score.double().square().sum()

            with torch.autocast(
                device_type=device.type, dtype=autocast_type, enabled=True
            ):
                mask = variable_topk_mask(router_score, tokens, f_1d.dtype)
                prediction = model.llamagen_vq.decoder(
                    (1.0 - mask) * f_1d + mask * f_2d
                )
            real_features = fid_extractor.inception(
                to_uint8(target, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            dynamic_features = fid_extractor.inception(
                to_uint8(prediction, args.llamagen_input_range)
            ).reshape(batch, -1).float()
            real_chunks.append(real_features.cpu().numpy())
            dynamic_chunks.append(dynamic_features.cpu().numpy())
            metrics = per_image_metrics(
                prediction.float(),
                target.float(),
                args.llamagen_input_range,
                lpips_metric,
            )
            for index, name in enumerate(METRIC_NAMES):
                metric_sums[index] += float(
                    np.asarray(metrics[name], dtype=np.float64).sum()
                )
            metric_sums[-1] += ssim_batch_sum(
                prediction.float(),
                target.float(),
                args.llamagen_input_range,
                structural_similarity,
            )
            local_images += batch
            if is_main:
                progress.set_postfix(images=local_images)

    if local_images != shard_end - shard_start:
        raise AssertionError("E82 local validation count changed")
    feature_arrays = [
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for chunks in (real_chunks, dynamic_chunks)
    ]
    del model, lpips_metric, fid_extractor, real_chunks, dynamic_chunks
    gc.collect()
    torch.cuda.empty_cache()
    sufficient = [
        feature_sufficient_statistics(values, device, args.fid_moment_chunk_size)
        for values in feature_arrays
    ]
    del feature_arrays
    gc.collect()

    if distributed:
        for feature_sum, feature_cross, feature_count in sufficient:
            dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_cross, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_count, op=dist.ReduceOp.SUM)
        for tensor in (
            metric_sums,
            allocation_histogram,
            score_sums,
            diagnostic_sums,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(diagnostic_maxima, op=dist.ReduceOp.MAX)

    if is_main:
        counts = [int(item[2].item()) for item in sufficient]
        if counts != [count, count]:
            raise AssertionError(f"E82 global FID counts disagree: {counts}")
        moments = [
            moments_from_sufficient_statistics(item[0], item[1], count)
            for item in sufficient
        ]
        fid = float(_compute_fid(*moments[0], *moments[1]).item())
        dynamic = {
            name: float(metric_sums[index].item() / count)
            for index, name in enumerate(METRIC_NAMES)
        }
        dynamic["ssim"] = float(metric_sums[-1].item() / count)
        dynamic["fid"] = fid
        fixed = e72_reference["metrics"]["fixed_router"]
        allocation = allocation_summary(
            allocation_histogram.cpu().numpy(), 96
        )
        if (
            int(allocation["count"]) != count
            or not set(int(value) for value in allocation["histogram"]).issubset(
                CANDIDATES
            )
        ):
            raise AssertionError("E82 allocation contract failed")
        relative = {
            "fid_reduction": 100.0 * (fixed["fid"] - dynamic["fid"]) / fixed["fid"],
            "lpips_reduction": 100.0
            * (fixed["lpips"] - dynamic["lpips"])
            / fixed["lpips"],
            "l1_reduction": 100.0
            * (fixed["l1_01"] - dynamic["l1_01"])
            / fixed["l1_01"],
            "mse_reduction": 100.0
            * (fixed["mse_01"] - dynamic["mse_01"])
            / fixed["mse_01"],
            "psnr_increase": dynamic["psnr"] - fixed["psnr"],
            "ssim_increase": dynamic["ssim"] - fixed["ssim"],
        }
        score_mean = float(score_sums[0].item() / count)
        score_std = float(
            max(score_sums[1].item() / count - score_mean**2, 0.0) ** 0.5
        )
        performance_evidence = not args.smoke
        success = (
            performance_evidence
            and 95.5 <= float(allocation["mean"]) <= 96.5
            and dynamic["fid"] < 1.2
        )
        result = {
            "format": FORMAT,
            "status": "completed",
            "stage": "smoke" if args.smoke else "formal_val50k",
            "not_performance_evidence": args.smoke,
            "experiment": "E82 E72 train-90 rate-drift compensation",
            "algorithm": ALGORITHM,
            "candidate_tokens": list(CANDIDATES),
            "train_target_tokens": TRAIN_TARGET_TOKENS,
            "deployed_price": price,
            "allocation": allocation,
            "metrics": {
                "e82_train90_dynamic": dynamic,
                "fixed96_frozen_e72_reference": fixed,
            },
            "relative_changes_vs_fixed96_percent": relative,
            "primary_success_gate": {
                "actual_mean_in_95p5_96p5": performance_evidence
                and 95.5 <= float(allocation["mean"]) <= 96.5,
                "fid_below_1p2": performance_evidence and dynamic["fid"] < 1.2,
                "all_pass": success,
            },
            "score_summary": {"mean": score_mean, "std": score_std},
            "endpoint_diagnostics": shared.finalize_diagnostics(
                diagnostic_sums, diagnostic_maxima
            ),
            "num_images": count,
            "fid_feature": 2048,
            "fid_input": "uint8",
            "probe_reconstructions_per_image": 1,
            "distinct_probe_k_values": [256],
            "candidate_k_search_reconstructions_per_image": 0,
            "candidate_k_search_loss_evaluations_per_image": 0,
            "final_dynamic_reconstructions_per_image": 1,
            "selection_is_per_image_independent_after_price_frozen": True,
            "selection_requires_batch_after_price_frozen": False,
            "test_batch_statistics_used_for_selection": False,
            "test_set_budget_rebalancing_used": False,
            "validation_used_to_fit_price": False,
            "train_full_1281167_used_to_fit_price": True,
            "target_90_motivated_by_previously_observed_val_rate_drift": True,
            "post_hoc_rate_compensation_hypothesis": True,
            "reported_reconstruction_metrics_used_to_choose_price": False,
            "features_saved_to_disk": False,
            "npz_written": False,
            "reconstructions_saved": False,
            "stats_pt_written": False,
            "output_json_only": True,
            "price_artifact": {
                "path": str(price_path.resolve()),
                "sha256": file_sha256(price_path),
                "allocation": price_artifact["allocation"],
            },
            "e70_confirmation_artifact": {
                "path": str(Path(args.e70_confirmation_json).resolve()),
                "sha256": file_sha256(args.e70_confirmation_json),
                "component_scales_train_only": component_scales,
            },
            "e72_reference_artifact": {
                "path": str(e72_path.resolve()),
                "sha256": file_sha256(e72_path),
                "teacher_ceiling": e72_reference["metrics"]["choice2_gap64"],
            },
            "diagonal_gaussian_ot_direction": {
                "path": str(Path(args.e70_direction_npz).resolve()),
                "sha256": file_sha256(args.e70_direction_npz),
                "metadata": direction_metadata,
                "population_statistics_frozen_complete_train": True,
                "population_images": 1_281_167,
            },
            "checkpoint": str(Path(args.ckpt).resolve()),
            "checkpoint_sha256": file_sha256(args.ckpt),
            "checkpoint_state": model_metadata["checkpoint_state"],
            "data_path": str(Path(args.data_path).resolve()),
            "world_size": world_size,
            "runtime_seconds": time.time() - started,
            "runtime_args": vars(args),
            "model": model_metadata,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stage": result["stage"],
                    "allocation": allocation,
                    "metrics": result["metrics"],
                    "primary_success_gate": result["primary_success_gate"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-json",
        default="results/e82_e72_train90_v1/fulltrain_price.json",
    )
    parser.add_argument("--price-sha256", required=True)
    parser.add_argument(
        "--e70-confirmation-json",
        default="results/e70_fulltrain_diagonal_ot_three_choice_v1/fresh_confirmation.json",
    )
    parser.add_argument(
        "--e70-direction-npz",
        default="results/e70_fulltrain_diagonal_ot_three_choice_v1/direction_fulltrain_1281167.npz",
    )
    parser.add_argument(
        "--e72-reference-json",
        default="results/e72_choice_count_val_rate_v1/eval_val50k.json",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--ckpt",
        default=(
            "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/dynamic/"
            "e54/checkpoints/base_latest.pt"
        ),
    )
    parser.add_argument("--data-path", default=str(EXPECTED_VAL_PATH))
    parser.add_argument("--num-images", type=int, default=FORMAL_VAL_IMAGES)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--fid-moment-chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips-net", choices=("alex",), default="alex")
    parser.add_argument("--smoke", action="store_true")
    add_base_model_arguments(parser)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

