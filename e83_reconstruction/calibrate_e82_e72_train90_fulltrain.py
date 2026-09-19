#!/usr/bin/env python3
"""Freeze E72's binary price at mean K=90 using complete ImageNet-train."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from generate_e74_binary_teacher_labels import (
    CANDIDATES,
    FORMAT as LABEL_FORMAT,
    FORMAL_TRAIN_IMAGES,
    calibrate_binary_score_price,
)


FORMAT = "e82_e72_train90_fulltrain_price_v1"
TARGET_TOKENS = 90.0
REFERENCE_TARGET_TOKENS = 96.0
REFERENCE_PRICE = 3.62009859085083
EXPECTED_SHARDS = (
    (
        0,
        640_583,
        "403aab87b16cc32314e96b219b053a125edd76a771024d07f5421e666f70f088",
    ),
    (
        640_583,
        FORMAL_TRAIN_IMAGES,
        "8f483d74b5166dc7d7e35bddcbcbc4e788b5451bd0e1349897ee9c7c2f5f4eff",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocation(
    tokens: np.ndarray, target_tokens: float = TARGET_TOKENS
) -> dict[str, object]:
    values, counts = np.unique(tokens.astype(np.int64), return_counts=True)
    histogram = {str(int(k)): int(v) for k, v in zip(values, counts)}
    mean = float(tokens.mean(dtype=np.float64))
    return {
        "count": int(tokens.size),
        "histogram": histogram,
        "mean": mean,
        "mean_delta_from_target": mean - float(target_tokens),
        "std": float(tokens.std(dtype=np.float64)),
        "min": int(tokens.min()),
        "max": int(tokens.max()),
    }


def load_scores(paths: list[Path]) -> tuple[np.ndarray, list[dict[str, object]]]:
    if len(paths) != len(EXPECTED_SHARDS):
        raise ValueError("E82 requires exactly the two complete E74 train shards")
    chunks: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for rank, (path, expected) in enumerate(zip(paths, EXPECTED_SHARDS)):
        expected_start, expected_end, expected_sha = expected
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"E82 label shard hash changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "relative_paths",
                "dataset_indices",
                "candidate_tokens",
                "prefix_gains",
                "binary_score",
                "metadata_json",
            }
            if set(archive.files) != required:
                raise ValueError(f"E82 label arrays changed: {path}")
            indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
            candidates = np.asarray(archive["candidate_tokens"], dtype=np.int64)
            prefixes = np.asarray(archive["prefix_gains"], dtype=np.float32)
            scores = np.asarray(archive["binary_score"], dtype=np.float32)
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        count = expected_end - expected_start
        if (
            metadata.get("format") != LABEL_FORMAT
            or metadata.get("validation_images_used") is not False
            or tuple(metadata.get("candidate_tokens", ())) != tuple(CANDIDATES)
            or indices.shape != (count,)
            or not np.array_equal(
                indices, np.arange(expected_start, expected_end, dtype=np.int64)
            )
            or not np.array_equal(candidates, np.asarray(CANDIDATES, dtype=np.int64))
            or prefixes.shape != (count, 2)
            or scores.shape != (count,)
            or not np.isfinite(prefixes).all()
            or not np.isfinite(scores).all()
            or not np.allclose(
                scores,
                (prefixes[:, 1] - prefixes[:, 0]) / 64.0,
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            raise ValueError(f"E82 invalid complete-train shard: {path}")
        chunks.append(scores)
        records.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "sha256": actual_sha,
                "start": expected_start,
                "end": expected_end,
                "count": count,
            }
        )
    combined = np.concatenate(chunks).astype(np.float32, copy=False)
    if combined.shape != (FORMAL_TRAIN_IMAGES,):
        raise AssertionError("E82 complete-train score coverage changed")
    return combined, records


def main(args: argparse.Namespace) -> None:
    output = Path(args.output_json)
    if not output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("E82 output JSON must remain under dynamic/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if float(args.target_tokens) != TARGET_TOKENS:
        raise ValueError(f"formal E82 freezes train target at {TARGET_TOKENS}")
    started = time.time()
    paths = [Path(value) for value in args.label_npz]
    scores, sources = load_scores(paths)
    reference_price, reference_tokens = calibrate_binary_score_price(
        scores, target_tokens=REFERENCE_TARGET_TOKENS
    )
    if reference_price != REFERENCE_PRICE:
        raise AssertionError("E82 loader failed to reproduce the E74 train-96 price")
    price, tokens = calibrate_binary_score_price(scores, target_tokens=TARGET_TOKENS)
    result = {
        "format": FORMAT,
        "status": "completed",
        "experiment": "E82 E72 full-train mean-90 rate-drift compensation",
        "algorithm": "E72_binary_average_marginal_fixed_fulltrain_price",
        "candidate_tokens": list(CANDIDATES),
        "score": "float32((G128-G64)/64)",
        "decision_rule": "K128 iff score > price; otherwise K64",
        "target_train_mean_tokens": TARGET_TOKENS,
        "price": float(price),
        "allocation": allocation(tokens),
        "reference_reproduction": {
            "target_tokens": REFERENCE_TARGET_TOKENS,
            "price": float(reference_price),
            "allocation": allocation(reference_tokens, REFERENCE_TARGET_TOKENS),
            "expected_price": REFERENCE_PRICE,
            "exact_price_reproduced": True,
        },
        "score_summary": {
            "count": int(scores.size),
            "mean": float(scores.mean(dtype=np.float64)),
            "std": float(scores.std(dtype=np.float64)),
            "min": float(scores.min()),
            "max": float(scores.max()),
        },
        "price_sources": sources,
        "complete_imagenet_train_coverage": True,
        "train_images": FORMAL_TRAIN_IMAGES,
        "validation_images_used_for_price": 0,
        "validation_statistics_used_for_price": False,
        "reported_reconstruction_metrics_used_to_choose_price": False,
        "target_90_motivated_by_previously_observed_val_rate_drift": True,
        "post_hoc_rate_compensation_hypothesis": True,
        "selection_is_per_image_independent_after_price_frozen": True,
        "selection_requires_batch_after_price_frozen": False,
        "features_saved": False,
        "reconstructions_saved": False,
        "npz_written": False,
        "stats_pt_written": False,
        "output_json_only": True,
        "runtime_seconds": time.time() - started,
        "runtime_args": vars(args),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(
        "/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/dynamic/"
        "e74/teacher_fulltrain_v1"
    )
    parser.add_argument(
        "--label-npz",
        action="append",
        default=None,
        help="Pass exactly two complete E74 full-train shards in rank order.",
    )
    parser.add_argument("--target-tokens", type=float, default=TARGET_TOKENS)
    parser.add_argument(
        "--output-json",
        default="results/e82_e72_train90_v1/fulltrain_price.json",
    )
    args = parser.parse_args()
    if args.label_npz is None:
        args.label_npz = [
            str(root / "teacher_rank00.npz"),
            str(root / "teacher_rank01.npz"),
        ]
    return args


if __name__ == "__main__":
    main(build_parser())

