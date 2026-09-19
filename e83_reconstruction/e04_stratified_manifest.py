#!/usr/bin/env python3
"""Build and validate deterministic class-balanced E04 train manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


FORMAT = "e04_stratified_index_manifest_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def stable_digest(seed: int, namespace: str, value: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    return digest.digest()


def entries_sha256(
    dataset_indices: np.ndarray,
    relative_paths: Sequence[str],
    class_ids: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for index, path, class_id in zip(dataset_indices, relative_paths, class_ids):
        digest.update(int(index).to_bytes(8, "little", signed=True))
        digest.update(int(class_id).to_bytes(4, "little", signed=True))
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_plan_sha256(
    seed: int,
    class_names: Sequence[str],
    split_entries: Sequence[tuple[str, np.ndarray, Sequence[str], np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    digest.update(FORMAT.encode("ascii"))
    digest.update(int(seed).to_bytes(8, "little", signed=True))
    for class_name in class_names:
        digest.update(str(class_name).encode("utf-8"))
        digest.update(b"\n")
    for split_name, indices, paths, class_ids in split_entries:
        digest.update(split_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entries_sha256(indices, paths, class_ids).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    path: Path
    dataset_indices: np.ndarray
    relative_paths: np.ndarray
    class_ids: np.ndarray
    metadata: dict[str, object]

    @property
    def entries_sha256(self) -> str:
        return str(self.metadata["entries_sha256"])

    @property
    def split_plan_sha256(self) -> str:
        return str(self.metadata["split_plan_sha256"])


def load_index_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    dataset_paths: Sequence[Path],
    expected_split: str | None = None,
) -> IndexManifest:
    path = Path(manifest_path)
    with np.load(path, allow_pickle=False) as archive:
        required = ("dataset_indices", "relative_paths", "class_ids", "metadata_json")
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"index manifest {path} is missing {missing}")
        indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        relative_paths = np.asarray(archive["relative_paths"]).astype(str)
        class_ids = np.asarray(archive["class_ids"], dtype=np.int64)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if metadata.get("format") != FORMAT:
        raise ValueError(f"unsupported index manifest format in {path}")
    if indices.ndim != 1 or relative_paths.shape != indices.shape or class_ids.shape != indices.shape:
        raise ValueError(f"manifest arrays have inconsistent shapes in {path}")
    if indices.size == 0:
        raise ValueError(f"index manifest {path} is empty")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"index manifest {path} contains duplicate dataset indices")
    if indices.min() < 0 or indices.max() >= len(dataset_paths):
        raise IndexError(f"index manifest {path} contains an out-of-range dataset index")
    if int(metadata.get("dataset_size", -1)) != len(dataset_paths):
        raise ValueError(f"manifest dataset size differs from the current dataset in {path}")
    if int(metadata.get("num_images", -1)) != indices.size:
        raise ValueError(f"manifest image count disagrees with its arrays in {path}")
    unique_classes, class_counts = np.unique(class_ids, return_counts=True)
    if unique_classes.size != int(metadata.get("num_classes", -1)):
        raise ValueError(f"manifest class coverage is incomplete in {path}")
    if unique_classes.min() != 0 or unique_classes.max() != unique_classes.size - 1:
        raise ValueError(f"manifest class ids are not contiguous from zero in {path}")
    if (
        int(metadata.get("class_count_min", -1)) != int(class_counts.min())
        or int(metadata.get("class_count_max", -1)) != int(class_counts.max())
    ):
        raise ValueError(f"manifest class counts disagree with metadata in {path}")
    saved_root = Path(str(metadata["data_root"])).resolve()
    if saved_root != Path(data_root).resolve():
        raise ValueError(f"manifest data root {saved_root} differs from {Path(data_root).resolve()}")
    if expected_split is not None and metadata.get("split") != expected_split:
        raise ValueError(
            f"manifest split {metadata.get('split')} differs from expected {expected_split}"
        )
    resolved_root = Path(data_root).resolve()
    actual_paths = np.asarray(
        [
            Path(dataset_paths[int(index)]).resolve().relative_to(resolved_root).as_posix()
            for index in indices
        ]
    )
    if not np.array_equal(relative_paths, actual_paths):
        mismatch = int(np.flatnonzero(relative_paths != actual_paths)[0])
        raise ValueError(
            f"manifest path mismatch at row {mismatch}: "
            f"{relative_paths[mismatch]} != {actual_paths[mismatch]}"
        )
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != unique_classes.size:
        raise ValueError(f"manifest class names are missing or inconsistent in {path}")
    if len(set(map(str, class_names))) != len(class_names):
        raise ValueError(f"manifest class names are not unique in {path}")
    actual_class_names = np.asarray(
        [relative_path.split("/", 1)[0] for relative_path in relative_paths]
    )
    expected_class_names = np.asarray(
        [str(class_names[int(class_id)]) for class_id in class_ids]
    )
    if not np.array_equal(actual_class_names, expected_class_names):
        mismatch = int(np.flatnonzero(actual_class_names != expected_class_names)[0])
        raise ValueError(
            f"manifest class mismatch at row {mismatch}: "
            f"{actual_class_names[mismatch]} != {expected_class_names[mismatch]}"
        )
    computed_hash = entries_sha256(indices, relative_paths.tolist(), class_ids)
    if computed_hash != metadata.get("entries_sha256"):
        raise ValueError(f"manifest entry hash mismatch in {path}")
    return IndexManifest(path.resolve(), indices, relative_paths, class_ids, metadata)


def save_manifest(
    output_path: Path,
    data_root: Path,
    dataset_size: int,
    split_name: str,
    dataset_indices: np.ndarray,
    relative_paths: Sequence[str],
    class_ids: np.ndarray,
    class_names: Sequence[str],
    seed: int,
    plan_hash: str,
    selection: dict[str, object],
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    entry_hash = entries_sha256(dataset_indices, relative_paths, class_ids)
    unique, counts = np.unique(class_ids, return_counts=True)
    metadata = {
        "format": FORMAT,
        "split": split_name,
        "data_root": str(data_root.resolve()),
        "dataset_size": int(dataset_size),
        "num_images": int(dataset_indices.size),
        "num_classes": int(len(class_names)),
        "class_names": list(class_names),
        "class_count_min": int(counts.min()),
        "class_count_max": int(counts.max()),
        "represented_class_ids": unique.astype(int).tolist(),
        "seed": int(seed),
        "entries_sha256": entry_hash,
        "split_plan_sha256": plan_hash,
        "selection": selection,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        dataset_indices=dataset_indices.astype(np.int64),
        relative_paths=np.asarray(relative_paths),
        class_ids=class_ids.astype(np.int16),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "path": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "split": split_name,
        "num_images": int(dataset_indices.size),
        "entries_sha256": entry_hash,
        "class_count_min": int(counts.min()),
        "class_count_max": int(counts.max()),
    }


def round_robin_order(
    selected_by_class: list[list[tuple[int, str]]],
    class_names: Sequence[str],
    seed: int,
    namespace: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    per_class = {len(entries) for entries in selected_by_class}
    if len(per_class) != 1:
        raise ValueError(f"class quotas disagree: {sorted(per_class)}")
    rounds = per_class.pop()
    ordered: list[tuple[int, str, int]] = []
    for round_index in range(rounds):
        class_order = sorted(
            range(len(class_names)),
            key=lambda class_id: stable_digest(
                seed, f"{namespace}:round:{round_index}", class_names[class_id]
            ),
        )
        for class_id in class_order:
            index, relative_path = selected_by_class[class_id][round_index]
            ordered.append((index, relative_path, class_id))
    return (
        np.asarray([entry[0] for entry in ordered], dtype=np.int64),
        [entry[1] for entry in ordered],
        np.asarray([entry[2] for entry in ordered], dtype=np.int64),
    )


def validate_balanced_allocation_groups(
    class_ids: np.ndarray,
    num_classes: int,
    allocation_group_size: int,
    split_name: str,
) -> int:
    if allocation_group_size <= 0:
        raise ValueError("--allocation-group-size must be positive")
    if allocation_group_size % num_classes != 0:
        raise ValueError(
            "--allocation-group-size must be divisible by the number of classes"
        )
    if class_ids.size % allocation_group_size != 0:
        raise ValueError(
            f"{split_name} size {class_ids.size} is not divisible by "
            f"allocation group size {allocation_group_size}"
        )
    expected_per_class = allocation_group_size // num_classes
    for group_start in range(0, class_ids.size, allocation_group_size):
        counts = np.bincount(
            class_ids[group_start : group_start + allocation_group_size],
            minlength=num_classes,
        )
        if not np.all(counts == expected_per_class):
            raise AssertionError(
                f"{split_name} allocation group at row {group_start} is not class-balanced"
            )
    return expected_per_class


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir)
    if args.teacher_per_class <= 0 or args.calibration_per_class <= 0:
        raise ValueError("teacher and calibration quotas must be positive")
    if args.teacher_shards <= 0:
        raise ValueError("--teacher-shards must be positive")
    if args.teacher_per_class % args.teacher_shards != 0:
        raise ValueError("--teacher-per-class must be divisible by --teacher-shards")
    if args.allocation_group_size <= 0:
        raise ValueError("--allocation-group-size must be positive")
    planned_outputs = [
        output_dir / f"teacher_shard_{index:02d}_manifest.npz"
        for index in range(args.teacher_shards)
    ]
    planned_outputs.extend(
        [output_dir / "calibration_manifest.npz", output_dir / "split_summary.json"]
    )
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            f"refusing to overwrite existing split outputs: {existing_outputs}"
        )
    paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no images found under {data_root}")
    relative_paths = [path.relative_to(data_root).as_posix() for path in paths]
    grouped: dict[str, list[tuple[int, str]]] = {}
    for index, relative_path in enumerate(relative_paths):
        class_name = relative_path.split("/", 1)[0]
        grouped.setdefault(class_name, []).append((index, relative_path))
    class_names = sorted(grouped)
    if len(class_names) != args.expected_classes:
        raise ValueError(
            f"found {len(class_names)} classes, expected {args.expected_classes}"
        )
    required_per_class = args.teacher_per_class + args.calibration_per_class
    selected_teacher: list[list[tuple[int, str]]] = []
    selected_calibration: list[list[tuple[int, str]]] = []
    for class_name in class_names:
        ranked = sorted(
            grouped[class_name],
            key=lambda entry: stable_digest(args.seed, "sample", entry[1]),
        )
        if len(ranked) < required_per_class:
            raise ValueError(
                f"class {class_name} has {len(ranked)} images, needs {required_per_class}"
            )
        selected_teacher.append(ranked[: args.teacher_per_class])
        selected_calibration.append(ranked[args.teacher_per_class : required_per_class])

    teacher_splits: list[tuple[str, np.ndarray, list[str], np.ndarray]] = []
    shard_quota = args.teacher_per_class // args.teacher_shards
    for shard_index in range(args.teacher_shards):
        shard_by_class = [
            entries[shard_index * shard_quota : (shard_index + 1) * shard_quota]
            for entries in selected_teacher
        ]
        indices, split_paths, class_ids = round_robin_order(
            shard_by_class,
            class_names,
            args.seed,
            f"teacher_shard_{shard_index:02d}",
        )
        teacher_splits.append(
            (f"teacher_shard_{shard_index:02d}", indices, split_paths, class_ids)
        )
    allocation_group_class_count = args.allocation_group_size // len(class_names)
    for split_name, _indices, _split_paths, class_ids in teacher_splits:
        allocation_group_class_count = validate_balanced_allocation_groups(
            class_ids,
            len(class_names),
            args.allocation_group_size,
            split_name,
        )
    calibration_indices, calibration_paths, calibration_class_ids = round_robin_order(
        selected_calibration, class_names, args.seed, "calibration"
    )
    all_splits = [
        *teacher_splits,
        ("calibration", calibration_indices, calibration_paths, calibration_class_ids),
    ]
    all_indices = np.concatenate([split[1] for split in all_splits])
    if np.unique(all_indices).size != all_indices.size:
        raise AssertionError("teacher/calibration manifests overlap")
    plan_hash = split_plan_sha256(args.seed, class_names, all_splits)
    outputs = []
    for split_name, indices, split_paths, class_ids in teacher_splits:
        outputs.append(
            save_manifest(
                output_dir / f"{split_name}_manifest.npz",
                data_root,
                len(paths),
                split_name,
                indices,
                split_paths,
                class_ids,
                class_names,
                args.seed,
                plan_hash,
                {
                    "teacher_per_class": int(args.teacher_per_class),
                    "teacher_shards": int(args.teacher_shards),
                    "per_class_in_this_shard": int(shard_quota),
                    "allocation_group_size": int(args.allocation_group_size),
                    "per_class_in_each_allocation_group": int(
                        allocation_group_class_count
                    ),
                    "ordering": "round_robin_one_per_class_with_sha256_class_order",
                },
            )
        )
    outputs.append(
        save_manifest(
            output_dir / "calibration_manifest.npz",
            data_root,
            len(paths),
            "calibration",
            calibration_indices,
            calibration_paths,
            calibration_class_ids,
            class_names,
            args.seed,
            plan_hash,
            {
                "calibration_per_class": int(args.calibration_per_class),
                "ordering": "round_robin_one_per_class_with_sha256_class_order",
            },
        )
    )
    summary = {
        "format": "e04_stratified_split_summary_v1",
        "status": "completed",
        "data_root": str(data_root),
        "dataset_size": len(paths),
        "num_classes": len(class_names),
        "teacher_per_class": int(args.teacher_per_class),
        "teacher_shards": int(args.teacher_shards),
        "calibration_per_class": int(args.calibration_per_class),
        "allocation_group_size": int(args.allocation_group_size),
        "per_class_in_each_teacher_allocation_group": int(
            allocation_group_class_count
        ),
        "teacher_calibration_overlap": 0,
        "split_plan_sha256": plan_hash,
        "seed": int(args.seed),
        "outputs": outputs,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="/var/tmp/heyefei_ImageNet/train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-per-class", type=int, default=100)
    parser.add_argument("--teacher-shards", type=int, default=2)
    parser.add_argument("--calibration-per-class", type=int, default=10)
    parser.add_argument("--expected-classes", type=int, default=1000)
    parser.add_argument("--allocation-group-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
