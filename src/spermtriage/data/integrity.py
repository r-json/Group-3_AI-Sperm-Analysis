"""Build a hashed manifest for a dataset and enforce its integrity.

The manifest is the single source of truth for which images exist, what their labels are,
and which are excluded. Policy (documented in docs/data_card.md):

* exact duplicates are detected on *decoded pixels*, not file bytes, so a re-encoded copy
  of an image is still caught;
* a duplicate group whose members share one label keeps its first member (sorted path) and
  marks the rest ``duplicate``;
* a duplicate group whose members carry different labels is a label conflict; every member
  is marked ``label_conflict`` and excluded, because no single label can be trusted.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spermtriage.config import DatasetSpec, project_root
from spermtriage.data.download import IntegrityError
from spermtriage.data.images import SUPPORTED_SUFFIXES, dhash64, pixel_sha256, read_rgb
from spermtriage.repro import sha256_file

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "image_id",
    "relpath",
    "label",
    "label_idx",
    "file_sha256",
    "pixel_sha256",
    "width",
    "height",
    "dhash",
    "dup_group",
    "status",
]


def manifest_path(dataset: str) -> Path:
    return project_root() / "data" / "manifests" / f"{dataset}_manifest.csv"


def scan_dataset(spec: DatasetSpec, image_dir: Path) -> pd.DataFrame:
    """Hash every image under ``image_dir`` whose parent folder is a known class."""
    rows = []
    for folder, label in spec.folder_to_label.items():
        cls_dir = image_dir / folder
        if not cls_dir.is_dir():
            raise IntegrityError(f"Missing class folder: {cls_dir}")
        for path in sorted(cls_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            arr = read_rgb(path)
            rel = path.relative_to(image_dir).as_posix()
            rows.append(
                {
                    "image_id": f"{spec.name}/{rel}",
                    "relpath": rel,
                    "label": label,
                    "label_idx": spec.label_index(label),
                    "file_sha256": sha256_file(path),
                    "pixel_sha256": pixel_sha256(arr),
                    "width": arr.shape[1],
                    "height": arr.shape[0],
                    "dhash": f"{dhash64(arr):016x}",
                }
            )
    return pd.DataFrame(rows)


def apply_duplicate_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``dup_group`` and ``status`` columns following the module policy."""
    df = df.sort_values("relpath").reset_index(drop=True).copy()
    df["dup_group"] = ""
    df["status"] = "ok"
    for pix, idx in df.groupby("pixel_sha256").groups.items():
        if len(idx) < 2:
            continue
        members = sorted(idx)
        df.loc[members, "dup_group"] = str(pix)[:16]
        if df.loc[members, "label"].nunique() > 1:
            df.loc[members, "status"] = "label_conflict"
        else:
            df.loc[members[1:], "status"] = "duplicate"
    return df[MANIFEST_COLUMNS]


def check_expected_counts(spec: DatasetSpec, df: pd.DataFrame) -> None:
    """Fail loudly if the raw per-class counts differ from the official release."""
    counts = df["label"].value_counts().to_dict()
    if counts != spec.expected_counts:
        raise IntegrityError(
            f"{spec.display_name}: class counts {counts} differ from the official "
            f"{spec.expected_counts}"
        )


def build_manifest(spec: DatasetSpec, image_dir: Path | None = None) -> pd.DataFrame:
    image_dir = image_dir or spec.extracted_dir
    df = scan_dataset(spec, image_dir)
    check_expected_counts(spec, df)
    return apply_duplicate_policy(df)


def verify_against_manifest(spec: DatasetSpec, manifest: pd.DataFrame, image_dir: Path) -> None:
    """Re-hash files on disk and compare with a committed manifest (fails on any drift)."""
    problems = []
    for relpath, expected in zip(manifest["relpath"], manifest["file_sha256"], strict=True):
        path = image_dir / str(relpath)
        if not path.exists():
            problems.append(f"missing: {relpath}")
        elif sha256_file(path) != expected:
            problems.append(f"hash mismatch: {relpath}")
    if problems:
        head = "\n  ".join(problems[:10])
        raise IntegrityError(f"{len(problems)} manifest problems, e.g.\n  {head}")


@dataclass(frozen=True)
class IntegrityReport:
    dataset: str
    n_files: int
    n_unique_pixels: int
    n_duplicate_groups: int
    n_conflict_groups: int
    n_excluded_duplicates: int
    n_excluded_conflicts: int
    n_kept: int
    kept_counts: dict[str, int]
    near_duplicate_pairs: int
    near_duplicate_cross_class_pairs: int

    def as_markdown(self) -> str:
        lines = [
            f"### {self.dataset}",
            "",
            "| Quantity | Value |",
            "| --- | --- |",
            f"| Files in official release | {self.n_files} |",
            f"| Unique decoded images | {self.n_unique_pixels} |",
            f"| Pixel-identical groups | {self.n_duplicate_groups} |",
            f"| ... of which carry conflicting labels | {self.n_conflict_groups} |",
            f"| Excluded as redundant duplicates | {self.n_excluded_duplicates} |",
            f"| Excluded as label conflicts | {self.n_excluded_conflicts} |",
            f"| Images kept | {self.n_kept} |",
            f"| Kept per class | {self.kept_counts} |",
            f"| Near-duplicate pairs (identical dHash, heuristic) | {self.near_duplicate_pairs} |",
            f"| ... across classes | {self.near_duplicate_cross_class_pairs} |",
        ]
        return "\n".join(lines)


def near_duplicate_pairs(manifest: pd.DataFrame, max_distance: int = 0) -> tuple[int, int]:
    """Count pixel-distinct pairs whose dHash differs by at most ``max_distance`` bits."""
    uniq = manifest.drop_duplicates("pixel_sha256")
    hashes = np.array([int(h, 16) for h in uniq["dhash"]], dtype=np.uint64)
    labels = uniq["label"].to_numpy()
    total = cross = 0
    for i in range(len(hashes) - 1):
        x = np.bitwise_xor(hashes[i + 1 :], hashes[i])
        dist = np.array([bin(int(v)).count("1") for v in x]) if len(x) else np.array([])
        hits = np.flatnonzero(dist <= max_distance)
        total += len(hits)
        cross += int(np.sum(labels[i + 1 :][hits] != labels[i]))
    return total, cross


def integrity_report(dataset: str, manifest: pd.DataFrame) -> IntegrityReport:
    groups: dict[str, set[str]] = defaultdict(set)
    for group, label in zip(manifest["dup_group"], manifest["label"], strict=True):
        if group:
            groups[str(group)].add(str(label))
    kept = manifest[manifest["status"] == "ok"]
    near, near_cross = near_duplicate_pairs(manifest)
    return IntegrityReport(
        dataset=dataset,
        n_files=len(manifest),
        n_unique_pixels=manifest["pixel_sha256"].nunique(),
        n_duplicate_groups=len(groups),
        n_conflict_groups=sum(1 for labels in groups.values() if len(labels) > 1),
        n_excluded_duplicates=int((manifest["status"] == "duplicate").sum()),
        n_excluded_conflicts=int((manifest["status"] == "label_conflict").sum()),
        n_kept=len(kept),
        kept_counts={str(k): int(v) for k, v in kept["label"].value_counts().sort_index().items()},
        near_duplicate_pairs=near,
        near_duplicate_cross_class_pairs=near_cross,
    )


def load_manifest(dataset: str, include_excluded: bool = False) -> pd.DataFrame:
    path = manifest_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: spermtriage data --dataset {dataset}")
    df = pd.read_csv(path, dtype={"dup_group": str, "dhash": str}, keep_default_na=False)
    return df if include_excluded else df[df["status"] == "ok"].reset_index(drop=True)
