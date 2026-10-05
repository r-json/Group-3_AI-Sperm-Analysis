"""Fetch the official dataset archives and verify them by SHA-256.

The datasets are not redistributed in this repository. They are downloaded from their
canonical Mendeley Data records and checked against the hash Mendeley publishes.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from spermtriage.config import DatasetSpec, data_root
from spermtriage.repro import sha256_file

log = logging.getLogger(__name__)


class IntegrityError(RuntimeError):
    """Raised when a file or dataset does not match its recorded fingerprint."""


def download_archive(spec: DatasetSpec, dest_dir: Path | None = None, force: bool = False) -> Path:
    dest_dir = dest_dir or data_root() / spec.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / spec.archive.filename
    if target.exists() and not force and sha256_file(target) == spec.archive.sha256:
        log.info("%s already downloaded and verified", target)
        return target
    log.info("Downloading %s from %s", spec.display_name, spec.archive.url)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(spec.archive.url, timeout=300) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    digest = sha256_file(tmp)
    if digest != spec.archive.sha256:
        tmp.unlink(missing_ok=True)
        raise IntegrityError(
            f"{spec.archive.filename}: SHA-256 {digest} does not match the published "
            f"{spec.archive.sha256}"
        )
    tmp.replace(target)
    return target


def extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract .zip natively; .rar via libarchive-c or a system extractor."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
        return
    try:
        import libarchive

        cwd = Path.cwd()
        try:
            os.chdir(dest_dir)
            libarchive.extract_file(str(archive.resolve()))
        finally:
            os.chdir(cwd)
        return
    except ImportError:
        pass
    for tool in (["unrar", "x", "-o+"], ["7z", "x", "-y"], ["bsdtar", "-xf"], ["unar", "-f"]):
        if shutil.which(tool[0]):
            cmd = [*tool, str(archive.resolve())]
            if tool[0] == "7z":
                cmd.append(f"-o{dest_dir}")
            result = subprocess.run(cmd, cwd=dest_dir, capture_output=True, check=False)
            if result.returncode == 0:
                return
    raise RuntimeError(
        f"Cannot extract {archive.name}. Install the optional extra "
        "(pip install 'spermtriage[data]') or extract it manually into "
        f"{dest_dir}."
    )


def fetch_dataset(spec: DatasetSpec, force: bool = False) -> Path:
    """Download (if needed), verify and extract one dataset; return its image folder."""
    if spec.extracted_dir.exists() and not force:
        return spec.extracted_dir
    archive = download_archive(spec, force=force)
    extract_archive(archive, archive.parent)
    if not spec.extracted_dir.exists():
        raise IntegrityError(f"Expected folder {spec.extracted_dir} after extracting {archive}")
    return spec.extracted_dir
