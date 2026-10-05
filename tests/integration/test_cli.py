"""CLI paths: data download (file:// archive), train/evaluate/report, benchmark,
register and predict, all on the synthetic fixture."""

from __future__ import annotations

import json
import shutil
import textwrap
import zipfile

import pandas as pd
import pytest
import yaml

from spermtriage.cli import build_parser, main
from spermtriage.data.download import IntegrityError, download_archive
from spermtriage.repro import sha256_file

pytestmark = pytest.mark.slow


def _zip_fixture(root):
    """Move the fixture images into a zip and point the dataset config at it."""
    src = root / "data" / "raw" / "tiny" / "Tiny"
    archive = root.parent / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in sorted(src.rglob("*.bmp")):
            zf.write(p, p.relative_to(src.parent).as_posix())
    shutil.rmtree(root / "data" / "raw")
    cfg_path = root / "configs" / "datasets" / "tiny.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["archive"] = {
        "filename": "tiny.zip",
        "url": archive.resolve().as_uri(),
        "sha256": sha256_file(archive),
        "root": "Tiny",
    }
    cfg_path.write_text(yaml.safe_dump(cfg))
    return archive


def test_parser_lists_all_commands():
    text = build_parser().format_help()
    for cmd in ("data", "train", "evaluate", "report", "benchmark", "register", "predict", "gui"):
        assert cmd in text


def test_download_rejects_wrong_hash(tiny_project, tiny_spec):
    _zip_fixture(tiny_project)
    from spermtriage.config import load_dataset_spec

    spec = load_dataset_spec("tiny")
    bad = type(spec.archive)(spec.archive.filename, spec.archive.url, "0" * 64, spec.archive.root)
    with pytest.raises(IntegrityError, match="does not match"):
        download_archive(type(spec)(**{**spec.__dict__, "archive": bad}))


def test_full_cli_workflow(tiny_project, tiny_spec, tmp_path, capsys):
    _zip_fixture(tiny_project)
    assert main(["data", "--dataset", "tiny"]) == 0
    assert (tiny_project / "data" / "INTEGRITY.md").exists()
    assert (tiny_project / "data" / "splits" / "tiny_splits_v1.csv").exists()

    exp = tiny_project / "configs" / "experiments" / "cli.yaml"
    exp.write_text(
        textwrap.dedent(
            """
            name: cli
            datasets: [tiny]
            split: {n_folds: 5}
            eval: {alphas: [0.1], bootstrap_resamples: 20}
            models:
              - id: tiny-ft
                backbone: tiny_test
                train: {mode: finetune, pretrained: false, image_size: 32, batch_size: 8,
                        warmup_epochs: 1, max_epochs: 1, min_epochs: 1, patience: 1}
            """
        )
    )
    assert main(["train", "--experiment", str(exp), "--fold", "0", "1"]) == 0
    assert main(["evaluate", "--experiment", "cli"]) == 0
    assert main(["report", "--experiment", "cli"]) == 0
    assert main(["benchmark", "--experiment", "cli", "--runs", "3"]) == 0
    eff = pd.read_csv(tiny_project / "results" / "cli" / "efficiency.csv")
    assert eff.loc[0, "latency_ms_median"] > 0

    run = tiny_project / "results" / "runs" / "cli" / "tiny" / "tiny-ft" / "fold0"
    assert main(["register", "--run", str(run), "--version", "0.1.0"]) == 0
    registry = yaml.safe_load((tiny_project / "models" / "registry.yaml").read_text())
    entry = registry["models"][0]
    assert entry["id"] == "tiny-tiny-ft-fold0" and len(entry["weights_sha256"]) == 64
    assert entry["metrics"]["source"].endswith("(5-fold CV)")

    capsys.readouterr()
    img_dir = tiny_project / "data" / "raw" / "tiny" / "Tiny" / "A_folder"
    out_csv = tmp_path / "pred.csv"
    assert main(["predict", str(img_dir), "--model", entry["id"], "--csv", str(out_csv)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 12 and all(
        x["decision"] in ("Auto-classified", "Refer to expert") for x in lines
    )
    assert len(pd.read_csv(out_csv)) == 12
