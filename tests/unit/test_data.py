import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spermtriage.config import ExperimentConfig, SplitConfig, load_dataset_spec, project_root
from spermtriage.data.download import IntegrityError
from spermtriage.data.images import (
    ImageValidationError,
    load_square,
    pad_to_square,
    pixel_sha256,
    read_rgb,
)
from spermtriage.data.integrity import (
    apply_duplicate_policy,
    build_manifest,
    integrity_report,
    verify_against_manifest,
)
from spermtriage.data.splits import LeakageError, check_no_leakage, fold_ids, make_splits


# ---------------------------------------------------------------- images
def test_read_rgb_rejects_bad_inputs(tmp_path):
    with pytest.raises(ImageValidationError, match="Unsupported"):
        read_rgb(tmp_path / "x.gif")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageValidationError, match="Cannot decode"):
        read_rgb(bad)
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (4, 4)).save(tiny)
    with pytest.raises(ImageValidationError, match="outside"):
        read_rgb(tiny)
    with pytest.raises(ImageValidationError, match="not found"):
        read_rgb(tmp_path / "missing.png")


def test_grayscale_is_converted_to_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.new("L", (30, 20), 128).save(p)
    arr = read_rgb(p)
    assert arr.shape == (20, 30, 3) and arr.dtype == np.uint8


def test_pad_to_square_preserves_content_and_aspect(tmp_path):
    arr = np.zeros((20, 30, 3), np.uint8)
    arr[:, 10] = 255
    sq = pad_to_square(arr)
    assert sq.shape == (30, 30, 3)
    assert np.array_equal(sq[5:25], arr)
    p = tmp_path / "r.png"
    Image.fromarray(arr).save(p)
    assert load_square(p, 64).shape == (64, 64, 3)


def test_pixel_hash_ignores_encoding(tmp_path):
    arr = np.random.default_rng(0).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.bmp")
    Image.fromarray(arr).save(tmp_path / "a.png")
    assert pixel_sha256(read_rgb(tmp_path / "a.bmp")) == pixel_sha256(read_rgb(tmp_path / "a.png"))


# ---------------------------------------------------------------- integrity
def _rows(spec):
    return [
        {"relpath": "a/1", "label": "Alpha", "pixel_sha256": "h1"},
        {"relpath": "a/2", "label": "Alpha", "pixel_sha256": "h1"},  # same-class duplicate
        {"relpath": "a/3", "label": "Alpha", "pixel_sha256": "h2"},  # conflict with b/1
        {"relpath": "b/1", "label": "Beta", "pixel_sha256": "h2"},
        {"relpath": "b/2", "label": "Beta", "pixel_sha256": "h3"},
    ]


def test_duplicate_policy(tiny_spec):
    df = pd.DataFrame(_rows(tiny_spec))
    for col in ("image_id", "label_idx", "file_sha256", "width", "height", "dhash"):
        df[col] = 0
    out = apply_duplicate_policy(df).set_index("relpath")["status"].to_dict()
    assert out == {
        "a/1": "ok",
        "a/2": "duplicate",
        "a/3": "label_conflict",
        "b/1": "label_conflict",
        "b/2": "ok",
    }


def test_build_manifest_and_report(tiny_spec):
    manifest = build_manifest(tiny_spec)
    assert len(manifest) == 36 and (manifest["status"] == "ok").all()
    assert manifest["file_sha256"].str.len().eq(64).all()
    report = integrity_report("Tiny", manifest)
    assert report.n_kept == 36 and report.n_conflict_groups == 0
    assert "Images kept | 36" in report.as_markdown()
    verify_against_manifest(tiny_spec, manifest, tiny_spec.extracted_dir)


def test_count_drift_fails_loudly(tiny_spec):
    (tiny_spec.extracted_dir / "A_folder" / "img_00.bmp").unlink()
    with pytest.raises(IntegrityError, match="differ from the official"):
        build_manifest(tiny_spec)


def test_tampered_file_fails_manifest_check(tiny_spec):
    manifest = build_manifest(tiny_spec)
    target = tiny_spec.extracted_dir / manifest.loc[0, "relpath"]
    Image.new("RGB", (40, 40), "red").save(target)
    with pytest.raises(IntegrityError, match="hash mismatch"):
        verify_against_manifest(tiny_spec, manifest, tiny_spec.extracted_dir)


# ---------------------------------------------------------------- splits
def test_splits_are_leakage_free_and_stratified(tiny_spec):
    manifest = build_manifest(tiny_spec)
    cfg = SplitConfig(n_folds=3, val_fraction=0.25, calib_fraction=0.25, seed=1)
    splits = make_splits(manifest, cfg)
    check_no_leakage(splits, manifest)
    for k in range(3):
        ids = fold_ids(splits, k)
        assert sum(len(v) for v in ids.values()) == 36
        test_labels = manifest.set_index("image_id").loc[ids["test"], "label"]
        assert test_labels.value_counts().min() == 4  # 12 per class / 3 folds
    assert make_splits(manifest, cfg).equals(splits)  # deterministic


def test_leakage_is_detected(tiny_spec):
    manifest = build_manifest(tiny_spec)
    splits = make_splits(manifest, SplitConfig(n_folds=3, val_fraction=0.25, calib_fraction=0.25))
    tampered = manifest.copy()
    # Make a val image and a test image of fold 0 identical in content.
    f0 = fold_ids(splits, 0)
    tampered.loc[tampered["image_id"] == f0["val"][0], "pixel_sha256"] = tampered.loc[
        tampered["image_id"] == f0["test"][0], "pixel_sha256"
    ].iloc[0]
    with pytest.raises(LeakageError):
        check_no_leakage(splits, tampered)
    with pytest.raises(LeakageError, match="exactly one test fold"):
        check_no_leakage(splits[splits["image_id"] != f0["test"][0]], manifest)


def test_duplicates_must_be_removed_before_splitting(tiny_spec):
    manifest = build_manifest(tiny_spec)
    manifest.loc[1, "pixel_sha256"] = manifest.loc[0, "pixel_sha256"]
    with pytest.raises(LeakageError):
        make_splits(manifest, SplitConfig())


# ---------------------------------------------------------------- real configs
def test_repository_dataset_configs_are_consistent(monkeypatch):
    monkeypatch.delenv("SPERMTRIAGE_ROOT", raising=False)
    for name, n in (("hushem", 216), ("smids", 3000)):
        spec = load_dataset_spec(name)
        assert sum(spec.expected_counts.values()) == n
        assert set(spec.expected_counts) == set(spec.classes)
    smids = load_dataset_spec("smids")
    assert smids.classes == ["Normal", "Abnormal", "Non-sperm"]
    assert smids.legacy_folder_map["Boya"] == "Non-sperm"


def test_experiment_config_dataset_overrides(monkeypatch):
    monkeypatch.delenv("SPERMTRIAGE_ROOT", raising=False)
    exp = ExperimentConfig.from_yaml(project_root() / "configs" / "experiments" / "main.yaml")
    ft = next(m for m in exp.models if m.id == "mobilenetv3-ft")
    assert exp.train_config(ft, "hushem").max_epochs == 60
    assert exp.train_config(ft, "smids").max_epochs == ft.train.max_epochs
