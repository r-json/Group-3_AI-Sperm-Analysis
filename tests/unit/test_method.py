"""AniFA: invariance and correctness tests (synthetic images; offline)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spermtriage.method.analysis import confident_learning, mcnemar_exact
from spermtriage.method.canonical import (
    N_VIEWS,
    frame_weights,
    head_mask,
    moment_frame,
    rotate_image,
    sample_views,
)
from spermtriage.method.classify import Inputs, deep_lr, fit_predict, shape_svm
from spermtriage.method.shape import FEATURE_NAMES, shape_features
from spermtriage.models.heads import Classifier


def pear(size: int = 131, angle: float = 0.3, seed: int = 0) -> np.ndarray:
    """A dark pear-shaped 'head' with a thin 'tail' on a light background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2
    x, y = xx - c, yy - c
    u = math.cos(angle) * x + math.sin(angle) * y
    v = -math.sin(angle) * x + math.cos(angle) * y
    half_width = 9 + 5 * np.clip((u + 22) / 44, 0, 1)  # wider at one end: pear shape
    head = (np.abs(u) <= 22) & (np.abs(v) <= half_width * np.sqrt(np.clip(1 - (u / 23) ** 2, 0, 1)))
    tail = (u < -22) & (u > -60) & (np.abs(v) < 1.2)
    img = np.full((size, size, 3), (235, 200, 215), float)
    img[head] = (90, 30, 110)
    img[tail] = (160, 120, 160)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_mask_finds_head_not_tail():
    img = pear()
    mask = head_mask(img)
    fr = moment_frame(mask)
    assert fr.mask_found and 800 < mask.sum() < 2500
    assert fr.anisotropy > 0.3


@pytest.mark.parametrize("angle,flip", [(0.7, False), (2.1, True), (4.4, False)])
def test_frame_is_equivariant_and_anisotropy_invariant(angle, flip):
    img = pear()
    f0 = moment_frame(head_mask(img))
    f1 = moment_frame(head_mask(rotate_image(img, angle, flip)))
    assert f1.anisotropy == pytest.approx(f0.anisotropy, abs=0.03)


@pytest.mark.parametrize("angle,flip", [(0.7, False), (2.1, True), (5.0, True)])
def test_shape_descriptors_are_invariant(angle, flip):
    img = pear()
    a = shape_features(img, head_mask(img))
    t = rotate_image(img, angle, flip)
    b = shape_features(t, head_mask(t))
    assert len(a) == len(FEATURE_NAMES)
    # Discretisation of a rotated raster changes pixel counts slightly; compare relative error
    # on the well-conditioned descriptors.
    rel = np.abs(a - b) / (np.abs(a) + 1e-3)
    assert np.median(rel) < 0.05


def test_frame_weights_limits_and_normalisation():
    w_inf = frame_weights(np.array([0.5, 0.0]), math.inf)
    assert np.allclose(w_inf.sum(1), 1) and np.count_nonzero(w_inf[0]) == 4
    w0 = frame_weights(np.array([0.9]), 0.0)
    assert np.allclose(w0, 1 / N_VIEWS)
    w = frame_weights(np.array([0.1, 0.8]), 4.0)
    # More anisotropic heads put more weight on major-axis views.
    assert w[1, 0] > w[0, 0]


@pytest.mark.parametrize("angle,flip", [(0.9, False), (3.3, True)])
def test_frame_averaged_features_are_rotation_invariant(angle, flip):
    """The invariance test the method claims: features averaged over the weighted frame of a
    rotated (and mirrored) input match those of the original."""
    torch.manual_seed(0)
    net = Classifier.build("resnet50", 4, 96, dropout=0.0, pretrained=False).eval()
    img = pear()

    def averaged(x: np.ndarray) -> np.ndarray:
        fr = moment_frame(head_mask(x))
        views = sample_views(x, fr, window=131, out_size=96)
        with torch.no_grad():
            f = net.features(views).numpy()
        return frame_weights(np.array(fr.anisotropy), 1.0) @ f

    a, b = averaged(img), averaged(rotate_image(img, angle, flip))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.98

    # Without the frame (raw input), the same network is far from invariant.
    def raw(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net.features(torch.from_numpy(x).permute(2, 0, 1)[None].contiguous()).numpy()[0]

    ra, rb = raw(img), raw(rotate_image(img, angle, flip))
    assert cos > float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def test_mcnemar_exact_matches_binomial():
    a = np.array([True] * 3 + [False] * 0 + [True] * 10)
    b = np.array([False] * 3 + [True] * 10)
    bb, cc, p = mcnemar_exact(a, b)
    assert (bb, cc) == (3, 0) and p == pytest.approx(0.25)


def test_confident_learning_flags_flipped_labels():
    rng = np.random.default_rng(0)
    n, k = 600, 3
    true = rng.integers(0, k, n)
    proba = np.full((n, k), 0.05)
    proba[np.arange(n), true] = 0.9
    labels = true.copy()
    flipped = rng.choice(n, 30, replace=False)
    labels[flipped] = (labels[flipped] + 1) % k
    res = confident_learning(proba, labels)
    assert res["n_suspected_label_issues"] == 30
    assert set(res["suspect_rows"]) == set(flipped.tolist())


def test_stacked_fit_predict_on_separable_synthetic_data():
    rng = np.random.default_rng(0)
    n, d = 120, 16
    y = np.repeat(np.arange(3), n // 3)
    canon = rng.normal(0, 1, (n, 8, d)) + y[:, None, None] * 1.5
    shape = rng.normal(0, 1, (n, len(FEATURE_NAMES))) + y[:, None]
    inp = Inputs(y=y, anisotropy=rng.uniform(0, 1, n), canon=canon, shape=shape)
    idx = rng.permutation(n)
    res = fit_predict([deep_lr("canon", Cs=[0.1]), shape_svm()], inp, idx[:90], idx[90:], k=3)
    assert res.proba.shape == (30, 3) and np.allclose(res.proba.sum(1), 1)
    assert (res.proba.argmax(1) == y[idx[90:]]).mean() > 0.9
    assert "stack" in res.inner_acc and "beta" in res.selected["deep_canon"]


def test_cbam_resnet50_backbone_shapes():
    from spermtriage.method.gpu_baselines import CBAM, cbam_resnet50

    x = torch.randn(2, 64, 8, 8)
    assert CBAM(64)(x).shape == x.shape
    net = Classifier(cbam_resnet50(pretrained=False), 3, 64, backbone_key="cbam_resnet50")
    assert net(torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)).shape == (2, 3)


def test_view_feature_extraction_resumes_from_checkpoint(tiny_project, tiny_spec):
    from spermtriage.data.integrity import build_manifest, manifest_path
    from spermtriage.method.features import cache_dir, frames_and_shape, view_features

    manifest = build_manifest(tiny_spec)
    manifest_path("tiny").parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path("tiny"), index=False)
    frames, shape = frames_and_shape("tiny")
    assert shape.shape == (36, len(FEATURE_NAMES))
    full = view_features("tiny", "tiny_test", True, frames, checkpoint_every=10)
    assert full.shape[:2] == (36, 8)
    # Simulate a crash after 20 images: only the checkpoint survives.
    final = cache_dir() / "tiny__tiny_test__canon.npy"
    partial = cache_dir() / "tiny__tiny_test__canon.partial.npy"
    np.save(partial, full[:20])
    final.unlink()
    resumed = view_features("tiny", "tiny_test", True, frames, checkpoint_every=10)
    assert np.allclose(resumed, full, atol=1e-5) and not partial.exists()


def test_capped_pca_on_tiny_training_sets():
    from spermtriage.method.classify import CappedPCA

    X = np.random.default_rng(0).normal(size=(20, 50))
    assert CappedPCA(n_components=64).fit_transform(X).shape == (20, 19)
    assert CappedPCA(n_components=8).fit(X).n_components_ == 8
