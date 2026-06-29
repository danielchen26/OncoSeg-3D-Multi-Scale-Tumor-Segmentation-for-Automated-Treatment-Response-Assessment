"""Regression tests for the MC-Dropout uncertainty path (findings F04, F16).

F04: the path must not crash on the inline ``train_all`` architecture, which
exposes ``self.decoders`` (plural) rather than ``self.decoder``.
F16: the uncertainty must be per-channel binary entropy (multi-label sigmoid),
not categorical entropy.

These tests use lightweight torch stubs so they run without monai/nibabel;
they are skipped entirely if torch is unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# src.inference pulls in monai + nibabel at import time; skip if absent.
pytest.importorskip("monai")
pytest.importorskip("nibabel")
nn = torch.nn


class _InlineLike(nn.Module):
    """Mimics train_all.OncoSeg: forward() applies mc_dropout, exposes
    ``decoders`` (plural) and has NO ``decoder`` attribute."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
        self.mc_dropout = nn.Dropout3d(p=0.5)
        self.decoders = nn.ModuleList([nn.Identity()])  # plural, like the trained model

    def forward(self, x):
        # Apply dropout so MC sampling produces variance, then map to `channels`.
        h = self.mc_dropout(x)
        pred = h[:, : self.channels] if x.shape[1] >= self.channels else h.repeat(
            1, self.channels, 1, 1, 1
        )
        return {"pred": pred}


def _make_predictor(model):
    # Import here so the module-level importorskip on torch guards everything.
    from src.inference import Predictor

    return Predictor(model=model, device=torch.device("cpu"), mc_samples=4)


def test_mc_dropout_does_not_crash_on_inline_model():
    """F04: inline architecture (decoders, no decoder) must not raise."""
    model = _InlineLike(channels=3)
    predictor = _make_predictor(model)
    image = torch.rand(1, 3, 8, 8, 8)
    unc = predictor._estimate_uncertainty(image)  # would AttributeError before the fix
    assert isinstance(unc, np.ndarray)
    assert unc.shape == (8, 8, 8)
    assert np.all(np.isfinite(unc))


def test_uncertainty_is_binary_entropy_bounded():
    """F16: per-channel binary entropy is bounded by ln(2) ~= 0.693."""
    model = _InlineLike(channels=3)
    predictor = _make_predictor(model)
    image = torch.rand(1, 3, 8, 8, 8)
    unc = predictor._estimate_uncertainty(image)
    # mean over channels of values each <= ln2 must itself be <= ln2 (+eps).
    assert unc.max() <= np.log(2) + 1e-4
    assert unc.min() >= 0.0
