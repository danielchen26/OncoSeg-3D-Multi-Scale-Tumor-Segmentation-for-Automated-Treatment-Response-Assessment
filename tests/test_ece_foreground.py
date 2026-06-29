"""Test that ECE can be computed foreground-only and that pooling over
background hides miscalibration (findings F02, F24).

The expected_calibration_error helper lives in
scripts/uncertainty_qualitative_analysis.py, which imports matplotlib / monai /
nibabel at module load, so this test skips when those are unavailable. The ECE
math itself is pure numpy.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("monai")
pytest.importorskip("nibabel")

from scripts.uncertainty_qualitative_analysis import expected_calibration_error


def _toy_volume():
    """A volume that is well-calibrated on the (vast) background but badly
    over-confident on the small foreground, mimicking the real artifact."""
    n_bg, n_fg = 9800, 200
    # Background: label 0, confidently predicted near 0 -> well calibrated.
    bg_probs = np.full(n_bg, 0.01)
    bg_labels = np.zeros(n_bg)
    # Foreground: label 1, but predicted with high confidence 0.9 only 40% right
    # -> strongly over-confident.
    fg_probs = np.full(n_fg, 0.9)
    fg_labels = np.zeros(n_fg)
    fg_labels[: int(0.4 * n_fg)] = 1  # 40% accuracy at conf 0.9

    probs = np.concatenate([bg_probs, fg_probs])
    labels = np.concatenate([bg_labels, fg_labels])
    fg_mask = np.concatenate([np.zeros(n_bg, bool), np.ones(n_fg, bool)])
    return probs, labels, fg_mask


def test_pooled_ece_hides_foreground_miscalibration():
    probs, labels, fg_mask = _toy_volume()

    ece_all, _ = expected_calibration_error(probs, labels, n_bins=15)
    ece_fg, _ = expected_calibration_error(
        probs, labels, n_bins=15, foreground_mask=fg_mask
    )

    # Pooled ECE is small (background dominates); foreground ECE is large.
    assert ece_all < 0.05, f"pooled ECE unexpectedly large: {ece_all}"
    assert ece_fg > 0.4, f"foreground ECE should expose overconfidence: {ece_fg}"
    assert ece_fg > 10 * ece_all  # foreground is dramatically worse


def test_foreground_ece_matches_hand_value():
    """Foreground: confidence 0.9, accuracy 0.4 -> |0.9-0.4| = 0.5 in one bin."""
    probs, labels, fg_mask = _toy_volume()
    ece_fg, _ = expected_calibration_error(
        probs, labels, n_bins=15, foreground_mask=fg_mask
    )
    assert ece_fg == pytest.approx(0.5, abs=1e-6)


def test_empty_mask_returns_zero():
    probs = np.array([0.2, 0.8])
    labels = np.array([0.0, 1.0])
    ece, bins = expected_calibration_error(
        probs, labels, foreground_mask=np.zeros(2, bool)
    )
    assert ece == 0.0
    assert bins == []
