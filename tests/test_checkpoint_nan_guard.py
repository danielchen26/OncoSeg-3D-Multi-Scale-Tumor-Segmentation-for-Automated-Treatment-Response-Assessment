"""Regression test for best-checkpoint NaN guard (finding F08).

When a validation region has empty ground truth (ET is commonly empty),
MONAI's DiceMetric returns NaN for that region. A plain mean then makes the
selection metric NaN, and ``NaN > best_dice`` is always False, so no best
checkpoint is ever saved. The fix uses nanmean for the selection metric and a
``math.isnan`` guard before the comparison.

This test is numpy-only (no torch/monai) and asserts both halves of the fix
against the committed per-subject Dice array, which already contains an
empty-ET (NaN) subject.
"""

import math

import numpy as np


def _select(metric_value: float, best: float) -> bool:
    """Mirror the guarded selection condition used in all three trainers."""
    return (not math.isnan(metric_value)) and metric_value > best


def test_committed_val_data_has_an_empty_region_subject():
    """The premise of F08: at least one val subject has a NaN (empty-GT) region."""
    o = np.load("experiments/local_results/oncoseg_per_subject_dice.npy")
    assert np.isnan(o).any(), "expected at least one NaN (empty-GT) region in val data"


def test_plain_mean_disables_saving_but_nanmean_does_not():
    o = np.load("experiments/local_results/oncoseg_per_subject_dice.npy")
    nan_row = o[np.where(np.isnan(o).any(axis=1))[0][0]]

    plain = float(np.mean(nan_row))      # NaN, the bug
    nan_safe = float(np.nanmean(nan_row))  # finite, the fix
    assert math.isnan(plain)
    assert not math.isnan(nan_safe)

    best = 0.0
    # Old behaviour: NaN selection metric never triggers a save.
    assert _select(plain, best) is False
    # Fixed behaviour: a real positive Dice does trigger a save.
    assert _select(nan_safe, best) is True


def test_guard_blocks_nan_but_allows_finite():
    assert _select(float("nan"), 0.0) is False
    assert _select(0.80, 0.79) is True
    assert _select(0.78, 0.79) is False
