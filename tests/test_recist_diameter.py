"""Regression tests for RECIST longest-diameter measurement (finding F06).

The longest diameter must be the longest *in-plane* extent over ALL axial
slices, not just the largest-area slice. These tests use numpy-only phantoms
(no torch/monai) so they run in any environment.
"""

import numpy as np
import pytest

from src.response.recist import RECISTMeasurer


@pytest.fixture
def measurer():
    return RECISTMeasurer()


def test_longest_diameter_on_non_max_area_slice(measurer):
    """The longest diameter lies on a small-area slice; the old max-area-only
    code would under-measure it.

    Slice 0: 30x30 solid square -> area 900, diagonal ~= 29*sqrt(2) ~= 41.0 mm.
    Slice 1: a 1-voxel-thick line 50 voxels long -> area 50, length 49 mm.
    The correct longest diameter is the 49 mm line on slice 1.
    """
    mask = np.zeros((64, 64, 8), dtype=np.uint8)
    mask[10:40, 10:40, 0] = 1          # big area, modest diagonal
    mask[30, 5:55, 1] = 1              # small area, long extent (49 mm)

    diameter = measurer.longest_axial_diameter(mask, pixdim=(1.0, 1.0, 1.0))

    # Endpoints of the line span columns 5..54 -> 49 mm. The old (max-area)
    # implementation would have returned the slice-0 diagonal ~= 41 mm.
    assert diameter == pytest.approx(49.0, abs=0.5)
    assert diameter > 41.5  # strictly larger than the max-area slice diagonal


def test_single_long_line(measurer):
    """A single line measures end-to-end length."""
    mask = np.zeros((64, 64, 8), dtype=np.uint8)
    mask[32, 12:53, 0] = 1  # 41 voxels -> 40 mm between endpoints
    assert measurer.longest_axial_diameter(mask, pixdim=(1.0, 1.0, 1.0)) == pytest.approx(40.0, abs=0.5)


def test_anisotropic_spacing_applied(measurer):
    """In-plane pixel spacing scales the diameter."""
    mask = np.zeros((64, 64, 8), dtype=np.uint8)
    mask[10:31, 15, 0] = 1  # 21 voxels along H -> 20 voxel gap
    # H spacing 2.0 mm -> 40 mm
    assert measurer.longest_axial_diameter(mask, pixdim=(2.0, 1.0, 1.0)) == pytest.approx(40.0, abs=0.5)


def test_empty_mask_returns_zero(measurer):
    assert measurer.longest_axial_diameter(np.zeros((16, 16, 4), dtype=np.uint8)) == 0.0


def test_single_voxel_returns_zero(measurer):
    mask = np.zeros((16, 16, 4), dtype=np.uint8)
    mask[8, 8, 2] = 1
    assert measurer.longest_axial_diameter(mask) == 0.0
