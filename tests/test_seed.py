"""Tests for the reproducibility seed helper (finding F12)."""

import random

import numpy as np

from src.utils.seed import set_seed


def test_set_seed_returns_seed():
    assert set_seed(42) == 42
    assert set_seed(7) == 7


def test_python_random_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_reproducible():
    set_seed(2024)
    a = np.random.rand(8)
    set_seed(2024)
    b = np.random.rand(8)
    assert np.allclose(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = np.random.rand(8)
    set_seed(2)
    b = np.random.rand(8)
    assert not np.allclose(a, b)


def test_torch_reproducible_if_available():
    import pytest

    torch = pytest.importorskip("torch")
    set_seed(99)
    a = torch.rand(8)
    set_seed(99)
    b = torch.rand(8)
    assert torch.equal(a, b)
