"""tests/test_normalize.py — Mathematical invariants and unit tests for normalization pipelines."""
import pytest
import numpy as np

from core.normalize import (
    normalize_ftir,
    normalize_eem,
    normalize_fticr,
    _asls_baseline,
)
from core.synthetic import (
    generate_synthetic_ftir,
    generate_synthetic_eem,
    generate_synthetic_fticr,
)


def test_ftir_normalization_invariants():
    wn, abs_y = generate_synthetic_ftir("humic_acid", seed=42)
    vec = normalize_ftir(wn, abs_y)

    # 1. Output dimension invariant: shape must be (1801,)
    assert vec.shape == (1801,)
    assert vec.dtype == np.float32

    # 2. L2 Unit Norm Invariant: ||v||_2 = 1.0 ± 1e-5
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-5)

    # 3. Non-negativity invariant: all elements >= 0
    assert np.all(vec >= 0.0)


def test_asls_baseline_offset_removal():
    wn = np.arange(400, 4002, 2)
    # Peak on top of large constant offset
    y = np.exp(-0.5 * ((wn - 1600) / 50) ** 2) + 2.5
    baseline = _asls_baseline(y, lam=1e6, p=0.01)
    corrected = y - baseline

    # The baseline should absorb the ~2.5 offset
    assert np.mean(baseline) > 2.0
    assert np.min(corrected) < 0.2


def test_eem_normalization_invariants():
    ex, em, mat = generate_synthetic_eem("humic_acid", seed=42)
    vec = normalize_eem(ex, em, mat)

    # 1. Output dimension invariant: 43 x 161 = 6923
    assert vec.shape == (6923,)
    assert vec.dtype == np.float32

    # 2. Max Norm Invariant: max(v) = 1.0
    assert np.isclose(np.max(vec), 1.0, atol=1e-5)

    # 3. Non-negativity
    assert np.all(vec >= 0.0)


def test_fticr_normalization_invariants():
    mz, intensity = generate_synthetic_fticr("lake_dom", seed=42)
    vec = normalize_fticr(mz, intensity)

    # 1. Output dimension invariant: 1300 bins
    assert vec.shape == (1300,)
    assert vec.dtype == np.float32

    # 2. L2 Unit Norm
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_empty_spectrum_raises_exception():
    wn = np.arange(400, 4002, 2)
    zeros = np.zeros_like(wn)
    with pytest.raises(ValueError):
        normalize_ftir(wn, zeros)
