"""tests/test_indices.py — Validation of chemometric descriptors calculation."""
import numpy as np

from core.indices import compute_ftir_indices, compute_eem_indices, compute_fticr_indices
from core.synthetic import generate_synthetic_ftir, generate_synthetic_eem, generate_synthetic_fticr


def test_ftir_indices():
    wn, abs_y = generate_synthetic_ftir("sludge_lignin", seed=42)
    indices = compute_ftir_indices(wn, abs_y)

    assert "LI" in indices
    assert "AI" in indices
    assert "CI" in indices
    # Lignin sample should exhibit high Lignin Index (LI)
    assert indices["LI"] > 0.5


def test_eem_indices():
    ex, em, mat = generate_synthetic_eem("humic_acid", seed=42)
    indices = compute_eem_indices(ex, em, mat)

    assert "FI" in indices
    assert "HIX" in indices
    assert "BIX" in indices
    assert "Peak_A" in indices
    assert "Peak_C" in indices
    assert indices["FI"] > 0.0
    assert indices["HIX"] > 0.0


def test_fticr_indices():
    mz, intensity = generate_synthetic_fticr("lake_dom", seed=42)
    indices = compute_fticr_indices(mz, intensity)

    assert "Richness" in indices
    assert "Mw" in indices
    assert "Mn" in indices
    assert "PD" in indices
    assert indices["Richness"] > 100
    assert 200.0 < indices["Mw"] < 700.0
    assert indices["PD"] >= 1.0  # Mw >= Mn by definition
