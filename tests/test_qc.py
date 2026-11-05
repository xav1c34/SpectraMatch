"""tests/test_qc.py — Quality Control unit tests and edge cases."""
import numpy as np

from core.models import ModalityEnum
from core.qc import run_qc
from core.synthetic import generate_synthetic_ftir


def test_qc_normal_spectrum_passes():
    wn, abs_y = generate_synthetic_ftir("fulvic_acid", seed=42)
    report = run_qc(wn, abs_y, ModalityEnum.FTIR)

    assert report.passed is True
    assert report.score >= 0.85
    assert len(report.errors) == 0
    assert report.coverage_pct >= 90.0


def test_qc_empty_spectrum_fails():
    wn = np.arange(400, 4002, 2)
    flat_y = np.ones_like(wn) * 0.5  # flatline (zero variance)
    report = run_qc(wn, flat_y, ModalityEnum.FTIR)

    assert report.passed is False
    assert report.score == 0.0
    assert any("empty or has zero variance" in err for err in report.errors)


def test_qc_too_few_points_fails():
    wn = np.array([1000.0, 1500.0, 2000.0])
    y = np.array([0.1, 0.5, 0.2])
    report = run_qc(wn, y, ModalityEnum.FTIR)

    assert report.passed is False
    assert any("insufficient data points" in err for err in report.errors)


def test_qc_low_coverage_warning():
    # Only 1000 to 1500 cm^-1 (width 500 out of 3600 -> ~14% coverage)
    wn = np.linspace(1000, 1500, 100)
    y = np.sin(np.linspace(0, 3, 100)) + 0.5
    report = run_qc(wn, y, ModalityEnum.FTIR)

    assert report.passed is True  # Warning, not fatal error
    assert any("Low canonical window coverage" in w for w in report.warnings)
    assert report.score < 1.0
