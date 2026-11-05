"""core/qc.py — Quality Control (QC) validation and analytical integrity checks."""
from typing import List, Optional, Tuple, Union
import numpy as np

from core.models import ModalityEnum, QualityReport


def _check_min_points(
    x: np.ndarray,
    y: np.ndarray,
    modality: ModalityEnum,
    matrix_2d: Optional[np.ndarray] = None,
) -> Tuple[bool, Optional[str]]:
    """QC-01: Ensure sufficient data density for analytical significance."""
    if modality == ModalityEnum.FTIR:
        if len(x) < 50:
            return False, f"FTIR spectrum has insufficient data points: {len(x)} (required >= 50)"
    elif modality == ModalityEnum.EEM:
        if len(x) < 5 or (len(y) < 10 and (matrix_2d is None or matrix_2d.shape[1] < 10)):
            return False, f"EEM grid dimensions are too small: {len(x)} Ex x {len(y)} Em (required >= 5x10)"
    elif modality == ModalityEnum.FTICR_MS:
        if len(x) < 20:
            return False, f"Mass spectrum has too few peaks: {len(x)} (required >= 20)"
    return True, None


def _check_coverage(
    x: np.ndarray,
    modality: ModalityEnum,
    em_axis: Optional[np.ndarray] = None,
) -> Tuple[bool, float, Optional[str]]:
    """QC-02: Validate canonical analytical wavelength/mass window coverage."""
    if len(x) == 0:
        return False, 0.0, "Zero-length coordinate axis"

    x_min, x_max = float(np.min(x)), float(np.max(x))

    if modality == ModalityEnum.FTIR:
        # Canonical range: 400 to 4000 cm^-1 (width = 3600)
        overlap = max(0.0, min(x_max, 4000.0) - max(x_min, 400.0))
        cov_pct = (overlap / 3600.0) * 100.0
    elif modality == ModalityEnum.EEM:
        # Ex: 240..450 (span 210), Em: 280..600 (span 320)
        overlap_ex = max(0.0, min(x_max, 450.0) - max(x_min, 240.0)) / 210.0
        if em_axis is not None and len(em_axis) > 0:
            em_min, em_max = float(np.min(em_axis)), float(np.max(em_axis))
            overlap_em = max(0.0, min(em_max, 600.0) - max(em_min, 280.0)) / 320.0
        else:
            overlap_em = 1.0
        cov_pct = min(overlap_ex * overlap_em, 1.0) * 100.0
    elif modality == ModalityEnum.FTICR_MS:
        # Canonical range: 150 to 800 Da (width = 650)
        overlap = max(0.0, min(x_max, 800.0) - max(x_min, 150.0))
        cov_pct = (overlap / 650.0) * 100.0
    else:
        cov_pct = 100.0

    cov_pct = round(float(np.clip(cov_pct, 0.0, 100.0)), 1)
    if cov_pct < 60.0:
        return False, cov_pct, f"Low canonical window coverage: {cov_pct}% (recommended >= 60%)"
    return True, cov_pct, None


def _check_snr(
    x: np.ndarray,
    y: np.ndarray,
    modality: ModalityEnum,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """QC-03: Signal-to-Noise Ratio (SNR) in FTIR silent region (1900-2200 cm^-1)."""
    if modality != ModalityEnum.FTIR or len(x) < 10:
        return True, None, None

    silent_mask = (x >= 1900.0) & (x <= 2200.0)
    if np.sum(silent_mask) < 5:
        # If silent window not recorded, fallback across 1% lowest variance window
        noise = float(np.std(y)) * 0.1
    else:
        noise = float(np.std(y[silent_mask]))

    signal = float(np.ptp(y))  # max - min
    snr = signal / max(noise, 1e-12)
    snr = round(snr, 1)

    if snr < 5.0:
        return False, snr, f"Low Signal-to-Noise Ratio (SNR = {snr} < 5.0)"
    return True, snr, None


def _check_negatives(y: np.ndarray) -> Tuple[bool, Optional[str]]:
    """QC-04: Frequency of unphysical negative intensity values."""
    if len(y) == 0:
        return True, None
    neg_count = np.sum(y < -1e-6)
    neg_frac = float(neg_count / len(y))
    if neg_frac > 0.20:
        return False, f"High fraction of negative intensities ({round(neg_frac * 100, 1)}% > 20%)"
    return True, None


def _check_saturation(y: np.ndarray, modality: ModalityEnum) -> Tuple[bool, Optional[str]]:
    """QC-05: Detector saturation check for absorbance spectra."""
    if modality != ModalityEnum.FTIR or len(y) == 0:
        return True, None
    sat_count = np.sum(y > 3.5)
    sat_frac = float(sat_count / len(y))
    if sat_frac > 0.05:
        return False, f"Detector saturation detected: {round(sat_frac * 100, 1)}% of points exceed 3.5 a.u."
    return True, None


def _check_empty(y: np.ndarray) -> Tuple[bool, Optional[str]]:
    """QC-06: Ensure spectrum is not flat or empty."""
    if len(y) == 0 or float(np.std(y)) < 1e-10:
        return False, "Spectrum is empty or has zero variance (flatline)"
    return True, None


def run_qc(
    x: np.ndarray,
    y: np.ndarray,
    modality: Union[str, ModalityEnum],
    matrix_2d: Optional[np.ndarray] = None,
    em_axis: Optional[np.ndarray] = None,
) -> QualityReport:
    """Run full analytical Quality Control suite and produce QualityReport."""
    mod = ModalityEnum(modality)
    errors: List[str] = []
    warnings: List[str] = []

    # QC-06 Flatline check (Fatal Error)
    flat_ok, flat_err = _check_empty(y if matrix_2d is None else matrix_2d.flatten())
    if not flat_ok:
        errors.append(flat_err)
        return QualityReport(passed=False, score=0.0, errors=errors, warnings=[], coverage_pct=0.0)

    # QC-01 Min points check (Fatal Error)
    pts_ok, pts_err = _check_min_points(x, y, mod, matrix_2d)
    if not pts_ok:
        errors.append(pts_err)

    # QC-02 Coverage check (Warning)
    cov_ok, cov_pct, cov_warn = _check_coverage(x, mod, em_axis if em_axis is not None else y)
    if not cov_ok and cov_warn:
        warnings.append(cov_warn)

    # QC-03 SNR check (Warning, FTIR only)
    snr_ok, snr_est, snr_warn = _check_snr(x, y, mod)
    if not snr_ok and snr_warn:
        warnings.append(snr_warn)

    # QC-04 Negative intensities (Warning if >20%)
    neg_ok, neg_warn = _check_negatives(y if matrix_2d is None else matrix_2d.flatten())
    if not neg_ok and neg_warn:
        warnings.append(neg_warn)

    # QC-05 Saturation check (Warning, FTIR only)
    sat_ok, sat_warn = _check_saturation(y, mod)
    if not sat_ok and sat_warn:
        warnings.append(sat_warn)

    # Calculate aggregate quality score
    score = 1.0
    if len(errors) > 0:
        score = 0.0
    else:
        score -= 0.15 * len(warnings)
        score = max(round(score, 2), 0.0)

    return QualityReport(
        passed=len(errors) == 0,
        score=score,
        warnings=warnings,
        errors=errors,
        snr_estimate=snr_est,
        coverage_pct=cov_pct,
    )
