"""core/indices.py — Chemometric index calculation across FTIR, EEM, and FT-ICR MS modalities."""
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.integrate import trapezoid

from core.models import ModalityEnum


def _find_peak_in_window(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    window: float = 4.0,
) -> float:
    """Find local maximum within [target_x - window, target_x + window]."""
    mask = (x >= target_x - window) & (x <= target_x + window)
    if not np.any(mask):
        # Fallback to nearest point if window has no exact data
        idx = int(np.argmin(np.abs(x - target_x)))
        return float(y[idx])
    return float(np.max(y[mask]))


def compute_ftir_indices(wavenumbers: np.ndarray, absorbance: np.ndarray) -> Dict[str, float]:
    """Compute characteristic FTIR structural and functional group indices."""
    wn = np.asarray(wavenumbers, dtype=np.float64)
    abs_y = np.asarray(absorbance, dtype=np.float64)

    a1715 = _find_peak_in_window(wn, abs_y, 1715.0, 8.0)  # C=O
    a1600 = _find_peak_in_window(wn, abs_y, 1600.0, 10.0) # Aromatic ring C=C
    a1510 = _find_peak_in_window(wn, abs_y, 1510.0, 6.0)  # Lignin ring
    a1450 = _find_peak_in_window(wn, abs_y, 1450.0, 8.0)  # Aliphatic C-H bend
    a1270 = _find_peak_in_window(wn, abs_y, 1270.0, 10.0) # Phenolic C-O
    a1030 = _find_peak_in_window(wn, abs_y, 1030.0, 10.0) # Polysaccharide C-O

    eps = 1e-6
    li = a1510 / (a1600 + eps)
    ai = a1510 / (a1030 + eps)
    ci = a1600 / (a1715 + eps)
    aliph_arom = a1450 / (a1510 + eps)

    return {
        "LI": round(float(li), 4),
        "AI": round(float(ai), 4),
        "CI": round(float(ci), 4),
        "Aliph_Arom": round(float(aliph_arom), 4),
        "A1510_A1030": round(float(ai), 4),
        "A1715": round(float(a1715), 4),
        "A1600": round(float(a1600), 4),
        "A1510": round(float(a1510), 4),
    }


def _get_eem_slice(
    ex: np.ndarray,
    em: np.ndarray,
    matrix_2d: np.ndarray,
    target_ex: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 1D emission slice at nearest excitation wavelength."""
    idx_ex = int(np.argmin(np.abs(ex - target_ex)))
    return em, matrix_2d[idx_ex, :]


def compute_eem_indices(
    ex: np.ndarray,
    em: np.ndarray,
    matrix_2d: np.ndarray,
) -> Dict[str, float]:
    """Compute bio-geochemical fluorescence indices (FI, HIX, BIX) and Coble peaks."""
    ex_arr = np.asarray(ex, dtype=np.float64)
    em_arr = np.asarray(em, dtype=np.float64)
    mat = np.asarray(matrix_2d, dtype=np.float64)

    if mat.shape == (len(em_arr), len(ex_arr)):
        mat = mat.T

    # 1. Fluorescence Index (FI): I(Em=450) / I(Em=500) at Ex=370
    em_370, slice_370 = _get_eem_slice(ex_arr, em_arr, mat, 370.0)
    idx_450 = int(np.argmin(np.abs(em_370 - 450.0)))
    idx_500 = int(np.argmin(np.abs(em_370 - 500.0)))
    val_500 = max(float(slice_370[idx_500]), 1e-6)
    fi = float(slice_370[idx_450]) / val_500

    # 2. Humification Index (HIX): int(435-480) / int(300-345) at Ex=254
    em_254, slice_254 = _get_eem_slice(ex_arr, em_arr, mat, 254.0)
    mask_high = (em_254 >= 435.0) & (em_254 <= 480.0)
    mask_low = (em_254 >= 300.0) & (em_254 <= 345.0)

    if np.sum(mask_high) >= 2 and np.sum(mask_low) >= 2:
        int_high = float(trapezoid(slice_254[mask_high], em_254[mask_high]))
        int_low = float(trapezoid(slice_254[mask_low], em_254[mask_low]))
        hix = int_high / max(int_low, 1e-6)
    else:
        hix = 1.0

    # 3. Biological Index (BIX): I(Em=380) / I(Em=430) at Ex=310
    em_310, slice_310 = _get_eem_slice(ex_arr, em_arr, mat, 310.0)
    idx_380 = int(np.argmin(np.abs(em_310 - 380.0)))
    idx_430 = int(np.argmin(np.abs(em_310 - 430.0)))
    val_430 = max(float(slice_310[idx_430]), 1e-6)
    bix = float(slice_310[idx_380]) / val_430

    # 4. Coble Fluorophores
    def _coble_peak(ex_min: float, ex_max: float, em_min: float, em_max: float) -> float:
        mask_ex = (ex_arr >= ex_min) & (ex_arr <= ex_max)
        mask_em = (em_arr >= em_min) & (em_arr <= em_max)
        if not np.any(mask_ex) or not np.any(mask_em):
            return 0.0
        submat = mat[np.ix_(mask_ex, mask_em)]
        return float(np.max(submat)) if submat.size > 0 else 0.0

    peak_a = _coble_peak(240, 260, 400, 460)  # UV humic-like
    peak_c = _coble_peak(320, 360, 420, 480)  # Visible humic-like
    peak_m = _coble_peak(290, 320, 380, 420)  # Marine humic-like
    peak_t = _coble_peak(270, 280, 320, 350)  # Tryptophan-like
    peak_b = _coble_peak(270, 280, 300, 310)  # Tyrosine-like

    return {
        "FI": round(float(fi), 3),
        "HIX": round(float(hix), 3),
        "BIX": round(float(bix), 3),
        "Peak_A": round(float(peak_a), 3),
        "Peak_C": round(float(peak_c), 3),
        "Peak_M": round(float(peak_m), 3),
        "Peak_T": round(float(peak_t), 3),
        "Peak_B": round(float(peak_b), 3),
    }


def compute_fticr_indices(mz: np.ndarray, intensity: np.ndarray) -> Dict[str, float]:
    """Compute molecular mass distribution and heterogeneity descriptors."""
    mz_arr = np.asarray(mz, dtype=np.float64)
    int_arr = np.asarray(intensity, dtype=np.float64)

    valid = (mz_arr > 0) & (int_arr > 0) & np.isfinite(mz_arr) & np.isfinite(int_arr)
    m = mz_arr[valid]
    i = int_arr[valid]

    if len(m) == 0:
        return {
            "Richness": 0,
            "Mw": 0.0,
            "Mn": 0.0,
            "PD": 1.0,
            "LMW_pct": 0.0,
            "MMW_pct": 0.0,
            "HMW_pct": 0.0,
        }

    total_i = np.sum(i)
    eps = 1e-12

    # Weight-average molecular weight (Mw)
    mw = np.sum(m * i) / max(total_i, eps)

    # Number-average molecular weight (Mn)
    mn = total_i / max(np.sum(i / m), eps)

    # Polydispersity (PD = Mw / Mn)
    pd = mw / max(mn, eps)

    # Molecular pool percentages
    lmw_pct = (np.sum(i[m < 300.0]) / max(total_i, eps)) * 100.0
    mmw_pct = (np.sum(i[(m >= 300.0) & (m <= 600.0)]) / max(total_i, eps)) * 100.0
    hmw_pct = (np.sum(i[m > 600.0]) / max(total_i, eps)) * 100.0

    return {
        "Richness": int(len(m)),
        "Mw": round(float(mw), 2),
        "Mn": round(float(mn), 2),
        "PD": round(float(pd), 3),
        "LMW_pct": round(float(lmw_pct), 1),
        "MMW_pct": round(float(mmw_pct), 1),
        "HMW_pct": round(float(hmw_pct), 1),
    }


def compute_indices(
    x: np.ndarray,
    y: np.ndarray,
    modality: Union[str, ModalityEnum],
    matrix_2d: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Unified entrypoint for chemometric indices computation."""
    mod = ModalityEnum(modality)
    if mod == ModalityEnum.FTIR:
        return compute_ftir_indices(x, y)
    elif mod == ModalityEnum.EEM:
        if matrix_2d is None:
            raise ValueError("matrix_2d required for EEM index calculation")
        return compute_eem_indices(x, y, matrix_2d)
    elif mod == ModalityEnum.FTICR_MS:
        return compute_fticr_indices(x, y)
    return {}
