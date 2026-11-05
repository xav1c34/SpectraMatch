"""core/synthetic.py — Deterministic synthetic spectra generation for zero-dependency testing and seed library."""
from typing import Dict, List, Optional, Tuple
import numpy as np


# Standard FTIR Peak Parameters: (nu_0, FWHM)
FTIR_BANDS = [
    (3380.0, 450.0),  # O-H stretch
    (2925.0, 60.0),   # -CH2- stretch (asymm)
    (2850.0, 45.0),   # -CH3 stretch (symm)
    (1720.0, 50.0),   # C=O stretch (COOH)
    (1610.0, 70.0),   # Aromatic C=C / quinone C=O
    (1510.0, 35.0),   # Lignin aromatic ring
    (1450.0, 45.0),   # C-H bend (aliphatic)
    (1270.0, 55.0),   # Phenolic C-O / guaiacyl
    (1035.0, 80.0),   # Polysaccharide C-O-C
]

# Amplitudes per sample type
FTIR_TYPE_PROFILES = {
    "humic_acid": [0.70, 0.35, 0.20, 0.45, 0.65, 0.20, 0.25, 0.30, 0.40],
    "fulvic_acid": [0.85, 0.20, 0.10, 0.75, 0.45, 0.08, 0.18, 0.25, 0.50],
    "sludge_lignin": [0.60, 0.40, 0.25, 0.30, 0.55, 0.85, 0.35, 0.60, 0.25],
    "lignin": [0.55, 0.42, 0.28, 0.28, 0.58, 0.90, 0.38, 0.65, 0.20],
    "polysaccharide": [0.90, 0.25, 0.15, 0.05, 0.02, 0.00, 0.20, 0.10, 0.95],
    "lake_dom": [0.75, 0.28, 0.16, 0.50, 0.52, 0.12, 0.22, 0.28, 0.48],
    "nom": [0.78, 0.30, 0.18, 0.55, 0.54, 0.15, 0.24, 0.32, 0.46],
}


def generate_synthetic_ftir(
    sample_type: str = "humic_acid",
    wn_start: float = 400.0,
    wn_end: float = 4000.0,
    step: float = 2.0,
    noise_sigma: float = 0.008,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic 1D FTIR absorbance spectrum using Gaussian profile superposition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    wn = np.arange(wn_start, wn_end + step, step, dtype=np.float64)
    heights = FTIR_TYPE_PROFILES.get(sample_type, FTIR_TYPE_PROFILES["humic_acid"])

    spectrum = np.zeros_like(wn, dtype=np.float64)
    for (center, fwhm), h in zip(FTIR_BANDS, heights):
        # Introduce subtle variation per instance (±5%)
        inst_h = h * (1.0 + rng.uniform(-0.06, 0.06))
        inst_c = center + rng.uniform(-4.0, 4.0)
        inst_w = fwhm * (1.0 + rng.uniform(-0.04, 0.04))
        sigma = inst_w / 2.35482
        spectrum += inst_h * np.exp(-0.5 * ((wn - inst_c) / sigma) ** 2)

    # Slight sloping background offset
    slope = rng.uniform(-0.03, 0.03) * ((wn - 400.0) / 3600.0)
    spectrum += slope

    # Add Gaussian noise
    noise = rng.normal(0.0, noise_sigma, len(spectrum))
    spectrum += noise

    return wn, np.maximum(spectrum, 0.0)


# EEM Fluorophores: (Ex0, Em0, sig_ex, sig_em)
EEM_CENTERS = {
    "A": (250.0, 440.0, 20.0, 35.0),  # UV Humic-like
    "C": (330.0, 450.0, 25.0, 30.0),  # Visible Humic-like
    "M": (310.0, 400.0, 20.0, 25.0),  # Marine Humic-like
    "T": (275.0, 340.0, 15.0, 20.0),  # Tryptophan Protein-like
    "B": (275.0, 305.0, 12.0, 15.0),  # Tyrosine Protein-like
}

EEM_TYPE_WEIGHTS = {
    "humic_acid": {"A": 1.0, "C": 0.85, "M": 0.20, "T": 0.08, "B": 0.05},
    "fulvic_acid": {"A": 0.90, "C": 0.65, "M": 0.40, "T": 0.12, "B": 0.06},
    "protein": {"A": 0.15, "C": 0.10, "M": 0.15, "T": 0.95, "B": 0.70},
    "lake_dom": {"A": 0.70, "C": 0.55, "M": 0.50, "T": 0.35, "B": 0.20},
    "sludge_lignin": {"A": 0.85, "C": 0.90, "M": 0.30, "T": 0.40, "B": 0.15},
    "nom": {"A": 0.80, "C": 0.70, "M": 0.35, "T": 0.20, "B": 0.10},
}


def generate_synthetic_eem(
    sample_type: str = "humic_acid",
    ex_start: float = 240.0,
    ex_end: float = 450.0,
    ex_step: float = 5.0,
    em_start: float = 280.0,
    em_end: float = 600.0,
    em_step: float = 2.0,
    noise_sigma: float = 0.005,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic 2D Excitation-Emission Matrix (EEM).

    Returns:
        ex: 1D array of excitation wavelengths
        em: 1D array of emission wavelengths
        matrix_2d: 2D array of shape (len(ex), len(em))
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    ex = np.arange(ex_start, ex_end + ex_step, ex_step, dtype=np.float64)
    em = np.arange(em_start, em_end + em_step, em_step, dtype=np.float64)
    mesh_ex, mesh_em = np.meshgrid(ex, em, indexing="ij")

    weights = EEM_TYPE_WEIGHTS.get(sample_type, EEM_TYPE_WEIGHTS["humic_acid"])
    matrix = np.zeros_like(mesh_ex, dtype=np.float64)

    for fluor_name, (c_ex, c_em, s_ex, s_em) in EEM_CENTERS.items():
        w = weights.get(fluor_name, 0.0)
        if w <= 0.0:
            continue
        inst_w = w * (1.0 + rng.uniform(-0.05, 0.05))
        inst_cex = c_ex + rng.uniform(-2.0, 2.0)
        inst_cem = c_em + rng.uniform(-3.0, 3.0)
        gauss = inst_w * np.exp(
            -0.5 * (((mesh_ex - inst_cex) / s_ex) ** 2 + ((mesh_em - inst_cem) / s_em) ** 2)
        )
        matrix += gauss

    # Rayleigh scattering ridge (added to mimic raw instrument readout before cleaning)
    diff = np.abs(mesh_em - mesh_ex)
    rayleigh_1 = 0.4 * np.exp(-0.5 * (diff / 4.0) ** 2)
    diff_2 = np.abs(mesh_em - 2.0 * mesh_ex)
    rayleigh_2 = 0.2 * np.exp(-0.5 * (diff_2 / 5.0) ** 2)
    matrix += rayleigh_1 + rayleigh_2

    # Noise
    matrix += rng.normal(0.0, noise_sigma, matrix.shape)
    matrix = np.maximum(matrix, 0.0)

    return ex, em, matrix


def generate_synthetic_fticr(
    sample_type: str = "lake_dom",
    n_peaks: int = 1200,
    center_mz: float = 420.0,
    spread_mz: float = 110.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic FT-ICR MS peak list with Kendrick-like chemical series."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Shift center mass depending on type
    if sample_type == "fulvic_acid":
        center_mz = 370.0
        spread_mz = 90.0
    elif sample_type == "humic_acid":
        center_mz = 460.0
        spread_mz = 120.0
    elif sample_type == "sludge_lignin":
        center_mz = 510.0
        spread_mz = 130.0

    raw_mz = rng.normal(center_mz, spread_mz, n_peaks * 2)
    raw_mz = raw_mz[(raw_mz >= 155.0) & (raw_mz <= 795.0)]
    raw_mz = raw_mz[:n_peaks]

    # Snap to typical CH2 / O homologous series increments with mass defect
    mz_list = []
    int_list = []
    for m in raw_mz:
        # Organic mass defect ~ 0.05..0.25 Da
        defect = (m % 1.0) * 0.15 + 0.04
        nominal = np.round(m)
        actual_mz = nominal + defect + rng.uniform(-0.002, 0.002)
        # Intensity follows log-normal distribution
        envelope = np.exp(-0.5 * ((actual_mz - center_mz) / spread_mz) ** 2)
        intensity = envelope * rng.lognormal(mean=2.0, sigma=0.8)
        mz_list.append(actual_mz)
        int_list.append(intensity)

    sort_idx = np.argsort(mz_list)
    return np.array(mz_list)[sort_idx], np.array(int_list)[sort_idx]
