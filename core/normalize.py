"""core/normalize.py — Preprocessing, baseline correction, canonical grid interpolation, and normalization."""
from typing import Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d, griddata, RegularGridInterpolator

from core.models import ModalityEnum


CANONICAL_FTIR_GRID = np.arange(400, 4002, 2, dtype=np.float64)  # 1801 points
CANONICAL_EEM_EX = np.arange(240, 455, 5, dtype=np.float64)      # 43 points
CANONICAL_EEM_EM = np.arange(280, 602, 2, dtype=np.float64)      # 161 points
CANONICAL_FTICR_EDGES = np.arange(150.0, 800.0 + 0.001, 0.5, dtype=np.float64)  # 1301 edges -> 1300 bins


def _asls_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares (AsLS) baseline estimation (Eilers & Boelens, 2005).

    Parameters:
        y: Raw 1D intensity array.
        lam: Smoothing parameter (default 1e6).
        p: Asymmetry penalty parameter (default 0.01).
        niter: Maximum number of iterations (default 10).

    Returns:
        baseline: Fitted baseline array matching the shape of y.
    """
    n = len(y)
    if n < 3:
        return np.zeros_like(y)

    e = np.ones(n)
    # Construct 2nd difference operator matrix D of shape (n-2, n)
    D = sp.diags([e[:-2], -2 * e[:-1], e], [0, 1, 2], shape=(n - 2, n))
    DTD = D.T.dot(D)

    w = np.ones(n)
    z = np.zeros(n)

    for _ in range(niter):
        W = sp.diags(w, 0, shape=(n, n))
        A = W + lam * DTD
        b = w * y
        z = spsolve(A.tocsc(), b)
        w = p * (y > z) + (1.0 - p) * (y <= z)

    return z


def _interpolate_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear 1D interpolation onto a regular grid with zero-padding outside data bounds."""
    # Ensure strictly ascending order of x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Remove non-finite values if present
    valid = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    x_valid = x_sorted[valid]
    y_valid = y_sorted[valid]

    if len(x_valid) < 2:
        return np.zeros_like(grid, dtype=np.float64)

    # Remove duplicated x values by keeping first
    _, unique_idx = np.unique(x_valid, return_index=True)
    x_clean = x_valid[unique_idx]
    y_clean = y_valid[unique_idx]

    f = interp1d(x_clean, y_clean, kind="linear", bounds_error=False, fill_value=0.0)
    return f(grid)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 unit-norm scaling (||v||_2 = 1.0). Raises ValueError if vector is all zero."""
    v_clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(v_clean)
    if norm < 1e-12:
        raise ValueError("empty_spectrum: L2 norm is below 1e-12")
    return (v_clean / norm).astype(np.float32)


def _remove_rayleigh(ex: np.ndarray, em: np.ndarray, matrix: np.ndarray, tolerance: float = 15.0) -> np.ndarray:
    """Remove 1st and 2nd order Rayleigh scattering from EEM matrix and interpolate."""
    # matrix is assumed to have shape (n_ex, n_em)
    mesh_ex, mesh_em = np.meshgrid(ex, em, indexing="ij")

    # Rayleigh mask: 1st order |Em - Ex| < tol, 2nd order |Em - 2*Ex| < tol
    mask_1st = np.abs(mesh_em - mesh_ex) < tolerance
    mask_2nd = np.abs(mesh_em - 2.0 * mesh_ex) < tolerance
    scatter_mask = mask_1st | mask_2nd

    matrix_clean = matrix.copy().astype(np.float64)
    matrix_clean[scatter_mask] = np.nan

    valid_mask = ~np.isnan(matrix_clean)
    if np.sum(valid_mask) < 4:
        return np.nan_to_num(matrix_clean, nan=0.0)

    # 2D Delaunay interpolation over scattering regions
    pts = np.column_stack((mesh_ex[valid_mask], mesh_em[valid_mask]))
    vals = matrix_clean[valid_mask]

    interp_vals = griddata(pts, vals, (mesh_ex, mesh_em), method="linear", fill_value=0.0)
    # Fallback any remaining NaNs with 0.0
    return np.nan_to_num(interp_vals, nan=0.0)


def normalize_ftir(wavenumbers: np.ndarray, absorbance: np.ndarray) -> np.ndarray:
    """Preprocess, AsLS-baseline correct, interpolate (1801 points), and L2 normalize FTIR spectrum.

    Returns:
        canonical_vector: ndarray shape (1801,), dtype float32, ||v||_2 = 1.0.
    """
    wn = np.asarray(wavenumbers, dtype=np.float64)
    abs_y = np.asarray(absorbance, dtype=np.float64)

    # Check ascending order
    if len(wn) > 1 and wn[0] > wn[-1]:
        wn = wn[::-1]
        abs_y = abs_y[::-1]

    # Baseline correction via AsLS
    baseline = _asls_baseline(abs_y, lam=1e6, p=0.01, niter=10)
    corrected = abs_y - baseline

    # Interpolate to canonical grid [400..4000 cm^-1, step 2]
    canonical_y = _interpolate_to_grid(wn, corrected, CANONICAL_FTIR_GRID)
    canonical_y = np.maximum(canonical_y, 0.0)

    return _l2_normalize(canonical_y)


def normalize_eem(ex: np.ndarray, em: np.ndarray, matrix_2d: np.ndarray) -> np.ndarray:
    """Clean Rayleigh scattering, interpolate to canonical grid (43x161), and max-normalize EEM.

    Returns:
        canonical_vector: ndarray shape (6923,), dtype float32, max = 1.0.
    """
    ex_arr = np.asarray(ex, dtype=np.float64)
    em_arr = np.asarray(em, dtype=np.float64)
    mat = np.asarray(matrix_2d, dtype=np.float64)

    # Check orientation: ensure shape is (len(ex), len(em))
    if mat.shape == (len(em_arr), len(ex_arr)):
        mat = mat.T
    elif mat.shape != (len(ex_arr), len(em_arr)):
        raise ValueError(
            f"Shape mismatch: matrix shape {mat.shape} does not match (len(ex), len(em)) = ({len(ex_arr)}, {len(em_arr)})"
        )

    # Remove Rayleigh scatter
    cleaned = _remove_rayleigh(ex_arr, em_arr, mat, tolerance=15.0)

    # 2D Interpolation onto canonical grid: Ex (43 points), Em (161 points)
    rgi = RegularGridInterpolator((ex_arr, em_arr), cleaned, bounds_error=False, fill_value=0.0)
    grid_ex, grid_em = np.meshgrid(CANONICAL_EEM_EX, CANONICAL_EEM_EM, indexing="ij")
    matrix_canonical = rgi((grid_ex, grid_em))
    matrix_canonical = np.maximum(matrix_canonical, 0.0)

    max_val = np.max(matrix_canonical)
    if max_val < 1e-12:
        raise ValueError("empty_spectrum: EEM intensity max is below 1e-12")

    matrix_normed = (matrix_canonical / max_val).astype(np.float32)
    return matrix_normed.flatten(order="C")


def normalize_fticr(mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Bin raw peak list into 1300 mass channels (150.0..800.5 Da, 0.5 Da bins) and L2 normalize.

    Returns:
        canonical_vector: ndarray shape (1300,), dtype float32, ||v||_2 = 1.0.
    """
    mz_arr = np.asarray(mz, dtype=np.float64)
    int_arr = np.asarray(intensity, dtype=np.float64)

    valid = (mz_arr > 0) & (int_arr > 0) & np.isfinite(mz_arr) & np.isfinite(int_arr)
    mz_valid = mz_arr[valid]
    int_valid = int_arr[valid]

    if len(mz_valid) == 0:
        raise ValueError("empty_spectrum: No valid positive peaks for FT-ICR MS")

    binned, _ = np.histogram(mz_valid, bins=CANONICAL_FTICR_EDGES, weights=int_valid)
    return _l2_normalize(binned)


def normalize(
    x: np.ndarray,
    y: np.ndarray,
    modality: Union[str, ModalityEnum],
    matrix_2d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generic normalization entrypoint for any supported modality."""
    mod = ModalityEnum(modality)
    if mod == ModalityEnum.FTIR:
        return normalize_ftir(x, y)
    elif mod == ModalityEnum.EEM:
        if matrix_2d is None:
            raise ValueError("matrix_2d must be provided for EEM normalization")
        return normalize_eem(x, y, matrix_2d)
    elif mod == ModalityEnum.FTICR_MS:
        return normalize_fticr(x, y)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
