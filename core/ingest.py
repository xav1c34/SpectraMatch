"""core/ingest.py — Resilient multi-format parser, encoding decoder, and modality detector."""
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from core.models import ModalityEnum, QualityReport, SpectralRecord
from core.normalize import normalize
from core.indices import compute_indices
from core.qc import run_qc
from core.jcamp import is_jcamp_file, parse_jcamp_dx


ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "latin-1"]


def _decode_bytes(file_bytes: bytes) -> str:
    """Decode raw bytes trying encodings in priority cascade."""
    for enc in ENCODINGS:
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return file_bytes.decode("latin-1", errors="replace")


def _sniff_and_read_table(text: str) -> pd.DataFrame:
    """Robustly parse tabular text into a DataFrame handling various separators and decimals."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("File is empty")

    # Check first non-empty lines to detect semicolon / tab / comma
    sample_text = "\n".join(lines[:40])

    # Check for semicolon delimiter with European comma decimal
    if ";" in sample_text and not ("," in sample_text and sample_text.count(",") > sample_text.count(";") * 2):
        try:
            df = pd.read_csv(io.StringIO(text), sep=";", decimal=",", comment="#", engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    # Try standard sniff with python engine
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, comment="#", engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Try whitespace delimiter
    try:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", comment="#", engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Fallback comma
    return pd.read_csv(io.StringIO(text), comment="#")


def detect_modality(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    filename: str = "",
    is_2d: bool = False,
    matrix_shape: Optional[Tuple[int, int]] = None,
) -> Optional[ModalityEnum]:
    """Determine spectral modality according to the 4-level priority rules in Section 4."""
    fname = filename.lower()

    # Priority 1: 2D Matrix structure
    if is_2d and matrix_shape is not None:
        r, c = matrix_shape
        if r >= 3 and c >= 3:
            return ModalityEnum.EEM

    # Priority 2: Coordinate X-axis range heuristics
    if x is not None and len(x) > 0:
        x_clean = x[np.isfinite(x)]
        if len(x_clean) > 0:
            x_min = float(np.min(x_clean))
            x_max = float(np.max(x_clean))

            # FTIR: typical wavenumbers 350-600 to 3500-5000 cm^-1
            if (350.0 <= x_min <= 700.0) and (3400.0 <= x_max <= 5500.0):
                return ModalityEnum.FTIR

            # FT-ICR MS: m/z 100-250 to 400-1200 Da, discrete peaks (< 50,000)
            if (80.0 <= x_min <= 260.0) and (400.0 <= x_max <= 1500.0) and len(x) < 50000:
                return ModalityEnum.FTICR_MS

    # Priority 3: Filename patterns
    if any(k in fname for k in [".dpt", "ftir", "ir_"]):
        return ModalityEnum.FTIR
    if any(k in fname for k in ["eem", "fluor", "excitation"]):
        return ModalityEnum.EEM
    if any(k in fname for k in ["fticr", "ft-icr", "mass", "ms_"]):
        return ModalityEnum.FTICR_MS

    # Priority 4: Undetermined
    return None


def _try_parse_eem(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Try interpreting DataFrame as 2D Excitation-Emission Matrix."""
    # Check if df has >= 5 rows and >= 5 columns
    if df.shape[0] < 4 or df.shape[1] < 4:
        return None

    try:
        # Check if first column has emission/excitation numbers
        col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        # Check if header columns (excluding first) have excitation numbers
        cols = pd.to_numeric(df.columns[1:], errors="coerce")

        if col0.notnull().sum() >= 4 and cols.notnull().sum() >= 3:
            em = col0.dropna().values.astype(np.float64)
            ex = cols.dropna().values.astype(np.float64)
            data_sub = df.iloc[: len(em), 1 : len(ex) + 1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            matrix = data_sub.values.astype(np.float64)

            # Check typical wavelength range: between 200 and 800 nm
            if (200.0 <= np.min(ex) <= 500.0) and (250.0 <= np.min(em) <= 700.0):
                # Ensure matrix is (len(ex), len(em))
                if matrix.shape == (len(em), len(ex)):
                    matrix = matrix.T
                return ex, em, matrix
    except Exception:
        pass

    return None


def _try_parse_1d(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 1D X and Y arrays from DataFrame with column name sniffing."""
    col_names = [str(c).lower().strip() for c in df.columns]

    x_col_idx = None
    y_col_idx = None

    # Search for Mass Spec headers
    for idx, c in enumerate(col_names):
        if any(term in c for term in ["m/z", "mz", "mass", "wavenumber", "cm-1", "cm^-1"]):
            if x_col_idx is None:
                x_col_idx = idx
        elif any(term in c for term in ["int", "intensity", "absorbance", "abundance", "transmittance", "%t"]):
            if y_col_idx is None:
                y_col_idx = idx

    if x_col_idx is None:
        x_col_idx = 0
    if y_col_idx is None:
        y_col_idx = 1 if df.shape[1] > 1 else 0

    x_s = pd.to_numeric(df.iloc[:, x_col_idx], errors="coerce")
    y_s = pd.to_numeric(df.iloc[:, y_col_idx], errors="coerce")

    valid = x_s.notnull() & y_s.notnull()
    x = x_s[valid].values.astype(np.float64)
    y = y_s[valid].values.astype(np.float64)

    # Sort by X ascending
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Convert Transmittance %T to Absorbance if detected
    if len(y) > 0 and 0.0 <= np.min(y) and np.max(y) <= 105.0 and np.mean(y) > 15.0:
        # Check if spectrum looks like %T (most values > 10)
        # Formula: A = 2 - log10(%T)
        t_clean = np.clip(y, 1e-4, 100.0)
        y = 2.0 - np.log10(t_clean)

    return x, y


def ingest_file(
    file_bytes: bytes,
    filename: str = "spectrum.csv",
    modality_hint: Optional[Union[str, ModalityEnum]] = None,
) -> SpectralRecord:
    """Full parsing, modality identification, QC, normalization, and index extraction pipeline."""
    text = _decode_bytes(file_bytes)

    # Check for JCAMP-DX standard format
    is_2d = False
    matrix_2d = None
    em_axis = None
    jcamp_meta = {}

    if is_jcamp_file(text):
        x, y, jcamp_meta = parse_jcamp_dx(text)
    else:
        df = _sniff_and_read_table(text)
        # First attempt 2D EEM parsing
        eem_res = _try_parse_eem(df)
        is_2d = eem_res is not None

        if is_2d:
            ex, em, matrix_2d = eem_res
            x, y = ex, em
            em_axis = em
        else:
            x, y = _try_parse_1d(df)

    # Determine modality
    if modality_hint is not None:
        resolved_modality = ModalityEnum(modality_hint)
    else:
        mat_shape = matrix_2d.shape if matrix_2d is not None else None
        detected = detect_modality(x=x, y=y, filename=filename, is_2d=is_2d, matrix_shape=mat_shape)
        resolved_modality = detected or ModalityEnum.FTIR

    # Run Analytical Quality Control suite
    qc_report = run_qc(x, y, resolved_modality, matrix_2d=matrix_2d, em_axis=em_axis)

    # Normalize onto canonical vector representation
    try:
        canonical_vec = normalize(x, y, resolved_modality, matrix_2d=matrix_2d)
    except Exception as e:
        # If normalization failed due to QC or math failure, assign zeros and record error
        qc_report.passed = False
        qc_report.score = 0.0
        qc_report.errors.append(f"Normalization failed: {str(e)}")
        if resolved_modality == ModalityEnum.FTIR:
            canonical_vec = np.zeros(1801, dtype=np.float32)
        elif resolved_modality == ModalityEnum.EEM:
            canonical_vec = np.zeros(6923, dtype=np.float32)
        else:
            canonical_vec = np.zeros(1300, dtype=np.float32)

    # Compute chemometric indices
    indices = {}
    if qc_report.passed or len(qc_report.errors) == 0:
        try:
            indices = compute_indices(x, y, resolved_modality, matrix_2d=matrix_2d)
        except Exception as e:
            qc_report.warnings.append(f"Indices computation notice: {str(e)}")

    base_name = re.sub(r"\.[^.]+$", "", filename).replace("_", " ").title()

    return SpectralRecord(
        id=f"query-{hash(filename) % 1000000:06d}",
        name=base_name,
        modality=resolved_modality,
        sample_type="unknown",
        x_axis=x,
        y_axis=y,
        canonical_vector=canonical_vec,
        matrix_2d=matrix_2d,
        indices=indices,
        quality=qc_report,
        metadata={"filename": filename, "points": len(x)},
    )
