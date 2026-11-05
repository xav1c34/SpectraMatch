"""core/models.py — Data contracts and typed representations for SpectraMatch."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class ModalityEnum(str, Enum):
    """Supported analytical spectral modalities."""
    FTIR = "FTIR"
    EEM = "EEM"
    FTICR_MS = "FTICR_MS"


class MetricEnum(str, Enum):
    """Similarity metrics for spectral matching."""
    COSINE = "cosine"
    PEARSON = "pearson"
    EUCLIDEAN = "euclidean"
    TCC = "tcc"
    AUTO = "auto"


@dataclass
class QualityReport:
    """Spectral quality validation assessment from QC module."""
    passed: bool
    score: float  # 0.0 to 1.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    snr_estimate: Optional[float] = None
    coverage_pct: float = 100.0


@dataclass
class SpectralRecord:
    """Canonical representation of an analytical spectrum in memory."""
    id: str
    name: str
    modality: ModalityEnum
    sample_type: str
    x_axis: np.ndarray             # Raw or processed X-axis (wavenumbers, Em or m/z)
    y_axis: np.ndarray             # Raw or 1D intensity signal
    canonical_vector: np.ndarray   # Interpolated & normalized vector for fast matrix search
    matrix_2d: Optional[np.ndarray] = None  # 2D intensity grid for EEM (n_ex, n_em)
    indices: Dict[str, float] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=lambda: QualityReport(True, 1.0))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Individual ranked match from library query."""
    spectrum_id: str
    name: str
    sample_type: str
    origin: str
    source_doi: str
    score: float
    metric: str
    rank: int
    indices: Dict[str, float] = field(default_factory=dict)
    canonical_vector: Optional[np.ndarray] = None
    matrix_2d: Optional[np.ndarray] = None


@dataclass
class CompareResult:
    """Direct binary comparison between two analytical spectra."""
    modality: ModalityEnum
    cosine: float
    pearson: float
    euclidean: float
    tcc: Optional[float]
    indices_a: Dict[str, float]
    indices_b: Dict[str, float]
