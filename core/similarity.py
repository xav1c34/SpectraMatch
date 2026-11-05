"""core/similarity.py — Chemometric similarity metrics and high-throughput vector search."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.models import MetricEnum, ModalityEnum, SearchResult, CompareResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_f = np.asarray(a, dtype=np.float64).flatten()
    b_f = np.asarray(b, dtype=np.float64).flatten()
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    dot = np.dot(a_f, b_f)
    val = dot / (norm_a * norm_b)
    return float(np.clip(val, -1.0, 1.0))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient (baseline offset invariant)."""
    a_f = np.asarray(a, dtype=np.float64).flatten()
    b_f = np.asarray(b, dtype=np.float64).flatten()
    a_c = a_f - np.mean(a_f)
    b_c = b_f - np.mean(b_f)
    norm_a = np.linalg.norm(a_c)
    norm_b = np.linalg.norm(b_c)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    val = np.dot(a_c, b_c) / (norm_a * norm_b)
    return float(np.clip(val, -1.0, 1.0))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean similarity: 1 / (1 + ||a - b||_2). Range (0, 1]."""
    a_f = np.asarray(a, dtype=np.float64).flatten()
    b_f = np.asarray(b, dtype=np.float64).flatten()
    dist = np.linalg.norm(a_f - b_f)
    return float(1.0 / (1.0 + dist))


def tcc_similarity(
    a_flat: np.ndarray,
    b_flat: np.ndarray,
    n_ex: int = 43,
    n_em: int = 161,
) -> float:
    """Tucker Congruence Coefficient (TCC) between two flattened 2D EEM matrices.

    TCC = min(cosine(q_ex, r_ex), cosine(q_em, r_em)),
    where profiles are obtained by marginal sums along axes.
    """
    a_2d = np.asarray(a_flat, dtype=np.float64).reshape((n_ex, n_em))
    b_2d = np.asarray(b_flat, dtype=np.float64).reshape((n_ex, n_em))

    # Ex profiles (sum over emission channels)
    a_ex = a_2d.sum(axis=1)
    b_ex = b_2d.sum(axis=1)

    # Em profiles (sum over excitation channels)
    a_em = a_2d.sum(axis=0)
    b_em = b_2d.sum(axis=0)

    tcc_ex = cosine_similarity(a_ex, b_ex)
    tcc_em = cosine_similarity(a_em, b_em)

    return float(max(min(tcc_ex, tcc_em), 0.0))


def _batch_cosine(library: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Fast batch cosine similarity over library matrix of shape (N, D)."""
    q = np.asarray(query, dtype=np.float64).flatten()
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return np.zeros(len(library), dtype=np.float64)

    # Dot products
    dots = library.dot(q)
    lib_norms = np.linalg.norm(library, axis=1)
    denoms = np.maximum(lib_norms * q_norm, 1e-12)
    scores = dots / denoms
    return np.clip(scores, -1.0, 1.0)


def _batch_pearson(library: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Fast batch Pearson correlation over library matrix (N, D)."""
    q = np.asarray(query, dtype=np.float64).flatten()
    q_c = q - np.mean(q)
    q_norm = np.linalg.norm(q_c)
    if q_norm < 1e-12:
        return np.zeros(len(library), dtype=np.float64)

    lib_mean = np.mean(library, axis=1, keepdims=True)
    lib_c = library - lib_mean
    dots = lib_c.dot(q_c)
    lib_norms = np.linalg.norm(lib_c, axis=1)
    denoms = np.maximum(lib_norms * q_norm, 1e-12)
    scores = dots / denoms
    return np.clip(scores, -1.0, 1.0)


def _batch_euclidean(library: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Fast batch Euclidean similarity over library matrix (N, D)."""
    q = np.asarray(query, dtype=np.float64).flatten()
    dists = np.linalg.norm(library - q, axis=1)
    return 1.0 / (1.0 + dists)


def _batch_tcc(
    library: np.ndarray,
    query: np.ndarray,
    n_ex: int = 43,
    n_em: int = 161,
) -> np.ndarray:
    """Fast batch Tucker Congruence Coefficient over library matrix (N, 6923)."""
    n_samples = len(library)
    q_2d = np.asarray(query, dtype=np.float64).reshape((n_ex, n_em))
    q_ex = q_2d.sum(axis=1)
    q_em = q_2d.sum(axis=0)

    lib_3d = np.asarray(library, dtype=np.float64).reshape((n_samples, n_ex, n_em))
    lib_ex = lib_3d.sum(axis=2)  # (N, n_ex)
    lib_em = lib_3d.sum(axis=1)  # (N, n_em)

    scores_ex = _batch_cosine(lib_ex, q_ex)
    scores_em = _batch_cosine(lib_em, q_em)

    return np.maximum(np.minimum(scores_ex, scores_em), 0.0)


def resolve_metric(metric: Union[str, MetricEnum], modality: Union[str, ModalityEnum]) -> MetricEnum:
    """Resolve AUTO metric or string to a concrete MetricEnum."""
    m = MetricEnum(metric)
    mod = ModalityEnum(modality)
    if m == MetricEnum.AUTO:
        if mod == ModalityEnum.FTIR:
            return MetricEnum.PEARSON
        elif mod == ModalityEnum.EEM:
            return MetricEnum.TCC
        elif mod == ModalityEnum.FTICR_MS:
            return MetricEnum.COSINE
    return m


def search(
    query_vec: np.ndarray,
    library_matrix: np.ndarray,
    library_meta: List[Dict[str, Any]],
    modality: Union[str, ModalityEnum],
    metric: Union[str, MetricEnum] = MetricEnum.AUTO,
    top_k: int = 10,
) -> List[SearchResult]:
    """Search query vector against pre-loaded library matrix.

    Returns:
        List of SearchResult dataclasses sorted in descending order of similarity score.
    """
    if len(library_matrix) == 0:
        return []

    mod = ModalityEnum(modality)
    concrete_metric = resolve_metric(metric, mod)

    if concrete_metric == MetricEnum.PEARSON:
        scores = _batch_pearson(library_matrix, query_vec)
    elif concrete_metric == MetricEnum.COSINE:
        scores = _batch_cosine(library_matrix, query_vec)
    elif concrete_metric == MetricEnum.EUCLIDEAN:
        scores = _batch_euclidean(library_matrix, query_vec)
    elif concrete_metric == MetricEnum.TCC:
        if mod != ModalityEnum.EEM:
            # Fallback to cosine if non-EEM requested TCC
            scores = _batch_cosine(library_matrix, query_vec)
        else:
            scores = _batch_tcc(library_matrix, query_vec)
    else:
        scores = _batch_cosine(library_matrix, query_vec)

    # Sort indices by descending score
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: List[SearchResult] = []
    for rank_idx, idx in enumerate(ranked_indices, start=1):
        meta = library_meta[idx] if idx < len(library_meta) else {}
        lib_vec = library_matrix[idx]
        mat_2d = None
        if mod == ModalityEnum.EEM and len(lib_vec) == 43 * 161:
            mat_2d = lib_vec.reshape((43, 161))

        results.append(
            SearchResult(
                spectrum_id=meta.get("id", f"spec_{idx}"),
                name=meta.get("name", "Unknown Sample"),
                sample_type=meta.get("sample_type", "unknown"),
                origin=meta.get("origin", "N/A"),
                source_doi=meta.get("source_doi", ""),
                score=round(float(scores[idx]), 4),
                metric=concrete_metric.value,
                rank=rank_idx,
                indices=meta.get("indices", {}),
                canonical_vector=lib_vec,
                matrix_2d=mat_2d,
            )
        )

    return results


def compare_two(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    modality: Union[str, ModalityEnum],
    indices_a: Optional[Dict[str, float]] = None,
    indices_b: Optional[Dict[str, float]] = None,
) -> CompareResult:
    """Perform direct 1-to-1 comparison between two normalized spectra across all metrics."""
    mod = ModalityEnum(modality)
    cos_val = cosine_similarity(vec_a, vec_b)
    pearson_val = pearson_correlation(vec_a, vec_b)
    euc_val = euclidean_similarity(vec_a, vec_b)
    tcc_val = None
    if mod == ModalityEnum.EEM and len(vec_a) == 43 * 161 and len(vec_b) == 43 * 161:
        tcc_val = tcc_similarity(vec_a, vec_b)

    return CompareResult(
        modality=mod,
        cosine=round(cos_val, 4),
        pearson=round(pearson_val, 4),
        euclidean=round(euc_val, 4),
        tcc=round(tcc_val, 4) if tcc_val is not None else None,
        indices_a=indices_a or {},
        indices_b=indices_b or {},
    )
