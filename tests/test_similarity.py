"""tests/test_similarity.py — Symmetry, identity, and batch invariants of similarity metrics."""
import pytest
import numpy as np

from core.similarity import (
    cosine_similarity,
    pearson_correlation,
    euclidean_similarity,
    tcc_similarity,
    _batch_cosine,
    _batch_pearson,
    _batch_tcc,
)


def test_metric_identity_invariants():
    rng = np.random.RandomState(42)
    vec = rng.uniform(0.1, 1.0, 1801).astype(np.float32)

    # Identical vectors must have similarity 1.0
    assert np.isclose(cosine_similarity(vec, vec), 1.0, atol=1e-5)
    assert np.isclose(pearson_correlation(vec, vec), 1.0, atol=1e-5)
    assert np.isclose(euclidean_similarity(vec, vec), 1.0, atol=1e-5)


def test_metric_symmetry_invariants():
    rng = np.random.RandomState(42)
    a = rng.uniform(0.1, 1.0, 1801).astype(np.float32)
    b = rng.uniform(0.1, 1.0, 1801).astype(np.float32)

    # Symmetry: S(a, b) == S(b, a)
    assert np.isclose(cosine_similarity(a, b), cosine_similarity(b, a), atol=1e-6)
    assert np.isclose(pearson_correlation(a, b), pearson_correlation(b, a), atol=1e-6)
    assert np.isclose(euclidean_similarity(a, b), euclidean_similarity(b, a), atol=1e-6)


def test_tcc_similarity_properties():
    rng = np.random.RandomState(42)
    # EEM matrix flat vector (43 x 161 = 6923)
    a = rng.uniform(0.1, 1.0, 6923).astype(np.float32)
    b = rng.uniform(0.1, 1.0, 6923).astype(np.float32)

    # Identity: TCC(a, a) == 1.0
    assert np.isclose(tcc_similarity(a, a), 1.0, atol=1e-5)

    # Symmetry: TCC(a, b) == TCC(b, a)
    assert np.isclose(tcc_similarity(a, b), tcc_similarity(b, a), atol=1e-6)


def test_batch_equivalence():
    rng = np.random.RandomState(42)
    library = rng.uniform(0.1, 1.0, (10, 500)).astype(np.float32)
    query = rng.uniform(0.1, 1.0, 500).astype(np.float32)

    batch_cos = _batch_cosine(library, query)
    elem_cos = [cosine_similarity(query, lib_row) for lib_row in library]
    assert np.allclose(batch_cos, elem_cos, atol=1e-5)

    batch_pearson = _batch_pearson(library, query)
    elem_pearson = [pearson_correlation(query, lib_row) for lib_row in library]
    assert np.allclose(batch_pearson, elem_pearson, atol=1e-5)
