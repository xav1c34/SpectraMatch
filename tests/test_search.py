"""tests/test_search.py — Integration search and retrieval accuracy tests."""
import numpy as np

from core.models import ModalityEnum, MetricEnum
from core.similarity import search
from core.synthetic import generate_synthetic_ftir
from core.normalize import normalize_ftir


def test_self_identification_rank1():
    """Invariant: A library sample queried against the library must return itself as rank #1 with score 1.0."""
    wn_grid = np.arange(400, 4002, 2)
    # Generate 5 reference spectra
    lib_vecs = []
    lib_meta = []
    types = ["humic_acid", "fulvic_acid", "sludge_lignin", "polysaccharide", "lake_dom"]

    for idx, t in enumerate(types):
        wn, y = generate_synthetic_ftir(t, seed=idx + 10)
        vec = normalize_ftir(wn, y)
        lib_vecs.append(vec)
        lib_meta.append({"id": f"spec_{idx}", "name": f"Sample {t}", "sample_type": t})

    library_matrix = np.array(lib_vecs, dtype=np.float32)

    # Query with exactly sample #2 (sludge_lignin)
    query_vec = lib_vecs[2]
    results = search(
        query_vec=query_vec,
        library_matrix=library_matrix,
        library_meta=lib_meta,
        modality=ModalityEnum.FTIR,
        metric=MetricEnum.PEARSON,
        top_k=5,
    )

    assert len(results) == 5
    assert results[0].rank == 1
    assert results[0].spectrum_id == "spec_2"
    assert np.isclose(results[0].score, 1.0, atol=1e-4)


def test_noisy_query_robustness():
    """A query corrupted with Gaussian noise should still retrieve the true origin as rank #1."""
    wn, y = generate_synthetic_ftir("humic_acid", seed=99)
    true_vec = normalize_ftir(wn, y)

    # Add Gaussian noise
    rng = np.random.RandomState(123)
    noisy_y = y + rng.normal(0, 0.05, len(y))
    query_vec = normalize_ftir(wn, noisy_y)

    other_wn, other_y = generate_synthetic_ftir("polysaccharide", seed=100)
    other_vec = normalize_ftir(other_wn, other_y)

    library_matrix = np.array([other_vec, true_vec], dtype=np.float32)
    library_meta = [
        {"id": "poly", "name": "Cellulose"},
        {"id": "target_ha", "name": "Humic Acid True Match"},
    ]

    results = search(
        query_vec=query_vec,
        library_matrix=library_matrix,
        library_meta=library_meta,
        modality=ModalityEnum.FTIR,
        metric=MetricEnum.PEARSON,
        top_k=2,
    )

    assert results[0].rank == 1
    assert results[0].spectrum_id == "target_ha"
    assert results[0].score > 0.90
