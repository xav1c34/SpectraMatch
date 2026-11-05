"""tests/test_chemspace_and_kendrick.py — Tests for PCA mapping and Kendrick mass defect."""
import numpy as np

from core.models import ModalityEnum
from core.chemspace import fit_library_pca, project_query_to_pca
from core.kendrick import compute_kendrick_mass_defect


def test_pca_fitting_and_projection():
    rng = np.random.RandomState(42)
    lib = rng.uniform(0.1, 1.0, (15, 1801)).astype(np.float32)

    pca = fit_library_pca(lib, modality=ModalityEnum.FTIR, n_components=3)
    assert pca is not None
    assert pca.library_coords.shape == (15, 3)
    assert len(pca.explained_variance_ratio) == 3
    assert np.all(pca.explained_variance_ratio > 0.0)
    assert np.sum(pca.explained_variance_ratio) <= 1.0001

    # Project query
    query = rng.uniform(0.1, 1.0, 1801).astype(np.float32)
    coords = project_query_to_pca(query, pca)
    assert coords.shape == (3,)


def test_kendrick_mass_defect():
    # Simulate a true CH2 homologous pair
    # Base peak m/z = 300.1234, plus CH2 (14.01565 Da) -> m/z = 314.13905
    mz = np.array([300.1234, 314.13905, 450.2000])
    intensity = np.array([1000.0, 800.0, 500.0])

    df = compute_kendrick_mass_defect(mz, intensity, base="CH2")
    assert len(df) == 3
    assert "KMD" in df.columns
    assert "NKM" in df.columns

    # The two homologues should have nearly identical KMD
    kmd_1 = df.iloc[0]["KMD"]
    kmd_2 = df.iloc[1]["KMD"]
    assert np.isclose(kmd_1, kmd_2, atol=1e-3)
