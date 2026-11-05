"""library/seed.py — Pre-computation pipeline: generates reference spectra, populates SQLite catalog, and builds vector index."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path so scripts can run standalone
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.models import ModalityEnum
from core.normalize import normalize_ftir, normalize_eem, normalize_fticr
from core.indices import compute_ftir_indices, compute_eem_indices, compute_fticr_indices
from core.synthetic import (
    generate_synthetic_ftir,
    generate_synthetic_eem,
    generate_synthetic_fticr,
)
from db.database import init_db, insert_spectrum, DEFAULT_DB_PATH


METADATA_PATH = ROOT_DIR / "library" / "metadata.json"
VECTORS_NPZ_PATH = ROOT_DIR / "library" / "vectors.npz"
SPECTRA_DIR = ROOT_DIR / "library" / "spectra"
DEMO_DIR = ROOT_DIR / "user_spectra"


def seed_library():
    """Build and save catalog database, raw spectra CSV files, and in-memory .npz vector matrix."""
    print("=" * 60)
    print("SpectraMatch — Seeding Reference Chemical Library")
    print("=" * 60)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta_records = json.load(f)

    db_conn = init_db(DEFAULT_DB_PATH)

    os.makedirs(SPECTRA_DIR / "ftir", exist_ok=True)
    os.makedirs(SPECTRA_DIR / "eem", exist_ok=True)
    os.makedirs(SPECTRA_DIR / "fticr", exist_ok=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    ftir_vectors = []
    ftir_ids = []
    eem_vectors = []
    eem_ids = []
    fticr_vectors = []
    fticr_ids = []

    print(f"Total catalog entries to build: {len(meta_records)}")

    for rec in meta_records:
        rec_id = rec["id"]
        modality = rec["modality"]
        sample_type = rec.get("sample_type", "nom")
        seed = rec.get("seed", 42)

        if modality == ModalityEnum.FTIR.value:
            wn, abs_y = generate_synthetic_ftir(sample_type=sample_type, seed=seed)
            # Save raw spectrum file
            df_spec = pd.DataFrame({"wavenumber": np.round(wn, 2), "absorbance": np.round(abs_y, 5)})
            csv_path = SPECTRA_DIR / "ftir" / f"{rec_id}.csv"
            df_spec.to_csv(csv_path, index=False)

            # Normalize & compute indices
            vec = normalize_ftir(wn, abs_y)
            indices = compute_ftir_indices(wn, abs_y)

            ftir_vectors.append(vec)
            ftir_ids.append(rec_id)

        elif modality == ModalityEnum.EEM.value:
            ex, em, matrix_2d = generate_synthetic_eem(sample_type=sample_type, seed=seed)
            # Save raw EEM matrix: rows = Em, cols = Ex (standard instrument output)
            df_eem = pd.DataFrame(matrix_2d.T, index=em, columns=ex)
            csv_path = SPECTRA_DIR / "eem" / f"{rec_id}.csv"
            df_eem.to_csv(csv_path, index=True)

            # Normalize & compute indices
            vec = normalize_eem(ex, em, matrix_2d)
            indices = compute_eem_indices(ex, em, matrix_2d)

            eem_vectors.append(vec)
            eem_ids.append(rec_id)

        elif modality == ModalityEnum.FTICR_MS.value:
            mz, intensity = generate_synthetic_fticr(sample_type=sample_type, seed=seed)
            # Save raw peak-list
            df_ms = pd.DataFrame({"mz": np.round(mz, 5), "intensity": np.round(intensity, 2)})
            csv_path = SPECTRA_DIR / "fticr" / f"{rec_id}.csv"
            df_ms.to_csv(csv_path, index=False)

            # Normalize & compute indices
            vec = normalize_fticr(mz, intensity)
            indices = compute_fticr_indices(mz, intensity)

            fticr_vectors.append(vec)
            fticr_ids.append(rec_id)

        else:
            print(f"Skipping unknown modality: {modality}")
            continue

        # Save to SQLite database
        rec_data = {
            "id": rec_id,
            "name": rec["name"],
            "modality": modality,
            "sample_type": sample_type,
            "origin": rec.get("origin", ""),
            "source_doi": rec.get("source_doi", ""),
            "indices": indices,
            "metadata": {
                "seed": seed,
                "file_path": str(csv_path.relative_to(ROOT_DIR)),
            },
        }
        insert_spectrum(db_conn, rec_data)

    # Save NumPy pre-indexed matrices
    np.savez_compressed(
        VECTORS_NPZ_PATH,
        ftir_matrix=np.array(ftir_vectors, dtype=np.float32),
        ftir_ids=np.array(ftir_ids),
        eem_matrix=np.array(eem_vectors, dtype=np.float32),
        eem_ids=np.array(eem_ids),
        fticr_matrix=np.array(fticr_vectors, dtype=np.float32),
        fticr_ids=np.array(fticr_ids),
    )

    print("\n--- Generating User Demonstration Test Spectra ---")
    # 1. FTIR Test Sample with realistic noise and slight shift
    wn_test, abs_test = generate_synthetic_ftir("fulvic_acid", seed=777, noise_sigma=0.015)
    pd.DataFrame({"wavenumber": np.round(wn_test, 2), "absorbance": np.round(abs_test, 5)}).to_csv(
        DEMO_DIR / "Test_FTIR_Suwannee_River.csv", index=False
    )

    # 2. FTIR Bruker .dpt file (tab-separated, reversed wavenumbers 4000 to 400)
    df_dpt = pd.DataFrame({"wavenumber": wn_test[::-1], "absorbance": abs_test[::-1]})
    df_dpt.to_csv(DEMO_DIR / "Test_FTIR_Kraft_Lignin.dpt", sep="\t", header=False, index=False)

    # 3. EEM Test Sample
    ex_test, em_test, eem_test_mat = generate_synthetic_eem("lake_dom", seed=888, noise_sigma=0.01)
    pd.DataFrame(eem_test_mat.T, index=em_test, columns=ex_test).to_csv(
        DEMO_DIR / "Test_EEM_Lake_DOM.csv", index=True
    )

    # 4. FT-ICR MS Test Sample with European comma decimal
    mz_test, int_test = generate_synthetic_fticr("sludge_lignin", seed=999)
    # Write with semicolon and comma decimal
    with open(DEMO_DIR / "Test_FTICR_Baikal_Pulp_Lignin.csv", "w", encoding="utf-8") as f:
        f.write("m/z;intensity\n")
        for m, i in zip(mz_test, int_test):
            f.write(f"{m:.5f};{i:.2f}\n".replace(".", ","))

    print("\n" + "=" * 60)
    print(f"✅ Seeding successfully completed!")
    print(f"   FTIR matrix:      {len(ftir_vectors)} spectra (dim {ftir_vectors[0].shape})")
    print(f"   EEM matrix:       {len(eem_vectors)} spectra (dim {eem_vectors[0].shape})")
    print(f"   FT-ICR MS matrix: {len(fticr_vectors)} spectra (dim {fticr_vectors[0].shape})")
    print(f"   Database:         {DEFAULT_DB_PATH}")
    print(f"   Vector Index:     {VECTORS_NPZ_PATH}")
    print(f"   Demo Files:       {DEMO_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    seed_library()
