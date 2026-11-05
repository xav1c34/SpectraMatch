"""tests/test_ingest.py — Parsing tests across various formats, delimiters, and encodings."""
import numpy as np
import pandas as pd

from core.models import ModalityEnum
from core.ingest import ingest_file, detect_modality


def test_ingest_standard_csv():
    csv_content = b"wavenumber,absorbance\n4000.0,0.05\n3000.0,0.25\n1600.0,0.65\n400.0,0.1\n"
    # Create sufficient points for FTIR
    wn = np.linspace(400, 4000, 100)
    abs_y = np.exp(-0.5 * ((wn - 1600) / 100) ** 2)
    df = pd.DataFrame({"wavenumber": wn, "absorbance": abs_y})
    bytes_data = df.to_csv(index=False).encode("utf-8")

    rec = ingest_file(bytes_data, filename="sample_ftir.csv")
    assert rec.modality == ModalityEnum.FTIR
    assert len(rec.canonical_vector) == 1801
    assert rec.quality.passed is True


def test_ingest_european_semicolon_format():
    lines = ["wavenumber;absorbance"]
    for w in np.linspace(400, 4000, 80):
        val = 0.5 * np.exp(-0.5 * ((w - 1500) / 80) ** 2)
        lines.append(f"{w:.2f};{val:.4f}".replace(".", ","))
    raw_bytes = "\n".join(lines).encode("cp1251")

    rec = ingest_file(raw_bytes, filename="bruker_export.csv")
    assert rec.modality == ModalityEnum.FTIR
    assert len(rec.x_axis) == 80


def test_modality_detection():
    # FTIR detection by range
    wn = np.linspace(400, 4000, 100)
    y = np.ones(100)
    assert detect_modality(x=wn, y=y) == ModalityEnum.FTIR

    # FT-ICR MS detection by range & point count
    mz = np.linspace(150, 750, 150)
    assert detect_modality(x=mz, y=y) == ModalityEnum.FTICR_MS

    # Filename fallback
    assert detect_modality(filename="sample_eem_scan.csv") == ModalityEnum.EEM
