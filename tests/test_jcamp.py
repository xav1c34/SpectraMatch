"""tests/test_jcamp.py — Validation of native IUPAC JCAMP-DX parser."""
import numpy as np

from core.jcamp import is_jcamp_file, parse_jcamp_dx
from core.ingest import ingest_file
from core.models import ModalityEnum


SAMPLE_JCAMP = """##TITLE= Suwannee River Fulvic Acid Reference
##JCAMP-DX= 4.24
##DATA TYPE= INFRARED SPECTRUM
##XUNITS= 1/CM
##YUNITS= ABSORBANCE
##FIRSTX= 400.0
##LASTX= 4000.0
##NPOINTS= 5
##XYDATA= (X++(Y..Y))
400.0 0.12 0.25 0.65
2500.0 0.40 0.10
##END=
"""


def test_is_jcamp():
    assert is_jcamp_file(SAMPLE_JCAMP) is True
    assert is_jcamp_file("wavenumber,absorbance\n400,0.1\n") is False


def test_parse_jcamp():
    x, y, meta = parse_jcamp_dx(SAMPLE_JCAMP)
    assert len(x) == 5
    assert len(y) == 5
    assert np.isclose(x[0], 400.0)
    assert meta["TITLE"] == "Suwannee River Fulvic Acid Reference"


def test_ingest_jcamp_file():
    raw_bytes = SAMPLE_JCAMP.encode("utf-8")
    rec = ingest_file(raw_bytes, filename="sample.jdx")
    assert rec.modality == ModalityEnum.FTIR
    assert "Suwannee River" in rec.name or "Sample" in rec.name
