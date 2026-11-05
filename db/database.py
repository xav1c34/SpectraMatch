"""db/database.py — SQLite database interface for SpectraMatch spectral catalog."""
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "spectramatch.db")


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database, creating schema and indexes if not present."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spectra (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                modality TEXT NOT NULL,
                sample_type TEXT NOT NULL,
                origin TEXT,
                source_doi TEXT,
                indices_json TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_modality ON spectra (modality)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_type ON spectra (sample_type)")

    return conn


def insert_spectrum(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    """Insert or replace spectral catalog entry."""
    indices_str = json.dumps(record.get("indices", {}))
    meta_str = json.dumps(record.get("metadata", {}))

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO spectra (
                id, name, modality, sample_type, origin, source_doi, indices_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["name"],
                record["modality"],
                record.get("sample_type", "unknown"),
                record.get("origin", ""),
                record.get("source_doi", ""),
                indices_str,
                meta_str,
            ),
        )


def get_all_spectra(
    conn: sqlite3.Connection,
    modality: Optional[str] = None,
    sample_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query catalog records with optional modality and sample_type filtering."""
    query = "SELECT * FROM spectra WHERE 1=1"
    params = []

    if modality and modality != "All":
        query += " AND modality = ?"
        params.append(modality)

    if sample_type and sample_type != "All":
        query += " AND sample_type = ?"
        params.append(sample_type)

    query += " ORDER BY name ASC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["indices"] = json.loads(d.get("indices_json") or "{}")
        except Exception:
            d["indices"] = {}
        try:
            d["metadata"] = json.loads(d.get("metadata_json") or "{}")
        except Exception:
            d["metadata"] = {}
        result.append(d)

    return result


def get_spectrum_by_id(conn: sqlite3.Connection, spectrum_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve detailed record by unique spectrum ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM spectra WHERE id = ?", (spectrum_id,))
    row = cursor.fetchone()
    if not row:
        return None

    d = dict(row)
    try:
        d["indices"] = json.loads(d.get("indices_json") or "{}")
    except Exception:
        d["indices"] = {}
    try:
        d["metadata"] = json.loads(d.get("metadata_json") or "{}")
    except Exception:
        d["metadata"] = {}
    return d


def get_modality_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    """Return dictionary of record counts grouped by modality."""
    cursor = conn.cursor()
    cursor.execute("SELECT modality, COUNT(*) as cnt FROM spectra GROUP BY modality")
    return {row["modality"]: row["cnt"] for row in cursor.fetchall()}
