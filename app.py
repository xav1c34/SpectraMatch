"""app.py — SpectraMatch Application Entrypoint and State Orchestrator."""
import os
from pathlib import Path
import streamlit as st
import numpy as np

from db.database import init_db, get_all_spectra, get_modality_counts, DEFAULT_DB_PATH
from views.search_view import render_search_page
from views.browse_view import render_browse_page
from views.compare_view import render_compare_page
from views.about_view import render_about_page
from views.batch_view import render_batch_page
from views.add_view import render_add_page
from views.generator_view import render_generator_page


ROOT_DIR = Path(__file__).resolve().parent
NPZ_PATH = ROOT_DIR / "library" / "vectors.npz"


def load_application_state():
    """Load SQLite database and precomputed NumPy matrix vectors into st.session_state."""
    if st.session_state.get("library_loaded", False):
        return

    # 1. Connect SQLite database
    conn = init_db(DEFAULT_DB_PATH)
    st.session_state["db_conn"] = conn
    st.session_state["library_meta"] = get_all_spectra(conn)
    st.session_state["modality_counts"] = get_modality_counts(conn)

    # 2. Load pre-indexed matrix vectors
    if NPZ_PATH.exists():
        data = np.load(NPZ_PATH, allow_pickle=True)
        st.session_state["ftir_matrix"] = data["ftir_matrix"]
        st.session_state["ftir_ids"] = list(data["ftir_ids"])
        st.session_state["eem_matrix"] = data["eem_matrix"]
        st.session_state["eem_ids"] = list(data["eem_ids"])
        st.session_state["fticr_matrix"] = data["fticr_matrix"]
        st.session_state["fticr_ids"] = list(data["fticr_ids"])
    else:
        st.session_state["ftir_matrix"] = np.empty((0, 1801), dtype=np.float32)
        st.session_state["ftir_ids"] = []
        st.session_state["eem_matrix"] = np.empty((0, 6923), dtype=np.float32)
        st.session_state["eem_ids"] = []
        st.session_state["fticr_matrix"] = np.empty((0, 1300), dtype=np.float32)
        st.session_state["fticr_ids"] = []

    st.session_state["library_loaded"] = True


def main():
    st.set_page_config(
        page_title="SpectraMatch — Multimodal Chemometric Identification",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_application_state()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <span style="font-size: 32px;">🔬</span>
                <div>
                    <h2 style="margin: 0; padding: 0; font-size: 20px; font-weight: 700; color: #0284C7;">SpectraMatch</h2>
                    <span style="font-size: 11px; color: #64748B;">v1.0.0 • Multimodal Chemometrics</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_choice = st.radio(
            "Разделы платформы",
            [
                "🔍 Search (Поиск и QC)",
                "📦 Batch (Пакетный скрининг)",
                "📋 Browse (Каталог спектров)",
                "⚖️ Compare (Парное сравнение)",
                "➕ Add (Пополнить библиотеку)",
                "🎛️ Simulator (Химическая песочница)",
                "📖 About (Теория и формулы)",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Статус библиотеки в RAM:**")
        counts = st.session_state.get("modality_counts", {})
        total_spectra = sum(counts.values())

        st.markdown(
            f"""
            - 🟢 **Всего записей:** `{total_spectra}`
            - 📈 **FTIR (1801 pt):** `{counts.get('FTIR', 0)}`
            - 🌊 **EEM (6923 pt):** `{counts.get('EEM', 0)}`
            - ⚛️ **FT-ICR MS (1300 pt):** `{counts.get('FTICR_MS', 0)}`
            """
        )

        st.markdown("---")
        st.caption(
            "Designed for Environmental Molecular Informatics, "
            "Natural Organic Matter (NOM/DOM), and Technical Lignins."
        )

    # Route navigation
    if "Search" in nav_choice:
        render_search_page()
    elif "Batch" in nav_choice:
        render_batch_page()
    elif "Browse" in nav_choice:
        render_browse_page()
    elif "Compare" in nav_choice:
        render_compare_page()
    elif "Add" in nav_choice:
        render_add_page()
    elif "Simulator" in nav_choice:
        render_generator_page()
    elif "About" in nav_choice:
        render_about_page()


if __name__ == "__main__":
    main()
