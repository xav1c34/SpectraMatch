"""views/browse_view.py — Catalog browser tab: filter, inspect spectra, check indices, and export raw data."""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from core.models import ModalityEnum
from db.database import get_all_spectra
from views.plot_utils import create_preview_plot


ROOT_DIR = Path(__file__).resolve().parent.parent


def render_browse_page():
    st.markdown("## 📋 Каталог эталонных спектров (Browse)")
    st.markdown(
        "Исследуйте базу эталонных спектров природного органического вещества: "
        "стандарты IHSS, лигнины, полисахариды, байкальские пробы и синтетические модели."
    )

    db_conn = st.session_state.get("db_conn")
    if db_conn is None:
        st.error("База данных не подключена.")
        return

    # Filter controls
    c_mod, c_type, c_search = st.columns([1, 1, 1.5])
    with c_mod:
        modality_filter = st.selectbox(
            "Фильтр по модальности",
            ["All", "FTIR", "EEM", "FTICR_MS"],
            index=0,
        )
    with c_type:
        sample_types = [
            "All",
            "fulvic_acid",
            "humic_acid",
            "lignin",
            "sludge_lignin",
            "polysaccharide",
            "lake_dom",
            "nom",
            "protein",
        ]
        type_filter = st.selectbox("Фильтр по типу вещества", sample_types, index=0)

    with c_search:
        search_query = st.text_input("Поиск по названию или происхождению", placeholder="например, Suwannee или Baikal")

    # Query spectra
    spectra_list = get_all_spectra(db_conn, modality=modality_filter, sample_type=type_filter)

    if search_query:
        q = search_query.lower()
        spectra_list = [
            s
            for s in spectra_list
            if q in s["name"].lower() or q in (s.get("origin") or "").lower() or q in s["id"].lower()
        ]

    st.markdown(f"**Найдено записей:** `{len(spectra_list)}`")

    if not spectra_list:
        st.info("Нет записей, удовлетворяющих заданным критериям фильтрации.")
        return

    # Display catalog table
    df_catalog = pd.DataFrame(
        [
            {
                "ID": s["id"],
                "Название образца": s["name"],
                "Модальность": s["modality"],
                "Тип образца": s["sample_type"],
                "Происхождение": s.get("origin") or "—",
                "Источник (DOI)": s.get("source_doi") or "—",
            }
            for s in spectra_list
        ]
    )
    st.dataframe(df_catalog, use_container_width=True, hide_index=True)

    # Detailed Spectrum Inspector
    st.markdown("---")
    st.subheader("🔬 Инспектор выбранного образца")

    spec_names = [f"{s['id']} — {s['name']}" for s in spectra_list]
    selected_idx = st.selectbox("Выберите спектр для детального просмотра:", range(len(spec_names)), format_func=lambda i: spec_names[i])

    selected_spec = spectra_list[selected_idx]

    # Load spectrum file from library/spectra/
    file_rel = selected_spec.get("metadata", {}).get("file_path")
    csv_bytes = None
    df_raw = None

    if file_rel:
        file_path = ROOT_DIR / file_rel
        if file_path.exists():
            csv_bytes = file_path.read_bytes()
            try:
                df_raw = pd.read_csv(file_path, index_col=0 if selected_spec["modality"] == "EEM" else None)
            except Exception:
                df_raw = None

    col_meta, col_idx = st.columns([1.5, 1.5])
    with col_meta:
        st.markdown(
            f"""
            - **ID:** `{selected_spec['id']}`
            - **Модальность:** `{selected_spec['modality']}`
            - **Тип:** `{selected_spec['sample_type']}`
            - **Происхождение:** {selected_spec.get('origin') or '—'}
            - **DOI:** `{selected_spec.get('source_doi') or '—'}`
            """
        )
    with col_idx:
        indices = selected_spec.get("indices", {})
        if indices:
            st.markdown("**Хемометрические индексы:**")
            idx_df = pd.DataFrame([{"Индекс": k, "Значение": v} for k, v in indices.items()])
            st.dataframe(idx_df, use_container_width=True, hide_index=True)

    # Plot spectrum
    if df_raw is not None:
        mod = ModalityEnum(selected_spec["modality"])
        if mod == ModalityEnum.FTIR:
            fig = create_preview_plot(
                x=df_raw.iloc[:, 0].values,
                y=df_raw.iloc[:, 1].values,
                modality=mod,
                title=f"{selected_spec['name']} (FTIR)",
            )
        elif mod == ModalityEnum.EEM:
            ex = pd.to_numeric(df_raw.columns, errors="coerce").values
            em = pd.to_numeric(df_raw.index, errors="coerce").values
            mat_2d = df_raw.values
            fig = create_preview_plot(
                x=ex,
                y=em,
                modality=mod,
                title=f"{selected_spec['name']} (EEM)",
                matrix_2d=mat_2d,
            )
        elif mod == ModalityEnum.FTICR_MS:
            fig = create_preview_plot(
                x=df_raw.iloc[:, 0].values,
                y=df_raw.iloc[:, 1].values,
                modality=mod,
                title=f"{selected_spec['name']} (FT-ICR MS)",
            )

        st.plotly_chart(fig, use_container_width=True)

    # Download Button
    if csv_bytes:
        st.download_button(
            label=f"📥 Скачать спектр ({selected_spec['id']}.csv)",
            data=csv_bytes,
            file_name=f"{selected_spec['id']}.csv",
            mime="text/csv",
        )
