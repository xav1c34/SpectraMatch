"""views/add_view.py — Interactive ingestion tool to append user-provided spectra to the active library."""
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

from core.models import ModalityEnum
from core.ingest import ingest_file
from db.database import insert_spectrum, get_all_spectra, get_modality_counts
from scripts.fetch_real_spectra import run_full_spectra_enrichment


ROOT_DIR = Path(__file__).resolve().parent.parent
SPECTRA_DIR = ROOT_DIR / "library" / "spectra"
NPZ_PATH = ROOT_DIR / "library" / "vectors.npz"


def render_add_page():
    st.markdown("## ➕ Добавить спектр в библиотеку (Add to Library)")
    st.markdown(
        "Пополните локальную базу данных вашим экспериментальным образцом. "
        "Спектр пройдет автоматическую каноническую нормализацию, расчет хемометрических индексов "
        "и будет мгновенно проиндексирован в оперативной памяти (RAM) для последующего векторного поиска."
    )

    with st.expander("🌐 Автоматическое пополнение из открытых научных источников (NIST, OpenFluor, WHONDRS)", expanded=False):
        st.markdown(
            "Нажмите кнопку ниже, чтобы автоматически выкачать сертифицированные ИК-спектры мономеров лигнина и гуминовых веществ "
            "из **NIST Chemistry WebBook** (в формате IUPAC JCAMP-DX), флуоресцентные компоненты **OpenFluor** "
            "и ультравысокоразрешающие пик-листы FT-ICR MS."
        )
        if st.button("🚀 Запустить автопополнение базы из NIST & OpenFluor"):
            with st.spinner("Загрузка и индексация эталонов..."):
                run_full_spectra_enrichment()
                # Reload application state
                db_conn = st.session_state.get("db_conn")
                if db_conn:
                    st.session_state["library_meta"] = get_all_spectra(db_conn)
                    st.session_state["modality_counts"] = get_modality_counts(db_conn)
                if NPZ_PATH.exists():
                    data = np.load(NPZ_PATH, allow_pickle=True)
                    st.session_state["ftir_matrix"] = data["ftir_matrix"]
                    st.session_state["ftir_ids"] = list(data["ftir_ids"])
                    st.session_state["eem_matrix"] = data["eem_matrix"]
                    st.session_state["eem_ids"] = list(data["eem_ids"])
                    st.session_state["fticr_matrix"] = data["fticr_matrix"]
                    st.session_state["fticr_ids"] = list(data["fticr_ids"])
                st.success("🎉 База успешно обогащена и переиндексирована в памяти!")

    with st.form("add_spectrum_form"):
        col1, col2 = st.columns(2)

        with col1:
            spec_id = st.text_input("Уникальный ID образца", placeholder="например, ftir-yenisei-fa-01")
            spec_name = st.text_input("Название образца", placeholder="например, Yenisei River Fulvic Acid")
            modality = st.selectbox("Модальность", ["FTIR", "EEM", "FTICR_MS"])
            sample_type = st.selectbox(
                "Тип природного вещества",
                [
                    "fulvic_acid",
                    "humic_acid",
                    "lignin",
                    "sludge_lignin",
                    "polysaccharide",
                    "lake_dom",
                    "nom",
                    "protein",
                    "other",
                ],
            )

        with col2:
            origin = st.text_input("Происхождение / Геолокация", placeholder="например, Yenisei River, 10m depth, pristine water")
            source_doi = st.text_input("Источник / DOI публикации", placeholder="например, 10.1016/j.apgeochem.2024.101112")
            uploaded_file = st.file_uploader("Спектральный файл (.csv, .dpt, .txt, .jdx)", type=["csv", "dpt", "txt", "jdx", "tsv"])

        submitted = st.form_submit_button("💾 Проиндексировать и добавить в базу", type="primary")

    if submitted:
        if not spec_id or not spec_name or uploaded_file is None:
            st.error("Пожалуйста, заполните все обязательные поля (ID, Название) и загрузите файл.")
            return

        db_conn = st.session_state.get("db_conn")
        if db_conn is None:
            st.error("База данных недоступна.")
            return

        # Clean ID
        clean_id = spec_id.strip().lower().replace(" ", "-")

        # Ingest file
        try:
            rec = ingest_file(
                uploaded_file.getvalue(),
                filename=uploaded_file.name,
                modality_hint=ModalityEnum(modality),
            )
        except Exception as e:
            st.error(f"Ошибка при обработке спектра: {str(e)}")
            return

        if not rec.quality.passed:
            st.error("Образец не прошел проверку качества QC. Исправьте ошибки в спектре перед добавлением.")
            for err in rec.quality.errors:
                st.write(f"- {err}")
            return

        # Save raw spectrum to library/spectra/
        sub_dir = SPECTRA_DIR / modality.lower().replace("_", "")
        sub_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sub_dir / f"{clean_id}.csv"

        if rec.matrix_2d is not None:
            df_out = pd.DataFrame(rec.matrix_2d.T, index=rec.y_axis, columns=rec.x_axis)
            df_out.to_csv(csv_path, index=True)
        else:
            df_out = pd.DataFrame({"x": rec.x_axis, "y": rec.y_axis})
            df_out.to_csv(csv_path, index=False)

        # Insert into SQLite
        rec_dict = {
            "id": clean_id,
            "name": spec_name.strip(),
            "modality": modality,
            "sample_type": sample_type,
            "origin": origin.strip(),
            "source_doi": source_doi.strip(),
            "indices": rec.indices,
            "metadata": {
                "file_path": str(csv_path.relative_to(ROOT_DIR)),
                "added_by": "user",
            },
        }
        insert_spectrum(db_conn, rec_dict)

        # Update in-memory session state and save vectors.npz
        if modality == "FTIR":
            curr_mat = st.session_state.get("ftir_matrix", np.empty((0, 1801), dtype=np.float32))
            curr_ids = st.session_state.get("ftir_ids", [])
            st.session_state["ftir_matrix"] = np.vstack([curr_mat, rec.canonical_vector])
            curr_ids.append(clean_id)
            st.session_state["ftir_ids"] = curr_ids
        elif modality == "EEM":
            curr_mat = st.session_state.get("eem_matrix", np.empty((0, 6923), dtype=np.float32))
            curr_ids = st.session_state.get("eem_ids", [])
            st.session_state["eem_matrix"] = np.vstack([curr_mat, rec.canonical_vector])
            curr_ids.append(clean_id)
            st.session_state["eem_ids"] = curr_ids
        elif modality == "FTICR_MS":
            curr_mat = st.session_state.get("fticr_matrix", np.empty((0, 1300), dtype=np.float32))
            curr_ids = st.session_state.get("fticr_ids", [])
            st.session_state["fticr_matrix"] = np.vstack([curr_mat, rec.canonical_vector])
            curr_ids.append(clean_id)
            st.session_state["fticr_ids"] = curr_ids

        # Update vectors.npz on disk
        np.savez_compressed(
            NPZ_PATH,
            ftir_matrix=st.session_state["ftir_matrix"],
            ftir_ids=np.array(st.session_state["ftir_ids"]),
            eem_matrix=st.session_state["eem_matrix"],
            eem_ids=np.array(st.session_state["eem_ids"]),
            fticr_matrix=st.session_state["fticr_matrix"],
            fticr_ids=np.array(st.session_state["fticr_ids"]),
        )

        # Refresh catalog metadata & counts
        st.session_state["library_meta"] = get_all_spectra(db_conn)
        st.session_state["modality_counts"] = get_modality_counts(db_conn)

        st.success(f"🎉 Спектр **{spec_name}** ({clean_id}) успешно сохранен и проиндексирован!")
        st.markdown(
            f"""
            - **Модальность:** `{modality}`
            - **QC Score:** `{rec.quality.score:.2f}` (Валидация пройдена)
            - **Число точек:** `{len(rec.x_axis)}`
            - **Канонический размер вектора:** `{len(rec.canonical_vector)}`
            """
        )
