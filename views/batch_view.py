"""views/batch_view.py — Multi-file batch screening, batch QC, and batch report generator."""
import io
import zipfile
from typing import Any, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.models import ModalityEnum, MetricEnum
from core.ingest import ingest_file
from core.similarity import search


def render_batch_page():
    st.markdown("## 📦 Пакетный скрининг спектров (Batch Screening)")
    st.markdown(
        "Загрузите серию файлов или ZIP-архив с экспедиционными образцами для автоматической "
        "пакетной предобработки, оценки QC, классификации и генерации сводного отчета."
    )

    uploaded_files = st.file_uploader(
        "Выберите спектральные файлы для пакетной обработки (мультивыбор)",
        type=["csv", "dpt", "txt", "jdx", "tsv", "zip"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Пожалуйста, выберите один или несколько файлов для запуска пакетного скрининга.")
        return

    # Unpack ZIP if present, or collect files
    file_list = []
    for uf in uploaded_files:
        if uf.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(uf.getvalue())) as zf:
                    for zname in zf.namelist():
                        if any(zname.lower().endswith(ext) for ext in [".csv", ".dpt", ".txt", ".jdx"]):
                            file_list.append((zname, zf.read(zname)))
            except Exception as e:
                st.error(f"Ошибка при распаковке ZIP {uf.name}: {e}")
        else:
            file_list.append((uf.name, uf.getvalue()))

    if not file_list:
        st.warning("В загруженных файлах не найдено подходящих спектральных таблиц.")
        return

    st.write(f"Найдено спектральных файлов в пакете: **{len(file_list)}**")

    metric_choice = st.selectbox(
        "Метрика сопоставления",
        ["Auto (рекомендуемая)", "Cosine", "Pearson", "Euclidean", "TCC"],
        index=0,
    )
    metric_enum = MetricEnum.AUTO
    if metric_choice == "Cosine":
        metric_enum = MetricEnum.COSINE
    elif metric_choice == "Pearson":
        metric_enum = MetricEnum.PEARSON
    elif metric_choice == "Euclidean":
        metric_enum = MetricEnum.EUCLIDEAN
    elif metric_choice == "TCC":
        metric_enum = MetricEnum.TCC

    if not st.button("🚀 Запустить пакетную обработку", type="primary"):
        return

    # Execute Batch Pipeline with progress bar
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    batch_rows = []
    db_meta_dict = {m["id"]: m for m in st.session_state.get("library_meta", [])}

    for idx, (fname, fbytes) in enumerate(file_list):
        status_text.text(f"Обработка {idx + 1}/{len(file_list)}: {fname}")
        progress_bar.progress((idx + 1) / len(file_list))

        try:
            rec = ingest_file(fbytes, filename=fname)
        except Exception as e:
            batch_rows.append(
                {
                    "Filename": fname,
                    "Modality": "Error",
                    "QC Score": 0.0,
                    "QC Status": "Failed",
                    "Top Match": f"Parse Error: {str(e)[:30]}",
                    "Score": 0.0,
                    "Class": "Error",
                    "Key Index": "—",
                }
            )
            continue

        # Search against library
        mod = rec.modality
        if mod == ModalityEnum.FTIR:
            lib_matrix = st.session_state.get("ftir_matrix")
            ids = st.session_state.get("ftir_ids", [])
        elif mod == ModalityEnum.EEM:
            lib_matrix = st.session_state.get("eem_matrix")
            ids = st.session_state.get("eem_ids", [])
        else:
            lib_matrix = st.session_state.get("fticr_matrix")
            ids = st.session_state.get("fticr_ids", [])

        lib_meta = [db_meta_dict.get(spec_id, {"id": spec_id, "name": spec_id}) for spec_id in ids]

        results = search(
            query_vec=rec.canonical_vector,
            library_matrix=lib_matrix,
            library_meta=lib_meta,
            modality=mod,
            metric=metric_enum,
            top_k=1,
        )

        top_match_name = results[0].name if results else "No Match"
        top_score = results[0].score if results else 0.0
        top_type = results[0].sample_type if results else "unknown"

        # Representative key index per modality
        key_idx_str = "—"
        if mod == ModalityEnum.FTIR and "LI" in rec.indices:
            key_idx_str = f"LI={rec.indices['LI']:.3f}"
        elif mod == ModalityEnum.EEM and "FI" in rec.indices:
            key_idx_str = f"FI={rec.indices['FI']:.2f}"
        elif mod == ModalityEnum.FTICR_MS and "Mw" in rec.indices:
            key_idx_str = f"Mw={rec.indices['Mw']:.1f}"

        qc_status = "🟢 Passed" if rec.quality.passed and rec.quality.score >= 0.85 else ("🟡 Warn" if rec.quality.passed else "🔴 Failed")

        batch_rows.append(
            {
                "Filename": fname,
                "Modality": mod.value,
                "QC Score": rec.quality.score,
                "QC Status": qc_status,
                "Top Match": top_match_name,
                "Similarity Score": top_score,
                "Identified Class": top_type,
                "Key Index": key_idx_str,
            }
        )

    progress_bar.empty()
    status_text.empty()

    df_batch = pd.DataFrame(batch_rows)

    # Metrics Summary
    st.markdown("### 📊 Сводные результаты серии")
    c1, c2, c3, c4 = st.columns(4)
    total_count = len(df_batch)
    passed_qc = sum(1 for r in batch_rows if "Passed" in str(r["QC Status"]))
    mean_sim = float(df_batch["Similarity Score"].mean()) if total_count > 0 else 0.0

    c1.metric("Всего образцов", total_count)
    c2.metric("Успешно прошли QC", f"{passed_qc} ({passed_qc/total_count*100:.0f}%)")
    c3.metric("Среднее сходство", f"{mean_sim:.4f}")
    c4.metric("Модальностей в серии", df_batch["Modality"].nunique())

    # Visual Distribution of Identified Classes
    if "Identified Class" in df_batch:
        class_counts = df_batch["Identified Class"].value_counts().reset_index()
        class_counts.columns = ["Класс вещества", "Количество образцов"]
        fig_pie = px.pie(
            class_counts,
            names="Класс вещества",
            values="Количество образцов",
            title="Распределение идентифицированных химических классов",
            color_discrete_sequence=px.colors.qualitative.Prism,
            hole=0.4,
        )
        fig_pie.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Table
    st.dataframe(df_batch, use_container_width=True)

    # CSV Download Button
    csv_bytes = df_batch.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Скачать полный сводный отчет серии (CSV)",
        data=csv_bytes,
        file_name="spectramatch_batch_screening_report.csv",
        mime="text/csv",
    )
