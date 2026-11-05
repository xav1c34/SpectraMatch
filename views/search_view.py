"""views/search_view.py — Main search tab: upload, autodetect, QC, vector search, overlay visualization, and export."""
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

from core.models import ModalityEnum, MetricEnum
from core.ingest import ingest_file
from core.similarity import search
from core.chemspace import fit_library_pca, project_query_to_pca, create_pca_scatter_plot
from core.kendrick import compute_kendrick_mass_defect, create_kendrick_plot
from views.plot_utils import (
    create_preview_plot,
    create_ftir_overlay_plot,
    create_eem_comparison_plot,
    create_fticr_mirror_plot,
)
from views.export_utils import (
    export_matches_to_csv,
    export_passport_to_json,
    export_standalone_html,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT_DIR / "user_spectra"


def render_search_page():
    st.markdown("## 🔍 Спектральный поиск и идентификация (Search)")
    st.markdown(
        "Загрузите спектральный файл образца неизвестного происхождения (DOM / NOM) "
        "для автоматической предобработки, валидации качества (QC) и сопоставления с эталонной базой."
    )

    # Demo quick-loader bar
    with st.expander("⚡ Быстрый тест: Загрузить готовый демонстрационный образец", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("📁 FTIR Suwannee River"):
            demo_file = DEMO_DIR / "Test_FTIR_Suwannee_River.csv"
            if demo_file.exists():
                st.session_state["demo_bytes"] = demo_file.read_bytes()
                st.session_state["demo_name"] = "Test_FTIR_Suwannee_River.csv"
        if c2.button("📁 FTIR Kraft Lignin (.dpt)"):
            demo_file = DEMO_DIR / "Test_FTIR_Kraft_Lignin.dpt"
            if demo_file.exists():
                st.session_state["demo_bytes"] = demo_file.read_bytes()
                st.session_state["demo_name"] = "Test_FTIR_Kraft_Lignin.dpt"
        if c3.button("📁 EEM Lake Baikal DOM"):
            demo_file = DEMO_DIR / "Test_EEM_Lake_DOM.csv"
            if demo_file.exists():
                st.session_state["demo_bytes"] = demo_file.read_bytes()
                st.session_state["demo_name"] = "Test_EEM_Lake_DOM.csv"
        if c4.button("📁 FT-ICR MS Lignin (EU format)"):
            demo_file = DEMO_DIR / "Test_FTICR_Baikal_Pulp_Lignin.csv"
            if demo_file.exists():
                st.session_state["demo_bytes"] = demo_file.read_bytes()
                st.session_state["demo_name"] = "Test_FTICR_Baikal_Pulp_Lignin.csv"

    # Search Configuration Controls
    col_upload, col_settings = st.columns([1.8, 1.2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Перетащите спектральный файл сюда",
            type=["csv", "dpt", "txt", "jdx", "tsv", "xy"],
            help="Поддерживаются 1D спектры (FTIR, MS) и 2D матрицы флуоресценции (EEM)",
        )

    # Handle uploaded file or demo selection
    file_bytes = None
    filename = "spectrum.csv"

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif "demo_bytes" in st.session_state:
        file_bytes = st.session_state["demo_bytes"]
        filename = st.session_state["demo_name"]
        st.info(f"Загружен демонстрационный файл: **{filename}**")

    with col_settings:
        modality_option = st.selectbox(
            "Модальность",
            ["🔄 Auto-detect", "FTIR", "EEM", "FT-ICR MS"],
            index=0,
            help="Автоопределение по диапазону осей и структуре данных или явное указание",
        )
        metric_option = st.selectbox(
            "Метрика сходства",
            ["Auto (рекомендуемая)", "Cosine", "Pearson", "Euclidean", "TCC"],
            index=0,
            help="Pearson для FTIR (инвариантен к сдвигу), TCC для EEM, Cosine для масс-спектров",
        )
        top_k = st.slider("Количество результатов (Top-K)", min_value=1, max_value=25, value=5)

    if file_bytes is None:
        st.markdown(
            """
            > [!NOTE]
            > Загрузите файл спектрометра или выберите демонстрационный файл выше, чтобы начать идентификацию.
            """
        )
        return

    # Map selected options to Enums
    modality_hint = None
    if modality_option == "FTIR":
        modality_hint = ModalityEnum.FTIR
    elif modality_option == "EEM":
        modality_hint = ModalityEnum.EEM
    elif modality_option == "FT-ICR MS":
        modality_hint = ModalityEnum.FTICR_MS

    metric_enum = MetricEnum.AUTO
    if metric_option == "Cosine":
        metric_enum = MetricEnum.COSINE
    elif metric_option == "Pearson":
        metric_enum = MetricEnum.PEARSON
    elif metric_option == "Euclidean":
        metric_enum = MetricEnum.EUCLIDEAN
    elif metric_option == "TCC":
        metric_enum = MetricEnum.TCC

    # Parse & Ingest Spectrum
    with st.spinner("Анализ, предобработка и контроль качества спектра..."):
        try:
            record = ingest_file(file_bytes=file_bytes, filename=filename, modality_hint=modality_hint)
        except Exception as e:
            st.error(f"❌ Ошибка разбора спектрального файла: {str(e)}")
            return

    st.session_state["last_query_record"] = record

    # Quality Control & Preview Section
    st.markdown("---")
    st.subheader("📊 Аналитический паспорт запроса и контроль качества (QC)")

    qc_col, meta_col = st.columns([1.5, 2.5])

    with qc_col:
        score = record.quality.score
        if score >= 0.85 and record.quality.passed:
            st.success(f"🟢 **Quality Score: {score:.2f} — Валидация пройдена**")
        elif score >= 0.50 and record.quality.passed:
            st.warning(f"🟡 **Quality Score: {score:.2f} — Предупреждения QC**")
        else:
            st.error(f"🔴 **Quality Score: {score:.2f} — Ошибка качества (QC Failed)**")

        st.markdown(
            f"""
            - **Модальность:** `{record.modality.value}`
            - **Покрытие диапазона:** `{record.quality.coverage_pct}%`
            - **SNR:** `{record.quality.snr_estimate if record.quality.snr_estimate is not None else 'N/A'}`
            - **Точек данных:** `{len(record.x_axis)}`
            """
        )

        if record.quality.warnings:
            with st.expander("⚠️ Предупреждения QC"):
                for w in record.quality.warnings:
                    st.write(f"- {w}")

        if record.quality.errors:
            with st.expander("❌ Критические ошибки"):
                for err in record.quality.errors:
                    st.write(f"- {err}")

    with meta_col:
        # Show mini indices summary
        if record.indices:
            idx_cols = st.columns(min(len(record.indices), 4))
            for i, (k, v) in enumerate(list(record.indices.items())[:4]):
                idx_cols[i % len(idx_cols)].metric(label=k, value=v)

    # Preview plot of the query
    with st.expander("📈 График загруженного спектра запроса", expanded=True):
        preview_fig = create_preview_plot(
            x=record.x_axis,
            y=record.y_axis,
            modality=record.modality,
            title=f"Query: {record.name} ({record.modality.value})",
            matrix_2d=record.matrix_2d,
        )
        st.plotly_chart(preview_fig, use_container_width=True)

    if not record.quality.passed:
        st.error("Поиск остановлен: спектр не прошел критическую валидацию QC.")
        return

    # In-memory Vector Search Execution
    mod = record.modality
    lib_matrix = None
    lib_meta = []

    if mod == ModalityEnum.FTIR:
        lib_matrix = st.session_state.get("ftir_matrix")
        ids = st.session_state.get("ftir_ids", [])
    elif mod == ModalityEnum.EEM:
        lib_matrix = st.session_state.get("eem_matrix")
        ids = st.session_state.get("eem_ids", [])
    elif mod == ModalityEnum.FTICR_MS:
        lib_matrix = st.session_state.get("fticr_matrix")
        ids = st.session_state.get("fticr_ids", [])

    db_meta_dict = {m["id"]: m for m in st.session_state.get("library_meta", [])}
    if ids is not None:
        lib_meta = [db_meta_dict.get(spec_id, {"id": spec_id, "name": spec_id}) for spec_id in ids]

    if lib_matrix is None or len(lib_matrix) == 0:
        st.error("Библиотека векторов пуста. Выполните 'python library/seed.py'.")
        return

    results = search(
        query_vec=record.canonical_vector,
        library_matrix=lib_matrix,
        library_meta=lib_meta,
        modality=mod,
        metric=metric_enum,
        top_k=top_k,
    )

    st.session_state["last_search_results"] = results

    # Results Section
    st.markdown("---")
    st.subheader(f"🎯 Результаты сопоставления (Top-{len(results)} совпадений)")

    if not results:
        st.warning("Совпадений в библиотеке не обнаружено.")
        return

    top_result = results[0]

    # Show top match summary card
    score_pct = max(top_result.score * 100.0, 0.0)
    st.success(
        f"🏆 **Лучшее совпадение (#1): {top_result.name}** | "
        f"Метрика: `{top_result.metric}` = **{top_result.score:.4f}** ({score_pct:.1f}% сходство)"
    )

    # Detailed overlay visualization for the top match
    if mod == ModalityEnum.FTIR:
        overlay_fig = create_ftir_overlay_plot(
            query_vec=record.canonical_vector,
            match_vec=top_result.canonical_vector,
            match_name=top_result.name,
            score=top_result.score,
            metric_name=top_result.metric,
        )
    elif mod == ModalityEnum.EEM:
        overlay_fig = create_eem_comparison_plot(
            query_vec=record.canonical_vector,
            match_vec=top_result.canonical_vector,
            match_name=top_result.name,
            score=top_result.score,
            metric_name=top_result.metric,
        )
    elif mod == ModalityEnum.FTICR_MS:
        overlay_fig = create_fticr_mirror_plot(
            query_vec=record.canonical_vector,
            match_vec=top_result.canonical_vector,
            match_name=top_result.name,
            score=top_result.score,
            metric_name=top_result.metric,
        )

    st.plotly_chart(overlay_fig, use_container_width=True)

    # Side-by-side Chemometric Indices Comparison
    with st.expander("📋 Сравнение хемометрических индексов (Query vs Match)", expanded=True):
        if record.indices and top_result.indices:
            idx_names = sorted(list(set(record.indices.keys()).intersection(top_result.indices.keys())))
            idx_rows = []
            for name in idx_names:
                q_val = record.indices.get(name, 0.0)
                m_val = top_result.indices.get(name, 0.0)
                delta = round(q_val - m_val, 4)
                idx_rows.append(
                    {
                        "Хемометрический индекс": name,
                        "Query (Ваш образец)": q_val,
                        "Best Match (Библиотека)": m_val,
                        "Разность (Δ)": delta,
                    }
                )
            st.dataframe(pd.DataFrame(idx_rows), use_container_width=True)

    # Top-K Matches Ranked Table
    st.markdown("#### Полная таблица кандидатов Top-K")
    table_rows = [
        {
            "Rank": r.rank,
            "Sample Name": r.name,
            "Type": r.sample_type,
            "Score": f"{r.score:.4f}",
            "Metric": r.metric,
            "Origin": r.origin,
            "DOI": r.source_doi or "—",
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Advanced Chemometrics Tabs
    with st.expander("🌟 Карта химического пространства (PCA Chemical Space Mapping)", expanded=False):
        st.markdown(
            "Проекция спектрального запроса на плоскость главных компонент эталонной базы. "
            "Позволяет визуально определить принадлежность неизвестного образца к макроклассам органики."
        )
        pca_model = fit_library_pca(lib_matrix, modality=mod, n_components=2)
        if pca_model is not None:
            q_coords = project_query_to_pca(record.canonical_vector, pca_model)
            pca_fig = create_pca_scatter_plot(
                pca_model=pca_model,
                library_meta=lib_meta,
                query_coords=q_coords,
                query_name=record.name,
            )
            st.plotly_chart(pca_fig, use_container_width=True)
        else:
            st.info("Недостаточно образцов в библиотеке для расчета PCA.")

    # Kendrick Mass Defect for FT-ICR MS
    if mod == ModalityEnum.FTICR_MS:
        with st.expander("🔬 Диаграмма дефекта массы Кендрика (KMD Analysis)", expanded=False):
            st.markdown(
                "Анализ дефекта массы Кендрика для гомологических серий ($-\\text{CH}_2-$, $\\Delta m = 14.01565\\text{ Да}$). "
                "Горизонтальные серии точек отражают алкилированные гомологи."
            )
            df_kmd = compute_kendrick_mass_defect(record.x_axis, record.y_axis, base="CH2")
            kmd_fig = create_kendrick_plot(df_kmd, title=f"KMD Plot (CH₂) — {record.name}")
            st.plotly_chart(kmd_fig, use_container_width=True)

    # Export Section
    st.markdown("---")
    st.subheader("📥 Экспорт результатов и формирование отчетов")
    exp_c1, exp_c2, exp_c3 = st.columns(3)

    csv_data = export_matches_to_csv(results)
    exp_c1.download_button(
        label="📊 Скачать CSV сводку",
        data=csv_data,
        file_name=f"spectramatch_matches_{record.id}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    json_data = export_passport_to_json(record, results)
    exp_c2.download_button(
        label="📄 Скачать JSON-паспорт",
        data=json_data,
        file_name=f"spectral_passport_{record.id}.json",
        mime="application/json",
        use_container_width=True,
    )

    html_report = export_standalone_html(record, top_result, overlay_fig)
    exp_c3.download_button(
        label="🌐 Скачать автономный HTML-отчет",
        data=html_report,
        file_name=f"spectral_report_{record.id}.html",
        mime="text/html",
        use_container_width=True,
    )
