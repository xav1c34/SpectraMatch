"""views/compare_view.py — Direct 1-to-1 comparison between two user spectra or library entries."""
import streamlit as st
import pandas as pd

from core.models import ModalityEnum
from core.ingest import ingest_file
from core.similarity import compare_two
from views.plot_utils import (
    create_ftir_overlay_plot,
    create_eem_comparison_plot,
    create_fticr_mirror_plot,
)


def render_compare_page():
    st.markdown("## ⚖️ Прямое парное сравнение (Compare)")
    st.markdown(
        "Загрузите два спектра одинаковой модальности для вычисления полного набора метрик сходства "
        "(Cosine, Pearson, Euclidean, TCC), построения совместного графика и дифференциального анализа индексов."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Спектр A (Образец 1)")
        file_a = st.file_uploader("Загрузите первый файл", type=["csv", "dpt", "txt", "jdx", "tsv"], key="file_a")

    with col_b:
        st.subheader("Спектр B (Образец 2)")
        file_b = st.file_uploader("Загрузите второй файл", type=["csv", "dpt", "txt", "jdx", "tsv"], key="file_b")

    if file_a is None or file_b is None:
        st.info("Пожалуйста, загрузите оба спектральных файла для выполнения сравнительного анализа.")
        return

    # Ingest both files
    try:
        rec_a = ingest_file(file_a.getvalue(), filename=file_a.name)
        rec_b = ingest_file(file_b.getvalue(), filename=file_b.name)
    except Exception as e:
        st.error(f"Ошибка при разборе файлов: {str(e)}")
        return

    # Validate Modalities match
    if rec_a.modality != rec_b.modality:
        st.error(
            f"❌ Несовпадение модальностей: Спектр A имеет тип `{rec_a.modality.value}`, "
            f"а Спектр B — `{rec_b.modality.value}`. Прямое сопоставление возможно только в рамках одной модальности."
        )
        return

    mod = rec_a.modality

    # Compute comparison
    res = compare_two(
        vec_a=rec_a.canonical_vector,
        vec_b=rec_b.canonical_vector,
        modality=mod,
        indices_a=rec_a.indices,
        indices_b=rec_b.indices,
    )

    st.markdown("---")
    st.subheader(f"📊 Метрики сходства ({mod.value})")

    # Display Metrics in metric cards
    if mod == ModalityEnum.EEM:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tucker Congruence (TCC)", f"{res.tcc:.4f}" if res.tcc is not None else "—")
        m2.metric("Pearson Correlation (r)", f"{res.pearson:.4f}")
        m3.metric("Cosine Similarity", f"{res.cosine:.4f}")
        m4.metric("Euclidean Similarity", f"{res.euclidean:.4f}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson Correlation (r)", f"{res.pearson:.4f}")
        m2.metric("Cosine Similarity", f"{res.cosine:.4f}")
        m3.metric("Euclidean Similarity", f"{res.euclidean:.4f}")

    # Overlay Visualization
    st.markdown("### 📈 Совместный спектральный график")
    if mod == ModalityEnum.FTIR:
        fig = create_ftir_overlay_plot(
            query_vec=rec_a.canonical_vector,
            match_vec=rec_b.canonical_vector,
            match_name=f"B: {rec_b.name}",
            score=res.pearson,
            metric_name="Pearson",
        )
    elif mod == ModalityEnum.EEM:
        fig = create_eem_comparison_plot(
            query_vec=rec_a.canonical_vector,
            match_vec=rec_b.canonical_vector,
            match_name=f"B: {rec_b.name}",
            score=res.tcc or res.cosine,
            metric_name="TCC" if res.tcc else "Cosine",
        )
    elif mod == ModalityEnum.FTICR_MS:
        fig = create_fticr_mirror_plot(
            query_vec=rec_a.canonical_vector,
            match_vec=rec_b.canonical_vector,
            match_name=f"B: {rec_b.name}",
            score=res.cosine,
            metric_name="Cosine",
        )

    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side Index Comparison
    if rec_a.indices and rec_b.indices:
        st.markdown("### 📋 Дифференциальный анализ хемометрических индексов")
        all_keys = sorted(list(set(rec_a.indices.keys()).union(rec_b.indices.keys())))
        idx_diff = []
        for k in all_keys:
            val_a = rec_a.indices.get(k, 0.0)
            val_b = rec_b.indices.get(k, 0.0)
            idx_diff.append(
                {
                    "Индекс": k,
                    f"Спектр A ({rec_a.name})": val_a,
                    f"Спектр B ({rec_b.name})": val_b,
                    "Разность (Δ = A - B)": round(val_a - val_b, 4),
                }
            )
        st.dataframe(pd.DataFrame(idx_diff), use_container_width=True)
