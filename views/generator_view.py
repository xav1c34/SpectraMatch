"""views/generator_view.py — Interactive synthetic spectrum modeling sandbox and simulator."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.models import ModalityEnum
from core.indices import compute_ftir_indices, compute_eem_indices, compute_fticr_indices
from core.synthetic import FTIR_BANDS, EEM_CENTERS
from views.plot_utils import create_preview_plot


def render_generator_page():
    st.markdown("## 🎛️ Интерактивная химическая песочница (Spectrum Generator)")
    st.markdown(
        "Моделируйте спектры с заданными физико-химическими свойствами. "
        "Управляйте функциональным составом в реальном времени, наблюдайте за динамикой хемометрических "
        "индексов и экспортируйте синтетические спектры для калибровки и обучения."
    )

    gen_modality = st.radio("Выберите модальность для симуляции:", ["FTIR", "EEM", "FT-ICR MS"], horizontal=True)

    if gen_modality == "FTIR":
        col_ctrl, col_chart = st.columns([1.2, 1.8])

        with col_ctrl:
            st.markdown("**Амплитуды аналитических полос (Гауссианы):**")
            h_oh = st.slider("Гидроксилы O-H (3380 см⁻¹)", 0.0, 1.0, 0.70, 0.05)
            h_aliph = st.slider("Алифатика -CH₂- / -CH₃ (2925/2850 см⁻¹)", 0.0, 1.0, 0.35, 0.05)
            h_carb = st.slider("Карбонилы C=O (1720 см⁻¹)", 0.0, 1.0, 0.45, 0.05)
            h_arom = st.slider("Ароматический каркас C=C (1610 см⁻¹)", 0.0, 1.0, 0.65, 0.05)
            h_lignin = st.slider("Лигниновое кольцо (1510 см⁻¹)", 0.0, 1.0, 0.20, 0.05)
            h_poly = st.slider("Полисахариды C-O (1035 см⁻¹)", 0.0, 1.0, 0.40, 0.05)
            noise_lvl = st.slider("Уровень шума детектора (σ)", 0.0, 0.05, 0.005, 0.005)

        # Generate spectrum
        wn = np.arange(400, 4002, 2, dtype=np.float64)
        spec = np.zeros_like(wn)

        # Mapping sliders to bands
        height_map = [h_oh, h_aliph, h_aliph * 0.6, h_carb, h_arom, h_lignin, h_aliph * 0.7, h_lignin * 1.2, h_poly]
        for (c, fwhm), h in zip(FTIR_BANDS, height_map):
            sig = fwhm / 2.35482
            spec += h * np.exp(-0.5 * ((wn - c) / sig) ** 2)

        rng = np.random.RandomState(42)
        spec += rng.normal(0, noise_lvl, len(spec))
        spec = np.maximum(spec, 0.0)

        indices = compute_ftir_indices(wn, spec)

        with col_chart:
            fig = create_preview_plot(wn, spec, ModalityEnum.FTIR, title="Моделируемый FTIR спектр")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Расчетные хемометрические индексы:")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Lignin Index (LI)", f"{indices['LI']:.3f}")
            m2.metric("Aromaticity (AI)", f"{indices['AI']:.3f}")
            m3.metric("Condensation (CI)", f"{indices['CI']:.3f}")
            m4.metric("Aliph / Arom", f"{indices['Aliph_Arom']:.3f}")

            # Download CSV
            df_ftir = pd.DataFrame({"wavenumber": wn, "absorbance": np.round(spec, 5)})
            st.download_button(
                "📥 Скачать сгенерированный FTIR (.csv)",
                data=df_ftir.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_ftir_model.csv",
                mime="text/csv",
            )

    elif gen_modality == "EEM":
        col_ctrl, col_chart = st.columns([1.2, 1.8])

        with col_ctrl:
            st.markdown("**Интенсивность флуорофоров Coble:**")
            w_a = st.slider("Peak A (UV Humic, Ex 250 / Em 440)", 0.0, 1.5, 1.0, 0.1)
            w_c = st.slider("Peak C (Visible Humic, Ex 330 / Em 450)", 0.0, 1.5, 0.85, 0.1)
            w_t = st.slider("Peak T (Tryptophan Protein, Ex 275 / Em 340)", 0.0, 1.5, 0.20, 0.1)
            w_m = st.slider("Peak M (Marine Humic, Ex 310 / Em 400)", 0.0, 1.5, 0.30, 0.1)

        ex = np.arange(240, 455, 5, dtype=np.float64)
        em = np.arange(280, 602, 2, dtype=np.float64)
        mesh_ex, mesh_em = np.meshgrid(ex, em, indexing="ij")

        mat = np.zeros_like(mesh_ex)
        fluor_weights = {"A": w_a, "C": w_c, "T": w_t, "M": w_m}
        for fn, w in fluor_weights.items():
            if w <= 0:
                continue
            cex, cem, sex, sem = EEM_CENTERS[fn]
            mat += w * np.exp(-0.5 * (((mesh_ex - cex) / sex) ** 2 + ((mesh_em - cem) / sem) ** 2))

        indices = compute_eem_indices(ex, em, mat)

        with col_chart:
            fig = create_preview_plot(ex, em, ModalityEnum.EEM, title="Моделируемая EEM матрица", matrix_2d=mat)
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Fluorescence Index (FI)", f"{indices['FI']:.2f}")
            m2.metric("Humification Index (HIX)", f"{indices['HIX']:.2f}")
            m3.metric("Biological Index (BIX)", f"{indices['BIX']:.2f}")

            df_eem = pd.DataFrame(mat.T, index=em, columns=ex)
            st.download_button(
                "📥 Скачать сгенерированную EEM матрицу (.csv)",
                data=df_eem.to_csv(index=True).encode("utf-8"),
                file_name="synthetic_eem_matrix.csv",
                mime="text/csv",
            )

    elif gen_modality == "FT-ICR MS":
        col_ctrl, col_chart = st.columns([1.2, 1.8])

        with col_ctrl:
            st.markdown("**Параметры молекулярно-массового распределения:**")
            center_m = st.slider("Центр распределения массы (m/z)", 250.0, 600.0, 420.0, 10.0)
            spread_m = st.slider("Ширина распределения (σ, Da)", 40.0, 200.0, 100.0, 10.0)
            n_peaks = st.slider("Число дискретных пиков", 100, 2000, 600, 100)

        rng = np.random.RandomState(42)
        raw_mz = rng.normal(center_m, spread_m, n_peaks * 2)
        raw_mz = raw_mz[(raw_mz >= 150.0) & (raw_mz <= 800.0)][:n_peaks]

        # Apply realistic defect
        mz_arr = []
        int_arr = []
        for m in raw_mz:
            defect = (m % 1.0) * 0.15 + 0.05
            actual = np.round(m) + defect
            envelope = np.exp(-0.5 * ((actual - center_m) / spread_m) ** 2)
            intensity = envelope * rng.lognormal(mean=2.0, sigma=0.6)
            mz_arr.append(actual)
            int_arr.append(intensity)

        mz_arr = np.array(mz_arr)
        int_arr = np.array(int_arr)
        sort_i = np.argsort(mz_arr)
        mz_arr = mz_arr[sort_i]
        int_arr = int_arr[sort_i]

        indices = compute_fticr_indices(mz_arr, int_arr)

        with col_chart:
            fig = create_preview_plot(mz_arr, int_arr, ModalityEnum.FTICR_MS, title="Моделируемый FT-ICR масс-спектр")
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Richness (Пиков)", f"{indices['Richness']}")
            m2.metric("Mw (Средневзвеш.)", f"{indices['Mw']:.1f} Da")
            m3.metric("Mn (Среднечисл.)", f"{indices['Mn']:.1f} Da")
            m4.metric("Полидисперсность (PD)", f"{indices['PD']:.2f}")

            df_ms = pd.DataFrame({"mz": np.round(mz_arr, 5), "intensity": np.round(int_arr, 2)})
            st.download_button(
                "📥 Скачать сгенерированный масс-спектр (.csv)",
                data=df_ms.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_fticr_mass_spectrum.csv",
                mime="text/csv",
            )
