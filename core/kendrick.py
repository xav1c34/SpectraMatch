"""core/kendrick.py — Kendrick Mass Defect (KMD) analysis for FT-ICR mass spectrometry."""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Exact IUPAC masses for Kendrick homologous bases
KENDRICK_BASES = {
    "CH2": 14.01565,    # Methylene homologous series (standard)
    "COO": 43.98983,    # Carboxyl / decarboxylation series
    "H2": 2.01565,      # Degree of unsaturation / hydrogenation
    "O": 15.99491,      # Oxidation series
}


def compute_kendrick_mass_defect(
    mz: np.ndarray,
    intensity: np.ndarray,
    base: str = "CH2",
) -> pd.DataFrame:
    """Calculate Kendrick Mass (KM), Nominal Kendrick Mass (NKM), and KMD for discrete peaks.

    KM = (m/z) * (Nominal_Base / Exact_Base)
    NKM = round(KM)
    KMD = NKM - KM
    """
    mz_arr = np.asarray(mz, dtype=np.float64)
    int_arr = np.asarray(intensity, dtype=np.float64)

    valid = (mz_arr > 50.0) & (int_arr > 0) & np.isfinite(mz_arr) & np.isfinite(int_arr)
    mz_v = mz_arr[valid]
    int_v = int_arr[valid]

    if len(mz_v) == 0:
        return pd.DataFrame(columns=["mz", "intensity", "KM", "NKM", "KMD", "log_int"])

    exact_base = KENDRICK_BASES.get(base, KENDRICK_BASES["CH2"])
    nominal_base = round(exact_base)

    factor = nominal_base / exact_base
    km = mz_v * factor
    nkm = np.round(km).astype(int)
    kmd = nkm - km

    # For standard chemometric convention, if KMD is negative, offset by +1 or keep signed
    log_int = np.log10(np.clip(int_v, 1.0, None))

    return pd.DataFrame(
        {
            "mz": mz_v,
            "intensity": int_v,
            "KM": km,
            "NKM": nkm,
            "KMD": kmd,
            "log_int": log_int,
        }
    )


def create_kendrick_plot(
    df_kmd: pd.DataFrame,
    title: str = "Kendrick Mass Defect (KMD) Analysis (CH₂ Series)",
) -> go.Figure:
    """Create interactive Plotly scatter chart for KMD vs m/z."""
    fig = go.Figure()

    if len(df_kmd) == 0:
        fig.add_annotation(text="No valid mass peaks for KMD calculation", showarrow=False)
        return fig

    fig.add_trace(
        go.Scatter(
            x=df_kmd["mz"],
            y=df_kmd["KMD"],
            mode="markers",
            marker=dict(
                size=5,
                color=df_kmd["log_int"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Log₁₀(Int)", len=0.75, x=1.02),
                opacity=0.75,
            ),
            hovertemplate=(
                "<b>m/z:</b> %{x:.4f} Da<br>"
                "<b>KMD:</b> %{y:.4f}<br>"
                "<b>Intensity:</b> %{customdata:.1f}<extra></extra>"
            ),
            customdata=df_kmd["intensity"],
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Observed m/z (Da)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="Kendrick Mass Defect (KMD, Da)", showgrid=True, gridcolor="#E2E8F0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=50, r=40, t=40, b=40),
    )

    return fig
