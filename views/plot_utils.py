"""views/plot_utils.py — Scientific Plotly charting templates following Section 13 styling specifications."""
from typing import Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import ModalityEnum
from core.normalize import CANONICAL_FTIR_GRID, CANONICAL_EEM_EX, CANONICAL_EEM_EM, CANONICAL_FTICR_EDGES


COLOR_QUERY = "#0284C7"      # Sky Blue
COLOR_MATCH = "#EF4444"      # Rose Red
COLOR_RESIDUAL = "#64748B"   # Slate Gray
COLOR_GRID = "#E2E8F0"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


def get_base_layout() -> dict:
    """Return common minimalist scientific layout options."""
    return dict(
        font=dict(family=FONT_FAMILY, size=12, color="#0F172A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )


def create_preview_plot(
    x: np.ndarray,
    y: np.ndarray,
    modality: ModalityEnum,
    title: str = "Query Spectrum Preview",
    matrix_2d: Optional[np.ndarray] = None,
) -> go.Figure:
    """Create a single spectrum preview chart for raw uploaded data."""
    fig = go.Figure()

    if modality == ModalityEnum.FTIR:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Signal",
                line=dict(color=COLOR_QUERY, width=2.0),
                hovertemplate="Wavenumber: %{x:.1f} cm⁻¹<br>Signal: %{y:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(
            title_text="Wavenumber (cm⁻¹)",
            autorange="reversed",
            showgrid=True,
            gridcolor=COLOR_GRID,
        )
        fig.update_yaxes(title_text="Absorbance (a.u.)", showgrid=True, gridcolor=COLOR_GRID)

    elif modality == ModalityEnum.EEM:
        if matrix_2d is not None:
            # Heatmap / Contour
            ex_axis = x
            em_axis = y
            z = matrix_2d.T if matrix_2d.shape == (len(ex_axis), len(em_axis)) else matrix_2d
            fig.add_trace(
                go.Contour(
                    z=z,
                    x=ex_axis,
                    y=em_axis,
                    colorscale="Viridis",
                    contours=dict(coloring="heatmap"),
                    hovertemplate="Ex: %{x} nm<br>Em: %{y} nm<br>Intensity: %{z:.2f}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Excitation (nm)", showgrid=True, gridcolor=COLOR_GRID)
            fig.update_yaxes(title_text="Emission (nm)", showgrid=True, gridcolor=COLOR_GRID)

    elif modality == ModalityEnum.FTICR_MS:
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name="Intensity",
                marker_color=COLOR_QUERY,
                width=0.4,
                hovertemplate="m/z: %{x:.4f}<br>Intensity: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="m/z (Da)", showgrid=True, gridcolor=COLOR_GRID)
        fig.update_yaxes(title_text="Intensity", showgrid=True, gridcolor=COLOR_GRID)

    layout = get_base_layout()
    layout["title"] = dict(text=title, font=dict(size=14, color="#1E293B"))
    fig.update_layout(**layout)
    return fig


def create_ftir_overlay_plot(
    query_vec: np.ndarray,
    match_vec: np.ndarray,
    match_name: str,
    score: float,
    metric_name: str,
) -> go.Figure:
    """Create FTIR overlay with residual subplot and characteristic band markers."""
    wn_grid = CANONICAL_FTIR_GRID

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=[
            f"FTIR Overlay vs {match_name} ({metric_name} = {score:.4f})",
            "Residuals (Δ = Query - Match)",
        ],
    )

    # Query Trace (Sky Blue, dotted or solid)
    fig.add_trace(
        go.Scatter(
            x=wn_grid,
            y=query_vec,
            mode="lines",
            name="Query Spectrum",
            line=dict(color=COLOR_QUERY, width=2.5, dash="solid"),
            hovertemplate="Wavenumber: %{x:.1f} cm⁻¹<br>Query: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Match Trace (Rose Red, solid)
    fig.add_trace(
        go.Scatter(
            x=wn_grid,
            y=match_vec,
            mode="lines",
            name=f"Match: {match_name}",
            line=dict(color=COLOR_MATCH, width=2.0),
            hovertemplate="Wavenumber: %{x:.1f} cm⁻¹<br>Match: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Residuals
    residuals = query_vec - match_vec
    fig.add_trace(
        go.Scatter(
            x=wn_grid,
            y=residuals,
            mode="lines",
            name="Residuals Δ",
            line=dict(color=COLOR_RESIDUAL, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(100, 116, 139, 0.15)",
            hovertemplate="Wavenumber: %{x:.1f} cm⁻¹<br>Residual: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Characteristic diagnostic vibrational bands
    key_bands = [
        (1715, "1715 (C=O)"),
        (1600, "1600 (Arom)"),
        (1510, "1510 (Lignin)"),
        (1030, "1030 (Poly)"),
    ]
    for band_wn, label in key_bands:
        fig.add_vline(
            x=band_wn,
            line_width=1,
            line_dash="dot",
            line_color="#94A3B8",
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="#475569",
            row=1,
            col=1,
        )

    # Reverse X axis according to spectroscopic standard
    fig.update_xaxes(autorange="reversed", showgrid=True, gridcolor=COLOR_GRID, row=1, col=1)
    fig.update_xaxes(
        autorange="reversed",
        title_text="Wavenumber (cm⁻¹)",
        showgrid=True,
        gridcolor=COLOR_GRID,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Norm. Absorbance", showgrid=True, gridcolor=COLOR_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Residual", showgrid=True, gridcolor=COLOR_GRID, row=2, col=1)

    layout = get_base_layout()
    layout["height"] = 520
    fig.update_layout(**layout)
    return fig


def create_eem_comparison_plot(
    query_vec: np.ndarray,
    match_vec: np.ndarray,
    match_name: str,
    score: float,
    metric_name: str,
) -> go.Figure:
    """Create 3-panel 2D contour comparison: Query, Match, and Absolute Difference."""
    ex_axis = CANONICAL_EEM_EX
    em_axis = CANONICAL_EEM_EM

    q_2d = query_vec.reshape((43, 161)).T
    m_2d = match_vec.reshape((43, 161)).T
    diff_2d = np.abs(q_2d - m_2d)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Query EEM",
            f"Match ({metric_name} = {score:.4f})",
            "Absolute Difference |Δ|",
        ],
        horizontal_spacing=0.08,
    )

    # Panel 1: Query (Viridis)
    fig.add_trace(
        go.Contour(
            z=q_2d,
            x=ex_axis,
            y=em_axis,
            colorscale="Viridis",
            showscale=False,
            contours=dict(coloring="heatmap"),
            hovertemplate="Ex: %{x} nm<br>Em: %{y} nm<br>Query: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Match (Viridis)
    fig.add_trace(
        go.Contour(
            z=m_2d,
            x=ex_axis,
            y=em_axis,
            colorscale="Viridis",
            showscale=False,
            contours=dict(coloring="heatmap"),
            hovertemplate="Ex: %{x} nm<br>Em: %{y} nm<br>Match: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Panel 3: Difference (|Delta|, RdBu_r)
    fig.add_trace(
        go.Contour(
            z=diff_2d,
            x=ex_axis,
            y=em_axis,
            colorscale="RdBu_r",
            showscale=True,
            contours=dict(coloring="heatmap"),
            colorbar=dict(title="|Δ|", len=0.8, x=1.02),
            hovertemplate="Ex: %{x} nm<br>Em: %{y} nm<br>|Δ|: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    for c in range(1, 4):
        fig.update_xaxes(title_text="Excitation (nm)", showgrid=True, gridcolor=COLOR_GRID, row=1, col=c)
        if c == 1:
            fig.update_yaxes(title_text="Emission (nm)", showgrid=True, gridcolor=COLOR_GRID, row=1, col=c)
        else:
            fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, row=1, col=c)

    layout = get_base_layout()
    layout["height"] = 420
    layout["title"] = dict(text=f"EEM Contour Matching vs {match_name}", font=dict(size=14))
    fig.update_layout(**layout)
    return fig


def create_fticr_mirror_plot(
    query_vec: np.ndarray,
    match_vec: np.ndarray,
    match_name: str,
    score: float,
    metric_name: str,
) -> go.Figure:
    """Create Head-to-Tail (Mirror) mass spectra plot (+Query vs -Match)."""
    # Mass bin centers: 150.25 .. 799.75
    bin_centers = (CANONICAL_FTICR_EDGES[:-1] + CANONICAL_FTICR_EDGES[1:]) / 2.0

    fig = go.Figure()

    # Query (upper half, positive)
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=query_vec,
            name="Query Mass Spectrum (+)",
            marker_color=COLOR_QUERY,
            width=0.45,
            hovertemplate="m/z: %{x:.2f} Da<br>Query Intensity: %{y:.4f}<extra></extra>",
        )
    )

    # Match (lower half, negative)
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=-match_vec,
            name=f"Match (-): {match_name}",
            marker_color=COLOR_MATCH,
            width=0.45,
            hovertemplate="m/z: %{x:.2f} Da<br>Match Intensity: %{y:.4f}<extra></extra>",
        )
    )

    # Zero dividing baseline
    fig.add_hline(y=0.0, line_width=1.5, line_color="#0F172A")

    layout = get_base_layout()
    layout["height"] = 460
    layout["title"] = dict(
        text=f"Head-to-Tail FT-ICR MS Mirror vs {match_name} ({metric_name} = {score:.4f})",
        font=dict(size=14),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="m/z (Da)", showgrid=True, gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="Relative Intensity", showgrid=True, gridcolor=COLOR_GRID)
    return fig
