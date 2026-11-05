"""core/chemspace.py — Principal Component Analysis (PCA) and Chemical Space Projection."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go

from core.models import ModalityEnum


COLOR_PALETTE = {
    "humic_acid": "#D97706",     # Amber
    "fulvic_acid": "#10B981",    # Emerald
    "lignin": "#8B5CF6",         # Violet
    "sludge_lignin": "#EC4899",  # Pink
    "polysaccharide": "#F59E0B", # Yellow-orange
    "lake_dom": "#06B6D4",       # Cyan
    "nom": "#3B82F6",            # Blue
    "protein": "#84CC16",        # Lime
    "unknown": "#94A3B8",        # Slate
}


@dataclass
class PCAModel:
    """Pre-fitted PCA projection model for a spectral library."""
    modality: ModalityEnum
    mean_vector: np.ndarray          # shape (D,)
    components: np.ndarray           # shape (n_components, D)
    explained_variance_ratio: np.ndarray  # shape (n_components,)
    library_coords: np.ndarray       # shape (N, n_components)


def fit_library_pca(
    library_matrix: np.ndarray,
    modality: Union[str, ModalityEnum],
    n_components: int = 3,
) -> Optional[PCAModel]:
    """Fit SVD-based Principal Component Analysis on library matrix (N, D)."""
    if len(library_matrix) < 3:
        return None

    mod = ModalityEnum(modality)
    X = np.asarray(library_matrix, dtype=np.float64)
    n_samples, n_features = X.shape

    # Mean centering
    mean = np.mean(X, axis=0)
    Xc = X - mean

    # Efficient SVD: Xc = U * S * Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, n_samples - 1, n_features)

    library_coords = U[:, :k] * S[:k]
    components = Vt[:k, :]

    total_var = np.sum(S**2)
    var_ratio = (S[:k] ** 2) / max(total_var, 1e-12)

    return PCAModel(
        modality=mod,
        mean_vector=mean,
        components=components,
        explained_variance_ratio=var_ratio,
        library_coords=library_coords,
    )


def project_query_to_pca(query_vec: np.ndarray, pca_model: PCAModel) -> np.ndarray:
    """Project a 1D query vector into the fitted library PC space."""
    q = np.asarray(query_vec, dtype=np.float64).flatten()
    qc = q - pca_model.mean_vector
    return qc.dot(pca_model.components.T)


def create_pca_scatter_plot(
    pca_model: PCAModel,
    library_meta: List[Dict[str, Any]],
    query_coords: Optional[np.ndarray] = None,
    query_name: str = "Query Sample",
) -> go.Figure:
    """Generate an interactive Plotly 2D PCA scatter plot showing chemical space clusters."""
    fig = go.Figure()

    evr = pca_model.explained_variance_ratio
    pc1_label = f"PC1 ({evr[0]*100:.1f}% var)" if len(evr) > 0 else "PC1"
    pc2_label = f"PC2 ({evr[1]*100:.1f}% var)" if len(evr) > 1 else "PC2"

    coords = pca_model.library_coords

    # Group library samples by sample_type for clean legend
    types_found = sorted(list(set(m.get("sample_type", "unknown") for m in library_meta)))

    for stype in types_found:
        indices = [i for i, m in enumerate(library_meta) if m.get("sample_type") == stype]
        if not indices:
            continue

        x_pts = coords[indices, 0]
        y_pts = coords[indices, 1]
        names = [library_meta[i].get("name", f"Sample {i}") for i in indices]
        origins = [library_meta[i].get("origin", "N/A") for i in indices]
        hover_texts = [
            f"<b>{name}</b><br>Type: {stype}<br>Origin: {origin}<br>PC1: {x:.2f}, PC2: {y:.2f}"
            for name, origin, x, y in zip(names, origins, x_pts, y_pts)
        ]

        color = COLOR_PALETTE.get(stype, "#64748B")

        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name=stype.replace("_", " ").title(),
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1, color="#FFFFFF"),
                    opacity=0.85,
                ),
                text=hover_texts,
                hoverinfo="text",
            )
        )

    # If user query coordinates are provided, render as large glowing Star
    if query_coords is not None and len(query_coords) >= 2:
        qx, qy = float(query_coords[0]), float(query_coords[1])
        fig.add_trace(
            go.Scatter(
                x=[qx],
                y=[qy],
                mode="markers+text",
                name="⭐ Your Query",
                text=[f"⭐ {query_name}"],
                textposition="top center",
                textfont=dict(size=12, color="#0284C7", family="sans-serif"),
                marker=dict(
                    symbol="star",
                    size=20,
                    color="#0284C7",
                    line=dict(width=2, color="#0369A1"),
                ),
                hoverinfo="text",
                hovertext=f"<b>⭐ Query: {query_name}</b><br>PC1: {qx:.2f}, PC2: {qy:.2f}",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Chemical Space Mapping — PCA Embedding ({pca_model.modality.value})",
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(title=pc1_label, showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#CBD5E1"),
        yaxis=dict(title=pc2_label, showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#CBD5E1"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig
