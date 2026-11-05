"""views/export_utils.py — Export generators for CSV summaries, JSON passports, and standalone HTML reports."""
from datetime import datetime, timezone
import json
from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go

from core.models import SpectralRecord, SearchResult


def export_matches_to_csv(results: List[SearchResult]) -> str:
    """Generate CSV string of ranked match results."""
    rows = []
    for r in results:
        row = {
            "rank": r.rank,
            "spectrum_id": r.spectrum_id,
            "sample_name": r.name,
            "sample_type": r.sample_type,
            "origin": r.origin,
            "metric": r.metric,
            "similarity_score": r.score,
            "source_doi": r.source_doi,
        }
        # Add indices if present
        for k, v in r.indices.items():
            row[f"idx_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_passport_to_json(record: SpectralRecord, results: List[SearchResult]) -> str:
    """Generate comprehensive JSON Analytical Sample Passport."""
    top_matches_data = [
        {
            "rank": r.rank,
            "id": r.spectrum_id,
            "name": r.name,
            "sample_type": r.sample_type,
            "origin": r.origin,
            "score": r.score,
            "metric": r.metric,
            "indices": r.indices,
        }
        for r in results[:10]
    ]

    passport = {
        "spectramatch_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_id": record.id,
        "sample_name": record.name,
        "modality": record.modality.value,
        "quality_control": {
            "passed": record.quality.passed,
            "quality_score": record.quality.score,
            "snr_estimate": record.quality.snr_estimate,
            "coverage_pct": record.quality.coverage_pct,
            "warnings": record.quality.warnings,
            "errors": record.quality.errors,
        },
        "computed_indices": record.indices,
        "metadata": record.metadata,
        "top_matches": top_matches_data,
    }

    return json.dumps(passport, indent=2, ensure_ascii=False)


def export_standalone_html(
    record: SpectralRecord,
    top_result: SearchResult,
    fig: go.Figure,
) -> str:
    """Generate a standalone self-contained HTML report with embedded interactive Plotly chart."""
    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SpectraMatch Analytical Report — {record.name}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 30px;
      background-color: #f8fafc;
      color: #0f172a;
    }}
    .container {{
      max-width: 1050px;
      margin: 0 auto;
      background: #ffffff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    h1 {{
      margin-top: 0;
      color: #0284c7;
      font-size: 24px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      background: #e0f2fe;
      color: #0369a1;
      margin-right: 8px;
    }}
    .meta-table, .indices-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 14px;
    }}
    .meta-table th, .meta-table td, .indices-table th, .indices-table td {{
      padding: 8px 12px;
      border: 1px solid #e2e8f0;
      text-align: left;
    }}
    .meta-table th, .indices-table th {{
      background: #f1f5f9;
      color: #475569;
    }}
    .chart-box {{
      margin: 30px 0;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🔬 SpectraMatch Analytical Identification Report</h1>
    <p>
      <span class="badge">Modality: {record.modality.value}</span>
      <span class="badge">QC Score: {record.quality.score:.2f}</span>
      <span class="badge">Top Match: {top_result.name} ({top_result.metric} = {top_result.score:.4f})</span>
    </p>

    <h3>Sample Overview</h3>
    <table class="meta-table">
      <tr><th>Query Sample</th><td>{record.name}</td><th>Report Generated</th><td>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
      <tr><th>Best Library Match</th><td>{top_result.name}</td><th>Origin</th><td>{top_result.origin}</td></tr>
      <tr><th>Match Score</th><td><strong>{top_result.score:.4f}</strong> ({top_result.metric})</td><th>Source DOI</th><td>{top_result.source_doi or 'N/A'}</td></tr>
    </table>

    <div class="chart-box">
      {chart_html}
    </div>

    <h3>Chemometric Indices Comparison</h3>
    <table class="indices-table">
      <thead>
        <tr><th>Index</th><th>Query Value</th><th>Best Match Value</th></tr>
      </thead>
      <tbody>
        {"".join(f"<tr><td><strong>{k}</strong></td><td>{record.indices.get(k, '—')}</td><td>{top_result.indices.get(k, '—')}</td></tr>" for k in record.indices.keys())}
      </tbody>
    </table>

    <p style="font-size: 12px; color: #94a3b8; text-align: center; margin-top: 40px;">
      Generated autonomously by <strong>SpectraMatch Chemometric Engine</strong>.
    </p>
  </div>
</body>
</html>
"""
    return html_template
