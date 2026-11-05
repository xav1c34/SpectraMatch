"""core/jcamp.py — Native IUPAC JCAMP-DX (.jdx, .dx) spectroscopic file format parser."""
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np


def is_jcamp_file(text: str) -> bool:
    """Check if file starts with or contains standard JCAMP-DX header tags."""
    return ("##TITLE=" in text) or ("##JCAMP-DX=" in text) or ("##DATA TYPE=" in text)


def parse_jcamp_dx(text: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Parse JCAMP-DX standard spectroscopic file.

    Extracts (x_axis, y_axis, metadata_dict).
    Handles standard (X++(Y..Y)) gridded tables and (XY..XY) peak tables.
    """
    metadata: Dict[str, Any] = {}
    lines = text.splitlines()

    data_lines = []
    in_data = False
    data_format = "(X++(Y..Y))"  # default

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        if line_s.startswith("##"):
            # Header tag
            if "=" in line_s:
                tag, val = line_s.split("=", 1)
                tag = tag[2:].strip().upper()
                val = val.strip()
                metadata[tag] = val

                if tag in ["XYDATA", "PEAK TABLE", "DATA POINTS"]:
                    in_data = True
                    data_format = val
                    continue
                elif tag == "END":
                    in_data = False
                    break
        elif in_data:
            data_lines.append(line_s)

    # Extract metadata properties
    first_x = float(metadata.get("FIRSTX", 0.0))
    last_x = float(metadata.get("LASTX", 0.0))
    n_points = int(float(metadata.get("NPOINTS", 0))) if "NPOINTS" in metadata else None
    x_factor = float(metadata.get("XFACTOR", 1.0))
    y_factor = float(metadata.get("YFACTOR", 1.0))

    x_list = []
    y_list = []

    # If format is (XY..XY) or simple pairs
    if "XY" in data_format or not in_data and not data_lines:
        for line in data_lines:
            tokens = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
            for i in range(0, len(tokens) - 1, 2):
                x_list.append(float(tokens[i]) * x_factor)
                y_list.append(float(tokens[i + 1]) * y_factor)

    else:
        # Standard (X++(Y..Y))
        # First number on line is X, subsequent numbers are Y values
        calc_dx = None
        if n_points and n_points > 1 and first_x != last_x:
            calc_dx = (last_x - first_x) / (n_points - 1)
        elif "DELTAX" in metadata:
            calc_dx = float(metadata["DELTAX"])

        for line in data_lines:
            tokens = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
            if not tokens:
                continue

            current_x = float(tokens[0]) * x_factor
            y_vals = [float(t) * y_factor for t in tokens[1:]]

            dx = calc_dx if calc_dx is not None else (1.0 if not y_vals else 0.0)

            for idx, y_val in enumerate(y_vals):
                x_list.append(current_x + idx * dx)
                y_list.append(y_val)

    if not x_list:
        raise ValueError("JCAMP-DX parser found no numeric coordinates in data section")

    x_arr = np.array(x_list, dtype=np.float64)
    y_arr = np.array(y_list, dtype=np.float64)

    # Sort X ascending
    sort_idx = np.argsort(x_arr)
    x_arr = x_arr[sort_idx]
    y_arr = y_arr[sort_idx]

    # Transmittance to Absorbance conversion if indicated in headers
    y_units = metadata.get("YUNITS", "").upper()
    if "TRANSMITTANCE" in y_units or "%T" in y_units:
        t_clean = np.clip(y_arr, 1e-4, 100.0)
        y_arr = 2.0 - np.log10(t_clean)

    return x_arr, y_arr, metadata
