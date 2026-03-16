#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmc_utils.py
Shared utilities for NMC Monte Carlo simulation notebooks.

Covers:
  - Paper-quality plot styling
  - Config file I/O  (.txt human-readable  +  *_H.mci machine-readable)
  - Executable runner  (real-time stdout streaming)
  - Output file discovery
  - Data loaders: metrics, rd_data, photons
"""

from __future__ import annotations

import os
import re
import sys
import glob
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

#: Tolerance (mm) used when deciding whether a z-coordinate sits on a surface.
Z_TOL: float = 1e-4

#: Photon type labels for plotting legends.
TYPE_LABELS: Dict[int, str] = {1: "Laser", 2: "Raman", 3: "SRS"}

#: Per-type colours (colorblind-safe, publication grade).
TYPE_COLORS: Dict[int, str] = {1: "#2C6FAC", 2: "#E8602C", 3: "#3DAA62"}

#: Ordered parameter keys written into *_H.mci  (must match C reader order).
_MCI_KEYS_ORDER: Tuple[str, ...] = (
    "mckernelflag",
    "width",
    "Nx", "Ny", "Nz",
    "raman_prob",
    "stim_raman_prob",
    "interaction_distance",
    "step_size",
    "laser_beam_radius",
    "laser_beam_pulse_width",
    "laser_beam_pulse_delay",
    "cutoff_radius",
    "zfocus",
    "numerial_aperture",
    "det_state",
    "Nt",
    "mu_a",
    "mu_s",
    "g",
    "index_of_refraction",
)


# =============================================================================
# Styling
# =============================================================================

def set_paper_style(
    dpi: int = 300,
    font_size: int = 12,
    context: str = "paper",
    style: str = "ticks",
) -> None:
    """
    Apply a Q1-journal-grade seaborn/matplotlib style.

    Characteristics
    ---------------
    - Clean ticks-only axes; top & right spines removed by default via
      ``sns.despine()`` (call it after each figure).
    - 3-colour qualitative palette:  Laser=#2C6FAC  Raman=#E8602C  SRS=#3DAA6
    - Single-column figure width 3.5 in, double-column 7.0 in.
    - 300 dpi save / 150 dpi screen.
    """
    sns.set_theme(style=style, context=context, font="sans-serif")
    sns.set_palette([TYPE_COLORS[1], TYPE_COLORS[2], TYPE_COLORS[3]])
    plt.rcParams.update(
        {
            # Resolution
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            # Typography
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            # Axes
            "axes.linewidth": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            # Layout
            "figure.constrained_layout.use": True,
            # Lines / markers
            "lines.linewidth": 2.0,
            "patch.linewidth": 0.8,
        }
    )


# =============================================================================
# Config I/O
# =============================================================================

#: Default simulation parameters.  All physical quantities are in mm / ps.
_BASE_PARAMS: Dict = {
    "fname":                 "test",
    "mckernelflag":          5,
    "width":                 0.5,      # mm   — simulation volume half-width
    "Nx":                    100,
    "Ny":                    100,
    "Nz":                    100,
    "raman_prob":            0.05,
    "stim_raman_prob":       0.1,
    "interaction_distance":  0.01,     # mm
    "step_size":             0.001,    # mm
    "laser_beam_radius":     0.05,     # mm
    "laser_beam_pulse_width":  5.0,    # ps
    "laser_beam_pulse_delay": 15.0,    # ps
    "cutoff_radius":         3.0,      # mm
    "zfocus":                1.0,      # mm
    "numerial_aperture":     1.0,      # n_exit · sin(θ_max)
    "det_state":             0,        # 0 = reflectance, 1 = transmittance
    "Nt":                    1,
    "mu_a":                  0.1,      # mm⁻¹
    "mu_s":                  10.0,     # mm⁻¹
    "g":                     0.6,
    "index_of_refraction":   1.6,
}


def init_config_file(
    cfg_dir: str | Path | None = None,
    mci_dir: str | Path | None = None,
    **params,
) -> Tuple[Path, Path]:
    """
    Create simulation config files.

    Parameters
    ----------
    cfg_dir:
        Directory for the human-readable ``config file <fname>.txt``.
        Defaults to the directory of the calling script / notebook
        (``Path.cwd()``).
    mci_dir:
        Directory for the machine-readable ``<fname>_H.mci`` consumed by the
        executable.  Defaults to ``cfg_dir``.
    **params:
        Any key from ``_BASE_PARAMS`` to override.

    Returns
    -------
    (path_to_mci, path_to_cfg)
    """
    p = dict(_BASE_PARAMS)           # fresh copy every call
    for k, v in params.items():
        if k in p:
            p[k] = v
        else:
            raise KeyError(
                f"Unknown parameter '{k}'. "
                f"Valid keys: {list(p.keys())}"
            )

    cfg_dir = Path(cfg_dir) if cfg_dir else Path.cwd()
    mci_dir = Path(mci_dir) if mci_dir else cfg_dir
    cfg_dir.mkdir(parents=True, exist_ok=True)
    mci_dir.mkdir(parents=True, exist_ok=True)

    # ── machine-readable *_H.mci  (values only, order matches _MCI_KEYS_ORDER)
    mci_path = mci_dir / f"{p['fname']}_H.mci"
    with mci_path.open("w") as fh:
        for key in _MCI_KEYS_ORDER:
            fh.write(f"{p[key]}\n")

    # ── human-readable config file <fname>.txt
    cfg_path = cfg_dir / f"config file {p['fname']}.txt"
    with cfg_path.open("w") as fh:
        for k, v in p.items():
            fh.write(f"{k}:\t{v}\n")

    return mci_path, cfg_path


def parse_config(file_path: str | Path) -> Dict[str, object]:
    """
    Parse a human-readable ``config file <fname>.txt``.

    Returns a dict with numeric values as ``float`` and ``fname`` as ``str``.
    """
    cfg: Dict[str, object] = {}
    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "fname":
                cfg[key] = value
            else:
                try:
                    cfg[key] = float(value)
                except ValueError:
                    cfg[key] = value   # keep as string if not numeric
    return cfg


def parse_mci(file_path: str | Path) -> Dict[str, float]:
    """
    Parse a values-only ``*_H.mci`` file into a dict keyed by
    ``_MCI_KEYS_ORDER``.
    """
    vals: List[float] = []
    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass   # skip comment lines if any

    if len(vals) != len(_MCI_KEYS_ORDER):
        raise ValueError(
            f"Unexpected line count in '{file_path}': "
            f"got {len(vals)}, expected {len(_MCI_KEYS_ORDER)}."
        )
    return dict(zip(_MCI_KEYS_ORDER, vals))


# =============================================================================
# Executable runner  (real-time streaming stdout)
# =============================================================================

def run(
    exe_path: str | Path,
    mci_name: str | None = None,
    workdir: str | Path | None = None,
    extra_args: Optional[List[str]] = None,
    verbose: bool = True,
) -> subprocess.CompletedProcess:
    """
    Launch the NMC executable and stream its output in real time.

    Parameters
    ----------
    exe_path:
        Absolute or relative path to the compiled NMC executable.
        Example (absolute):  ``/Users/you/.../NMC/Build/Products/Debug/nmc``
        Example (relative):  ``../NMC/Build/Products/Debug/nmc``
    mci_name:
        Base name of the ``*_H.mci`` file (with or without the ``_H.mci``
        suffix).  If ``None``, the first ``*_H.mci`` found in ``workdir``
        is used automatically.
    workdir:
        Working directory passed to the subprocess — this is where output
        CSV files will be written.  Defaults to the directory that contains
        the ``*_H.mci`` file.
    extra_args:
        Additional command-line arguments forwarded to the executable.
    verbose:
        If ``True`` (default), print the command and stream stdout/stderr
        line-by-line as the simulation runs.

    Returns
    -------
    ``subprocess.CompletedProcess``

    Raises
    ------
    FileNotFoundError
        If the executable or MCI file cannot be located.
    RuntimeError
        If the executable returns a non-zero exit code.
    """
    extra_args = extra_args or []
    exe_path = Path(exe_path).expanduser().resolve()

    if not exe_path.exists():
        raise FileNotFoundError(
            f"Executable not found: {exe_path}\n"
            "Check the exe_path argument or your build."
        )

    # ── resolve MCI file
    if mci_name is None:
        search_dir = Path(workdir).expanduser().resolve() if workdir else exe_path.parent
        mci_files = sorted(search_dir.glob("*_H.mci"))
        if not mci_files:
            raise FileNotFoundError(
                f"No *_H.mci files found in '{search_dir}'.\n"
                "Run init_config_file() first, or pass mci_name explicitly."
            )
        mci_path = mci_files[0]
        if verbose:
            print(f"[nmc] Auto-detected MCI: {mci_path.name}")
    else:
        # accept  "test_H.mci", "test_H", or "test"
        stem = Path(mci_name).stem
        if stem.endswith("_H"):
            stem = stem[:-2]
        candidate = (Path(workdir) if workdir else exe_path.parent) / f"{stem}_H.mci"
        if not candidate.exists():
            raise FileNotFoundError(f"MCI file not found: {candidate}")
        mci_path = candidate

    # fname = everything before _H
    stem = mci_path.stem
    fname = stem[:-2] if stem.endswith("_H") else stem

    # ── working directory defaults to the folder containing the MCI file
    cwd = Path(workdir).expanduser().resolve() if workdir else mci_path.parent

    cmd = [str(exe_path), fname, *extra_args]

    if verbose:
        print(f"[nmc] Command : {' '.join(cmd)}")
        print(f"[nmc] cwd     : {cwd}")
        print("[nmc] ── simulation output ──────────────────")

    # ── launch with real-time line-by-line streaming
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    with subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,          # line-buffered
    ) as proc:
        # stream stdout
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            stdout_lines.append(stripped)
            if verbose:
                print(stripped, flush=True)

        # capture stderr after stdout closes
        assert proc.stderr is not None
        for line in proc.stderr:
            stripped = line.rstrip("\n")
            stderr_lines.append(stripped)

        proc.wait()

    if verbose:
        print("[nmc] ─────────────────────────────────────")

    if proc.returncode != 0:
        raise RuntimeError(
            f"NMC executable failed (exit code {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{''.join(stderr_lines)}"
        )

    if verbose and stderr_lines:
        print("[nmc] stderr:")
        print("\n".join(stderr_lines))

    if verbose:
        print(f"[nmc] Done (exit code {proc.returncode}).")

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
    )


# =============================================================================
# Output file discovery
# =============================================================================

def find_outputs(
    pattern: str,
    workdir: str | Path | None = None,
) -> List[Path]:
    """
    Return a sorted list of paths matching ``pattern`` in ``workdir``.

    Typical use — skip simulation if output already exists::

        outputs = find_outputs("rd_data_mus-10.00_NA-1.00_zf-1.00.csv", workdir)
        if not outputs:
            run(exe_path, workdir=workdir)

    Parameters
    ----------
    pattern:
        Glob pattern, e.g. ``"rd_data_mus-*.csv"`` or an exact filename.
    workdir:
        Directory to search.  Defaults to ``Path.cwd()``.
    """
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
    return sorted(base.glob(pattern))


# =============================================================================
# Filename parameter extraction
# =============================================================================

def parse_params_from_filename(path: str | Path) -> Dict[str, float]:
    """
    Extract ``mus``, ``NA``, ``zf`` from output filenames such as::

        rd_data_mus-10.00_NA-1.00_zf-1.00.csv
        photons_data_mus-10.00_NA-1.00_zf-1.00.csv
        metrics_mus-10.00_NA-1.00_zf-1.00.csv

    Returns an empty dict if the pattern is not found.
    """
    name = Path(path).name
    m = re.search(
        r"mus-(?P<mus>\d+(?:\.\d+)?)_NA-(?P<na>\d+(?:\.\d+)?)_zf-(?P<zf>\d+(?:\.\d+)?)",
        name,
    )
    if not m:
        return {}
    return {k: float(v) for k, v in m.groupdict().items()}


# =============================================================================
# Data loaders
# =============================================================================

def load_metrics(path: str | Path) -> Dict[str, object]:
    """
    Load a ``metrics_mus-*_NA-*_zf-*.csv`` file.

    Returns a dict with keys:
    ``n_total``, ``n_batch``, ``sim_time_s``, ``photon_rate``,
    ``mus``, ``NA``, ``zf``  (last three from filename, may be NaN).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    row = df.iloc[0]
    result: Dict[str, object] = {
        "n_total":      int(row["N total"]),
        "n_batch":      int(row["N batch"]),
        "sim_time_s":   float(row["sim time (s)"]),
        "photon_rate":  float(row["photon rate (photons/s)"]),
    }
    result.update(parse_params_from_filename(path))
    return result


def load_rd_data(
    path: str | Path,
    Nx: int,
    Ny: int,
    use_log1p: bool = False,
) -> Dict[str, object]:
    """
    Load a ``rd_data_mus-*_NA-*_zf-*.csv`` file and reshape into 2-D arrays.

    Parameters
    ----------
    path:     Path to the CSV file.
    Nx, Ny:   Grid dimensions from the config.  Must satisfy ``Nx * Ny == rows``.
    use_log1p:
              Apply ``log1p`` transform to all channels before returning.
              Useful for heatmap visualisation of high-dynamic-range data.

    Returns
    -------
    dict with keys:
        ``laser``  (Nx x Ny ndarray),
        ``raman``  (Nx x Ny ndarray),
        ``srs``    (Nx x Ny ndarray),
        ``x``      (Nx  1-D coordinate array, mm, centred at 0),
        ``y``      (Ny  1-D coordinate array, mm, centred at 0),
        ``extent`` ([x_min, x_max, y_min, y_max]  for imshow),
        ``mus``, ``NA``, ``zf``  (from filename).
    """
    df = pd.read_csv(path, sep=",", skipinitialspace=True)
    df.columns = df.columns.str.strip()

    required = ["Laser RD", "Raman RD", "SRS RD"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in '{path}'.\n"
            f"Found: {list(df.columns)}"
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)

    expected = Nx * Ny
    if len(df) != expected:
        raise ValueError(
            f"Expected Nx*Ny = {Nx}*{Ny} = {expected} rows, "
            f"got {len(df)}.  Check Nx/Ny or file integrity."
        )

    def _sanitize(a: np.ndarray) -> np.ndarray:
        a = a.astype(float, copy=True)
        a = np.where(np.isfinite(a), a, 0.0)
        if use_log1p:
            a = np.log1p(np.maximum(a, 0.0))
        return a

    # reshape: row-major, index order matches C loop  i = ix * Ny + iy
    cfg = parse_params_from_filename(path)
    width = cfg.get("mus", np.nan)   # fallback; caller should pass width from config

    laser = _sanitize(df["Laser RD"].to_numpy().reshape((Nx, Ny)))
    raman = _sanitize(df["Raman RD"].to_numpy().reshape((Nx, Ny)))
    srs   = _sanitize(df["SRS RD"].to_numpy().reshape((Nx, Ny)))

    dx = 1.0   # placeholder — caller should override with  width/Nx * 2
    dy = 1.0
    x  = (np.arange(Nx) - Nx / 2.0) * dx
    y  = (np.arange(Ny) - Ny / 2.0) * dy

    return {
        "laser":  laser,
        "raman":  raman,
        "srs":    srs,
        "x":      x,
        "y":      y,
        "extent": [float(x.min()), float(x.max()),
                   float(y.min()), float(y.max())],
        **cfg,
    }


def _is_close(a, b, tol=Z_TOL):
    """Absolute-tolerance float comparison."""
    return abs(a - b) <= tol

def load_photons(
    path: str | Path,
    z_tol: float = Z_TOL,
    z_top: float = 0.0,
    z_bottom: float = 0.5,
    types_of_interest: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a ``photons_data_mus-*_NA-*_zf-*.csv`` file and build two tables.

    Raw trajectory data
    -------------------
    The CSV has columns::

        id, x [mm], y [mm], z [mm], time [ps], weight, type

    Each photon's path is a sequence of rows sharing the same ``id``
    (= *marker*).  The final row of every trajectory has ``type == 0``
    (termination event); its ``z`` value indicates the exit surface:

    * ``z ≈ z_top``    → escaped top surface
    * ``z ≈ z_bottom`` → escaped bottom surface
    * otherwise        → absorbed

    The *photon type* (laser / Raman / SRS) is taken from the **last
    non-zero type** seen before termination.

    Parameters
    ----------
    path :
        Path to the CSV.
    z_tol :
        Absolute tolerance (mm) for surface detection.
    z_top :
        z-coordinate of the top (entry) surface in mm.  Default ``0.0``.
    z_bottom :
        z-coordinate of the bottom surface in mm.
        Should equal ``cfg["width"]``.  Default ``0.5``.
    types_of_interest :
        Photon types to retain in ``df_summary``.  Trajectories whose
        ``prev_type`` is not in this tuple are dropped from the summary.
        Defaults to ``(1, 2, 3)`` — keep all types.

    Returns
    -------
    df_raw : pd.DataFrame
        Full trajectory table with typed columns:
        ``marker, x, y, z, t, W, type``.
        Rows are sorted by ``(marker, t)``.

    df_summary : pd.DataFrame
        One row per unique marker (trajectory) with columns:

        ============  =====================================================
        marker        photon id
        term_mode     ``"term"``  — explicit type-0 termination row found
                      ``"no_term"`` — diagnostic fallback, no type-0 row
        exit_state    ``"top"`` | ``"bottom"`` | ``"absorbed"``
        z_end         z-coordinate of the termination point (mm)
        prev_type     last non-zero photon type before termination
        max_z         maximum z reached during propagation (mm),
                      computed over non-termination rows only
        n_points      total number of rows for this marker
        ============  =====================================================

        Only trajectories with ``prev_type in types_of_interest`` are kept.
        Diagnostics are printed to stdout regardless of filtering.
    """
    path = Path(path)

    # ── 1. read raw CSV ───────────────────────────────────────────────────────
    col_names = ["marker", "x", "y", "z", "t", "W", "type"]

    df = pd.read_csv(
        path,
        sep=",",
        skipinitialspace=True,
        header=0,
        names=col_names,
        dtype=str,          # read as str first — robust against formatting quirks
    )

    for c in ["x", "y", "z", "t", "W"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["marker"] = pd.to_numeric(df["marker"], errors="coerce")
    df["type"]   = pd.to_numeric(df["type"],   errors="coerce")
    df = df.dropna(subset=["marker", "type", "z", "t"]).copy()

    df["marker"] = df["marker"].astype(np.int64)
    df["type"]   = df["type"].astype(np.int64)

    # sort within each trajectory by time
    df = df.sort_values(["marker", "t"], kind="mergesort").reset_index(drop=True)
    df_raw = df  # alias for return — same object, clear name

    # ── 2. per-trajectory summary ─────────────────────────────────────────────
    rows: List[Dict] = []

    n_total           = df["marker"].nunique()
    n_no_type0        = 0
    n_no_prev_nonzero = 0

    for marker, g in df.groupby("marker", sort=False):
        if len(g) < 2:
            continue

        types = g["type"].to_numpy()
        zvals = g["z"].to_numpy()

        term_indices = np.where(types == 0)[0]

        if len(term_indices) == 0:
            # ── diagnostic fallback: no explicit termination row
            n_no_type0 += 1
            end_idx   = len(g) - 1
            z_end     = float(zvals[end_idx])
            prev_type = int(types[end_idx])
            term_mode = "no_term"
        else:
            end_idx = int(term_indices[-1])
            z_end   = float(zvals[end_idx])

            # last non-zero type strictly before the termination row
            prev_candidates = np.where(types[:end_idx] != 0)[0]
            if len(prev_candidates) == 0:
                n_no_prev_nonzero += 1
                continue    # degenerate: only type-0 rows — skip
            prev_idx  = int(prev_candidates[-1])
            prev_type = int(types[prev_idx])
            term_mode = "term"

        # ── exit surface classification (full resolution, no placeholder)
        if _is_close(z_end, z_top, tol=z_tol):
            exit_state = "top"
        elif _is_close(z_end, z_bottom, tol=z_tol):
            exit_state = "bottom"
        else:
            exit_state = "absorbed"

        # ── max penetration depth (exclude termination rows)
        nonzero_mask = types != 0
        max_z = (
            float(np.max(zvals[nonzero_mask]))
            if np.any(nonzero_mask)
            else float(np.max(zvals))
        )

        rows.append(
            {
                "marker":    int(marker),
                "term_mode": term_mode,
                "exit_state": exit_state,
                "z_end":     z_end,
                "prev_type": prev_type,
                "max_z":     max_z,
                "n_points":  int(len(g)),
            }
        )

    # ── 3. assemble + filter summary ─────────────────────────────────────────
    df_summary = pd.DataFrame(rows)

    if df_summary.empty:
        raise RuntimeError(
            "Trajectory summary is empty — check the CSV file content, "
            "column names, and that marker/type columns are populated."
        )

    # apply type filter
    df_summary = df_summary[
        df_summary["prev_type"].isin(types_of_interest)
    ].reset_index(drop=True)

    # ── 4. diagnostics ────────────────────────────────────────────────────────
    print(f"[load_photons] Total rows       : {len(df_raw):,}")
    print(f"[load_photons] Unique markers   : {n_total:,}")
    print(f"[load_photons] Summarised       : {len(df_summary):,}  "
          f"(after type filter: {list(types_of_interest)})")
    print()
    print("[load_photons] Exit state counts (post-filter):")
    print(df_summary["exit_state"].value_counts(dropna=False).to_string(header=False))
    print()
    print("[load_photons] prev_type counts (post-filter):")
    print(df_summary["prev_type"].value_counts().sort_index().to_string(header=False))

    return df_raw, df_summary


# ══════════════════════════════════════════════════════════════════════════════
# Numeric helpers  (used internally and available to notebooks)
# ══════════════════════════════════════════════════════════════════════════════

def build_xy_coords(
    Nx: int, Ny: int, width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return centred (x, y) coordinate arrays matching the C-side index layout.

    The simulation volume spans ``[-width, +width]`` in x and y, divided into
    ``Nx`` / ``Ny`` bins respectively.

    Parameters
    ----------
    Nx, Ny:  Grid dimensions.
    width:   Half-width of the simulation volume in mm.
    """
    dx = 2.0 * width / Nx
    dy = 2.0 * width / Ny
    x  = (np.arange(Nx) - Nx / 2.0) * dx
    y  = (np.arange(Ny) - Ny / 2.0) * dy
    return x, y


def robust_clim(
    *arrays: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Tuple[float, float]:
    """
    Return (vmin, vmax) from the joint percentile distribution of all arrays.

    Useful for aligning colour scales across multiple heatmaps::

        vmin, vmax = robust_clim(laser, raman)
    """
    combined = np.concatenate([a.ravel() for a in arrays])
    finite   = combined[np.isfinite(combined)]
    return float(np.percentile(finite, p_low)), float(np.percentile(finite, p_high))


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    print("── init_config_file smoke-test ──")
    mci_path, cfg_path = init_config_file()
    print(f"  MCI : {mci_path}")
    print(f"  CFG : {cfg_path}")

    print("\n── parse_config ──")
    cfg = parse_config(cfg_path)
    print(cfg)

    print("\n── parse_mci ──")
    mci = parse_mci(mci_path)
    print(mci)