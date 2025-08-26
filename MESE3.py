#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile likelihood joint 2D analysis for neutrino flavors.

- Uses a 2D Poisson likelihood (energy × angle), profiles the overall
  normalization analytically, and computes test statistics:

  q(θ) = -2 [ ln L(θ, μ̂(θ)) - ln L(θ̂, μ̂) ]

- Provides:
  (A) Triangle (ternary) plot with Δχ² (2 dof) contours at 68%/95% (2.30, 5.99)
  (B) 1-dof profile scans q(fe), q(fμ), q(fτ) with thresholds 1.00 (68%),
      2.71 (90%).

Inputs
------
- Effective areas CSV at MC_outputs/effareasMESE.csv with columns:
  E_center, A_mu, A_tau, A_e   (header optional; comments start with '#')
- Optional observed data file MC_outputs/obs2d.npz with array 'O' (2D). If not
  present, an Asimov dataset is generated from the baseline (1/3,1/3,1/3).

Outputs
-------
- MC_outputs/profile_triangle.png      (2 dof contours on ternary)

Usage
-----
  python profile_joint2d_likelihood.py
"""
from __future__ import annotations

import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.patheffects as pe  # kept in case you annotate later

import time

def _normalize_bary(fe, fmu, ftau):
    s = fe + fmu + ftau
    if s <= 0:
        return fe, fmu, ftau
    return fe/s, fmu/s, ftau/s


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------- Configuration --------------------
CSV_PATH = "MC_outputs/effareasMESE.csv"
BIN_WIDTH_LOG10 = 0.2
PHI0 = 2.72e-18
T_EXPOSURE = 11.4 * 365.25 * 24 * 3600.0
OMEGA = 4 * np.pi
E0 = 1e5
GRID_STEP = 1 / 60.0   # finer for smoother triangle
OUTDIR = "MC_outputs"

# 2 dof Wilks (fe,fmu) with ftau = 1 - fe - fmu
WILKS_2DOF_LEVELS = (2.30, 5.99)  # 68%, 95%
# 1 dof Wilks thresholds for a single parameter-of-interest
WILKS_1DOF_LEVELS = (1.00, 2.71)  # 68%, 90%

# -------------------- I/O helpers --------------------
def read_effarea_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV "{path}" not found in current directory.')
    try:
        df = pd.read_csv(path, comment="#", header=None)
        if df.shape[1] < 4:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, comment="#")
        if df.shape[1] < 4:
            raise ValueError("CSV must have ≥4 columns: E, A_muon, A_tau, A_electron")

    E = np.asarray(df.iloc[:, 0], dtype=float)
    A_mu = np.asarray(df.iloc[:, 1], dtype=float)
    A_tau = np.asarray(df.iloc[:, 2], dtype=float)
    A_e = np.asarray(df.iloc[:, 3], dtype=float)

    order = np.argsort(E)
    return E[order], A_mu[order], A_tau[order], A_e[order]


def geometric_edges_from_centers(E: np.ndarray) -> np.ndarray:
    if len(E) < 2:
        raise ValueError("Need at least 2 energy points to build edges.")
    edges = np.zeros(len(E) + 1, dtype=float)
    edges[1:-1] = np.sqrt(E[:-1] * E[1:])
    edges[0] = E[0] / np.sqrt(E[1] / E[0])
    edges[-1] = E[-1] * np.sqrt(E[-1] / E[-2])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Computed edges are not strictly increasing.")
    return edges


def make_log_bins(edges: np.ndarray, bin_width_log10: float) -> np.ndarray:
    lo = math.log10(edges[0])
    hi = math.log10(edges[-1])
    start = lo
    stop = lo + math.ceil((hi - lo) / bin_width_log10) * bin_width_log10
    raw = 10.0 ** np.arange(start, stop + 1e-12, bin_width_log10)
    raw[0] = edges[0]
    return np.unique(raw)


def integrate_bin_I(edges: np.ndarray, centers: np.ndarray, A: np.ndarray, Emin: float, Emax: float) -> float:
    total = 0.0
    dE = np.diff(edges)
    for j in range(len(centers)):
        a = edges[j]; b = edges[j + 1]
        L = max(a, Emin); U = min(b, Emax)
        if L < U:
            total += A[j] * ((centers[j] / E0) ** -2.54) * dE[j]
    return float(total)


def events_per_coarse_bin(E, edges, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, norm):
    counts = []
    for i in range(len(coarse_edges) - 1):
        Emin = float(coarse_edges[i]); Emax = float(coarse_edges[i + 1])
        Ie = integrate_bin_I(edges, E, A_e, Emin, Emax)
        Imu = integrate_bin_I(edges, E, A_mu, Emin, Emax)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax)
        counts.append(norm * (Ie * fe + Imu * fmu + Itau * ftau))
    return np.asarray(counts)


def generate_flavor_grid(step: float = 0.2) -> List[Tuple[float, float, float]]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    out = []
    for fe in vals:
        for fmu in vals:
            ftau = 1.0 - fe - fmu
            if ftau < -1e-9:  # below simplex
                continue
            ftau = float(round(ftau / step) * step)
            if ftau < 0 or ftau > 1.0:
                continue
            if abs(fe + fmu + ftau - 1.0) < 1e-6:
                out.append((float(fe), float(fmu), float(ftau)))
    return sorted(set(out))

# -------------------- Angular templates --------------------
def _angular_templates():
    casc = {-0.9: 3.5e2, -0.7: 5.0e2, -0.5: 6.0e2, -0.3: 7.0e2, -0.1: 8.0e2,
             0.1: 1.0e3, 0.3: 5.0e2, 0.5: 2.5e2, 0.7: 1.5e2, 0.9: 6.0e1}
    tracks = {-0.9: 1.5e3, -0.7: 9.0e2, -0.5: 9.0e2, -0.3: 1.0e3, -0.1: 6.0e2,
               0.1: 1.5e2, 0.3: 1.5e1, 0.5: 1.7e1, 0.7: 5.0e0, 0.9: 1.5e1}
    db = {-0.7: 1.0, -0.5: 2.0, -0.1: 1.0, 0.1: 2.0, 0.3: 1.0, 0.5: 2.0}
    keys = sorted(set(casc.keys()) | set(tracks.keys()) | set(db.keys()))
    def to_array(d): return np.array([d.get(k, 0.0) for k in keys], dtype=float)
    return np.array(keys, float), to_array(casc), to_array(tracks), to_array(db)


def angular_expected_hist(fe: float, fmu: float, ftau: float):
    cosbins, casc, track, db = _angular_templates()
    frace = 1
    fracmu = 0
    fractau = 0
    counts = ((casc * frace) * fe + (casc * fracmu + track) * fmu + (casc * fractau + db) * ftau) / (1 / 3)
    return cosbins, counts

# -------------------- Likelihood --------------------
def make_template_2d(fe, fmu, ftau, E, edges, A_e, A_mu, A_tau, coarse_edges, norm):
    en = events_per_coarse_bin(E, edges, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, norm)
    _, ang = angular_expected_hist(fe, fmu, ftau)
    return np.outer(en, ang)


def profiled_deviance(obs2d: np.ndarray, templ2d: np.ndarray) -> float:
    O = np.ravel(np.asarray(obs2d, float))
    T = np.ravel(np.asarray(templ2d, float))
    m = T > 0
    O, T = O[m], T[m]
    if O.size == 0:
        return float("nan")
    mu_hat = O.sum() / T.sum()
    M = mu_hat * T
    term = np.where(O > 0, O * np.log(np.maximum(O, 1e-300) / M), 0.0)
    return float(2.0 * np.sum(M - O + term))

# -------------------- Geometry helpers --------------------
def bary_to_xy(fe, fmu, ftau):
    x = fe + 0.5 * fmu
    y = (np.sqrt(3) / 2.0) * fmu
    return x, y

# -------------------- Main pipeline --------------------
def main():

    POINTS = {
    "0:1:0": (0.17, 0.45, 0.37),
    "1:2:0": (0.30, 0.36, 0.34),
    "1:0:0": (0.55, 0.17, 0.28),
    "1:1:0": (0.36, 0.31, 0.33),
    }
    
    log('Starting main pipeline')
    os.makedirs(OUTDIR, exist_ok=True)

    # Effective areas and binning
    log('Reading effarea CSV')
    E, A_mu, A_tau, A_e = read_effarea_csv(CSV_PATH)
    edges = geometric_edges_from_centers(E)
    coarse_edges = make_log_bins(edges, BIN_WIDTH_LOG10)
    norm = PHI0 * T_EXPOSURE * OMEGA

    # Build observed 2D (use Asimov by default)
    asimov_flavors = (1/3, 1/3, 1/3)
    obs_path = os.path.join(OUTDIR, "obs2d.npz")
    if os.path.exists(obs_path):
        log('Loading observed dataset from npz')
        O = np.load(obs_path)["O"]
    else:
        log('Generating Asimov observed dataset')
        O = make_template_2d(*asimov_flavors, E, edges, A_e, A_mu, A_tau, coarse_edges, norm)

    # Flavor grid
    log('Generating flavor grid')
    grid = generate_flavor_grid(step=GRID_STEP)
    fe_arr = np.array([g[0] for g in grid], dtype=float)
    fmu_arr = np.array([g[1] for g in grid], dtype=float)
    ftau_arr = np.array([g[2] for g in grid], dtype=float)

    # Compute profiled deviance over the grid
    log('Scanning likelihood over grid')
    chi = np.empty(len(grid), dtype=float)
    for i, (fe, fmu, ftau) in enumerate(grid):
        T2 = make_template_2d(fe, fmu, ftau, E, edges, A_e, A_mu, A_tau, coarse_edges, norm)
        chi[i] = profiled_deviance(O, T2)

    # Δχ² relative to the global minimum
    chi_min = np.nanmin(chi)
    dchi = chi - chi_min
    # Clean up possible numerical noise
    dchi = np.where(np.isfinite(dchi), dchi, np.nan)

     # Triangle plot in exclusion confidence (%)
    log('Plotting triangle as exclusion confidence (%)')
    x, y = bary_to_xy(fe_arr, fmu_arr, ftau_arr)
    tri = Triangulation(x, y)

    # Convert Δχ² (2 dof) -> exclusion confidence fraction and percent
    conf_frac = 1.0 - np.exp(-0.5 * dchi)        # for k=2 dof
    conf_pct  = 100.0 * conf_frac

    plt.figure(figsize=(7.4, 7.2))
    # draw triangle border
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2.0], [0.0, 0.0]])
    plt.plot(verts[:,0], verts[:,1], lw=1, color="k")

    for tag, (pfe, pfmu, pftau) in POINTS.items():
        pfe, pfmu, pftau = _normalize_bary(pfe, pfmu, pftau)  # guard tiny rounding
        px, py = bary_to_xy(np.array([pfe]), np.array([pfmu]), np.array([pftau]))
        # marker
        plt.scatter(px, py, s=55, facecolors="white", edgecolors="black",
                    linewidths=0.9, zorder=6)
        # label with a light outline for readability
        plt.annotate(tag, (px[0], py[0]), textcoords="offset points", xytext=(6, 6),
                     fontsize=9, weight="bold", zorder=7,
                     path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        

    # smooth filled background in percent
    # (0 → 100% scale with ~2% steps)
    levels_bg = np.linspace(0.0, 100.0, 51)
    tpc = plt.tricontourf(tri, conf_pct, levels=levels_bg)
    cbar = plt.colorbar(tpc, pad=0.02, fraction=0.04)
    cbar.set_label("Exclusion confidence (%)")

    # --- Requested confidence contours (solid) ---
    conf_levels = (68.0, 95.0)
    cs = plt.tricontour(tri, conf_pct, levels=conf_levels,
                        colors='red', linewidths=2.0, linestyles="-")
    fmt = {conf_levels[0]: "68%", conf_levels[1]: "95%"}
    plt.clabel(cs, inline=True, fmt=fmt, fontsize=9, colors='red')

    # Legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color="red", lw=2, linestyle="-", label="68% / 95%"),
    ]
    plt.legend(handles=legend_lines, loc="upper right", frameon=False)

    # Annotate corners
    plt.text(-0.03, -0.03, r"$f_\tau=1$", ha="right", va="top")
    plt.text(0.95, -0.03,  r"$f_e=1$",   ha="left",  va="top")
    plt.text(0.5, np.sqrt(3)/2 + 0.06, r"$f_\mu=1$", ha="center", va="bottom")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Profile Likelihood: Exclusion confidence (%)")
    out_tri = os.path.join(OUTDIR, "profile_triangle.png")
    plt.tight_layout()
    plt.savefig(out_tri, dpi=220, bbox_inches="tight")
    plt.close()

    ibest = int(np.nanargmin(chi))
    print("Global best-fit (fe, fmu, ftau) =", (fe_arr[ibest], fmu_arr[ibest], ftau_arr[ibest]))
    print("Min profiled deviance:", float(chi_min))
    print("Saved:", out_tri)

    log('Plotting 2×2 energy and angle histograms')

    # Flavor cases: 001 (tau), 010 (mu), 100 (e), and 1/3-1/3-1/3
    cases = [
        (0.0, 0.0, 1.0, r"$f=(0,0,1)$"),
        (0.0, 1.0, 0.0, r"$f=(0,1,0)$"),
        (1.0, 0.0, 0.0, r"$f=(1,0,0)$"),
        (1/3, 1/3, 1/3, r"$f=(1/3,1/3,1/3)$"),
    ]

    # Helper to get normalized energy and angle 1D histograms from the 2D template
    def get_marginals_norm(fe, fmu, ftau):
        T2 = make_template_2d(fe, fmu, ftau, E, edges, A_e, A_mu, A_tau, coarse_edges, norm)
        # energy marginal: sum over angle axis (axis=1 in 2D outer convention)
        mE = T2.sum(axis=1)
        # angle marginal: sum over energy axis
        mA = T2.sum(axis=0)
        # normalize to unit area for shape comparison
        if mE.sum() > 0:
            mE = mE / mE.sum()
        if mA.sum() > 0:
            mA = mA / mA.sum()
        return mE, mA

    # Prepare binning
    # Energy: use coarse_edges for bins; plot at bin centers
    E_centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    E_widths = (coarse_edges[1:] - coarse_edges[:-1])
    # Angle: use the discrete cosθ bins from templates and give them a nominal width
    cosbins, _, _, _ = _angular_templates()
    # Build simple edges around the provided centers for bar widths
    cos_edges = np.zeros(len(cosbins) + 1, dtype=float)
    cos_edges[1:-1] = 0.5 * (cosbins[:-1] + cosbins[1:])
    # extrapolate end edges
    cos_edges[0]  = cosbins[0]  - (cos_edges[1] - cosbins[0])
    cos_edges[-1] = cosbins[-1] + (cosbins[-1] - cos_edges[-2])
    cos_widths = cos_edges[1:] - cos_edges[:-1]

    # ---------- Energy 2×2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (fe, fmu, ftau, label) in zip(axes, cases):
        mE, _ = get_marginals_norm(fe, fmu, ftau)
        ax.bar(E_centers, mE, width=E_widths, align='center', edgecolor='k', linewidth=0.5)
        ax.set_title(label)
        ax.set_xscale('log')
        ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.supylabel("Normalized counts")
    fig.supxlabel("Energy (GeV)")
    fig.tight_layout()
    out_energy = os.path.join(OUTDIR, "energy_2x2.png")
    fig.savefig(out_energy, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---------- Angle 2×2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (fe, fmu, ftau, label) in zip(axes, cases):
        _, mA = get_marginals_norm(fe, fmu, ftau)
        ax.bar(cosbins, mA, width=cos_widths, align='center', edgecolor='k', linewidth=0.5)
        ax.set_title(label)
        ax.set_xlim(cos_edges[0], cos_edges[-1])
        ax.grid(True, ls=':', alpha=0.5)
    fig.supylabel("Normalized counts")
    fig.supxlabel(r"$\cos\theta$")
    fig.tight_layout()
    out_angle = os.path.join(OUTDIR, "angle_2x2.png")
    fig.savefig(out_angle, dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
