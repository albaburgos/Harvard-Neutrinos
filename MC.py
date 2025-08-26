#!/usr/bin/env python3
"""
Four-configurations-in-one-figure (no changes to your original code required).

This standalone script:
- Loads effective areas from 'effareasmc.csv' (fallback to 'effareasicecube.csv').
- Recomputes edges and integrates A_i(E)*E^-2 in coarse log10 bins (Δlog10E=0.2).
- Applies the "limiting flavor" rule per bin.
- Produces ONE figure with 4 panels (2x2), for flavor mixes:
    (1/3,1/3,1/3), (1,0,0), (0,1,0), (0,0,1).
- Saves: 'four_configs_one_figure.png' in the current directory.

Run separately from your code; it does not modify your files.
"""

import os, math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Constants (match your setup) --------------------
POSSIBLE_CSVS = [ "MC_outputs/effareasMESE.csv"]
BIN_WIDTH_LOG10 = 0.5
PHI0 = 2.78e-18*3
T_EXPOSURE = 11.4*365.25 * 24 * 3600.0
OMEGA = 4*np.pi
E0 = 1e5

FLAVORS = [
    (1/3, 1/3, 1/3),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]
TITLES = [
    "fe=1/3, fμ=1/3, fτ=1/3",
    "fe=1, fμ=0, fτ=0",
    "fe=0, fμ=1, fτ=0",
    "fe=0, fμ=0, fτ=1",
]
OUTFILE = "MC_outputs/four_configs_one_figure.png"
# ---------------------------------------------------------------------

def read_effarea():
    for path in POSSIBLE_CSVS:
        if os.path.exists(path):
            # Try headerless first
            try:
                df = pd.read_csv(path, comment="#", header=None)
                if df.shape[1] >= 4:
                    E = np.asarray(df.iloc[:,0], dtype=float)
                    A_mu = np.asarray(df.iloc[:,1], dtype=float)
                    A_tau = np.asarray(df.iloc[:,2], dtype=float)
                    A_e  = np.asarray(df.iloc[:,3], dtype=float)
                    order = np.argsort(E)
                    print(f"[info] Loaded '{path}' (no header).")
                    return E[order], A_e[order], A_mu[order], A_tau[order]
            except Exception:
                pass
            # Try with header row
            df = pd.read_csv(path, comment="#")
            if df.shape[1] < 4:
                raise ValueError(f"CSV '{path}' must have ≥4 columns: E, A_mu, A_tau, A_e")
            E = np.asarray(df.iloc[:,0], dtype=float)
            A_mu = np.asarray(df.iloc[:,1], dtype=float)
            A_tau = np.asarray(df.iloc[:,2], dtype=float)
            A_e  = np.asarray(df.iloc[:,3], dtype=float)
            order = np.argsort(E)
            print(f"[info] Loaded '{path}' (with header).")
            return E[order], A_e[order], A_mu[order], A_tau[order]
    raise FileNotFoundError(
        "err"
    )

def geometric_edges_from_centers(E: np.ndarray) -> np.ndarray:
    if len(E) < 2:
        raise ValueError("Need at least 2 energy points to build edges.")
    edges = np.zeros(len(E) + 1, dtype=float)
    edges[1:-1] = np.sqrt(E[:-1] * E[1:])
    edges[0]  = E[0]  / np.sqrt(E[1] / E[0])
    edges[-1] = E[-1] * np.sqrt(E[-1] / E[-2])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Computed edges are not strictly increasing.")
    return edges


def make_log_bins(edges: np.ndarray, bin_width_log10: float) -> np.ndarray:
    lo = math.log10(edges[0])
    hi = math.log10(edges[-1])

    # start from the first edge's log10
    start = lo
    # extend in equal log10 steps until we cover the max edge (may exceed it)
    stop = lo + math.ceil((hi - lo) / bin_width_log10) * bin_width_log10

    # generate edges at fixed log10 spacing
    raw = 10.0 ** np.arange(start, stop + 1e-12, bin_width_log10)

    # ensure the very first edge is exactly edges[0]
    raw[0] = edges[0]

    # return as-is (no clipping to edges[-1])
    b = np.unique(raw)
    return b

def integrate_bin_I(edges: np.ndarray, centers: np.ndarray, A: np.ndarray, Emin: float, Emax: float) -> float:
    total = 0.0
    print (centers)
    dE = np.diff(edges)
    for j in range(len(centers)):
        a = edges[j]
        b = edges[j+1]
        L = max(a, Emin)
        U = min(b, Emax)
        if L < U:
            total += A[j] * ((centers[j]/E0)**-2.6) * dE[j] 
    return float(total)


def events_per_coarse_bin(E: np.ndarray, edges: np.ndarray, A_e: np.ndarray, A_mu: np.ndarray, A_tau: np.ndarray,
                          coarse_edges: np.ndarray, fe: float, fmu: float, ftau: float, norm: float):
    counts = []
    for i in range(len(coarse_edges) - 1):
        Emin = float(coarse_edges[i])
        Emax = float(coarse_edges[i + 1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax)
        Imu  = integrate_bin_I(edges, E, A_mu, Emin, Emax)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax)
        counts.append(norm*(Ie*fe+Imu*fmu+Itau*ftau))


    return np.asarray(counts)

def main():
    # Load data
    E, A_e, A_mu, A_tau = read_effarea()
    edges = geometric_edges_from_centers(E)
    coarse_edges = make_log_bins(edges, BIN_WIDTH_LOG10)
    norm = PHI0 * T_EXPOSURE * OMEGA 

    # Compute counts for all four mixes
    counts_list = []
    for (fe, fmu, ftau) in FLAVORS:
        counts = events_per_coarse_bin(E, edges, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, norm)
        counts_list.append(counts)

    # ---- One 2x2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    lefts = coarse_edges[:-1]
    rights = coarse_edges[1:]
    widths = rights - lefts
    centers = np.sqrt(lefts * rights)  # geometric centers for log bins

    for ax, counts, title in zip(axes, counts_list, TITLES):
        y = np.asarray(counts, dtype=float)
        yerr = np.sqrt(np.maximum(y, 0.0))  # error scales as sqrt(value)

        ax.errorbar(
            centers, y, yerr=yerr,
            fmt="o", markersize=4, linewidth=1.0,
            capsize=2, elinewidth=0.9
        )
        # If you also want to visualize bin widths horizontally, uncomment:
        # ax.errorbar(centers, y, xerr=widths/2, yerr=yerr, fmt="o", capsize=2, elinewidth=0.9)

        ax.set_xscale("log")
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel("Events/bin (10 yr)")
        ax.set_title(title)
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.set_xlim(lefts[0], rights[-1])
        ax.set_ylim(bottom=0)

    fig.suptitle("Monte Carlo Triangle Analysis — Four Flavor Configurations", y=0.98)
    fig.tight_layout(rect=[0, 0.00, 1, 0.97])
    fig.savefig(OUTFILE, dpi=170)
    print(f"[done] Saved: {OUTFILE}")

if __name__ == "__main__":
    main()
