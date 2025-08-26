#!/usr/bin/env python3


import os
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Configuration (from user's script) --------------------
CSV_PATH = "MC_outputs/effareasMESE.csv"
BIN_WIDTH_LOG10 = 0.33
PHI0 =  2.72e-18
T_EXPOSURE = 11.4*365.25 * 24 * 3600.0
OMEGA = 4*np.pi
E0 = 1e5

GRID_STEP = 1/30

# Compare shapes (normalize each histogram to unit area) vs raw rates.
SHAPE_NORMALIZE = True

# Toggle optional artifact outputs
WRITE_PER_COMPOSITION_CSV = False
WRITE_PER_COMPOSITION_PNG = False

OUTDIR = "MC_outputs"
# ---------------------------------------------------------------------------

def read_effarea_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV "{path}" not found in current directory.')
    # Try no header first
    try:
        df = pd.read_csv(path, comment="#", header=None)
        if df.shape[1] < 4:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, comment="#")
        if df.shape[1] < 4:
            raise ValueError("CSV must have ≥4 columns: master, A_muon, A_tau, A_electron")

    E = np.asarray(df.iloc[:, 0], dtype=float)
    A_mu = np.asarray(df.iloc[:, 1], dtype=float)
    A_tau = np.asarray(df.iloc[:, 2], dtype=float)
    A_e  = np.asarray(df.iloc[:, 3], dtype=float)

    order = np.argsort(E)
    return E[order], A_mu[order], A_tau[order], A_e[order]


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
    dE = np.diff(edges)
    for j in range(len(centers)):
        a = edges[j]
        b = edges[j+1]
        L = max(a, Emin)
        U = min(b, Emax)
        if L < U:
            total += A[j] * ((centers[j])**-2.54) * dE[j]
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

def chi2_pearson(obs: np.ndarray, exp: np.ndarray, shape_normalize: bool = True) -> Tuple[float, int]:
    """
    Compute the Pearson chi-square statistic between observed (obs) and expected (exp).

    Parameters
    ----------
    obs : np.ndarray
        Observed counts or proportions (nonnegative).
    exp : np.ndarray
        Expected counts or proportions (must be > 0 wherever compared).
    shape_normalize : bool, default True
        If True, normalize obs and exp to sum to 1 (compares shapes only).
        If False, use values as-is.

    Returns
    -------
    chi2 : float
        Pearson chi-square statistic (NaN if nothing to compare).
    ndof : int
        Degrees of freedom: (#bins - 1) if shape_normalize else #bins.
        Clamped at >= 0.
    """
    o = np.asarray(obs, dtype=float).ravel()
    e = np.asarray(exp, dtype=float).ravel()

    if o.shape != e.shape:
        raise ValueError(f"obs and exp must have the same shape, got {o.shape} vs {e.shape}")

    # Keep only bins with strictly positive expected value
    mask = e > 0
    o = o[mask]
    e = e[mask]

    if o.size == 0:
        return float("nan"), 0

    if shape_normalize:
        o_sum = o.sum()
        e_sum = e.sum()
        # If either sum is nonpositive, statistic is undefined
        if o_sum <= 0 or e_sum <= 0:
            return float("nan"), 0
        o = o / o_sum
        e = e / e_sum

    # Pearson chi-square
    chi2 = float(np.sum((o - e) ** 2 / e))

    # Degrees of freedom
    ndof = len(e) - 1 if shape_normalize else len(e)
    return chi2, max(ndof, 0)


def generate_flavor_grid(step: float = 0.2) -> List[Tuple[float, float, float]]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    out = []
    for fe in vals:
        for fmu in vals:
            ftau = 1.0 - fe - fmu
            if ftau < -1e-9:
                continue
            ftau = float(round(ftau / step) * step)
            if ftau < 0 or ftau > 1.0:
                continue
            if abs(fe + fmu + ftau - 1.0) < 1e-6:
                out.append((float(fe), float(fmu), float(ftau)))
    out = sorted(set(out))
    return out


def plot_histogram(coarse_edges: np.ndarray, counts: np.ndarray, fe: float, fmu: float, ftau: float, out_png: str):
    lefts = coarse_edges[:-1]
    rights = coarse_edges[1:]
    widths = rights - lefts
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(lefts, counts, width=widths, align="edge", edgecolor="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Number of events per bin")
    ax.set_title(f"Events per energy bin (10 years) — fe={fe:.1f}, fμ={fmu:.1f}, fτ={ftau:.1f}")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_counts_csv(coarse_edges: np.ndarray, counts: np.ndarray, path: str):
    centers = np.sqrt(coarse_edges[:-1] * coarse_edges[1:])
    df = pd.DataFrame({
        "Emin_GeV": coarse_edges[:-1],
        "Emax_GeV": coarse_edges[1:],
        "Ecenter_GeV": centers,
        "N_events": counts,
    })
    df.to_csv(path, index=False)



def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Read and prepare
    E, A_mu, A_tau, A_e = read_effarea_csv(CSV_PATH)
    edges = geometric_edges_from_centers(E)
    coarse_edges = make_log_bins(edges, BIN_WIDTH_LOG10)
    norm = PHI0 * T_EXPOSURE * OMEGA 

    # Baseline distribution (1/3,1/3,1/3)
    fe0 = fmu0 = ftau0 = 1.0/3.0
    base_counts = events_per_coarse_bin(
        E, edges, A_e, A_mu, A_tau, coarse_edges, fe0, fmu0, ftau0, norm
    )

    # Sweep grid
    flavors = generate_flavor_grid(step=GRID_STEP)

    # Optionally save the baseline for reference
    pd.DataFrame({
        "Emin_GeV": coarse_edges[:-1],
        "Emax_GeV": coarse_edges[1:],
        "N_events_baseline": base_counts
    }).to_csv(os.path.join(OUTDIR, "baseline_counts.csv"), index=False)

    rows = []
    for (fe, fmu, ftau) in flavors:
        counts = events_per_coarse_bin(
            E, edges, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, norm
        )

        chi2, ndof = chi2_pearson(counts, base_counts, shape_normalize = True)


        rows.append({
            "fe": fe,
            "fmu": fmu,
            "ftau": ftau,
            "chi2": chi2,
            "ndof": ndof,
            "shape_normalized": SHAPE_NORMALIZE
        })

    out_csv = os.path.join(OUTDIR, "chi2_vs_baseline.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")
    print("Done.")

if __name__ == "__main__":
    main()