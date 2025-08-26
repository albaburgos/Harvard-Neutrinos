#!/usr/bin/env python3
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

# ------------------------ Data ------------------------
E1 = np.array([4e4,6e4,1e5,2.1e5,3.5e5,5.5e5], dtype=float)
E2 = np.array([1.3e3,2.0e3,3.0e3,4.2e3,6.5e3,1.0e4,1.6e4,2.3e4,3.5e4,5.5e4,8.0e4,1.2e5,1.9e5,3.0e5,4.2e5,6.7e5,1.0e6,2.0e6], dtype=float)
E3 = np.array([1.5e3,2.0e3,6.0e3,1.3e4,2.5e4,5.0e4,1.0e5,2.0e5,4.0e5,3.5e6], dtype=float)

Ncasc = np.array([1.0e3,1.2e3,1.0e3,7.0e2,4.5e2,3.0e2,2.0e2,1.4e2,8.0e1,4.0e1,3.0e1,1.5e1,6.0,3.0,2.0,2.0,1.0,1.0], dtype=float)
Ndb   = np.array([3,1,2,1,1,1], dtype=float)
Ntrack= np.array([2.6e3,1.5e3,6.0e2,2.5e2,1.0e2,4.0e1,1.5e1,2.0,1.0,1.0], dtype=float)

STEP = 1/30       
bin_width_log10 = 0.1   
SHAPE_NORMALIZE = True
OUTPUT_CSV = "MC_outputs/MESE.csv"
OUTPUT_PNG = "MC_outputs/MESE_grid.png"

# ------------------------ Helpers ------------------------
def chi2_pearson(obs: np.ndarray, exp: np.ndarray, shape_normalize: bool = True):
    o = np.asarray(obs, dtype=float)
    e = np.asarray(exp, dtype=float)
    mask = e > 0
    o = o[mask]; e = e[mask]
    if o.size == 0:
        return float("nan"), 0
    if shape_normalize:
        o_sum = o.sum(); e_sum = e.sum()
        if o_sum > 0: o = o / o_sum
        if e_sum > 0: e = e / e_sum
    chi2 = float(np.sum((o - e) ** 2 / e))
    ndof = len(e) - (1 if shape_normalize else 0)
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

@dataclass
class CascadeRatios:
    tau: float = 127.0
    mu: float  = 22.0
    e: float   = 80.0
    def to_fracs(self):
        tot = self.tau + self.mu + self.e
        if tot <= 0:
            raise ValueError("Cascade ratio total must be positive.")
        return (self.e / tot, self.mu / tot, self.tau / tot)

def split_cascades(ncasc: np.ndarray, ratios: CascadeRatios):
    fe, fmu, ftau = ratios.to_fracs()
    return ncasc * fe, ncasc * fmu, ncasc * ftau

def build_common_grid(*arrays):
    return np.unique(np.concatenate(arrays))

def project_to_grid(E_src: np.ndarray, y_src: np.ndarray, E_grid: np.ndarray) -> np.ndarray:
    y = np.zeros_like(E_grid, dtype=float)
    idx = {float(e): i for i, e in enumerate(E_grid)}
    for e, v in zip(E_src, y_src):
        j = idx.get(float(e))
        if j is not None:
            y[j] = v
    return y

def reweight_distribution(fe: float, fmu: float, ftau: float,
                          Ntau: np.ndarray, Nmu: np.ndarray,
                          Ncasc_e: np.ndarray, Ncasc_mu: np.ndarray, Ncasc_tau: np.ndarray) -> np.ndarray:
    base = 1.0/3.0
    se, smu, stau = fe/base, fmu/base, ftau/base
    return stau*(Ntau + Ncasc_tau) + smu*(Nmu + Ncasc_mu) + se*(Ncasc_e)

# --- Provided bin helpers ---
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
    start = lo
    stop = lo + math.ceil((hi - lo) / bin_width_log10) * bin_width_log10
    raw = 10.0 ** np.arange(start, stop + 1e-12, bin_width_log10)
    raw[0] = edges[0]
    b = np.unique(raw)
    return b

def hist_counts(E_points: np.ndarray, weights: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # Histogram of discrete points with weights (sum per bin)
    h, _ = np.histogram(E_points, bins=bin_edges, weights=weights)
    return h.astype(float)

# ------------------------ Main ------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Split cascades
    ratios = CascadeRatios(127.0,22.0,80.0)
    Ncasc_e_src, Ncasc_mu_src, Ncasc_tau_src = split_cascades(Ncasc, ratios)

    # Union energy grid
    E_grid = build_common_grid(E1, E2, E3)
    Ncasc_e  = project_to_grid(E2, Ncasc_e_src,  E_grid)
    Ncasc_mu = project_to_grid(E2, Ncasc_mu_src, E_grid)
    Ncasc_tau= project_to_grid(E2, Ncasc_tau_src,E_grid)
    Nmu      = project_to_grid(E3, Ntrack,       E_grid)
    Ntau     = project_to_grid(E1, Ndb,          E_grid)

    print(Ntau, Nmu)

    baseline_counts = (Ntau + Ncasc_tau) + (Nmu + Ncasc_mu) + (Ncasc_e)

    # Build histogram binning
    edges_geom = geometric_edges_from_centers(E_grid)
    bin_edges  = make_log_bins(edges_geom, bin_width_log10)

    # Build histograms (sum counts inside bins)
    base_hist = hist_counts(E_grid, baseline_counts, bin_edges)

    def make_hist_for(fe,fmu,ftau):
        y = reweight_distribution(fe,fmu,ftau, Ntau,Nmu, Ncasc_e,Ncasc_mu,Ncasc_tau)
        return hist_counts(E_grid, y, bin_edges)

    pure_specs = {
        "ftau=1": (0.0, 0.0, 1.0),
        "fmu=1":  (0.0, 1.0, 0.0),
        "fe=1":   (1.0, 0.0, 0.0),
    }
    pure_hists = {name: make_hist_for(*triple) for name, triple in pure_specs.items()}

    # 2x2 plot: baseline + 3 pure flavors
    fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey=True)
    ax_list = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

    # Baseline
    ax_list[0].step(bin_edges[:-1], base_hist, where='post', label='baseline')
    ax_list[0].set_xscale('log'); ax_list[0].set_yscale('log')
    ax_list[0].set_title('Baseline')
    ax_list[0].grid(True, which='both', alpha=0.3); ax_list[0].legend()

    # Pure panels
    for ax, (name, triple) in zip(ax_list[1:], pure_specs.items()):
        hist = pure_hists[name]
        ax.step(bin_edges[:-1], hist, where='post', label=name)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(name); ax.grid(True, which='both', alpha=0.3); ax.legend()

    for ax in ax_list:
        ax.set_xlabel('Energy (GeV)'); ax.set_ylabel('Counts / bin')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)

    # Chi2 on the HISTOGRAM shapes (expected vs generated)
    rows = []
    grid = generate_flavor_grid(step=STEP)
    for fe,fmu,ftau in grid:
        print
        y_hist = make_hist_for(fe,fmu,ftau)
        chi2, ndof = chi2_pearson(obs=y_hist, exp=base_hist, shape_normalize=SHAPE_NORMALIZE)
        rows.append({
            "fe": fe,
            "fmu": fmu,
            "ftau": ftau,
            "chi2": chi2,
            "ndof": ndof,
            "shape_normalized": SHAPE_NORMALIZE
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote CSV to {OUTPUT_CSV}")
    print(f"Wrote plot to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()