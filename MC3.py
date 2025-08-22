#!/usr/bin/env python3
"""
Triangle (ternary) HPD contour plot from chi^2 grid
This is for Nevents without discriminating by flavor

- Converts chi^2 to likelihood L ∝ exp[-Δχ²/2]
- Computes 1σ (68.27%) and 2σ (95.45%) HPD thresholds by integrating the likelihood over the grid
- Colors the triangle by Δχ² (lower is better)
- Draws 1σ and 2σ iso-likelihood contours on a ternary triangle (fe, fμ, fτ)
- Saves figure to 'triangle.png'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os


def load_df() -> pd.DataFrame:
    return pd.read_csv("MC_outputs/chi2_vs_baseline.csv")


def hpd_likelihood_thresholds(L: np.ndarray, masses=(0.6827, 0.9545)):
    """Return likelihood thresholds so that {L >= t} encloses each mass on the discrete grid."""
    order = np.argsort(-L)
    L_sorted = L[order]
    cum = np.cumsum(L_sorted) / L_sorted.sum()
    thresholds = []
    for t in masses:
        idx = np.searchsorted(cum, t, side="left")
        idx = min(idx, len(L_sorted) - 1)
        thresholds.append(L_sorted[idx])
    return thresholds


def bary_to_xy(fe: np.ndarray, fmu: np.ndarray, ftau: np.ndarray):

    x = fe + 0.5 * fmu
    y = (np.sqrt(3) / 2.0) * fmu
    return x, y

def main():
    df = load_df()
    for col in ("fe", "fmu", "ftau", "chi2"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    fe = df["fe"].to_numpy(float)
    fmu = df["fmu"].to_numpy(float)
    ftau = df["ftau"].to_numpy(float)
    chi2 = df["chi2"].to_numpy(float)

    # Likelihood from Δχ²
    chi2_min = float(np.min(chi2))
    dchi2 = chi2 - chi2_min
    L = np.exp(-0.5 * dchi2)

    # HPD thresholds (likelihood levels)
    L1, L2 = hpd_likelihood_thresholds(L, masses=(0.6827, 0.9545))
    # Convert to equivalent Δχ² relative to max likelihood
    dchi2_1 = float(-2.0 * np.log(L1 / L.max()))
    dchi2_2 = float(-2.0 * np.log(L2 / L.max()))

    # Best-fit
    ibest = int(np.argmin(chi2))
    best = (float(fe[ibest]), float(fmu[ibest]), float(ftau[ibest]))

    # Ternary coords and triangulation
    x, y = bary_to_xy(fe, fmu, ftau)
    tri = Triangulation(x, y)

    # Plot (single figure)
    plt.figure(figsize=(7, 7))

    # Draw triangle frame
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2.0], [0.0, 0.0]])
    plt.plot(verts[:, 0], verts[:, 1], lw=1)

    # Use tripcolor so each triangle is shaded; Δχ² makes the scale intuitive (0 at best-fit)
    tpc = plt.tripcolor(tri, dchi2, cmap = 'rainbow')
    cbar = plt.colorbar(tpc, pad=0.02, fraction=0.04)
    cbar.set_label(r"$\Delta \chi^2$")

    # Draw iso-Δχ² contours (strictly increasing & unique)
    dlevels = np.array([dchi2_1, dchi2_2], dtype=float)
    # Drop duplicates caused by coarse/discrete HPD thresholds
    dlevels = np.unique(np.round(dlevels, 12))
    if dlevels.size > 1:
        dlevels.sort()

    cs = plt.tricontour(tri, dchi2, levels=dlevels, linewidths=1.25, colors = 'red')

    if dlevels.size == 2:
        plt.clabel(cs, inline=True, fmt={dlevels[0]: "1σ", dlevels[1]: "2σ"}, fontsize=9, colors = 'red')
    elif dlevels.size == 1:
        # If thresholds collapsed, just draw one line and label accordingly
        plt.clabel(cs, inline=True, fmt={dlevels[0]: "1σ/2σ"}, fontsize=9)

    # Ticks/labels
    plt.text(-0.03, -0.03, r"$f_\tau=1$", ha="right", va="top")
    plt.text(0.95, -0.03,  r"$f_e=1$",   ha="left",  va="top")
    plt.text(0.5, np.sqrt(3)/2 + 0.06, r"$f_\mu=1$", ha="center", va="bottom")

    # Aesthetics
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Likelihood analysis: 10e4 to 10e11 GeV")

    # Save
    out_path = "MC_outputs/triangle.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    # Summary to stdout
    print("Best-fit (fe, fmu, ftau) at min χ²:", best)
    print("Min χ²:", chi2_min)
    print("HPD thresholds (likelihood levels) and equivalent Δχ²:")
    print(f"  1σ (68.27%): L ≥ {L1:.6g}  -> Δχ² ≈ {dchi2_1:.6g}")
    print(f"  2σ (95.45%): L ≥ {L2:.6g}  -> Δχ² ≈ {dchi2_2:.6g}")
    print(f" Bin width: 0.2GeV")
    print(f"Saved figure to: {out_path}")

csv_path = "MC_outputs/baseline_counts.csv"
df = pd.read_csv(csv_path)
Emin = df["Emin_GeV"].to_numpy(float)
Emax = df["Emax_GeV"].to_numpy(float)
N = df["N_events_baseline"].to_numpy(float)
widths = Emax - Emin
centers = (Emin + Emax) / 2.0
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(Emin, N, width=widths, align="edge", edgecolor="black", alpha=0.85)
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel("Baseline event count")
ax.set_title("Baseline counts per energy bin")
# Use log x-scale when energies are positive (common for energy spectra)
if np.all(Emin > 0) and np.all(Emax > 0):
    ax.set_xscale("log")
ax.grid(True, which="both", axis="both", alpha=0.25)
fig.tight_layout()
fig.savefig("MC_outputs/baseline.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()
