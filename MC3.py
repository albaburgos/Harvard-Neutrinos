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
import matplotlib.patheffects as pe


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




def hpd_exclusion_percent(L: np.ndarray) -> np.ndarray:
    """
    For each grid point, return the HPD 'exclusion %' = 100 * (posterior mass with likelihood STRICTLY greater than L_i).
    Ties get the same exclusion %, equal to the mass before the entire tie group.
    """
    order = np.argsort(-L)                   # descending by L
    Ls = L[order]
    csum = np.cumsum(Ls)
    total = csum[-1]

    # Identify tie groups in the sorted array
    # For each group, exclusion mass is csum[start-1]/total (0 if start==0)
    # Assign the same value to all members of the tie group
    _, starts, counts = np.unique(Ls, return_index=True, return_counts=True)
    excl_sorted = np.empty_like(Ls)
    for s, c in zip(starts, counts):
        mass_before = 0.0 if s == 0 else csum[s-1] / total
        excl_sorted[s:s+c] = mass_before

    # Map back to original order
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    excl = excl_sorted[inv]
    return 100.0 * excl  # percent

def bary_to_xy(fe: np.ndarray, fmu: np.ndarray, ftau: np.ndarray):

    x = fe + 0.5 * fmu
    y = (np.sqrt(3) / 2.0) * fmu
    return x, y

POINTS = {
    "0:1:0": (0.17, 0.45, 0.37),
    "1:2:0": (0.30, 0.36, 0.34),
    "1:0:0": (0.55, 0.17, 0.28),
    "1:1:0": (0.36, 0.31, 0.33),
}

def _normalize_bary(fe, fmu, ftau):
    s = fe + fmu + ftau
    if s <= 0:
        return fe, fmu, ftau
    return fe/s, fmu/s, ftau/s

def main():
    df = load_df()
    for col in ("fe", "fmu", "ftau", "chi2"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    fe = df["fe"].to_numpy(float)
    fmu = df["fmu"].to_numpy(float)
    ftau = df["ftau"].to_numpy(float)
    chi2 = df["chi2"].to_numpy(float)

    chi2_min = float(np.min(chi2))
    dchi2 = chi2 - chi2_min
    L = np.exp(-0.5 * dchi2)

    # HPD thresholds (likelihood levels)
    L1, L2 = hpd_likelihood_thresholds(L, masses=(0.6827, 0.9545))
    # Equivalent Δχ² (for contours only)
    dchi2_1 = float(-2.0 * np.log(L1 / L.max()))
    dchi2_2 = float(-2.0 * np.log(L2 / L.max()))

    # Best-fit
    ibest = int(np.argmin(chi2))
    best = (float(fe[ibest]), float(fmu[ibest]), float(ftau[ibest]))
    print(ibest, float(fe[ibest]))

    # Ternary coords and triangulation
    x, y = bary_to_xy(fe, fmu, ftau)
    tri = Triangulation(x, y)

    # --- PLOT ---
    plt.figure(figsize=(7, 7))

    # Frame
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2.0], [0.0, 0.0]])
    plt.plot(verts[:, 0], verts[:, 1], lw=1)

    # NEW: color by HPD exclusion (%) instead of Δχ²
    excl_pct = hpd_exclusion_percent(L)           # 0% (best) → 100% (worst)
    tpc = plt.tripcolor(tri, excl_pct, vmin=0.0, vmax=100.0, cmap='nipy_spectral_r', alpha = 0.6)
    cbar = plt.colorbar(tpc, pad=0.02, fraction=0.04)
    cbar.set_label("Confidence Level Exclusion [%] ")

    # Keep contours at 1σ / 2σ using Δχ² thresholds so they match your HPD levels
    dlevels = np.array([dchi2_1, dchi2_2], dtype=float)
    dlevels = np.unique(np.round(dlevels, 12))
    dlevels.sort()
    cs = plt.tricontour(tri, dchi2, levels=dlevels, linewidths=1.25, colors='red')


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
        
    # Labels
    if dlevels.size == 2:
        plt.clabel(cs, inline=True, fmt={dlevels[0]: "1σ", dlevels[1]: "2σ"}, fontsize=9, colors='red')
    elif dlevels.size == 1:
        plt.clabel(cs, inline=True, fmt={dlevels[0]: "1σ/2σ"}, fontsize=9)

    # Corner labels, aesthetics, title unchanged...
    plt.text(-0.03, -0.03, r"$f_\tau=1$", ha="right", va="top")
    plt.text(0.95, -0.03,  r"$f_e=1$",   ha="left",  va="top")
    plt.text(0.5, np.sqrt(3)/2 + 0.06, r"$f_\mu=1$", ha="center", va="bottom")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Likelihood analysis (shaded by HPD exclusion %)")


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
    print(f" Bin width: 0.1GeV")
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
