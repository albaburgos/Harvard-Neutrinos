"""
Compute and plot IceCube-like A_eff from binned counts using an E^-2 astrophysical flux.
Assumes bin *centers* are given; reconstructs bin edges by the geometric-mean method.

This version plots TWO overlaid curves:
  - "original" from the provided counts
  - "scaled-last" where only the last bin's N_counts is divided by 1.3

Outputs:
  - CSV with both A_eff variants [cm^2 sr] and their direction-averaged values over Ω=2π [m^2]
  - PNG plots for both quantities (two curves each)
  - Additional compact CSV with Muon and Tau direction-averaged A_eff (m^2) per energy

Notes:
  - The compact Muon/Tau CSV assumes *no channel separation* is available; both columns are set equal to the
    all-channel direction-averaged A_eff. Replace with your own per-channel arrays if you have them.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- User inputs -----------------------
# Livetime in years (Julian year is used for conversion)
T_years: float = 9.5

# Flux model: phi(E) = phi0 * (E/E0)^m
phi0: float = 1.44e-18   # GeV^-1 cm^-2 s^-1 sr^-1  (per flavor)
m: float    = -2.0       # spectral exponent (E^-2)
E0: float   = 1e5        # GeV (pivot)

# Solid angle for averaging (default: hemisphere, e.g., upgoing)
Omega_sr: float = 2.0 * math.pi  # 2π sr

# Pairs of (Energy_center_GeV, N_counts)
pairs_raw: Iterable[Tuple[float, float]] = [
    (1.1e2, 9.0e3),
    (1.5e2, 1.2e4),
    (2.0e2, 1.8e4),
    (2.2e2, 2.3e4),
    (3.8e2, 4.0e4),
    (4.5e2, 6.0e4),
    (7.0e2, 8.0e4),
    (1.1e3, 6.0e4),
    (2.0e3, 4.0e4),
    (3.0e3, 2.0e4),
    (7.0e3, 5.0e3),
    (9.0e3, 2.5e3),
    (2.0e4, 8.2e2),
    (3.0e4, 3.7e2),
    (4.5e4, 2.0e2),
    (7.0e4, 1.0e2),
    (1.5e5, 3.3e1),
    (3.0e5, 1.0e1),
    (5.0e5, 3.0e0),
    (1.0e6, 2.0e0),
    (4.0e6, 1.0e0),
]

# ----------------------- Core math -----------------------

def integral_powerlaw(Elo: float, Ehi: float, m: float, E0: float) -> float:
    """Integral of (E/E0)^m dE over [Elo, Ehi]."""
    if not (Ehi > Elo > 0.0):
        return float("nan")
    if abs(m + 1.0) < 1e-14:
        # m = -1 special case: ∫ (E/E0)^(-1) dE = ln(Ehi/Elo)
        return math.log(Ehi / Elo)
    # General case: E0^{-m} * (E^{m+1}/(m+1)) |_{Elo}^{Ehi}
    return (Ehi ** (m + 1) - Elo ** (m + 1)) / ((m + 1) * (E0 ** m))


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Reconstruct bin edges from monotonic centers using geometric means.

    For centers c[0..n-1], interior edges e[1..n-1] = sqrt(c[i-1]*c[i]).
    End edges are extrapolated geometrically: e[0] = c0^2 / e[1]; e[n] = c_{n-1}^2 / e[n-1].
    """
    c = np.asarray(centers, dtype=float)
    if np.any(c <= 0):
        raise ValueError("All centers must be > 0.")
    if not np.all(np.diff(c) > 0):
        raise ValueError("Centers must be strictly increasing.")

    n = c.size
    e = np.empty(n + 1, dtype=float)
    # interior edges
    e[1:n] = np.sqrt(c[:-1] * c[1:])
    # extrapolated ends
    e[0] = c[0] ** 2 / e[1]
    e[n] = c[-1] ** 2 / e[n - 1]

    if not np.all(e[1:] > e[:-1]):
        raise RuntimeError("Reconstructed edges are not strictly increasing; check centers.")
    return e


@dataclass
class AeffResults:
    df: pd.DataFrame
    csv_path: str
    png_cm2sr: str
    png_m2avg: str
    compact_csv_path: str


def compute_aeff(centers: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    """Compute effective area per bin for given centers and counts.

    Returns a DataFrame with per-bin integrals, A_eff [cm^2 sr], and direction-averaged A_eff [m^2].
    """
    T_seconds = T_years * 365.25 * 86400.0
    edges = edges_from_centers(centers)

    I = np.array([integral_powerlaw(edges[i], edges[i + 1], m, E0) for i in range(len(centers))])
    denom = T_seconds * phi0 * I  # units: (s * (GeV^-1 cm^-2 s^-1 sr^-1) * GeV) => cm^-2 sr^-1
    Aeff_cm2_sr = counts / denom
    Aeff_avg_m2 = Aeff_cm2_sr / Omega_sr / 1.0e4

    df = pd.DataFrame({
        "E_center_GeV": centers,
        "E_low_GeV": edges[:-1],
        "E_high_GeV": edges[1:],
        "N_counts": counts,
        "I_bin_GeV": I,
        "Aeff_cm2_sr": Aeff_cm2_sr,
        "Aeff_avg_m2_Omega2pi": Aeff_avg_m2,
    })
    return df


def main() -> AeffResults:
    # Parse inputs
    centers, counts = np.array([p[0] for p in pairs_raw], dtype=float), np.array([p[1] for p in pairs_raw], dtype=float)

    # Variant 1: original counts
    df_orig = compute_aeff(centers, counts)

    # Variant 2: scaled-last (divide only the last bin by 1.3)
    counts_scaled = counts.copy()
    counts_scaled[-1] /= 1.3
    df_scaled = compute_aeff(centers, counts_scaled)

    # Merge for output
    df_out = pd.DataFrame({
        "E_center_GeV": centers,
        "E_low_GeV": df_orig["E_low_GeV"],
        "E_high_GeV": df_orig["E_high_GeV"],
        "N_counts_original": counts,
        "N_counts_scaled_last": counts_scaled,
        "I_bin_GeV": df_orig["I_bin_GeV"],
        "Aeff_cm2_sr_original": df_orig["Aeff_cm2_sr"],
        "Aeff_cm2_sr_scaled_last": df_scaled["Aeff_cm2_sr"],
        "Aeff_avg_m2_original": df_orig["Aeff_avg_m2_Omega2pi"],
        "Aeff_avg_m2_scaled_last": df_scaled["Aeff_avg_m2_Omega2pi"],
    })

    tag = f"T{T_years:g}yr_Omega{Omega_sr/math.pi:.0f}pi"
    csv_path = f"Aeff_{tag}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plots
    plt.figure()
    plt.loglog(centers, df_out["Aeff_cm2_sr_original"], marker="o", linestyle="-", label="original")
    plt.loglog(centers, df_out["Aeff_cm2_sr_scaled_last"], marker="s", linestyle="--", label="scaled-last (/1.3)")
    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"$A_{\\mathrm{eff}}$ [cm$^2$ sr]")
    plt.title("Effective Area from Counts (cm$^2$ sr)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    png_cm2sr = f"Aeff_cm2sr_{tag}.png"
    plt.savefig(png_cm2sr, dpi=160)
    print(f"Saved: {png_cm2sr}")

    plt.figure()
    plt.loglog(centers, df_out["Aeff_avg_m2_original"], marker="o", linestyle="-", label="original")
    plt.loglog(centers, df_out["Aeff_avg_m2_scaled_last"], marker="s", linestyle="--", label="scaled-last (/1.3)")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("$A_{\\mathrm{eff}}$ [cm$^2$ sr]")
    plt.ylabel("$\\overline{A}_{\\mathrm{eff}}$ over $\\Omega=2\\pi$ [m$^2$]")
    plt.title("Direction-averaged Effective Area (m$^2$)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    png_m2avg = f"Aeff_avg_m2_{tag}.png"
    plt.savefig(png_m2avg, dpi=160)
    print(f"Saved: {png_m2avg}")

    # Compact CSV with Muon/Tau placeholders (see note at top)
    compact = pd.DataFrame({
        "E_center_GeV": centers,
        "Muon_Aeff_avg_m2": df_out["Aeff_avg_m2_original"],
        "Tau_Aeff_avg_m2": df_out["Aeff_avg_m2_original"],
    })
    compact_csv_path = f"Aeff_avg_m2_compact_mu_tau_{tag}.csv"
    compact.to_csv(compact_csv_path, index=False)
    print(f"Saved compact CSV (mu/tau placeholders): {compact_csv_path}")

    return AeffResults(df_out, csv_path, png_cm2sr, png_m2avg, compact_csv_path)


if __name__ == "__main__":
    main()
