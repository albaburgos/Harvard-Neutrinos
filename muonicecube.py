"""
Compute and plot IceCube-like A_eff from binned counts using an E^-2 astrophysical flux.
Assumes bin *centers* are given; reconstructs bin edges by geometric-mean method.

This version plots TWO overlaid curves:
  - "original" from the provided counts
  - "scaled-last" where only the last bin's N_counts is divided by 1.3

Outputs:
  - CSV with both A_eff variants [cm^2 sr] and their direction-averaged values over Ω=2π [m^2]
  - PNG plots for both quantities (two curves each)
  - Additional compact CSV with Muon and Tau direction-averaged A_eff (m^2) per energy
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple

# ----------------------- User inputs -----------------------
# Livetime
T_years = 9.5

# Flux model: phi(E) = phi0 * (E/E0)^m
phi0 = 1.44e-18   # GeV^-1 cm^-2 s^-1 sr^-1  (per flavor)
m    = -2.0       # spectral exponent (E^-2)
E0   = 1e5        # GeV (pivot)

# Solid angle for averaging (default: hemisphere, upgoing)
Omega_sr = 2.0 * math.pi  # 2π sr

pairs_raw: Iterable[Tuple[float, float]] = [
    (1.1e2, 9.0e3),
    (1.5e2, 1.2e4),
    (2.0e2, 1.8e4),
    (2.2e2, 2.3e4),
    (3.8e2, 4.0e4),
    (4.5e2, 6.0e4),
    (7.0e2, 8.0e4),
    (1.1e3, 6.0e4),
    (2e3, 4.0e4),
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

OUT_CSV = "aeff_gamma_minus2_upgoing_2pi_WITH_SCALED_LAST.csv"
OUT_PNG_CM2SR = "aeff_cm2sr_gamma_minus2_two_curves.png"
OUT_PNG_M2 = "aeff_avg_m2_Omega2pi_gamma_minus2_two_curves.png"
# ----------------------------------------------------------

def integral_powerlaw(Elo: float, Ehi: float, m: float, E0: float) -> float:
    """Integral of (E/E0)^m dE from Elo to Ehi (GeV)."""
    if Ehi <= Elo or Elo <= 0.0:
        return float("nan")
    if abs(m + 1.0) < 1e-12:
        return math.log(Ehi / Elo)
    # General case:
    return (Ehi ** (m + 1.0) - Elo ** (m + 1.0)) / ((m + 1.0) * (E0 ** m))

def reconstruct_edges(E_centers: np.ndarray):
    """Reconstruct geometric bin edges from centers."""
    ratios = E_centers[1:] / E_centers[:-1]
    E_edges = np.empty(E_centers.size + 1, dtype=float)
    E_edges[1:-1] = np.sqrt(E_centers[1:] * E_centers[:-1])
    E_edges[0]    = E_centers[0] / (ratios[0] ** 0.5)
    E_edges[-1]   = E_centers[-1] * (ratios[-1] ** 0.5)
    return E_edges[:-1], E_edges[1:]

def compute_aeff_from_counts(E_centers: np.ndarray, N_counts: np.ndarray,
                             T_years: float, phi0: float, m: float, E0: float, Omega_sr: float):
    """Return dict of arrays for A_eff in cm^2 sr and its direction-averaged value over Ω=2π in m^2."""
    # Livetime in seconds (Julian year)
    T_seconds = T_years * 365.25 * 86400.0

    # Bin edges
    E_low, E_high = reconstruct_edges(E_centers)

    # Bin integrals
    I_bin = np.array([integral_powerlaw(el, eh, m, E0) for el, eh in zip(E_low, E_high)], dtype=float)

    # Effective area [cm^2 sr]
    denom = T_seconds * phi0 * I_bin   # [s] * [GeV^-1 cm^-2 s^-1 sr^-1] * [GeV]
    Aeff_cm2_sr = N_counts / denom
    Aeff_err_cm2_sr = np.sqrt(np.clip(N_counts, 0.0, None)) / denom

    # Direction-averaged over Omega_sr in m^2
    Aeff_avg_m2 = Aeff_cm2_sr / Omega_sr / 1.0e4
    Aeff_avg_err_m2 = Aeff_err_cm2_sr / Omega_sr / 1.0e4

    return {
        "E_low_GeV": E_low,
        "E_center_GeV": E_centers,
        "E_high_GeV": E_high,
        "N_counts": N_counts,
        "I_bin_GeV": I_bin,
        "Aeff_cm2_sr": Aeff_cm2_sr,
        "Aeff_err_cm2_sr": Aeff_err_cm2_sr,
        "Aeff_avg_m2": Aeff_avg_m2,
        "Aeff_avg_err_m2": Aeff_avg_err_m2,
    }

def main():
    # Sort pairs (ensure increasing energy)
    pairs_sorted = sorted(pairs_raw, key=lambda x: x[0])
    E_centers = np.array([p[0] for p in pairs_sorted], dtype=float)
    N_counts  = np.array([p[1] for p in pairs_sorted], dtype=float)

    # Variant with last bin scaled down by 1.3 (treated as "muon" curve)
    N_counts_scaled = N_counts.copy()
    N_counts_scaled[-1] = N_counts_scaled[-1] / 1.3

    # Compute both
    base = compute_aeff_from_counts(E_centers, N_counts, T_years, phi0, m, E0, Omega_sr)  # "all flavor"
    scl  = compute_aeff_from_counts(E_centers, N_counts_scaled, T_years, phi0, m, E0, Omega_sr)  # "muon"

    # ----------------------- CSV (full, as documented) -----------------------
    df = pd.DataFrame({
        "E_low_GeV": base["E_low_GeV"],
        "E_center_GeV": base["E_center_GeV"],
        "E_high_GeV": base["E_high_GeV"],
        "N_counts_base": base["N_counts"],
        "N_counts_scaled_last": scl["N_counts"],
        "I_bin_GeV": base["I_bin_GeV"],
        "Aeff_cm2_sr_base": base["Aeff_cm2_sr"],
        "Aeff_cm2_sr_scaled_last": scl["Aeff_cm2_sr"],
        "Aeff_avg_m2_base": base["Aeff_avg_m2"],
        "Aeff_avg_m2_scaled_last": scl["Aeff_avg_m2"],
    })
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # ----------------------- Plot 1: A_eff [cm^2 sr] -----------------------
    plt.figure()
    plt.loglog(base["E_center_GeV"], base["Aeff_cm2_sr"], marker="o", label="All flavor (original)")
    plt.loglog(scl["E_center_GeV"],  scl["Aeff_cm2_sr"],  marker="^", label="Muon (scaled-last)")
    plt.xlabel("Energy center (GeV)")
    plt.ylabel(r"$A_{\mathrm{eff}}$ (cm$^2$ sr)")
    plt.title(r"Ice-Cube-like Effective Area $A_{\mathrm{eff}}$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(OUT_PNG_CM2SR, dpi=160)
    print(f"Saved: {OUT_PNG_CM2SR}")

    # ----------------------- Plot 2: direction-averaged A_eff [m^2] -----------------------
    plt.figure()
    plt.loglog(base["E_center_GeV"], base["Aeff_avg_m2"], marker="o", label="All flavor (original)")
    plt.loglog(scl["E_center_GeV"],  scl["Aeff_avg_m2"],  marker="^", label="Muon (scaled-last)")
    plt.xlabel("Energy center (GeV)")
    plt.ylabel(r"Average $\overline{A}_\mathrm{eff}$ over $\Omega=2\pi$ (m$^2$)")
    plt.title(r"Ice-Cube Track Effective Areas $\overline{A}_\mathrm{eff}$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(OUT_PNG_M2, dpi=160)
    print(f"Saved: {OUT_PNG_M2}")

    # ----------------------- Compact CSV: Muon & Tau (m^2) -----------------------
    # Tau := (all - muon); numeric safety: clip to [0, all]
    tau_m2 = (base["Aeff_avg_m2"] - scl["Aeff_avg_m2"])
    tau_m2 = np.clip(tau_m2, 0.0, base["Aeff_avg_m2"])

    df_compact = pd.DataFrame({
        "E_center_GeV": base["E_center_GeV"],
        "Muon_Aeff_m2": scl["Aeff_avg_m2"],
        "Tau_Aeff_m2":  tau_m2,
    })

    OUT_CSV_M2 = OUT_PNG_M2.rsplit(".", 1)[0] + ".csv"
    df_compact.to_csv(OUT_CSV_M2, index=False)
    print(f"Saved CSV: {OUT_CSV_M2}")

if __name__ == "__main__":
    main()
