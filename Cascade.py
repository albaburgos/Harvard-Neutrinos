

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------- Configuration ----------------
phi0 = 1.44e-18     # GeV^-1 cm^-2 s^-1 sr^-1
m    = -2.0         # spectral exponent
E0   = 1e5          # GeV
T_years = 6.0
Omega_sr = 4.0 * math.pi  # upgoing hemisphere

# Bin-edge multipliers per decade (5 equal log bins)
MUL = np.array([1.0, 1.5849, 2.5119, 3.9811, 6.3096, 10.0], dtype=float)

# Datasets (A and B). Keys are decades as integer powers of 10 for GeV.
# Values are lists of bin counts; length can be < 5, in which case they map to the LAST bins.
dataset_A: Dict[int, List[float]] = {
    2: [1.0, 2.8],                             # 10^2 GeV decade, last 2 bins
    3: [5.0, 6.7, 7.2, 7.2, 6.5],             # 10^3 GeV decade
    4: [5.0, 4.0, 2.8, 1.8, 1.0],             # 10^4 GeV decade
    5: [0.6, 0.4, 0.26, 0.17],                # 10^5 GeV decade (4 bins -> last 4)
}

dataset_B: Dict[int, List[float]] = {
    2: [7.0, 20.0],                            # 10^2 GeV decade, last 2 bins
    3: [4e1, 5.5e1, 6e1, 6e1, 5.8e1],         # 10^3 GeV decade
    4: [4.9e1, 3.6e1, 2.2e1, 1.6e1, 1.0e1],     # 10^4 GeV decade
    5: [7.0, 4.0, 3.0, 1.8, 1.0],              # 10^5 GeV decade
    6: [0.7, 0.4, 0.36, 0.4],                  # 10^6 GeV decade (4 bins -> last 4)
}

# ------------- Core math -----------------
def integral_powerlaw(Elo: float, Ehi: float, m: float, E0: float) -> float:
    """Integral of (E/E0)^m dE over [Elo, Ehi]."""
    if Ehi <= Elo or Elo <= 0.0:
        return float('nan')
    if abs(m + 1.0) < 1e-14:
        # m = -1 special case: ∫ (E/E0)^(-1) dE = ln(Ehi/Elo)
        return math.log(Ehi / Elo)
    # General case
    return (Ehi**(m+1) - Elo**(m+1)) / ((m+1) * (E0**m))

def decade_edges(power10: int) -> np.ndarray:
    """Return the 6 bin edges for a given decade 10^power10 * MUL."""
    base = 10.0 ** power10
    return base * MUL

def map_counts_to_bins(power10: int, counts: List[float]) -> List[Tuple[float, float, float]]:
    """
    Map a possibly-short list of counts to bin intervals [Elo, Ehi] within the decade.
    If len(counts) < 5, assign them to the LAST bins (highest energies) per user guidance.
    Returns list of (Elo, Ehi, N) entries in increasing energy order.
    """
    edges = decade_edges(power10)  # 6 edges -> 5 bins
    n = len(counts)
    if n > 5:
        raise ValueError(f"Decade 10^{power10} supplied {n} counts; expected <= 5.")
    # Indices of bins to fill
    start_bin = 5 - n  # start at this bin index (0-based), fill to 4
    bins = []
    for i, val in enumerate(counts):
        b = start_bin + i
        Elo, Ehi = edges[b], edges[b+1]
        bins.append((Elo, Ehi, float(val)))
    return bins

def compute_aeff_for_dataset(dataset: Dict[int, List[float]], label: str) -> pd.DataFrame:
    """Compute A_eff arrays for all decades in a dataset; return a DataFrame."""
    rows = []
    # Convert livetime to seconds (Julian year)
    T_seconds = T_years * 365.25 * 86400.0

    # Gather bins across all decades
    for p10, counts in sorted(dataset.items()):
        bins = map_counts_to_bins(p10, counts)
        for Elo, Ehi, N in bins:
            I = integral_powerlaw(Elo, Ehi, m, E0)
            denom = T_seconds * phi0 * I
            A_cm2_sr = N / denom
            A_m2_avg = A_cm2_sr / Omega_sr / 1.0e4
            rows.append({
                "dataset": label,
                "decade": p10,
                "E_low_GeV": Elo,
                "E_high_GeV": Ehi,
                "E_center_GeV": math.sqrt(Elo * Ehi),
                "N_counts": N,
                "I_bin_GeV": I,
                "Aeff_cm2_sr": A_cm2_sr,
                "Aeff_avg_m2_Omega2pi": A_m2_avg,
            })
    df = pd.DataFrame(rows).sort_values(by=["E_low_GeV"]).reset_index(drop=True)
    return df

def main():
    dfA = compute_aeff_for_dataset(dataset_A, "A")
    dfB = compute_aeff_for_dataset(dataset_B, "B")

    frac_electron = 127 / (127 + 80)
    frac_tau = 80 / (127 + 80)

    # Create separate columns
    df_out = pd.DataFrame({
        "E_center_GeV": dfA["E_center_GeV"],
        "Muon_Aeff_m2_Omega2pi": dfA["Aeff_avg_m2_Omega2pi"],
        "Electron_Aeff_m2_Omega2pi": dfB["Aeff_avg_m2_Omega2pi"] * frac_electron,
        "Tau_Aeff_m2_Omega2pi": dfB["Aeff_avg_m2_Omega2pi"] * frac_tau
    })

    # Save to CSV
    csv_filename = "Aeff_avg_m2_T6yr_Omega2pi.csv"
    df_out.to_csv(csv_filename, index=False)
    print(f"Saved CSV: {csv_filename}")


    plt.figure()
    plt.loglog(dfA["E_center_GeV"], dfA["Aeff_avg_m2_Omega2pi"], marker="o", label="Muon")
    plt.loglog(dfB["E_center_GeV"], dfB["Aeff_avg_m2_Omega2pi"], marker="^", label="Electron+Taon")
    plt.loglog(dfB["E_center_GeV"], dfB["Aeff_avg_m2_Omega2pi"]*(127/(127+80)), marker=".", label="Electron")
    plt.loglog(dfB["E_center_GeV"], dfB["Aeff_avg_m2_Omega2pi"]*(80/(127+80)), marker="x", label="Taon")

    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"Average $\overline{A}_{\mathrm{eff}}$ ")
    plt.title("Ice-Cube Cascade Effective Areas ")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("Aeff_avg_m2_T6yr_Omega2pi.png", dpi=160)
    print("Saved: Aeff_avg_m2_T6yr_Omega2pi.png")

if __name__ == "__main__":
    main()

