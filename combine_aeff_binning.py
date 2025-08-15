
import argparse
import math
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csvs(csv1_path: str, csv2_path: str, divide_by_2pi: bool = True) -> pd.DataFrame:
    # CSV 1
    df1_raw = pd.read_csv(csv1_path)
    if "E_center_GeV" not in df1_raw.columns:
        raise ValueError("CSV1 must include 'E_center_GeV' column.")
    df1_raw = df1_raw.dropna(subset=["E_center_GeV"])

    # Safe get for species columns
    def safe_get(col): 
        return df1_raw[col] if col in df1_raw.columns else np.nan

    factor = (2.0 * math.pi) if divide_by_2pi else 1.0
    df1 = pd.DataFrame({
        "E_center_GeV": pd.to_numeric(df1_raw["E_center_GeV"], errors="coerce"),
        "Muon_Aeff_m2": pd.to_numeric(safe_get("Muon_Aeff_m2_Omega2pi"), errors="coerce") / factor,
        "Electron_Aeff_m2": pd.to_numeric(safe_get("Electron_Aeff_m2_Omega2pi"), errors="coerce") / factor,
        "Tau_Aeff_m2": pd.to_numeric(safe_get("Tau_Aeff_m2_Omega2pi"), errors="coerce") / factor,
    })

    # CSV 2
    df2_raw = pd.read_csv(csv2_path)
    for req in ["E_center_GeV", "Muon_Aeff_m2", "Tau_Aeff_m2"]:
        if req not in df2_raw.columns:
            raise ValueError(f"CSV2 must include '{req}' column.")
    df2 = pd.DataFrame({
        "E_center_GeV": pd.to_numeric(df2_raw["E_center_GeV"], errors="coerce"),
        "Muon_Aeff_m2": pd.to_numeric(df2_raw["Muon_Aeff_m2"], errors="coerce"),
        "Electron_Aeff_m2": np.nan,  # not provided in CSV2
        "Tau_Aeff_m2": pd.to_numeric(df2_raw["Tau_Aeff_m2"], errors="coerce"),
    })

    stacked = pd.concat([df1, df2], ignore_index=True)
    stacked = stacked.dropna(subset=["E_center_GeV"])
    return stacked


def make_log_bins(emin: float, emax: float, bins_per_decade: int) -> np.ndarray:
    if emin <= 0 or emax <= 0:
        raise ValueError("Energy values must be positive for log binning.")
    if emax <= emin:
        raise ValueError("emax must be greater than emin.")
    n_decades = math.log10(emax) - math.log10(emin)
    nbins = max(1, int(math.ceil(n_decades * bins_per_decade)))
    edges = np.logspace(math.log10(emin), math.log10(emax), nbins + 1)
    return edges


def sum_per_bin(stacked: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    # Assign bins
    stacked = stacked.copy()
    stacked["bin"] = pd.cut(stacked["E_center_GeV"], bins=edges, include_lowest=True)

    # Sum Aeff per species per bin
    sum_by_bin = stacked.groupby("bin").agg({
        "Muon_Aeff_m2": "sum",
        "Electron_Aeff_m2": "sum",
        "Tau_Aeff_m2": "sum",
    })

    # Count contributors per species per bin
    counts = stacked.groupby("bin").agg({
        "Muon_Aeff_m2": lambda s: s.notna().sum(),
        "Electron_Aeff_m2": lambda s: s.notna().sum(),
        "Tau_Aeff_m2": lambda s: s.notna().sum(),
    }).rename(columns={
        "Muon_Aeff_m2": "N_muon_sources",
        "Electron_Aeff_m2": "N_electron_sources",
        "Tau_Aeff_m2": "N_tau_sources"
    })

    combined = sum_by_bin.join(counts).reset_index()

    # Derive bin edges and geometric midpoints
    combined["E_bin_min_GeV"] = combined["bin"].apply(lambda iv: float(iv.left))
    combined["E_bin_max_GeV"] = combined["bin"].apply(lambda iv: float(iv.right))
    combined["E_center_GeV"]  = combined["bin"].apply(lambda iv: float(math.sqrt(iv.left * iv.right)))
    combined = combined.drop(columns=["bin"])

    # Sort by midpoint and reorder columns
    combined = combined.sort_values("E_center_GeV")[
        ["E_bin_min_GeV", "E_bin_max_GeV", "E_center_GeV",
         "Muon_Aeff_m2", "Electron_Aeff_m2", "Tau_Aeff_m2",
         "N_muon_sources", "N_electron_sources", "N_tau_sources"]
    ]
    return combined


def plot_overlay(combined: pd.DataFrame, out_png: str):
    x = pd.to_numeric(combined["E_center_GeV"], errors="coerce").to_numpy(dtype=float)

    def mask(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        return m, y

    m_mu, y_mu = mask(combined["Muon_Aeff_m2"])
    m_tau, y_tau = mask(combined["Tau_Aeff_m2"])
    m_el,  y_el  = mask(combined["Electron_Aeff_m2"])

    fig = plt.figure()
    plotted_any = False
    if m_mu.sum() > 0:
        plt.plot(x[m_mu], y_mu[m_mu], marker="o", label="Muon")
        plotted_any = True
    if m_tau.sum() > 0:
        plt.plot(x[m_tau], y_tau[m_tau], marker="o", label="Tau")
        plotted_any = True
    if m_el.sum() > 0:
        plt.plot(x[m_el],  y_el[m_el],  marker="o", label="Electron")
        plotted_any = True

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Effective Area (m$^2$)")
    plt.title("IceCube Effective Area Track+Cascade (per Flavor, binned)")
    plt.grid(True, which="both", linestyle=":")
    if plotted_any:
        plt.legend()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Combine two Aeff datasets, bin in energy, SUM per bin, and plot overlay.")
    p.add_argument("--csv1", required=True, help="Path to CSV 1")
    p.add_argument("--csv2", required=True, help="Path to CSV 2")
    p.add_argument("--out-csv", default="combined_aeff_binned_SUM.csv", help="Output CSV path")
    p.add_argument("--out-png", default="aeff_vs_energy_overlay.png", help="Output PNG path")
    p.add_argument("--bins-per-decade", type=int, default=12, help="Number of log bins per decade (default: 12)")
    p.add_argument("--no-divide-by-2pi", action="store_true", help="Keep CSV1 *_Omega2pi values as-is (do NOT divide by 2π)")
    args = p.parse_args()

    # Load and harmonize
    stacked = read_csvs(args.csv1, args.csv2, divide_by_2pi=(not args.no_divide_by_2pi))

    # Build bins
    emin = float(stacked["E_center_GeV"].min())
    emax = float(stacked["E_center_GeV"].max())
    edges = make_log_bins(emin, emax, args.bins_per_decade)

    # Sum per bin
    combined = sum_per_bin(stacked, edges)

    # Save and plot
    combined.to_csv(args.out_csv, index=False)
    plot_overlay(combined, args.out_png)

    # Console summary
    print(f"Saved combined table -> {args.out_csv}")
    print(f"Saved plot -> {args.out_png}")
    print("Non-empty points plotted:")
    for sp in ["Muon_Aeff_m2", "Tau_Aeff_m2", "Electron_Aeff_m2"]:
        nnz = int(((combined[sp].astype(float).to_numpy() > 0) & np.isfinite(combined[sp].to_numpy())).sum())
        print(f"  {sp}: {nnz} points")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        



'''
implementation

python combine_aeff_binning.py \
  --csv1 Csv1.csv \
  --csv2 csv2.csv \
  --out-csv icecube.csv \
  --out-png csv.png \
  --bins-per-decade 4


'''