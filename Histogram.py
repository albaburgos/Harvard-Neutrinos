
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_df(path: str | None) -> pd.DataFrame:
    """Load the decade-binned counts CSV."""
    if path is None:
        path = "nevents_per_decade.csv"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p)
    required = [
        "E_low_GeV","E_high_GeV",
        "N_e","N_mu","N_tau",
        "err_e","err_mu","err_tau",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def compute_centers(df: pd.DataFrame) -> np.ndarray:
    """Geometric bin centers via log10 to avoid overflow at large edges."""
    low = df["E_low_GeV"].astype(float).to_numpy()
    high = df["E_high_GeV"].astype(float).to_numpy()
    centers_log10 = (np.log10(low) + np.log10(high)) / 2.0
    return 10.0 ** centers_log10


def make_three_panel(df: pd.DataFrame, out_png: str = "three_panel_hist.png") -> None:
    centers = compute_centers(df)
    widths = df["E_high_GeV"].astype(float) - df["E_low_GeV"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    panels = [
        ("Electrons", "N_e", "err_e", axes[0]),
        ("Muons", "N_mu", "err_mu", axes[1]),
        ("Taus", "N_tau", "err_tau", axes[2]),
    ]

    for title, n_col, err_col, ax in panels:
        y = df[n_col].astype(float).to_numpy()
        yerr = df[err_col].astype(float).to_numpy()

        # Histogram bars spanning each decade
        ax.bar(df["E_low_GeV"].astype(float).to_numpy(), y, width=widths, align="edge")

        # Error bars at bin centers
        ax.errorbar(centers, y, yerr=yerr, fmt="o", capsize=3, linestyle="none")

        ax.set_xscale("log")
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Counts per bin")
        ax.set_title(title)
        ax.grid(True, which="both", axis="both", alpha=0.2)

    fig.suptitle("Electrons | Muons | Taus — Histogram")
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")


def main():
    in_csv = sys.argv[1] if len(sys.argv) >= 2 else None
    out_png = sys.argv[2] if len(sys.argv) >= 3 else "three_panel_hist.png"
    df = load_df(in_csv)
    make_three_panel(df, out_png)


if __name__ == "__main__":
    main()