# --------------------------- CONFIG ---------------------------
CSV_PATH     = "nevents_per_decade.csv"   # path to your CSV
INDICES      = list(range(9))             # <- plot row indices 0..8
NSAMPLES     = 3000
GRID_STEP    = 0.1
TICK_FONTSZ  = 8.0

# Scatter and contours
SHOW_BLOB    = True
CONTOUR_LINEWIDTH = 2.0

# Gaussian (elliptical) heat + contours
GAUSS_BINS             = 250
GAUSS_CLIP_TRI         = True
GAUSS_CONTOUR_LINEWIDTH= 2.0

OUT_PATH_GRID = "flavor_triangle_0-8_grid.png"  # output PNG for the 3x3 grid
# --------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SQ3_OVER_2 = math.sqrt(3) / 2.0

def bary_to_xy(f_e, f_mu, f_tau):
    # Swap e ↔ μ axes: x now tracks f_e (previously f_μ).
    x = f_e * 1.0 + f_tau * 0.5
    y = f_tau * SQ3_OVER_2
    return x, y


def inside_triangle_mask(xg, yg):
    left = yg / math.sqrt(3.0)
    right = 1.0 - yg / math.sqrt(3.0)
    return (yg >= 0) & (yg <= SQ3_OVER_2) & (xg >= left) & (xg <= right)

def draw_triangle(ax):
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, SQ3_OVER_2, 0.0]
    ax.plot(tri_x, tri_y)

def draw_fraction_grid(ax, step=0.1, linewidth=0.6, alpha=0.35):
    import numpy as np
    ticks = np.arange(step, 1.0, step)
    for p in ticks:  # f_e = p
        x1, y1 = bary_to_xy(p, 0.0, 1.0 - p)
        x2, y2 = bary_to_xy(p, 1.0 - p, 0.0)
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)
    for p in ticks:  # f_mu = p
        x1, y1 = bary_to_xy(0.0, p, 1.0 - p)
        x2, y2 = bary_to_xy(1.0 - p, p, 0.0)
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)
    for p in ticks:  # f_tau = p
        x1, y1 = bary_to_xy(0.0, 1.0 - p, p)
        x2, y2 = bary_to_xy(1.0 - p, 0.0, p)
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)

def annotate_fraction_ticks(ax, step=0.1, fontsize=8, alpha=0.75):
    import numpy as np
    ticks = np.arange(step, 1.0, step)
    for p in ticks:  # f_e ticks
        fe, fmu, ftau = p, (1.0 - p) / 2.0, (1.0 - p) / 2.0
        x, y = bary_to_xy(fe, fmu, ftau); ax.text(x, y, f"{p:.2g}", ha="center", va="center", fontsize=fontsize, alpha=alpha)
    for p in ticks:  # f_mu ticks
        fe, fmu, ftau = (1.0 - p) / 2.0, p, (1.0 - p) / 2.0
        x, y = bary_to_xy(fe, fmu, ftau); ax.text(x, y, f"{p:.2g}", ha="center", va="center", fontsize=fontsize, alpha=alpha)
    for p in ticks:  # f_tau ticks
        fe, fmu, ftau = (1.0 - p) / 2.0, (1.0 - p) / 2.0, p
        x, y = bary_to_xy(fe, fmu, ftau); ax.text(x, y, f"{p:.2g}", ha="center", va="center", fontsize=fontsize, alpha=alpha)

def label_axes(ax, *, show_long_names=True, fs_symbol=12, fs_long=9, pad=0.08):
    mid_y = SQ3_OVER_2 / 2.0

    # --- Bottom edge (y=0): f_e (swapped from f_μ) ---
    ax.text(0.5, -pad, r"$f_\mu$", ha="center", va="top", fontsize=fs_symbol)
    if show_long_names:
        ax.text(0.5, -(pad + 0.05), "fraction of muon neutrino", ha="center", va="top", fontsize=fs_long)

    # --- Left edge: f_τ ---
    ax.text(-pad, mid_y, r"$f_\tau$", ha="right", va="center",
            fontsize=fs_symbol, rotation=60)
    if show_long_names:
        ax.text(-(pad + 0.03), mid_y, "fraction of tau neutrino", ha="right", va="center",
                fontsize=fs_long, rotation=60)

    # --- Right edge: f_μ (swapped from f_e) ---
    ax.text(1.0 + pad, mid_y, r"$f_e$", ha="left", va="center",
            fontsize=fs_symbol, rotation=-60)
    if show_long_names:
        ax.text(1.0 + pad + 0.03, mid_y, "fraction of electron neutrino", ha="left", va="center",
                fontsize=fs_long, rotation=-60)

# ---- Gaussian (elliptical) helpers
def safe_cov_inv(xs, ys):
    import numpy as np
    X = np.vstack([xs, ys])
    cov = np.cov(X, bias=True)
    eps = 1e-10 * np.trace(cov) or 1e-12
    cov_r = cov + eps * np.eye(2)
    inv = np.linalg.inv(cov_r)
    return cov_r, inv

CHI2_2_68 = 2.279
CHI2_2_95 = 5.991

def add_gaussian_contours(ax, xs, ys, bins=250, clip_triangle=True, linewidth=2.0):
    import numpy as np
    mean = np.array([np.mean(xs), np.mean(ys)])
    cov, inv = safe_cov_inv(xs, ys)
    x_edges = np.linspace(0.0, 1.0, bins + 1)
    y_edges = np.linspace(0.0, SQ3_OVER_2, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers)
    xm = Xc - mean[0]; ym = Yc - mean[1]
    m2 = inv[0,0]*xm*xm + (inv[0,1]+inv[1,0])*xm*ym + inv[1,1]*ym*ym
    if clip_triangle:
        mask = inside_triangle_mask(Xc, Yc)
        m2 = np.where(mask, m2, np.nan)
    CS = ax.contour(x_centers, y_centers, m2,
                    levels=[CHI2_2_68, CHI2_2_95],
                    linestyles=["-", "--"], linewidths=linewidth)
    ax.clabel(CS, inline=True, fontsize=9, fmt={CHI2_2_68:"68%", CHI2_2_95:"95%"})

def plot_one_row(ax, r, rng):
    # Extract values
    Ne, Nmu, Ntau = float(r["N_e"]), float(r["N_mu"]), float(r["N_tau"])
    err_e, err_mu, err_tau = float(r["err_e"]), float(r["err_mu"]), float(r["err_tau"])
    E_low, E_high = float(r["E_low_GeV"]), float(r["E_high_GeV"])

    Ntot = Ne + Nmu + Ntau
    if Ntot <= 0:
        ax.set_title("No events"); ax.axis("off"); return

    fe, fmu, ftau = Ne/Ntot, Nmu/Ntot, Ntau/Ntot
    x0, y0 = bary_to_xy(fe, fmu, ftau)

    # Monte Carlo sampling (shared RNG for unique draws across panels)
    Ne_s  = rng.normal(loc=Ne,  scale=max(err_e,  1e-12), size=NSAMPLES)
    Nmu_s = rng.normal(loc=Nmu, scale=max(err_mu, 1e-12), size=NSAMPLES)
    Ntau_s= rng.normal(loc=Ntau,scale=max(err_tau,1e-12), size=NSAMPLES)
    Ne_s  = np.clip(Ne_s,  0, None)
    Nmu_s = np.clip(Nmu_s, 0, None)
    Ntau_s= np.clip(Ntau_s,0, None)
    Ntot_s = Ne_s + Nmu_s + Ntau_s
    mask = Ntot_s > 0
    Ne_s, Nmu_s, Ntau_s, Ntot_s = Ne_s[mask], Nmu_s[mask], Ntau_s[mask], Ntot_s[mask]
    fe_s, fmu_s, ftau_s = Ne_s/Ntot_s, Nmu_s/Ntot_s, Ntau_s/Ntot_s

    # Use the shared barycentric mapping (with e↔μ swapped)
    xs, ys = bary_to_xy(fe_s, fmu_s, ftau_s)

    # Triangle, grid, ticks
    draw_triangle(ax)
    draw_fraction_grid(ax, step=float(GRID_STEP))

    # Raw scatter samples (optional)
    if SHOW_BLOB:
        ax.scatter(xs, ys, s=5, alpha=0.08, edgecolors="none")

    # Central point
    ax.plot([x0], [y0], marker="o", markersize=8)

    # Gaussian 68% & 95% contours (elliptical)
    add_gaussian_contours(ax, xs, ys, bins=int(GAUSS_BINS),
                          clip_triangle=bool(GAUSS_CLIP_TRI),
                          linewidth=float(GAUSS_CONTOUR_LINEWIDTH))

    # Labels & formatting
    label_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, SQ3_OVER_2 + 0.05)
    ax.set_xticks([]); ax.set_yticks([])
    title_line1 = rf"Row {int(r.name)}: $E \in [{E_low:.3g}, {E_high:.3g}]\,\mathrm{{GeV}}$"
    title_line2 = rf"$f_e={fe:.3f},\ f_\mu={fmu:.3f},\ f_\tau={ftau:.3f}$"
    ax.set_title(title_line1 + "\n" + title_line2, fontsize=10)

def main():
    # Load CSV
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    required = {"E_low_GeV", "E_high_GeV", "N_e", "N_mu", "N_tau", "err_e", "err_mu", "err_tau"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Validate indices
    max_idx = len(df) - 1
    bad = [i for i in INDICES if i < 0 or i > max_idx]
    if bad:
        raise IndexError(f"Indices out of bounds for {len(df)} rows: {bad}")

    # Figure & axes (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    rng = np.random.default_rng(seed=42)  # shared RNG for reproducibility

    for ax, idx in zip(axes, INDICES):
        r = df.iloc[idx]; r.name = idx  # keep index label for titles
        plot_one_row(ax, r, rng)

    # If fewer than 9 requested (not here), hide extra axes
    for k in range(len(INDICES), 9):
        axes[k].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(OUT_PATH_GRID, dpi=230)
    print(f"Saved: {OUT_PATH_GRID}")
    print(f"Plotted rows: {INDICES}")

if __name__ == "__main__":
    main()


    