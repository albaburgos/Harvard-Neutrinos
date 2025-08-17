#!/usr/bin/env python3
"""
Flavor triangle plotter (no command-line args).

Edit the CONFIG section below and run:
    python plot_flavor_triangle_noargs.py

Features:
- Axis titles (“fraction of νe / νμ / ντ”)
- Interior fractional grid lines + numeric tick labels
- Monte Carlo “blob” from count uncertainties (optional)
- **Density heatmap inside the triangle (non-elliptical)** from a Gaussian-smoothed 2D histogram (pure NumPy)
- 68% & 95% iso-probability contours (HDR) overlaid on the heatmap
- Single matplotlib plot, no seaborn, no explicit colors
"""

# --------------------------- CONFIG ---------------------------
CSV_PATH     = "nevents_per_decade.csv"  # path to your CSV
ROW_INDEX    = None     # set to an integer row index, or None to auto-select the row with the largest total events
NSAMPLES     = 3000     # Monte Carlo samples for the uncertainty blob
GRID_STEP    = 0.1      # fractional grid spacing (e.g., 0.1 draws 0.1..0.9 lines)
TICK_FONTSZ  = 8.0      # font size for interior numeric tick labels
KDE_BINS     = 220      # number of bins per axis for the KDE proxy (resolution of heatmap)
KDE_SIGMA    = 1.2      # Gaussian sigma in *bins* for KDE smoothing
SHOW_BLOB    = True    # set True to show raw scatter; False to rely on heatmap + contours
SHOW_HEAT    = False   # render the density heatmap (non-elliptical)
SHOW_COLORBAR= False     # show colorbar for the heatmap
OUT_PATH     = "flavor_triangle.png"  # output PNG
CONTOUR_LINEWIDTH = 2.0  # line width for contour lines
SHOW_GAUSSIAN_HEAT = True      # turn on the fitted-Gaussian heatmap
GAUSS_BINS = 250               # grid resolution for Gaussian heat
GAUSS_CLIP_TRI = True          # clip heat/contours to triangle domain
GAUSS_CONTOUR_LINEWIDTH = 2.0
# --------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SQ3_OVER_2 = math.sqrt(3) / 2.0

def bary_to_xy(f_e, f_mu, f_tau):
    """Convert barycentric (fractions) to Cartesian in an equilateral triangle.
    Vertices: e=(0,0), mu=(1,0), tau=(0.5, sqrt(3)/2)
    """
    x = f_mu * 1.0 + f_tau * 0.5
    y = f_tau * SQ3_OVER_2
    return x, y

def inside_triangle_mask(xg, yg):
    """Return boolean mask for points (xg,yg) inside the triangle."""
    left = yg / math.sqrt(3.0)
    right = 1.0 - yg / math.sqrt(3.0)
    return (yg >= 0) & (yg <= SQ3_OVER_2) & (xg >= left) & (xg <= right)

def draw_triangle(ax):
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, SQ3_OVER_2, 0.0]
    ax.plot(tri_x, tri_y)

def draw_fraction_grid(ax, step=0.1, linewidth=0.6, alpha=0.35):
    ticks = np.arange(step, 1.0, step)

    # Lines of constant f_e = p (parallel to mu–tau side)
    for p in ticks:
        x1, y1 = bary_to_xy(p, 0.0, 1.0 - p)
        x2, y2 = bary_to_xy(p, 1.0 - p, 0.0)
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)

    # Lines of constant f_mu = p (parallel to e–tau side)
    for p in ticks:
        x1, y1 = bary_to_xy(0.0, p, 1.0 - p)
        x2, y2 = bary_to_xy(1.0 - p, p, 0.0)
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)

    # Lines of constant f_tau = p (parallel to e–mu side)
    for p in ticks:
        x1, y1 = bary_to_xy(0.0, 1.0 - p, p)
        x2, y2 = bary_to_xy(1.0 - p, 0.0, p)
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)

def annotate_fraction_ticks(ax, step=0.1, fontsize=8, alpha=0.75):
    ticks = np.arange(step, 1.0, step)

    # f_e ticks
    for p in ticks:
        fe, fmu, ftau = p, (1.0 - p) / 2.0, (1.0 - p) / 2.0
        x, y = bary_to_xy(fe, fmu, ftau)
        ax.text(x, y, f"{p:.2g}", ha="center", va="center", fontsize=fontsize, alpha=alpha)

    # f_mu ticks
    for p in ticks:
        fe, fmu, ftau = (1.0 - p) / 2.0, p, (1.0 - p) / 2.0
        x, y = bary_to_xy(fe, fmu, ftau)
        ax.text(x, y, f"{p:.2g}", ha="center", va="center", fontsize=fontsize, alpha=alpha)

    # f_tau ticks
    for p in ticks:
        fe, fmu, ftau = (1.0 - p) / 2.0, (1.0 - p) / 2.0, p
        x, y = bary_to_xy(fe, fmu, ftau)
        ax.text(x, y, f"{p:.2g}", ha="center", va="center", fontsize=fontsize, alpha=alpha)

def label_axes(ax):
    xe, ye   = bary_to_xy(0.0, 0.5, 0.5)   # “fraction of νe” near μ–τ side
    xmu, ymu = bary_to_xy(0.5, 0.0, 0.5)   # “fraction of νμ” near e–τ side
    xtau, ytau = bary_to_xy(0.5, 0.5, 0.0) # “fraction of ντ” near e–μ side

    ax.text(xe,  ye - 0.035, r"fraction of $\nu_e$", ha="center", va="top")
    ax.text(xmu + 0.035, ymu, r"fraction of $\nu_\mu$", ha="left", va="center")
    ax.text(xtau, ytau + 0.035, r"fraction of $\nu_\tau$", ha="center", va="bottom")

    # Vertex labels
    ax.text(0.0 - 0.04, 0.0 - 0.02, r"$\nu_e$", ha="right", va="top")
    ax.text(1.0 + 0.04, 0.0 - 0.02, r"$\nu_\mu$", ha="left", va="top")
    ax.text(0.5, SQ3_OVER_2 + 0.04, r"$\nu_\tau$", ha="center", va="bottom")

def gaussian_kernel_1d(sigma_bins):
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(round(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma_bins)) ** 2)
    k /= k.sum()
    return k

def convolve_separable(img, k1d):
    pad = len(k1d) // 2
    padded = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    out_x = np.empty_like(img, dtype=float)
    for i in range(img.shape[0]):
        row = padded[i]
        out_x[i] = np.convolve(row, k1d, mode="valid")
    padded_y = np.pad(out_x, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(img, dtype=float)
    for j in range(img.shape[1]):
        col = padded_y[:, j]
        out[:, j] = np.convolve(col, k1d, mode="valid")
    return out

def compute_kde_grid(xs, ys, bins=220, sigma=1.2):
    """Return (x_edges, y_edges, x_centers, y_centers, P) for KDE heatmap/contours."""
    x_edges = np.linspace(0.0, 1.0, bins + 1)
    y_edges = np.linspace(0.0, SQ3_OVER_2, bins + 1)
    H, xe, ye = np.histogram2d(ys, xs, bins=[y_edges, x_edges])  # H: (y_bins, x_bins)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Mask outside triangle
    Xc, Yc = np.meshgrid(x_centers, y_centers)
    mask = inside_triangle_mask(Xc, Yc)
    H = H * mask.astype(float)

    # Smooth
    k = gaussian_kernel_1d(sigma_bins=sigma)
    H_smooth = convolve_separable(H, k)

    total = H_smooth.sum()
    if total <= 0:
        P = H_smooth  # all zeros
    else:
        P = H_smooth / total

    return x_edges, y_edges, x_centers, y_centers, P

def highest_density_thresholds(density, levels=(0.68, 0.95)):
    flat = density.ravel()
    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    cdf = np.cumsum(sorted_vals)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    thrs = {}
    for q in levels:
        idx = np.searchsorted(cdf, q, side="left")
        thr = sorted_vals[min(idx, len(sorted_vals)-1)]
        thrs[q] = float(thr)
    return thrs




def safe_cov_inv(xs, ys):
    """Compute covariance and its inverse with small ridge for stability."""
    X = np.vstack([xs, ys])
    # ddof=0 for MLE covariance
    cov = np.cov(X, bias=True)
    # Ridge in case of near-singular covariance
    eps = 1e-10 * np.trace(cov)
    if eps <= 0:
        eps = 1e-12
    cov_r = cov + eps * np.eye(2)
    inv = np.linalg.inv(cov_r)
    return cov_r, inv

def gaussian_pdf_grid(mean, cov, x_edges, y_edges):
    """Return Gaussian pdf evaluated at cell centers for a grid defined by edges."""
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers)

    xm = Xc - mean[0]
    ym = Yc - mean[1]

    det = np.linalg.det(cov)
    if det <= 0:
        det = 1e-20
    inv = np.linalg.inv(cov)

    # Mahalanobis squared
    m2 = inv[0,0]*xm*xm + (inv[0,1]+inv[1,0])*0.5*2.0*xm*ym + inv[1,1]*ym*ym
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    pdf = norm * np.exp(-0.5 * m2)
    return x_centers, y_centers, pdf

def add_gaussian_heat(ax, xs, ys, bins=250, clip_triangle=True):
    """Add Gaussian heatmap from fitted 2D normal to samples, optionally clipped to triangle."""
    mean = np.array([np.mean(xs), np.mean(ys)])
    cov, inv = safe_cov_inv(xs, ys)

    x_edges = np.linspace(0.0, 1.0, bins + 1)
    y_edges = np.linspace(0.0, SQ3_OVER_2, bins + 1)
    x_centers, y_centers, pdf = gaussian_pdf_grid(mean, cov, x_edges, y_edges)

    if clip_triangle:
        Xc, Yc = np.meshgrid(x_centers, y_centers)
        mask = inside_triangle_mask(Xc, Yc)
        pdf = np.where(mask, pdf, np.nan)

    qm = ax.pcolormesh(x_edges, y_edges, pdf, shading="auto")
    return qm, mean, cov

# Precomputed chi-square quantiles for 2 dof: 68% and 95%
CHI2_2_68 = 2.279
CHI2_2_95 = 5.991

def add_gaussian_contours(ax, xs, ys, bins=250, clip_triangle=True, linewidth=2.0):
    """Add 68% (solid) and 95% (dashed) contours from fitted 2D Gaussian via Mahalanobis levels."""
    mean = np.array([np.mean(xs), np.mean(ys)])
    cov, inv = safe_cov_inv(xs, ys)

    x_edges = np.linspace(0.0, 1.0, bins + 1)
    y_edges = np.linspace(0.0, SQ3_OVER_2, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers)

    xm = Xc - mean[0]
    ym = Yc - mean[1]
    # Mahalanobis squared field
    m2 = inv[0,0]*xm*xm + (inv[0,1]+inv[1,0])*0.5*2.0*xm*ym + inv[1,1]*ym*ym

    if clip_triangle:
        mask = inside_triangle_mask(Xc, Yc)
        m2 = np.where(mask, m2, np.nan)

    CS = ax.contour(x_centers, y_centers, m2, levels=[CHI2_2_95, CHI2_2_68], linestyles=["--", "-"], linewidths=linewidth)
    fmt = {CS.levels[0]: "95%", CS.levels[1]: "68%"}
    ax.clabel(CS, inline=True, fontsize=9, fmt=fmt)
    return mean, cov, CS


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

    # Choose row
    if ROW_INDEX is None:
        totals = df[["N_e", "N_mu", "N_tau"]].sum(axis=1)
        idx = int(totals.idxmax())
    else:
        idx = int(ROW_INDEX)
        if idx < 0 or idx >= len(df):
            raise IndexError(f"ROW_INDEX={idx} is out of bounds for {len(df)} rows.")
    r = df.iloc[idx]

    # Extract values
    Ne, Nmu, Ntau = float(r["N_e"]), float(r["N_mu"]), float(r["N_tau"])
    err_e, err_mu, err_tau = float(r["err_e"]), float(r["err_mu"]), float(r["err_tau"])
    E_low, E_high = float(r["E_low_GeV"]), float(r["E_high_GeV"])

    Ntot = Ne + Nmu + Ntau
    if Ntot <= 0:
        raise ValueError("Total number of events is zero or negative for the selected decade.")

    fe = Ne / Ntot
    fmu = Nmu / Ntot
    ftau = Ntau / Ntot

    # Central point
    x0, y0 = bary_to_xy(fe, fmu, ftau)

    # Monte Carlo sampling
    rng = np.random.default_rng(seed=42)
    Ne_s  = rng.normal(loc=Ne,  scale=max(err_e,  1e-12), size=NSAMPLES)
    Nmu_s = rng.normal(loc=Nmu, scale=max(err_mu, 1e-12), size=NSAMPLES)
    Ntau_s= rng.normal(loc=Ntau,scale=max(err_tau,1e-12), size=NSAMPLES)

    Ne_s  = np.clip(Ne_s,  0, None)
    Nmu_s = np.clip(Nmu_s, 0, None)
    Ntau_s= np.clip(Ntau_s,0, None)

    Ntot_s = Ne_s + Nmu_s + Ntau_s
    mask = Ntot_s > 0
    Ne_s, Nmu_s, Ntau_s, Ntot_s = Ne_s[mask], Nmu_s[mask], Ntau_s[mask], Ntot_s[mask]

    fe_s  = Ne_s  / Ntot_s
    fmu_s = Nmu_s / Ntot_s
    ftau_s= Ntau_s/ Ntot_s

    xs = fmu_s * 1.0 + ftau_s * 0.5
    ys = ftau_s * SQ3_OVER_2

    # Plot
    fig, ax = plt.subplots(figsize=(8.6, 8.6))

    # Gaussian heatmap (elliptical-ish)
    if SHOW_GAUSSIAN_HEAT:
        _, mean_gauss, cov_gauss = add_gaussian_heat(ax, xs, ys, bins=int(GAUSS_BINS), clip_triangle=bool(GAUSS_CLIP_TRI))

    # Triangle, grid, ticks
    draw_triangle(ax)
    draw_fraction_grid(ax, step=float(GRID_STEP))
    annotate_fraction_ticks(ax, step=float(GRID_STEP), fontsize=float(TICK_FONTSZ))

    # Raw scatter blob (optional; off by default)
    if SHOW_BLOB:
        ax.scatter(xs, ys, s=5, alpha=0.08, edgecolors="none")

    # Central point
    ax.plot([x0], [y0], marker="o", markersize=8)

    # 68% & 95% contours from fitted 2D Gaussian (elliptical)
    add_gaussian_contours(ax, xs, ys, bins=int(GAUSS_BINS), clip_triangle=bool(GAUSS_CLIP_TRI), linewidth=float(GAUSS_CONTOUR_LINEWIDTH))

    # Axis titles & vertex labels
    label_axes(ax)

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQ3_OVER_2 + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])

    # Properly formatted LaTeX title (two lines)
    title_line1 = rf"Flavor Triangle ($E \in [{E_low:.3g}, {E_high:.3g}]\,\mathrm{{GeV}}$)"
    title_line2 = rf"central fractions:\ $f_e={fe:.3f},\ f_\mu={fmu:.3f},\ f_\tau={ftau:.3f}$"
    ax.set_title(title_line1 + "\n" + title_line2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=230)
    print(f"Saved: {OUT_PATH}")
    print(f"Selected row: {idx}   Energy decade: {E_low:.3g}–{E_high:.3g} GeV")
    print(f"Central fractions: fe={fe:.5f}, fmu={fmu:.5f}, ftau={ftau:.5f}")
    print(f"Samples used: {len(xs)} (after filtering non-physical draws)")

if __name__ == "__main__":
    main()