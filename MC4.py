import os
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.patheffects as pe

# -------------------- Configuration (from user's script) --------------------
CSV_PATH = "MC_outputs/effareasMESE.csv"
BIN_WIDTH_LOG10 = 0.2
PHI0 = 2.72e-18
T_EXPOSURE = 11.3*365.25 * 24 * 3600.0
OMEGA = 4*np.pi
E0 = 1e5
GRID_STEP = 1/30       
SHAPE_NORMALIZE = True
OUTDIR = "MC_outputs"

# ---------------------------------------------------------------------------



def LogLPoisson(obs, exp):

    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)

    mask = exp > 0
    o = obs[mask]
    e = exp[mask]

    '''
    
    obs = np.asarray(obs, dtype=float)
    n = len(obs)

    if mu is None:
        mu = np.mean(obs)
    if sigma is None:
        sigma = np.std(obs, ddof=0)

    logL = (
        -0.5 * n * np.log(2 * np.pi)
        -0.5 * n * np.log(sigma**2)
        -0.5 * np.sum((obs - mu)**2) / (sigma**2)
    )
    '''

    term = np.where(o > 0, o * np.log(o / e), 0.0)
    ts = 2.0 * np.sum(e - o + term)
    return ts


def gaussian_LR(obs, exp, sigma):
    """
    Gaussian likelihood-ratio test statistic
    (equivalent to chi^2) between observed and expected.

    Parameters
    ----------
    obs : array-like
        Observed values
    exp : array-like
        Expected values under hypothesis
    sigma : array-like
        Standard deviation per bin (must be >0)

    Returns
    -------
    ts : float
        -2 log λ test statistic
    """
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mask = sigma > 0
    o, e, s = obs[mask], exp[mask], sigma[mask]

    ts = -np.sum(((o - e)/s)**2+np.log(2*np.pi*s**2))
    return ts



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


def read_effarea_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV "{path}" not found in current directory.')
    # Try no header first
    try:
        df = pd.read_csv(path, comment="#", header=None)
        if df.shape[1] < 4:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, comment="#")
        if df.shape[1] < 4:
            raise ValueError("CSV must have ≥4 columns: master, A_muon, A_tau, A_electron")

    E = np.asarray(df.iloc[:, 0], dtype=float)
    A_mu = np.asarray(df.iloc[:, 1], dtype=float)
    A_tau = np.asarray(df.iloc[:, 2], dtype=float)
    A_e  = np.asarray(df.iloc[:, 3], dtype=float)

    order = np.argsort(E)
    return E[order], A_mu[order], A_tau[order], A_e[order]


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

    # start from the first edge's log10
    start = lo
    # extend in equal log10 steps until we cover the max edge (may exceed it)
    stop = lo + math.ceil((hi - lo) / bin_width_log10) * bin_width_log10

    # generate edges at fixed log10 spacing
    raw = 10.0 ** np.arange(start, stop + 1e-12, bin_width_log10)

    # ensure the very first edge is exactly edges[0]
    raw[0] = edges[0]

    # return as-is (no clipping to edges[-1])
    b = np.unique(raw)
    return b

def integrate_bin_I(edges: np.ndarray, centers: np.ndarray, A: np.ndarray, Emin: float, Emax: float) -> float:
    total = 0.0
    dE = np.diff(edges)
    for j in range(len(centers)):
        a = edges[j]
        b = edges[j+1]
        L = max(a, Emin)
        U = min(b, Emax)
        if L < U:
            total += A[j] * ((centers[j]/E0)**-2.54) * (U-L) * T_EXPOSURE * OMEGA * PHI0 
    return float(total)


def events_per_coarse_bin(E: np.ndarray, edges: np.ndarray, A_e: np.ndarray, A_mu: np.ndarray, A_tau: np.ndarray,
                          coarse_edges: np.ndarray, fe: float, fmu: float, ftau: float, norm: float):
    counts = []
    for i in range(len(coarse_edges) - 1):
        Emin = float(coarse_edges[i])
        Emax = float(coarse_edges[i + 1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax)
        Imu  = integrate_bin_I(edges, E, A_mu, Emin, Emax)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax)
        counts.append((Ie*fe+Imu*fmu+Itau*ftau)/(1/3))

    return np.asarray(counts)


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


def plot_histogram(coarse_edges: np.ndarray, counts: np.ndarray, fe: float, fmu: float, ftau: float, out_png: str):
    lefts = coarse_edges[:-1]
    rights = coarse_edges[1:]
    widths = rights - lefts
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(lefts, counts, width=widths, align="edge", edgecolor="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Number of events per bin")
    ax.set_title(f"Events per energy bin (10 years) — fe={fe:.1f}, fμ={fmu:.1f}, fτ={ftau:.1f}")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_2x2_histograms(coarse_edges: np.ndarray,
                         counts_list: List[np.ndarray],
                         labels: List[str],
                         out_png: str):
    """Make a 2x2 grid of energy histograms sharing axes.

    counts_list must contain exactly 4 arrays corresponding to labels.
    """
    assert len(counts_list) == 4 and len(labels) == 4, "Need 4 cases for a 2x2 grid"

    lefts = coarse_edges[:-1]
    rights = coarse_edges[1:]
    widths = rights - lefts

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, counts, lab in zip(axes, counts_list, labels):
        ax.bar(lefts, counts, width=widths, align="edge", edgecolor="black", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_title(lab)
        ax.grid(True, which="both", axis="both", alpha=0.3)

    axes[2].set_xlabel("Energy (GeV)")
    axes[3].set_xlabel("Energy (GeV)")
    axes[0].set_ylabel("Events / bin")
    axes[2].set_ylabel("Events / bin")

    fig.suptitle("Events per energy bin (10 years)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_counts_csv(coarse_edges: np.ndarray, counts: np.ndarray, path: str):
    centers = np.sqrt(coarse_edges[:-1] * coarse_edges[1:])
    df = pd.DataFrame({
        "Emin_GeV": coarse_edges[:-1],
        "Emax_GeV": coarse_edges[1:],
        "Ecenter_GeV": centers,
        "N_events": counts,
    })
    df.to_csv(path, index=False)

def log_likelihood_ratio(observed, expected):

    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    sigma_null = np.sqrt(expected)
    sigma_alt = np.asarray(observed)

    term1 = np.log(sigma_null / sigma_alt)
    term2 = (expected**2) / (2 * sigma_null**2)
    term3 = ((observed-expected)**2) / (2 * sigma_alt**2)

    Lg = np.sum(term1 + term2 - term3)
    return Lg
 
def lr_at_coverage(alpha, cum, order, LR):
    """
    Return LR threshold L_alpha such that the HPD region {LR >= L_alpha}
    contains fraction `alpha` of total weight.
    """
    idx = np.searchsorted(cum, alpha, side="left")
    idx = min(idx, len(order)-1)
    return LR[order[idx]]

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Read and prepare
    E, A_mu, A_tau, A_e = read_effarea_csv(CSV_PATH)
    edges = geometric_edges_from_centers(E)
    coarse_edges = make_log_bins(edges, BIN_WIDTH_LOG10)
    norm = 2

    # Baseline distribution (1/3,1/3,1/3)
    fe0 = fmu0 = ftau0 = 1.0/3.0
    base_counts = events_per_coarse_bin(
        E, edges, A_e, A_mu, A_tau, coarse_edges, fe0, fmu0, ftau0, norm
    )

    # Additional single-flavor extremes for 2x2 grid: 001, 010, 100, and 1/3-1/3-1/3
    counts_001 = events_per_coarse_bin(E, edges, A_e, A_mu, A_tau, coarse_edges, 0.0, 0.0, 1.0, norm)
    counts_010 = events_per_coarse_bin(E, edges, A_e, A_mu, A_tau, coarse_edges, 0.0, 1.0, 0.0, norm)
    counts_100 = events_per_coarse_bin(E, edges, A_e, A_mu, A_tau, coarse_edges, 1.0, 0.0, 0.0, norm)

    # Plot and save the 2x2 grid
    out_grid = os.path.join(OUTDIR, "hist_2x2_001_010_100_333.png")
    plot_2x2_histograms(
        coarse_edges,
        [counts_001, counts_010, counts_100, base_counts],
        labels=["fe=0, fμ=0, fτ=1 (001)", "fe=0, fμ=1, fτ=0 (010)", "fe=1, fμ=0, fτ=0 (100)", "fe=fμ=fτ=1/3"],
        out_png=out_grid,
    )

    # Sweep grid
    flavors = generate_flavor_grid(step=GRID_STEP)
    fe_arr = np.array([g[0] for g in flavors], dtype=float)
    fmu_arr = np.array([g[1] for g in flavors], dtype=float)
    ftau_arr = np.array([g[2] for g in flavors], dtype=float)
    
    LRs = np.empty(len(flavors))
    for i, (fe, fmu, ftau) in enumerate(flavors):
        counts = events_per_coarse_bin(
            E, edges, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, norm
        )

        LRs[i] = log_likelihood_ratio(counts, base_counts)
    
    LR = np.asarray(LRs, dtype=float) - min(LRs)
    print(LR)
    '''
    w_sum = np.sum(LR)
    order = np.argsort(LR)
    cum = np.cumsum(LR[order]) / w_sum
    cov_1s = 0.6827
    cov_2s = 0.9545
    L1 = lr_at_coverage(cov_1s, cum, order, LR)   # 1σ level
    L2 = lr_at_coverage(cov_2s, cum, order, LR)   # 2σ level
    rank_survival = np.empty_like(LR)
    rank_survival[order] = cum
    excl_pct = (1-rank_survival) * 100.0
    print(cum[-1])
    '''

    plt.figure(figsize=(8, 5))
    plt.hist(LR, bins=60, density=True, alpha=0.6, color="gray")

    plt.title("PDF of LR with 1σ and 2σ Coverage Thresholds")
    plt.xlabel("LR")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ---------- coverage thresholds & exclusion map ----------
    cov_1s, cov_2s = 0.6827, 0.9545

    # shift so LR is non-negative (you already do this above)
    w = LR - LR.min()
    if np.allclose(w.sum(), 0):
        # guard against degenerate case
        w = np.ones_like(w)

    # sort by descending LR "weight", compute cumulative mass
    order = np.argsort(w)[::-1]
    cum = np.cumsum(w[order]) / np.sum(w)

    def lr_at_coverage(cov, order, cum, field):
        idx = np.searchsorted(cum, cov, side="left")
        idx = np.clip(idx, 0, len(order) - 1)
        return field[order[idx]]

    L1 = lr_at_coverage(cov_1s, order, cum, w)  # 1σ level in LR units (after shift)
    L2 = lr_at_coverage(cov_2s, order, cum, w)  # 2σ level in LR units (after shift)

    # per-point survival / exclusion percentage
    rank_survival = np.empty_like(w, dtype=float)
    rank_survival[order] = cum
    excl_pct = (1.0 - rank_survival) * 100.0  # 0% = best, 100% = worst

    x, y = bary_to_xy(fe_arr, fmu_arr, ftau_arr)
    plt.figure(figsize=(7, 7))
    tri = Triangulation(x, y)

    # draw outer triangle
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2.0], [0.0, 0.0]])
    plt.plot(verts[:, 0], verts[:, 1], lw=1)

    # background: exclusion %
    tpc = plt.tripcolor(tri, excl_pct, shading="gouraud", vmin=0.0, vmax=100.0)
    cbar = plt.colorbar(tpc, pad=0.02, fraction=0.045)
    cbar.set_label("Exclusion [%] (0 = best)")

    # overlay 1σ / 2σ as contours of exclusion %
    levels = [(1.0 - cov_2s) * 100.0, (1.0 - cov_1s) * 100.0]  # [4.55, 31.73]
    cs = plt.tricontour(tri, excl_pct, levels=levels, linewidths=1.25)
    labels = {levels[0]: r"2$\sigma$", levels[1]: r"1$\sigma$"}
    plt.clabel(cs, inline=True, fmt=labels)
    

    # annotate reference points
    for tag, (pfe, pfmu, pftau) in POINTS.items():
        pfe, pfmu, pftau = _normalize_bary(pfe, pfmu, pftau)
        px, py = bary_to_xy(np.array([pfe]), np.array([pfmu]), np.array([pftau]))
        plt.scatter(px, py, s=55, facecolors="white", edgecolors="black",
                    linewidths=0.9, zorder=6)
        plt.annotate(tag, (px[0], py[0]), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, weight="bold", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # corner labels, aesthetics
    plt.text(-0.03, -0.03, r"$f_\tau=1$", ha="right", va="top")
    plt.text(0.95, -0.03,  r"$f_e=1$",   ha="left",  va="top")
    plt.text(0.5, np.sqrt(3)/2 + 0.06, r"$f_\mu=1$", ha="center", va="bottom")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Likelihood analysis (exclusion %, 1σ/2σ)")

    plt.tight_layout()
    plt.savefig("MC_outputs/triangle.png", dpi=200, bbox_inches="tight")

    print("Likelihood thresholds (shifted LR units):")
    print(f"  1σ (68.27%): LR ≥ {L1:.6g}")
    print(f"  2σ (95.45%): LR ≥ {L2:.6g}")
    print("Saved: MC_outputs/triangle.png")
if __name__ == "__main__":
    main()



