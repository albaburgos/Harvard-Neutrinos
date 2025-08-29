import os
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.patheffects as pe
import numpy as np # Perform mathematical operations on arrays
from scipy.stats import poisson # To calculate poisson probabilities 
import scipy.optimize as opt # Function optimizers and root finders
import matplotlib.pyplot as plt 

# -------------------- Configuration (from user's script) --------------------
CSV_PATH = "MC_outputs/effareasMESE.csv"
BIN_WIDTH_LOG10 = 0.2
PHI0 = 2.72e-18
T_EXPOSURE = 10*365.25 * 24 * 3600.0
OMEGA = 4*np.pi
E0 = 1e5
GRID_STEP = 1/60      
SHAPE_NORMALIZE = True
OUTDIR = "MC_outputs"

# ---------------------------------------------------------------------------


def _draw_const_lines(ax, interval):
    vals = np.arange(interval, 1.0, interval)
    # constant f_e = c  (lines parallel to f_e=0 edge)
    for c in vals:
        fe = c
        fmu0, ftau0 = 0.0, 1.0 - fe
        fmu1, ftau1 = 1.0 - fe, 0.0
        x12, y12 = bary_to_xy(
            np.array([fe, fe]),
            np.array([fmu0, fmu1]),
            np.array([ftau0, ftau1]),
        )
        ax.plot(x12, y12, ls="--", lw=0.7, color="0.75", zorder=1)

    # constant f_mu = c  (lines parallel to f_mu=0 edge)
    for c in vals:
        fmu = c
        fe0, ftau0 = 0.0, 1.0 - fmu
        fe1, ftau1 = 1.0 - fmu, 0.0
        x12, y12 = bary_to_xy(
            np.array([fe0, fe1]),
            np.array([fmu, fmu]),
            np.array([ftau0, ftau1]),
        )
        ax.plot(x12, y12, ls="--", lw=0.7, color="0.75", zorder=1)

    # constant f_tau = c  (lines parallel to f_tau=0 edge)
    for c in vals:
        ftau = c
        fe0, fmu0 = 0.0, 1.0 - ftau
        fe1, fmu1 = 1.0 - ftau, 0.0
        x12, y12 = bary_to_xy(
            np.array([fe0, fe1]),
            np.array([fmu0, fmu1]),
            np.array([ftau, ftau]),
        )
        ax.plot(x12, y12, ls="--", lw=0.7, color="0.75", zorder=1)

def bary_to_xy(fe: np.ndarray, fmu: np.ndarray, ftau: np.ndarray):

    x = fe + 0.5 * fmu
    y = (np.sqrt(3) / 2.0) * fmu
    return x, y

POINTS = {
    "0:1:0": (0.17, 0.45, 0.37),
    "1:2:0": (0.30, 0.36, 0.34),
    "1:0:0": (0.55, 0.17, 0.28),
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
    stop = lo + math.ceil((hi - lo))

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

from scipy.special import gammaln

def poisson_loglike(k, lam):
    lam = np.asarray(lam, dtype=float)
    lam = np.clip(lam, 1e-12, None)
    k = np.asarray(k, dtype=float)
    return np.sum(k * np.log(lam) - lam - gammaln(k + 1.0))

class LikelihoodFunction:
    
    def __init__(self,data,event_classes,binning):
        '''Sets up a likelihood function for some data and event classes
        
           data is a 1D array of quantities describing each data event
           event_classes is a list of 1D arrays with quantites for each event 
               class. These should be derived from simulation, and will be 
               used to generate PDFs for each event class.
           binning is a 1D array of bin edges describing how the data and PDFs 
               should be binned for this analysis.
        '''
        # First step is to bin the data into a histogram (k_i)
        self.data_counts = np.histogram(data,bins=binning)[0]
        # Create a list to store PDFs for each event class
        self.class_pdfs = []
        for event_class in event_classes:
            # Bin the MC data from each event class the same way as data
            pdf_counts = np.histogram(event_class,bins=binning)[0]
            # Normalized PDF (H_ij) such that sum of all bins is 1
            pdf_norm = pdf_counts/np.sum(pdf_counts)
            # Save for later
            self.class_pdfs.append(pdf_norm)
        
    def __call__(self,*params):
        '''Evaluates the likelihood function and returns likelihood
        
           params is a list of scale factors for each PDF (event_class) passed
               to the __init__ method.
        '''
        # Observed event histogram is always the binned data
        observed = self.data_counts
        # Expected events are normalized PDFs times scale factors (\mu_j) for each PDF
        expecteds = [scale*pdf for scale,pdf in zip(params,self.class_pdfs)]
        # Sum up all the expected event historgrams bin-by-bin (sum over j is axis 0)
        expected = np.sum(expecteds,axis=0)
        # Calculate the bin-by-bin poisson probabilities to observe `observed` events
        # with an average `expected` events in each bin (these poisson functions operate bin-by-bin)
        bin_probabilities = poisson.pmf(observed,expected)
        # multiply all the probabilities together
        return np.prod(bin_probabilities)




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
    
    counts_listplot = [counts_001, counts_010, counts_100, base_counts]
    labels_plot = ["001", "010", "100", "base"]

    plot_2x2_histograms(coarse_edges,
                            counts_listplot,
                            labels_plot,
                            "2x2hist.png")

    flavors = generate_flavor_grid(step=GRID_STEP)
    fe_arr = np.array([g[0] for g in flavors], dtype=float)
    fmu_arr = np.array([g[1] for g in flavors], dtype=float)
    ftau_arr = np.array([g[2] for g in flavors], dtype=float)

    counts_list = []
    for i, (fe, fmu, ftau) in enumerate(flavors):
        counts = events_per_coarse_bin(
            E, edges, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, norm
        )
        counts = np.asarray(counts)   # ensure NumPy array
        counts_list.append(counts)    # add to Python list
        print(np.shape(counts_list))
        print(np.shape(coarse_edges))

    ll_grid = np.array([poisson_loglike(base_counts, lam) for lam in counts_list])

    best_idx = np.nanargmax(ll_grid)
    best_fe, best_fmu, best_ftau = flavors[best_idx]
    print("Best (fe, fmu, ftau):", best_fe, best_fmu, best_ftau)
    print("Best log-likelihood:", ll_grid[best_idx])
    
    # --- barycentric -> 2D triangle coordinates for plotting ---
    x, y = bary_to_xy(fe_arr, fmu_arr, ftau_arr)  # each length N

    # mask any bad values just in case
    m = np.isfinite(ll_grid) & np.isfinite(x) & np.isfinite(y)
    x_plot, y_plot, z_plot = x[m], y[m], ll_grid[m]

    # plot relative log-likelihood (peak at 0)
    z_plot = z_plot - np.max(z_plot)

    # triangulate scattered grid
    tri = Triangulation(x_plot, y_plot)

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw outer triangle
    verts = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, np.sqrt(3)/2.0],
                    [0.0, 0.0]])
    ax.plot(verts[:, 0], verts[:, 1], lw=1)

    
    levels_inc = np.sort([np.log(1.0 - 0.98),  # ≈ -3.9120
                      np.log(1.0 - 0.65)]) # ≈ -1.0498

    # draw the contours (first level -> dashed for 98%, second -> solid for 65%)
    cs_inc = ax.tricontour(
        tri, z_plot,
        levels=levels_inc,
        colors='k',
        linewidths=1.8,
        linestyles=['--', '-']
    )

    # OPTION A: add a legend using proxy handles (robust across Matplotlib versions)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.8, label='98%'),
        Line2D([0], [0], color='k', linestyle='-',  linewidth=1.8, label='65%'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False, title="Likelihood Ratio Contours")

    interval = 0.1
    _draw_const_lines(ax, interval)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.8,
            label='98% Likelihood Ratio contour'),
        Line2D([0], [0], color='k', linestyle='-',  linewidth=1.8,
            label='65% Likelihood Ratio contour'),
        Line2D([0], [0],
            marker='o', markersize=6,
            markerfacecolor='white', markeredgecolor='black',
            linestyle='None',
            label=(r'$\nu_e:\nu_\mu:\nu_\tau$ at source → on Earth:'
                    '\n 0:1:0 → 0.17:0.45:0.37'
                    '\n 1:2:0 → 0.30:0.36:0.34'
                    '\n 1:0:0 → 0.55:0.17:0.28')),
        Line2D([0], [0],
            marker='*', markersize=10,
            markerfacecolor='C0', markeredgecolor='black',
            linestyle='None',
            label=r'Best fit'), 
    ]
    ax.legend(
        handles=legend_handles,
        loc='best',
        frameon=True,
        fancybox=True,        # rounded corners
        framealpha=0.9,       # transparency (1 = solid)
        edgecolor="black",    # box edge color
        facecolor="white"     # background color
    )

    # annotate reference points
    for tag, (pfe, pfmu, pftau) in POINTS.items():
        pfe, pfmu, pftau = _normalize_bary(pfe, pfmu, pftau)
        px, py = bary_to_xy(np.array([pfe]), np.array([pfmu]), np.array([pftau]))
        ax.scatter(px, py, s=55, facecolors="white", edgecolors="black",
                linewidths=0.9, zorder=6)
        ax.annotate(tag, (px[0], py[0]), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, weight="bold", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # highlight the best point
    bx, by = bary_to_xy(np.array([best_fe]), np.array([best_fmu]), np.array([best_ftau]))
    ax.scatter(bx, by, s=65, marker="*", edgecolors="black", zorder=8)
    ax.scatter(bx, by, s=65, marker="*", facecolors="C0", edgecolors="black", zorder=8)

    # corner labels, aesthetics
    ax.text(-0.03, -0.03, r"$f_\tau=1$", ha="right", va="top")
    ax.text(0.95, -0.03,  r"$f_e=1$",   ha="left",  va="top")
    ax.text(0.5, np.sqrt(3)/2 + 0.1, r"$f_\mu=1$", ha="center", va="bottom")

    ax.text(0.05, 0.25, "fτ fraction",
        rotation=60, ha="center", va="center", fontsize=10)

    # right edge: f_mu fraction (slanted along edge from bottom-right to top)
    ax.text(0.95-0.05, 0.25, "fμ fraction",
            rotation=-60, ha="center", va="center", fontsize=10)

    # bottom edge: f_e fraction (flat along bottom)
    ax.text(0.50, -0.08, "fe fraction",
            ha="center", va="center", fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title('\nFlavor Likelihood Analysis All-Experiment(10yr)'
                 '\n assuming SPL Astrophysical Flux (MESE Best-Fit γ = -2.54)')

    os.makedirs("MC_outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("MC_outputs/triangle.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()

