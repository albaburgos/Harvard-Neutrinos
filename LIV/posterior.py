#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------ Config (no CLI) ------------------------
D_OPERATOR = 4            # operator dimension d
ORDERING   = "NO"         # "NO" or "IO"
EMIN, EMAX = 1e4, 1e8     # GeV; integrate uniformly in logE
NE         = 300          # number of log-spaced energies
KNN_K      = 12           # neighbors for grid smoothing
MIN_LIVE   = 400          # UltraNest live points
DLOGZ      = 0.1          # UltraNest stopping criterion
MAXCALLS   = 200000
RESULTS_DIR = "results"
GRID_NPZ    = "LIV/llgrid.npz"  # optional cache: contains 'flavors' & 'll_grid_energy'

# ------------------------ PMNS & Hamiltonians ------------------------
def pmns_matrix():
    th12, th23, th13, dcp = np.deg2rad([33.44, 49.0, 8.57, 195.0])
    s12, c12 = np.sin(th12), np.cos(th12)
    s23, c23 = np.sin(th23), np.cos(th23)
    s13, c13 = np.sin(th13), np.cos(th13)
    U = np.array([
        [c12*c13,                s12*c13,                s13*np.exp(-1j*dcp)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*dcp),
          c12*c23 - s12*s23*s13*np.exp(1j*dcp),          s23*c13],
        [ s12*s23 - c12*c23*s13*np.exp(1j*dcp),
         -c12*s23 - s12*c23*s13*np.exp(1j*dcp),          c23*c13]
    ], dtype=complex)
    return U

def h_vacuum(E_GeV, ordering="NO"):
    U = pmns_matrix()
    dm21 = 7.42e-5   # eV^2
    dm31_NO = 2.517e-3
    dm31_IO = -2.498e-3
    if ordering.upper() == "NO":
        m1, m2, m3 = 0.0, dm21, dm31_NO
    else:
        m3, m1 = 0.0, abs(dm31_IO)
        m2 = m1 + dm21
    ev2_to_GeV2 = 1e-18
    m2_GeV2 = np.array([m1, m2, m3]) * ev2_to_GeV2
    Hm = np.diag(m2_GeV2 / (2.0 * E_GeV))
    return U @ Hm @ U.conj().T

def h_liv(E_GeV, d, a_tau_tau):
    scale = (E_GeV ** (d - 3))
    a = np.zeros((3,3), dtype=float)
    a[2,2] = a_tau_tau
    return scale * a

def decohered_P_from_H(H):
    evals, evecs = np.linalg.eigh(H)
    A = np.abs(evecs)**2
    return A @ A.T  # rows: detected α, cols: source β

# ------------------------ Energy-averaged flavor at Earth ------------------------
def average_earth_flavor(a_tau_tau, d=D_OPERATOR, ordering=ORDERING,
                         Emin=EMIN, Emax=EMAX, nE=NE):
    Es = np.logspace(np.log10(Emin), np.log10(Emax), nE)
    f_source = np.array([1/3, 2/3, 0.0], dtype=float)  # same as your script
    acc = np.zeros(3, dtype=float)
    for E in Es:
        H = h_vacuum(E, ordering=ordering).astype(complex) + h_liv(E, d=d, a_tau_tau=a_tau_tau)
        H = 0.5 * (H + H.conj().T)  # ensure Hermiticity
        P = decohered_P_from_H(H)
        f_det = P @ f_source
        f_det = np.real_if_close(f_det)
        f_det = f_det / np.sum(f_det)
        acc += f_det
    favg = acc / len(Es)
    favg = np.clip(favg, 0.0, 1.0)
    favg /= favg.sum()
    return favg  # (fe, fmu, ftau)

# ------------------------ Likelihood grid helper ------------------------
class TernaryLikelihood:
    """k-NN inverse-distance smoother on (fe,fmu,ftau)→logL."""
    def __init__(self, points3, loglikes, k=12, eps=1e-12):
        pts = np.asarray(points3, dtype=float)
        ll  = np.asarray(loglikes, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points3 must be (N,3) for (fe,fmu,ftau).")
        s = pts.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        self.points = np.clip(pts / s, 0.0, 1.0)
        self.ll = ll
        self.k = int(k)
        self.eps = float(eps)

    def query_loglike(self, f_vec):
        f = np.asarray(f_vec, dtype=float)
        f = np.clip(f, 0.0, 1.0)
        s = f.sum()
        if s <= 0: return -np.inf
        f = f / s
        d2 = np.sum((self.points - f)**2, axis=1)
        k = min(self.k, d2.size)
        idx = np.argpartition(d2, k-1)[:k]
        w = 1.0 / (d2[idx] + self.eps)
        w /= w.sum()
        return float(np.sum(w * self.ll[idx]))

# ------------------------ Prior & sampler ------------------------
def liv_prior_bounds(d):
    # a_tau_tau ∈ [-10^{-(4d+10)}, +10^{-(4d+10)}]
    w = 10.0 ** (-(4*int(d) + 10))
    return -w, +w

def build_sampler(grid_like, d=D_OPERATOR):
    from ultranest import ReactiveNestedSampler
    a_lo, a_hi = liv_prior_bounds(d)

    def transform(u):
        # u in [0,1] -> a in [a_lo, a_hi]
        a = a_lo + (a_hi - a_lo) * u[0]
        return [a]

    def loglike(theta):
        (a_tau_tau,) = theta
        fe, fmu, ftau = average_earth_flavor(a_tau_tau, d=d)
        return grid_like.query_loglike([fe, fmu, ftau])

    return ReactiveNestedSampler(['a_tau_tau'], loglike, transform)

# ------------------------ Posterior utilities ------------------------
def weighted_quantile(x, w, qs):
    x = np.asarray(x); w = np.asarray(w)
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w); c = c / c[-1]
    return np.interp(qs, c, x)

def plot_1d_posterior(samples, weights, outpng):
    w = weights / np.sum(weights)
    hist, edges = np.histogram(samples, bins=120, weights=w, density=True)
    ctrs = 0.5*(edges[1:] + edges[:-1])
    plt.figure(figsize=(6,4))
    plt.plot(ctrs, hist, lw=2)
    plt.xlabel(r"$a_{\tau\tau}$")
    plt.ylabel("Posterior density")
    plt.title("Posterior for LIV coefficient $a_{\\tau\\tau}$")
    plt.tight_layout()
    plt.savefig(outpng, dpi=160)
    plt.close()

# ------------------------ Grid loading ------------------------
def load_llgrid():
    g = globals()
    if 'flavors' in g and 'll_grid' in g:
        pts = np.asarray(g['flavors'], dtype=float)
        ll  = np.asarray(g['ll_grid'], dtype=float)
        return pts, ll
    if os.path.exists(GRID_NPZ):
        d = np.load(GRID_NPZ)
        return np.asarray(d['flavors']), np.asarray(d['ll_grid'])
    raise RuntimeError("Provide 'flavors' and 'll_grid' in the session, or save them to 'llgrid.npz'.")

# ------------------------ Main (no args) ------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load grid
    pts, ll = load_llgrid()
    grid_like = TernaryLikelihood(pts, ll, k=KNN_K)

    try:
        from ultranest import ReactiveNestedSampler  # noqa: F401 (import check)
    except Exception as e:
        raise SystemExit("UltraNest not available. Install with: pip install ultranest") from e

    sampler = build_sampler(grid_like, d=D_OPERATOR)
    a_lo, a_hi = liv_prior_bounds(D_OPERATOR)
    print(f"Prior on a_tau_tau: Uniform({a_lo:.3e}, {a_hi:.3e})")

    results = sampler.run(
    min_num_live_points=MIN_LIVE,
    dlogz=DLOGZ,
    max_ncalls=MAXCALLS,   # <-- rename
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    )

    # Save results JSON
    with open(os.path.join(RESULTS_DIR, "ultranest_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))

    post = sampler.results
    samples = np.array(post['samples'])[:,0]
    weights = np.array(post['weights'])
    weights = weights / weights.sum()

    np.savez(os.path.join(RESULTS_DIR, "post_equal_weights.npz"),
             a_tau_tau=samples, weights=weights)

    med, lo68, hi68 = weighted_quantile(samples, weights, [0.5, 0.16, 0.84])
    lo95, hi95 = weighted_quantile(samples, weights, [0.025, 0.975])
    map_like = samples[np.argmax(weights)]

    print("\nPosterior summary for a_tau_tau:")
    print(f"  median = {med:.3e}")
    print(f"  68% CI = [{lo68:.3e}, {hi68:.3e}]")
    print(f"  95% CI = [{lo95:.3e}, {hi95:.3e}]")
    print(f"  MAP-like ≈ {map_like:.3e}")

    outpng = os.path.join(RESULTS_DIR, "posterior_a_tau_tau.png")
    plot_1d_posterior(samples, weights, outpng)
    print("\nSaved:")
    print(f"  {os.path.join(RESULTS_DIR, 'ultranest_results.json')}")
    print(f"  {os.path.join(RESULTS_DIR, 'post_equal_weights.npz')}")
    print(f"  {outpng}")

if __name__ == "__main__":
    main()
