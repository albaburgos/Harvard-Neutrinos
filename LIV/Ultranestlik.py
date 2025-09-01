"""
UltraNest template for LIV coefficient fits + neutrino nuisance priors
-----------------------------------------------------------------------------
This script gives you a *working* skeleton to run UltraNest with the priors
from your table. Swap in your actual likelihood/model where marked TODO.

It supports two setups:
  1) Single-parameter LIV fit (default) — choose Re/Im of a(d)_{alpha beta} or c(d)_{alpha beta}
  2) Joint example for d=3 diagonal CPT-odd (Re[a^(3)]_{ee,μμ,ττ})

Notes on priors taken from your snippet (interpreted):
  - LIV coefficients a^(d) (CPT-odd): Uniform in [−10^{−(4d+10)}, +10^{−(4d+10)}]
  - LIV coefficients c^(d) (CPT-even) real component: Uniform in [−10^{−(4d+7)}, +10^{−(4d+7)}]
  - Nuisance (NuFIT 5.2, Normal priors):
      Δm21^2 ~ N(7.14e−5, 0.019e−5)
      Δm31^2 ~ N(2.507e−3, 0.026e−3)    # assume normal ordering
      sin^2θ12 ~ N(0.303, 0.012)
      sin^2θ13 ~ N(0.02225, 0.00056)
      sin^2θ23 ~ N(0.5?) see table: using N(0.5, 0.1) would be generic; your table shows 2D posteriors — we keep Normal(0.5,0.1) as a placeholder if you want a looser prior.
      δ_CP     ~ N(2.89, 0.23)          # radians
  - Astrophysical source fraction f_e,S:
      fixed to {1/3, 0, 1} for pion / muon-damped / neutron cases, otherwise Uniform[0,1]
  - Spectral index γ: left Uniform[1.5, 3.0] as a generic prior (adjust to your analysis)

You can flip between setups via CONFIG at the top.
UltraNest: https://johannesbuchner.github.io/UltraNest/
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple

import ultranest
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot

# ------------------------------
# User configuration
# ------------------------------
CONFIG = dict(d = 3,
    # Operator mass dimension d (integer >= 3 typically) 

    # Which fit to run: 'single' or 'joint_d3_diag'
    fit_mode='single',

    # For 'single' fit: choose operator and component
    operator='a',           # 'a' (CPT-odd) or 'c' (CPT-even)
    component='Re',         # 'Re' or 'Im'
    alpha='e',              # flavor index α in {e, mu, tau}
    beta='e',               # flavor index β in {e, mu, tau}

    # Source production mode for f_e,S (electron-neutrino fraction at source)
    # None -> free fe,S in [0,1] with muon carrying (1-fe,S) and tau = 0.
    # 'pion' -> (1/3, 2/3, 0); 'muon-damped' -> (0,1,0); 'neutron' -> (1,0,0)
    source_mode=None,

    # Prior choices
    prior_gamma=(1.5, 3.0), # Uniform prior for spectral index γ

    # Whether to use the tighter NuFIT priors for sin^2θ23 given 2D posteriors.
    use_loose_s23_prior=True,

    # Energy integration for P_{αβ} averaging (GeV)
    E_min_GeV=1e4,   # 10 TeV
    E_max_GeV=1e7,   # 10 PeV
    N_E=41,
)

    d=3,

    # Which fit to run: 'single' or 'joint_d3_diag'
    fit_mode='single',

    # For 'single' fit: choose operator and component
    operator='a',           # 'a' (CPT-odd) or 'c' (CPT-even)
    component='Re',         # 'Re' or 'Im'
    alpha='e',              # flavor index α in {e, mu, tau}
    beta='e',               # flavor index β in {e, mu, tau}

    # Source production mode for f_e,S (electron-neutrino fraction at source)
    source_mode=None,       # None (float prior U[0,1]) or one of {'pion','muon-damped','neutron'}

    # Prior choices
    prior_gamma=(1.5, 3.0), # Uniform prior for spectral index γ

    # Whether to use the tighter NuFIT priors for sin^2θ23 given 2D posteriors.
    # If False, we use a modest Normal prior N(0.5, 0.1) (bounded to [0,1]).
    use_loose_s23_prior=True,
)

# ------------------------------
# Helpers: distributions on [0,1] -> R
# ------------------------------

def uniform_ppf(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) * u

# Acklam's rational approximation for inverse normal CDF
# Reference: https://web.archive.org/web/20150910044719/http://home.online.no/~pjacklam/notes/invnorm/
# Implemented to avoid SciPy dependency

def _invnorm(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-12, 1-1e-12)
    # Coefficients
    a = [ -3.969683028665376e+01,
           2.209460984245205e+02,
          -2.759285104469687e+02,
           1.383577518672690e+02,
          -3.066479806614716e+01,
           2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,
           1.615858368580409e+02,
          -1.556989798598866e+02,
           6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03,
          -3.223964580411365e-01,
          -2.400758277161838e+00,
          -2.549732539343734e+00,
           4.374664141464968e+00,
           2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,
          3.224671290700398e-01,
          2.445134137142996e+00,
          3.754408661907416e+00 ]

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(u)
    # lower region
    mask = u < plow
    if mask.any():
        q = np.sqrt(-2*np.log(u[mask]))
        x[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    # central region
    mask = (u >= plow) & (u <= phigh)
    if mask.any():
        q = u[mask] - 0.5
        r = q*q
        x[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    # upper region
    mask = u > phigh
    if mask.any():
        q = np.sqrt(-2*np.log(1-u[mask]))
        x[mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return x

def normal_ppf(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return mu + sigma * _invnorm(u)

# ------------------------------
# LIV prior utilities
# ------------------------------

def liv_range(d: int, operator: str) -> float:
    """Return half-width R(d) for uniform prior of LIV coefficient.
    a^(d): R = 10^{-(4d+10)} ; c^(d): R = 10^{-(4d+7)}
    """
    if operator.lower() == 'a':
        k = 10
    elif operator.lower() == 'c':
        k = 7
    else:
        raise ValueError("operator must be 'a' or 'c'")
    return 10.0 ** (-(4*d + k))

flavors = ['e', 'mu', 'tau']

# ------------------------------
# Build parameter list + prior transform
# ------------------------------

def build_parameters_and_prior(config: Dict) -> Tuple[List[str], callable]:
    d = config['d']
    fit_mode = config['fit_mode']

    names: List[str] = []

    if fit_mode == 'single':
        op = config['operator']
        comp = config['component']
        a = config['alpha']
        b = config['beta']
        if a not in flavors or b not in flavors:
            raise ValueError("alpha/beta must be in {e, mu, tau}")
        names.append(f"{comp}[{op}^{d}]_{a}{b}")
        # 1 LIV param
    elif fit_mode == 'joint_d3_diag':
        # Example: real diagonal Re[a^(3)]_{ee,μμ,ττ}
        names += [f"Re[a^{d}]_ee", f"Re[a^{d}]_mumu", f"Re[a^{d}]_tautau"]
    else:
        raise ValueError("fit_mode must be 'single' or 'joint_d3_diag'")

    # Nuisance parameters common to both fits (NuFIT 5.2 inspired)
    # We parameterize s12 = sin^2 θ12 etc., directly.
    names += [
        'dm21_sq', 'dm31_sq',
        's12', 's13', 's23', 'delta_cp',
        'gamma', 'fe_S'
    ]

    # Precompute ranges
    R = liv_range(d, config['operator'])
    gamma_lo, gamma_hi = config['prior_gamma']

    # Fixed source fractions
    source_mode = config['source_mode']
    fixed_fe = None
    if source_mode is not None:
        mapping = {
            'pion': 1.0/3.0,
            'muon-damped': 0.0,
            'neutron': 1.0
        }
        try:
            fixed_fe = mapping[source_mode]
        except KeyError:
            raise ValueError("source_mode must be one of None, 'pion', 'muon-damped', 'neutron'")

    # Indices for convenience
    idx = {n: i for i, n in enumerate(names)}

    def prior_transform(u: np.ndarray) -> np.ndarray:
        # u is in [0,1]^ndim
        x = np.empty_like(u)
        k = 0
        # LIV params
        if fit_mode == 'single':
            x[k] = uniform_ppf(u[k], -R, R)
            k += 1
        else:  # joint_d3_diag -> three diagonal Re[a]
            for _ in range(3):
                x[k] = uniform_ppf(u[k], -R, R)
                k += 1

        # dm21^2, dm31^2 (eV^2)
        x[k] = normal_ppf(u[k], 7.14e-5, 0.019e-5); k += 1
        x[k] = normal_ppf(u[k], 2.507e-3, 0.026e-3); k += 1

        # sin^2 θ12, sin^2 θ13
        x[k] = np.clip(normal_ppf(u[k], 0.303, 0.012), 0.0, 1.0); k += 1
        x[k] = np.clip(normal_ppf(u[k], 0.02225, 0.00056), 0.0, 1.0); k += 1

        # sin^2 θ23
        if CONFIG['use_loose_s23_prior']:
            x[k] = np.clip(normal_ppf(u[k], 0.5, 0.1), 0.0, 1.0)
        else:
            # If you have the exact NuFIT 5.2 1D marginal, plug it here instead.
            x[k] = np.clip(normal_ppf(u[k], 0.57, 0.04), 0.0, 1.0)
        k += 1

        # δ_CP (radians)
        x[k] = normal_ppf(u[k], 2.89, 0.23); k += 1

        # γ spectral index
        x[k] = uniform_ppf(u[k], gamma_lo, gamma_hi); k += 1

        # f_e,S
        if fixed_fe is None:
            x[k] = uniform_ppf(u[k], 0.0, 1.0)
        else:
            x[k] = fixed_fe
        # k += 1 (not needed afterwards)

        return x

    return names, prior_transform

# ------------------------------
# Physics likelihood: PMNS + LIV (energy-averaged decohered transition)
# ------------------------------

FLAV_IDX = {'e': 0, 'mu': 1, 'tau': 2}


def pmns_matrix(s12: float, s13: float, s23: float, delta: float) -> np.ndarray:
    """Return standard PDG PMNS matrix given sin^2 angles and δ (radians)."""
    s12 = float(np.sqrt(np.clip(s12, 0.0, 1.0)))
    s13 = float(np.sqrt(np.clip(s13, 0.0, 1.0)))
    s23 = float(np.sqrt(np.clip(s23, 0.0, 1.0)))
    c12, c13, c23 = np.sqrt(1 - s12**2), np.sqrt(1 - s13**2), np.sqrt(1 - s23**2)
    e_minus_iδ = np.exp(-1j * delta)
    e_plus_iδ  = np.exp(+1j * delta)

    U = np.zeros((3,3), dtype=complex)
    U[0,0] = c12 * c13
    U[0,1] = s12 * c13
    U[0,2] = s13 * e_minus_iδ

    U[1,0] = -s12 * c23 - c12 * s23 * s13 * e_plus_iδ
    U[1,1] =  c12 * c23 - s12 * s23 * s13 * e_plus_iδ
    U[1,2] =  s23 * c13

    U[2,0] =  s12 * s23 - c12 * c23 * s13 * e_plus_iδ
    U[2,1] = -c12 * s23 - s12 * c23 * s13 * e_plus_iδ
    U[2,2] =  c23 * c13
    return U


def mass_hamiltonian(E_eV: float, dm21: float, dm31: float, U: np.ndarray) -> np.ndarray:
    """Vacuum term H_mass = (1/2E) U diag(0, dm21, dm31) U^† (units: eV)."""
    M2 = np.diag([0.0, dm21, dm31])
    return (U @ M2 @ U.conj().T) / (2.0 * E_eV)


def liv_hamiltonian(E_eV: float, theta: np.ndarray, names: List[str]) -> np.ndarray:
    """Build Hermitian LIV Hamiltonian in flavor basis.
    We use scaling ∝ E^{d-3}. Supports:
      - fit_mode 'single': single complex entry at (α,β)
      - fit_mode 'joint_d3_diag': three diagonal Re[a^(d)] entries
    """
    d = CONFIG['d']
    scale = (E_eV ** (d - 3))

    H = np.zeros((3,3), dtype=complex)
    fit_mode = CONFIG['fit_mode']

    if fit_mode == 'single':
        val = float(theta[0])
        i = FLAV_IDX[CONFIG['alpha']]
        j = FLAV_IDX[CONFIG['beta']]
        if CONFIG['component'].lower() == 're':
            H[i, j] += val
            H[j, i] += val
        else:  # 'Im' component -> purely imaginary off-diagonal; ignore diagonal imaginary to keep Hermiticity
            if i != j:
                H[i, j] += 1j * val
                H[j, i] -= 1j * val
    elif fit_mode == 'joint_d3_diag':
        # Interpret first three parameters as Re[a^(d)]_{ee,μμ,ττ}
        H[0,0] += float(theta[0])
        H[1,1] += float(theta[1])
        H[2,2] += float(theta[2])
    else:
        raise ValueError('Unknown fit_mode')

    return scale * H


def transition_matrix_avg(theta: np.ndarray, names: List[str]) -> np.ndarray:
    """Compute energy-averaged decohered transition matrix P_{αβ}.
    P(E) = |V(E)|^2 @ |V(E)|^2^T where columns of V are eigenvectors of H(E).
    Average over power-law E with index γ between E_min and E_max.
    """
    dm21 = float(theta[names.index('dm21_sq')])
    dm31 = float(theta[names.index('dm31_sq')])
    s12  = float(theta[names.index('s12')])
    s13  = float(theta[names.index('s13')])
    s23  = float(theta[names.index('s23')])
    delta= float(theta[names.index('delta_cp')])
    gamma= float(theta[names.index('gamma')])

    U = pmns_matrix(s12, s13, s23, delta)

    # Energy grid (GeV -> eV)
    E_GeV = np.logspace(np.log10(CONFIG['E_min_GeV']), np.log10(CONFIG['E_max_GeV']), CONFIG['N_E'])
    E_eV  = E_GeV * 1e9
    weights = E_GeV ** (-gamma)

    P_sum = np.zeros((3,3))
    w_sum = 0.0

    for EeV, w in zip(E_eV, weights):
        H = mass_hamiltonian(EeV, dm21, dm31, U) + liv_hamiltonian(EeV, theta, names)
        # Hermitian ensure numerical symmetry
        H = 0.5 * (H + H.conj().T)
        # Diagonalize
        evals, evecs = np.linalg.eigh(H)
        W = np.abs(evecs)**2  # shape (3,3)
        P = W @ W.T
        P_sum += w * P
        w_sum += w

    return P_sum / w_sum


def source_flavor_vector(theta: np.ndarray, names: List[str]) -> np.ndarray:
    mode = CONFIG['source_mode']
    if mode is None:
        fe = float(theta[names.index('fe_S')])
        src = np.array([fe, 1.0 - fe, 0.0])
    elif mode == 'pion':
        src = np.array([1/3, 2/3, 0.0])
    elif mode == 'muon-damped':
        src = np.array([0.0, 1.0, 0.0])
    elif mode == 'neutron':
        src = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError('Invalid source_mode')
    return src / src.sum()


def predicted_flavor_fractions_at_earth(theta: np.ndarray, names: List[str]) -> np.ndarray:
    Pavg = transition_matrix_avg(theta, names)
    src = source_flavor_vector(theta, names)
    earth = Pavg.T @ src
    earth = np.clip(earth, 1e-12, None)
    return earth / earth.sum()

# Example data vector (e.g. IceCube HESE-like flavor composition) — placeholder
OBS_FLAVOR = np.array([0.31, 0.35, 0.34])
OBS_COV = np.diag([0.12, 0.12, 0.12])**2   # placeholder uncertainties
INV_COV = np.linalg.inv(OBS_COV)


def loglike_single(theta: np.ndarray, names: List[str]) -> float:
    pred = predicted_flavor_fractions_at_earth(theta, names)
    r = pred - OBS_FLAVOR
    chi2 = float(r.T @ INV_COV @ r)
    return -0.5 * chi2


def vectorized_loglike(thetas: np.ndarray, names: List[str]) -> np.ndarray:
    thetas = np.atleast_2d(thetas)
    out = np.empty(thetas.shape[0])
    for i, th in enumerate(thetas):
        out[i] = loglike_single(th, names)
    return np.asarray(out, dtype=float).reshape(-1)

# ------------------------------
# Run
# ------------------------------
if __name__ == '__main__':
    param_names, prior_t = build_parameters_and_prior(CONFIG)

    sampler = ReactiveNestedSampler(param_names,
                                    lambda x: vectorized_loglike(x, param_names),
                                    prior_t)

    result = sampler.run()

    # Corner plot
    try:
        import matplotlib.pyplot as plt
        cornerplot(result, labels=param_names)
        plt.show()
    except Exception as e:
        print("Corner plot failed:", e)

    print("UltraNest run finished. Keys in result:", list(result.keys()))
    # You can also access posterior samples via result['samples'] and derived values.
