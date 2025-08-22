# Final EffAreas Plot divided explicitly by sensitivity

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEC_PER_YEAR = 365.25 * 24 * 3600.0

class FluxModel:

    def __init__(self, phi0_per_flavor=1.68e-18, gamma=2.58, E0=1e5):
        self.phi0_per_flavor = float(phi0_per_flavor)
        self.gamma = float(gamma)
        self.E0 = float(E0)

    def phi(self, E_GeV, flavor=None):
        E = np.asarray(E_GeV, dtype=float)
        return self.phi0_per_flavor * (E / self.E0) ** (-self.gamma)

def effective_area_from_sensitivity(E_pts, E2Phi_pts, *, E_units, T_years, mu90=2.3, DeltaOmega):
    E = np.asarray(E_pts, dtype=float)
    E2Phi = np.asarray(E2Phi_pts, dtype=float)

    if str(E_units).lower() == "ev":
        E_GeV = E / 1e9
    else:
        E_GeV = E.copy()

    order = np.argsort(E_GeV)
    E_GeV = E_GeV[order]
    E2Phi = E2Phi[order]

    # Geometric bin edges from centers
    edges = np.zeros(len(E_GeV) + 1)
    for i in range(1, len(E_GeV)):
        edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
    edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
    edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

    dlog10E = np.log10(edges[1:]) - np.log10(edges[:-1])
    dE = E_GeV * np.log(10.0) * dlog10E

    Phi = E2Phi / (E_GeV)**2

    T_sec = T_years * SEC_PER_YEAR

    # A_eff from the rearranged N=μ90 at sensitivity
    Aeff_cm2 = mu90 / (T_sec * DeltaOmega * Phi * dE)
    Aeff_m2 = Aeff_cm2 / 1e4
    return E_GeV, Aeff_m2

def log_interp(x, xp, fp):
    """Log-log linear interpolation. Returns 0 outside the positive range."""
    xp = np.asarray(xp); fp = np.asarray(fp)
    mask = (xp > 0) & (fp > 0)
    xp = xp[mask]; fp = fp[mask]
    x = np.asarray(x, dtype=float)
    if len(xp) < 2:
        return np.zeros_like(x)
    xlog  = np.log10(x)
    xplog = np.log10(xp)
    fplog = np.log10(fp)
    ylog = np.interp(xlog, xplog, fplog, left=np.nan, right=np.nan)
    y = 10**ylog
    y[np.isnan(y)] = 0.0
    return y

### IC Tau + Electron (Cascade)

E_icet = np.array([1.5e13, 2.0e14, 2.5e15])
E_icet2 = np.array([1.5e13, 2.0e14, 2.5e15])
E2Phi_et_max = np.array([1.7e-7, 4.0e-8, 1.5e-8])
E2Phi_et_min = np.array([9.5e-8, 2e-8, 6e-9])

# Analysis constants
T_YEARS = 6
MU90 = 2.3
OMEGA_SR = 4 * np.pi

# Convert sensitivities to effective areas at the provided grid
E_icet, A_et_max = effective_area_from_sensitivity(
    E_icet, E2Phi_et_max, E_units="eV",
    T_years=T_YEARS, mu90=MU90, DeltaOmega=OMEGA_SR
)
E_icet2, A_et_min = effective_area_from_sensitivity(
    E_icet2, E2Phi_et_min, E_units="eV",
    T_years=T_YEARS, mu90=MU90, DeltaOmega=OMEGA_SR
)

# Common log-spaced interpolation grid across the data range
E_icet_interp = np.logspace(np.log10(E_icet[0]), np.log10(E_icet[-1]), 100)
E_icet_interp2 = np.logspace(np.log10(E_icet2[0]), np.log10(E_icet2[-1]), 100)

# Interpolate (log-log) max/min bands onto the common grid
A_et_max_interp = log_interp(E_icet_interp, E_icet, A_et_max)
A_et_min_interp = log_interp(E_icet_interp2, E_icet2, A_et_min)

frac_e = 127 / (80 + 127)

A_ic_cascade_emax = frac_e * A_et_max_interp
A_ic_cascade_emin = frac_e * A_et_min_interp
A_ic_cascade_taumax = (1-frac_e) * A_et_max_interp
A_ic_cascade_taumin = (1-frac_e) * A_et_min_interp

## IC Muon Tracks

E_icmuon = np.array([1.5e13, 2.0e14, 5.0e15])
E_icmuon2 = np.array([1.5e13, 2.0e14, 5.0e15])
E2Phi_muon_max = np.array([1.3e-7, 4.0e-8, 1.6e-8])
E2Phi_muon_min = np.array([5e-8, 2.5e-8, 7e-9])

# Analysis constants
T_YEARS = 10
MU90 = 2.3
OMEGA_SR = 4 * np.pi

# Convert sensitivities to effective areas at the provided grid
E_icmuon, A_muon_max = effective_area_from_sensitivity(
    E_icmuon, E2Phi_muon_max, E_units="eV",
    T_years=T_YEARS, mu90=MU90, DeltaOmega=OMEGA_SR
)
E_icmuon2, A_muon_min = effective_area_from_sensitivity(
    E_icmuon2, E2Phi_muon_min, E_units="eV",
    T_years=T_YEARS, mu90=MU90, DeltaOmega=OMEGA_SR
)

# Common log-spaced interpolation grid across the data range
E_icmuon_interp = np.logspace(np.log10(E_icmuon[0]), np.log10(E_icmuon[-1]), 100)
E_icmuon_interp2 = np.logspace(np.log10(E_icmuon2[0]), np.log10(E_icmuon2[-1]), 100)

# Interpolate (log-log) max/min bands onto the common grid
A_muon_max_interp = log_interp(E_icmuon_interp, E_icmuon, A_muon_max)
A_muon_min_interp = log_interp(E_icmuon_interp2, E_icmuon2, A_muon_min)


# --------------------------- Plot: A_eff -------------------------------
fig, ax_left = plt.subplots(figsize=(9,6), dpi=140)  # Left axis: A_eff (m^2)

def errorline(ax, x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.errorbar(x, y, fmt='o-', capsize=3, markersize=1, linewidth=1.2, label=label)

# Add flavor components

master = np.logspace(4.1,6.1, 200)

ice_cube_muon = (A_muon_min_interp+A_muon_max_interp)/2
ice_cube_tau = (A_ic_cascade_emin+A_ic_cascade_emax)/2
ice_cube_e = (A_ic_cascade_taumax+A_ic_cascade_taumin)/2

A_muon_2 = log_interp(master, E_icmuon_interp, ice_cube_muon)

A_tau = log_interp(master, E_icet_interp, ice_cube_tau)

A_e_2 = log_interp(master, E_icet_interp, ice_cube_e)

errorline(ax_left,  master,   A_muon_2,      "Muon")
errorline(ax_left,  master,   A_tau,      "Taon")
errorline(ax_left,  master,   A_e_2,      "Electron")

# Scales, labels, grids
ax_left.set_xscale("log"); ax_left.set_yscale("log")
ax_left.set_xlabel("Energy E (GeV)")
ax_left.set_ylabel("All-Experiment Effective Area $A_{\\rm eff}$ (m$^2$)")
ax_left.set_title("All-Experiment Effective Area Per Flavor ")

ax_left.grid(True, which="both", alpha=0.3)

# Legends: one per axis, then combine
h1, l1 = ax_left.get_legend_handles_labels()
ax_left.legend(h1, l1, loc="lower right", fontsize=9, ncol=2)

fig.tight_layout()
out_path = "combined_plot.png"
plt.savefig(out_path)
print(f"Saved: {out_path}")
plt.show()

def save_csv(path, master, A_muon, A_tau, A_e):
    cols = {
        "master": np.asarray(master).ravel(),
        "A_muon": np.asarray(A_muon).ravel(),
        "A_tau":  np.asarray(A_tau).ravel(),
        "A_e":    np.asarray(A_e).ravel(),
    }

    # If lengths differ, truncate to the shortest so the CSV is rectangular.
    lengths = {k: len(v) for k, v in cols.items()}
    n = min(lengths.values())
    if len(set(lengths.values())) != 1:
        print(f"Warning: different lengths {lengths}; truncating all to {n}.")
        cols = {k: v[:n] for k, v in cols.items()}

    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"Saved {path}")

# Usage:
save_csv("effareasicecube.csv", master, A_muon_2, A_tau, A_e_2)