#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flavor ratio evolution with an LIV tau-tau term (no-args script)
----------------------------------------------------------------

This script computes normalized flavor ratios at Earth as functions of energy
for astrophysical neutrinos, assuming very long baselines (decohered limit).
We include Lorentz Invariance Violation (LIV) via an additive effective
Hamiltonian term:

    H_tot(E) = H_vac(E) + H_LIV(E)

with
    H_vac(E) = U · diag(m_i^2 / (2E)) · U^†
and an isotropic, diagonal LIV contribution in the flavor basis

    H_LIV(E) = E^(d-3) · a_eff^(d),     with (a_eff^(d))_{ττ} = a_tau_tau

By default we take a_tau_tau = 1e-34 (in units of GeV^(3-d)) and d = 5.
You can change the values in the USER-SET PARAMETERS block.

Because we show flavor *ratios*, the overall flux normalization cancels,
so the figure directly reflects the energy-dependent flavor composition.

Run:
    python flavor_ratios_LIV_noargs.py

It will write a figure 'flavor_ratios_LIV.png' in the working directory.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------ PMNS and constants ------------------------
def pmns_matrix():
    # Angles in degrees from NuFIT 5.x (illustrative)
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
    """
    Build H_vac in the flavor basis for a given energy E (GeV).
    Mass-squared in eV^2 converted to GeV^2. Normal ordering by default.
    """
    U = pmns_matrix()
    # Mass splittings (illustrative): Δm21^2, Δm31^2 in eV^2
    dm21 = 7.42e-5
    dm31_NO = 2.517e-3
    dm31_IO = -2.498e-3

    if ordering.upper() == "NO":
        m1, m2, m3 = 0.0, dm21, dm31_NO
    else:
        # For IO: m3 is lightest (set 0), then m1 ~ |dm31_IO|, m2 = m1 + dm21
        m3, m1 = 0.0, abs(dm31_IO)
        m2 = m1 + dm21

    # Convert eV^2 -> GeV^2
    ev2_to_GeV2 = 1e-18
    m2_GeV2 = np.array([m1, m2, m3]) * ev2_to_GeV2
    Hm = np.diag(m2_GeV2 / (2.0 * E_GeV))  # GeV
    Hvac = U @ Hm @ U.conj().T
    return Hvac

def h_liv(E_GeV, d=5, a_tau_tau=1e-34):
    """
    Isotropic LIV term in flavor basis with only the ττ element nonzero.
    Units: a_tau_tau has GeV^(3-d) so that H_LIV has GeV.
    """
    scale = (E_GeV ** (d - 3))
    a = np.zeros((3,3), dtype=float)
    a[2,2] = a_tau_tau
    return scale * a

def decohered_P_from_H(H):
    """
    Given a 3x3 Hermitian Hamiltonian H (flavor basis), compute the
    decohered transition matrix P with elements
        P_{αβ} = Σ_i |V_{αi}|^2 |V_{βi}|^2
    where H = V diag(λ_i) V^†.
    """
    evals, evecs = np.linalg.eigh(H)
    A = np.abs(evecs)**2
    return A @ A.T  # rows=detected α, cols=source β

# ====================== USER-SET PARAMETERS ======================
# Source flavor composition (sums to 1)
f_source = np.array([1/3, 2/3, 0.0])   # (f_e,S, f_mu,S, f_tau,S)

# LIV settings
a_tau_tau = 1e-34    # GeV^(3-d)
d_operator = 4      # operator dimension d (so H_LIV ~ E^(d-3))

# Energy grid (GeV)
Emin, Emax = 1e3, 1e8
nE = 300

# Mass ordering
ordering = "NO"

# Output figure path
outfile = "LIV/flavor_ratios_LIV.png"
# ================================================================

def main():
    Es = np.logspace(np.log10(Emin), np.log10(Emax), nE)
    f_e, f_mu, f_tau = [], [], []

    for E in Es:
        H = h_vacuum(E, ordering=ordering).astype(complex) + h_liv(E, d=d_operator, a_tau_tau=a_tau_tau)
        # Ensure Hermiticity (numerical guard)
        H = 0.5 * (H + H.conj().T)
        P = decohered_P_from_H(H)  # 3x3

        # Detected flavor fractions at Earth: f_det = P @ f_source
        f_det = P @ f_source
        # Numerical normalization to 1
        f_det = f_det / np.sum(f_det)

        f_e.append(f_det[0])
        f_mu.append(f_det[1])
        f_tau.append(f_det[2])

    f_e = np.array(f_e)
    f_mu = np.array(f_mu)
    f_tau = np.array(f_tau)

    # Plot (single chart; do not set explicit colors/styles beyond basics)
    plt.figure(figsize=(7,5))
    plt.semilogx(Es, f_e, label=r"$f_e$")
    plt.semilogx(Es, f_mu, label=r"$f_\mu$")
    plt.semilogx(Es, f_tau, label=r"$f_\tau$")
    plt.ylim(0, 1)
    plt.xlim(1e3, 1e7)
    plt.xlabel("Energy E (GeV)")
    plt.ylabel("Flavor fraction at Earth")
    plt.title(r"Flavor ratio evolution with single LIV parameter $".format(a_tau_tau, d_operator))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved figure to {outfile}")

if __name__ == "__main__":
    main()