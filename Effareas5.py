#!/usr/bin/env python3
# Interpolate effective areas on a common grid, plot, and save CSV/PNG.
# used for all Energy data MESE 11.3 years + Tambo

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ Config (edit as needed) ------------------------

T_years = 11.3
DeltaOmega = 4*np.pi 

OUTPUT_CSV = "MC_outputs/effareasMESE.csv"
PLOT_PNG   = "combined_plot.png"

SEC_PER_YEAR = 365.25 * 24 * 3600.0

# ------------------------ Data ------------------------
'''HESE DATA
# E1: tau double-bang, E2: cascades, E3: tracks
E1 = np.array([4e4,6e4,1e5,2.1e5,3.5e5,5.5e5], dtype=float)
E2 = np.array([1.3e3,2.0e3,3.0e3,4.2e3,6.5e3,1.0e4,1.6e4,2.3e4,3.5e4,5.5e4,
               8.0e4,1.2e5,1.9e5,3.0e5,4.2e5,6.7e5,1.0e6,2.0e6], dtype=float)
E3 = np.array([1.5e3,2.0e3,6.0e3,1.3e4,2.5e4,5.0e4,1.0e5,2.0e5,4.0e5,3.5e6], dtype=float)

Ncasc = np.array([1.0e3,1.2e3,1.0e3,7.0e2,4.5e2,3.0e2,2.0e2,1.4e2,8.0e1,4.0e1,
                  3.0e1,1.5e1,6.0,3.0,2.0,2.0,1.0,1.0], dtype=float)
Ndb   = np.array([3,1,2,1,1,1], dtype=float)
Ntrack= np.array([2.6e3,1.5e3,6.0e2,2.5e2,1.0e2,4.0e1,1.5e1,2.0,1.0,1.0], dtype=float)

'''

E1   = [2.5e4, 5e4, 1e5, 2e5, 4e5, 8e5, 1.8e6, 3.5e6, 7e6]
E1_Mu       = [5.5e-2, 2e-1, 1.5e-1, 8e-2, 4.5e-2, 1.5e-2, 1e-2, 3e-3, 1e-3]
E1_e        = [7e-2, 4e-1, 2.2e-1, 1.8e-1, 7e-2, 3e-2, 1.5e-2, 1.8e-2, 5e-2]
E1_Tau      = [2e-1, 7e-1, 1e0, 1.5e0, 1e0, 6e-1, 2.8e-1, 1e-1, 4e-2]

E2   = [1.3e3, 2e3, 3e3, 4.5e3, 6.5e3, 1e4, 1.6e4, 2.3e4, 3.5e4, 5.2e4,
               8e4, 1.3e5, 2e5, 2.9e5, 4.5e5, 7e5, 1e6, 1.6e6, 2.3e6, 2.5e6, 5e6, 8e6]
E2_Mu       = [2.5e1, 7, 8.5, 9, 9.5, 8, 6.5, 5, 3.5, 2, 1.5, 1, 6e-1, 5e-1, 3e-1, 1.6e-1, 1.2e-1, 7e-2, 3.5e-2, 3e-2, 1.8e-2, 1e-2]
E2_e        = [2e1, 4e1, 5e1, 6e1, 6e1, 5.5e1, 5e1, 3.5e1, 3e1, 2e1, 1.5e1, 10, 7, 4.5, 3, 2, 1.5, 1, 5.5e-1, 4.5e-1, 2, 4e-1]
E2_Tau      = [2e1, 3e1, 4e1, 4e1, 4e1, 4e1, 3e1, 2.6e1, 2e0, 1.3e1, 9, 6, 3.5, 2, 1.2, 7e-1, 4e-1, 2e-1, 1.4e-1, 8e-2, 4e-2, 2.2e-2]

E3   = [1.5e3, 2e3, 6e3, 1.3e4, 2.5e4, 5e4, 1e5, 2e5, 4e5, 9e5, 1.7e6, 3e6, 7e6]
E3_Mu       = [2.5e1, 2.7e1, 20, 18, 15, 10, 7, 4, 1, 0.8, 4e-1, 1.8e-1, 8e-2]
E3_e        = [2, 1.2, 3, 3, 1, 1, 1.2, 0.9, 0.6, 2.5, 9e-1, 1e-1, 2.5e-2]
E3_Tau      = [4, 5, 4.5, 4.5, 3.5, 3, 2, 1.1, 0.6, 2.5, 1.8e-1, 8e-3, 4e-2]


# ------------------------ Helpers ------------------------
def flux(E, phi0_per_flavor=2.72e-18, gamma=2.54, E0=1e5):
    """Per-flavor flux model: phi0 * (E/E0)^(-gamma), with E in GeV."""
    return phi0_per_flavor * ((E / E0) ** (-gamma))

def effective_area_from_Nev(E_GeV, N_ev, T_years, DeltaOmega):
    """
    Compute binned A_eff (m^2) from event counts:
    N = flux(E) * A_eff * T * dE * ΔΩ  -> A_eff = N / (flux*T*dE*ΔΩ)
    """
    E_GeV = np.asarray(E_GeV, dtype=float)
    N_ev  = np.asarray(N_ev, dtype=float)

    order = np.argsort(E_GeV)
    E_GeV = E_GeV[order]
    N_ev  = N_ev[order]
    print(E_GeV[0])

    # Geometric bin edges from centers
    edges = np.zeros(len(E_GeV) + 1)
    for i in range(1, len(E_GeV)):
        edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
    edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
    edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

    T_sec = T_years * SEC_PER_YEAR
    dE = np.diff(edges)

    Aeff_cm2 = N_ev / (T_sec * DeltaOmega * flux(E_GeV) * dE)
    Aeff_m2 = Aeff_cm2/10e4
    return E_GeV, Aeff_cm2

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

    Phi = E2Phi / (E_GeV)**2
    dE = np.diff(edges) 
    T_sec = T_years * SEC_PER_YEAR

    # A_eff from the rearranged N=μ90 at sensitivity
    Aeff_cm2 = mu90 / (T_sec * DeltaOmega * Phi *  dE )
    return E_GeV, Aeff_cm2


# ------------------------ Compute component A_eff ------------------------
E1mu, A1mu = effective_area_from_Nev(E1, E1_Mu, T_years, DeltaOmega)
E1e, A1e = effective_area_from_Nev(E1, E1_e, T_years, DeltaOmega)
E1tau, A1tau = effective_area_from_Nev(E1, E1_Tau, T_years, DeltaOmega)

E2mu, A2mu = effective_area_from_Nev(E2, E2_Mu, T_years, DeltaOmega)
E2e, A2e = effective_area_from_Nev(E2, E2_e, T_years, DeltaOmega)
E2tau, A2tau = effective_area_from_Nev(E2, E2_Tau, T_years, DeltaOmega)

E3mu, A3mu = effective_area_from_Nev(E3, E3_Mu, T_years, DeltaOmega)
E3e, A3e = effective_area_from_Nev(E3, E3_e, T_years, DeltaOmega)
E3tau, A3tau = effective_area_from_Nev(E3, E3_Tau, T_years, DeltaOmega)


# ------------------------------ TAMBO (A) ---------------------------------
tambo_angle = 4*np.pi
tambo_E_GeV_base = np.array([3e5, 4e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 1000, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 600, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

E_tambo = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 200)
AOm_tambo_tau = log_interp(E_tambo, tambo_E_GeV_base, tambo_tau_ap_m2sr_base)     
AOm_tambo_all = log_interp(E_tambo, tambo_E_GeV_base, tambo_all_ap_m2sr_base)       
AOm_tambo_e   = np.abs(AOm_tambo_all - AOm_tambo_tau)            

A_tambo_e = AOm_tambo_e *10e4 / (tambo_angle )
A_tambo_tau = AOm_tambo_tau*10e4  / (tambo_angle)

A_tambo_mu = A_tambo_e *0

# ------------------------ Common energy grid ------------------------
print(E1tau)
emin = min(min(E2e), min(E1tau), min(E3mu))
emax =  max(max(E1e), max(E2tau), max(E3mu))
print (emax, emin)
#emin = min(E_tambo)
#emax =  max(E_tambo)
master = np.logspace(np.log10(emin), np.log10(emax), 300)

#A_tau_master =  log_interp(master, E_tambo,  A_tambo_tau)*0.1
#A_mu_master  = log_interp(master, E_tambo,  A_tambo_mu)*0.1
#A_e_master   =  log_interp(master, E_tambo,  A_tambo_e)*0.1

A_mu_master = (log_interp(master, E1mu, A1mu) + log_interp(master, E2mu,  A2mu) + log_interp(master, E3mu,  A3mu))
A_tau_master = (log_interp(master, E1tau, A1tau) + log_interp(master, E2tau,  A2tau) + log_interp(master, E3tau,  A3tau)) 
A_e_master = (log_interp(master, E1e, A1e) + log_interp(master, E2e,  A2e) + log_interp(master, E3e,  A3e))

# ------------------------ Plot ------------------------
plt.figure(figsize=(9,6), dpi=140)
plt.loglog(master, A_mu_master,  label="Muon")
plt.loglog(master, A_tau_master, label="Tau")
plt.loglog(master, A_e_master,   label="Electron")
plt.xlabel("Energy E (GeV)")
plt.ylabel(r"Effective Area $A_{\rm eff}$ (cm$^2$)")
plt.title("Effective Area per-flavor MESE (21.3yr) ")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="lower right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(PLOT_PNG)
print(f"Saved plot: {PLOT_PNG}")

# ------------------------ Save CSV ------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame({
    "E_GeV": master,
    "A_mu_m2": A_mu_master,
    "A_tau_m2": A_tau_master,
    "A_e_m2": A_e_master,
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved CSV:  {OUTPUT_CSV}")