#!/usr/bin/env python3

# This script has all the experimental data and outputs a csv of Number of counts vs. energy
# Use for Histogram.py and triangle.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEC_PER_YEAR = 365.25 * 24 * 3600.0

def effective_area_from_sensitivity(E_pts, E2Phi_pts, *, E_units="GeV", T_years, mu90=2.3, DeltaOmega=2*np.pi):

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

    # Convert E^2 Phi -> Phi
    Phi = E2Phi / (E_GeV)**2

    T_sec = T_years * SEC_PER_YEAR

    # A_eff from the rearranged N=μ90 at sensitivity
    Aeff_cm2 = mu90 / (T_sec * DeltaOmega * Phi * dE)
    Aeff_m2 = Aeff_cm2 / 1e4
    return E_GeV, Aeff_m2


def log_interp(x, xp, fp):
    """Log-log linear interpolation.
    Returns 0 outside the positive range.
    """
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


def geom_edges_from_centers(E):
    E = np.asarray(E, dtype=float)
    E = np.sort(E)
    if E.size < 2:
        raise ValueError("Need at least 2 energy points to infer edges")
    edges = np.zeros(len(E) + 1)
    for i in range(1, len(E)):
        edges[i] = math.sqrt(E[i-1] * E[i])
    edges[0]  = E[0]  / math.sqrt(E[1] / E[0])
    edges[-1] = E[-1] * math.sqrt(E[-1] / E[-2])
    return edges


def integrate_binwise(E, y, edges):
    """Integrate y(E) over each bin in `edges` using trapezoidal rule on a dense grid.
    Assumes E is dense and monotonic increasing.
    """
    E = np.asarray(E); y = np.asarray(y)
    out = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        m = (E >= lo) & (E <= hi)
        if not np.any(m):
            out[i] = 0.0
            continue
        Ei = E[m]; yi = y[m]
        out[i] = np.trapz(yi, Ei)
    return out

class FluxModel:

    def __init__(self, phi0_per_flavor=1.68e-18, gamma=2.58, E0=1e5):
        self.phi0_per_flavor = float(phi0_per_flavor)
        self.gamma = float(gamma)
        self.E0 = float(E0)

    def phi(self, E_GeV, flavor=None):
        E = np.asarray(E_GeV, dtype=float)
        return self.phi0_per_flavor * (E / self.E0) ** (-self.gamma)
    
# --------------------------- Radio experiments ------------------------------
# Sensitivity → A_eff for RNOG, Gen2; GRAND directly tau-sensitive via sensitivity.

# RNOG (RNO-G): 90% CL, T=5 yr, μ90=2.3, Ω=2π. Energies given in eV
E_rnog_eV   = np.array([3e16, 1e17, 3e17, 1e18, 4e18, 1e19, 4e19])
E2Phi_rnog  = np.array([2e-8,  9e-9,  6.5e-9, 5.5e-9, 5e-9,  6e-9,  8e-9])
T_rnog      = 5.0; mu90_rnog   = 2.3; Omega_rnog  = 2*np.pi

# Gen2 Radio: (placeholder: using RNOG energy points with different sensitivity)
E_gen2_eV   = E_rnog_eV.copy()
E2Phi_gen2  = np.array([2e-9, 6e-10, 3e-10, 2.2e-10, 2e-10, 4e-10, 8e-10])
T_gen2      = 10.0; mu90_gen2   = 2.3; Omega_gen2  = 2*np.pi

# GRAND 200k: 90% CL, T=3 yr, μ90=2.44, Ω=2π. Energies in GeV. Tau-sensitive only.
E_grand_GeV   = np.array([4.5e7, 1e8, 5e8, 1e9, 3e9, 1e10, 4e10, 7e10, 1e11])
E2Phi_grand   = np.array([4e-9, 2.7e-9, 7e-10, 7.5e-10, 1.2e-9, 3e-9, 9e-9, 0.5e-8, 2e-8])
T_grand       = 3.0; mu90_grand = 2.44; Omega_grand = 2*np.pi

# Convert sensitivities → A_eff(E)
E_rnog_GeV,  Aeff_rnog  = effective_area_from_sensitivity(
    E_rnog_eV, E2Phi_rnog, E_units="eV", T_years=T_rnog, mu90=mu90_rnog, DeltaOmega=Omega_rnog
)
E_gen2_GeV,  Aeff_gen2  = effective_area_from_sensitivity(
    E_gen2_eV, E2Phi_gen2, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
)
E_grand_GeV, Aeff_grand = effective_area_from_sensitivity(
    E_grand_GeV, E2Phi_grand, E_units="GeV", T_years=T_grand, mu90=mu90_grand, DeltaOmega=Omega_grand
)

# Build a common log grid for high-energy radio range
E_low  = min(float(E_rnog_GeV.min()), float(E_gen2_GeV.min()), float(E_grand_GeV.min()))
E_high = max(float(E_rnog_GeV.max()), float(E_gen2_GeV.max()), float(E_grand_GeV.max()))
E_grid_radio = np.logspace(np.log10(E_low), np.log10(E_high), 300)

A_rnog_on = log_interp(E_grid_radio, E_rnog_GeV,  Aeff_rnog)   # on-axis shorthand
A_gen2_on = log_interp(E_grid_radio, E_gen2_GeV,  Aeff_gen2)
A_grand_on= log_interp(E_grid_radio, E_grand_GeV, Aeff_grand)


if os.path.exists("radioscaling.csv"):
    try:
        df_frac = pd.read_csv("radioscaling.csv")
        req = {"Energy (GeV)", "electron fraction", "muon fraction", "taon fraction"}
        if req.issubset(df_frac.columns):
            E_scale = df_frac["Energy (GeV)"].to_numpy(float)
            fe  = df_frac["electron fraction"].to_numpy(float)
            fmu = df_frac["muon fraction"].to_numpy(float)
            ft  = df_frac["taon fraction"].to_numpy(float)
            s = fe + fmu + ft; s[s==0] = 1.0
            fe, fmu, ft = fe/s, fmu/s, ft/s
            fe_g  = log_interp(E_grid_radio, E_scale, fe)
            fmu_g = log_interp(E_grid_radio, E_scale, fmu)
            ft_g  = log_interp(E_grid_radio, E_scale, ft)
        else:
            print("WARNING: radioscaling.csv missing required columns, using 1/3 each.")
    except Exception as e:
        print(f"WARNING: Could not read radioscaling.csv ({e}); using 1/3 each.")
else:
    print("INFO: radioscaling.csv not found; using equal flavor fractions for radio.")

# Split RNOG + Gen2 into flavors using fractions; GRAND is tau-only
Aeff_radio_e  = (A_rnog_on + A_gen2_on) * fe_g
Aeff_radio_mu = (A_rnog_on + A_gen2_on) * fmu_g
Aeff_radio_tau= (A_rnog_on + A_gen2_on) * ft_g + A_grand_on

# ------------------------------ RET-N / Others ------------------------------
# (Optional placeholder) RET-N radar all-flavor sensitivity band → here we skip
# adding it to avoid double counting unless you explicitly enable it later.

# ------------------------ POEMMA & Trinity (tau-only) -----------------------
log10E_poemma = np.array([7, 7.2, 7.6, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])
E_poemma_GeV  = 10**log10E_poemma
E2Phi_poemma  = np.array([3e-6, 5e-8, 1e-8, 8e-9, 6e-9, 7e-9, 2e-8, 2.4e-8, 7e-8])
T_poemma      = 5.0
mu90_poemma   = 2.3
Omega_poemma  = 4*np.pi/12  # example field-of-view
E_po_GeV, A_po = effective_area_from_sensitivity(
    E_poemma_GeV, E2Phi_poemma, E_units="GeV", T_years=T_poemma, mu90=mu90_poemma, DeltaOmega=Omega_poemma
)

E_trinity_GeV = np.array([1e5, 4e5, 1e6, 4e6, 1e7, 7e7, 4e8, 3e9, 1e10, 4e10])
E2Phi_trinity = np.array([2e-7, 3e-8, 1e-8, 2e-9, 1e-9, 5e-10, 7e-10, 3e-9, 8e-9, 4e-8])
T_trinity     = 3.0
mu90_trinity  = 2.3
Omega_trinity = 2*np.pi
E_tr_GeV, A_tr = effective_area_from_sensitivity(
    E_trinity_GeV, E2Phi_trinity, E_units="GeV", T_years=T_trinity, mu90=mu90_trinity, DeltaOmega=Omega_trinity
)

# ------------------------------ TAMBO (AΩ) ---------------------------------
# T = 10 yr, aperture AΩ given in m^2·sr for all and tau channels.
T_tambo_years = 10.0
omega_tambo   = 0.1      # sr

# Base (sparse) energy sampling and aperture curves (m^2·sr)
tambo_E_GeV_base = np.array([1e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 50, 100, 500, 600, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 50, 100, 500, 700, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

# Dense grid interpolation
E_tambo = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 200)
AOm_tambo_tau = log_interp(E_tambo, tambo_E_GeV_base, tambo_tau_ap_m2sr_base)   # m^2·sr
AOm_tambo_all = log_interp(E_tambo, tambo_E_GeV_base, tambo_all_ap_m2sr_base)
AOm_tambo_e   = np.abs(AOm_tambo_all - AOm_tambo_tau)               # m^2·sr

# -------------------------- IceCube Cascades (AΩ) ---------------------------
# T = 6 yr, 2π sr. Aeff_m2_Omega2pi means AΩ already (m^2·sr for 2π acceptance)
IC_cascade_T_years = 6.0

# (Only rows with valid energies are kept; trailing incomplete lines from the
# user paste are intentionally ignored.)
IC_casc_E_center = np.array([
    5.011900693349779e2, 7.943299062732058e2, 1.2589281155014372e3,
    1.9952719889779437e3, 3.1622974385721527e3, 5.011900693349779e3,
    7.943299062732058e3, 1.2589281155014372e4, 1.9952719889779437e4,
    3.162297438572153e4, 5.0119006933497796e4, 7.943299062732059e4,
    1.9952719889779435e5, 3.162297438572153e5, 5.0119006933497795e5,
    7.943299062732059e5,
])
IC_casc_AOm_mu = np.array([
    0.003148481157771042, 0.013972008590113834, 0.03954236811412915,
    0.08397919480714747, 0.14303075652198077, 0.22669064335951497,
    0.3243501994133569, 0.39542368114129145, 0.5013683272068507,
    0.5562307198077029, 0.5667266083987874, 0.49900030678977975,
    0.7520524908102761, 0.794615314011004, 0.8186051010204706,
    0.8483005215426258,
])
IC_casc_AOm_e = np.array([
    0.013521737919122976, 0.06122998933555752, 0.19408234784519426,
    0.42295383641785644, 0.7312764121695474, 1.1590061073533977,
    1.775669690731168, 2.3775087611036296, 2.7684251110986966,
    2.681346844621674, 3.0906829529423936, 3.0614994667778754,
    3.3964410872908988, 3.0760279012207756, 3.6563820608477364,
    3.4770183220601925,
])
IC_casc_AOm_tau = np.array([
    0.008517630185274317, 0.03857007202239844, 0.12225659706783892,
    0.26642761349156313, 0.460646558846959, 0.730082587309227,
    1.1185320886495547, 1.4976433140810266, 1.7438898337629587,
    1.6890373824388498, 1.9468868994912716, 1.9285036011199217,
    2.1394904486871806, 1.9376553708477326, 2.3032327942347948,
    2.1902477619276803,
])

IC_casc_edges = geom_edges_from_centers(IC_casc_E_center)


# --------------------------- IceCube Tracks (N) -----------------------------
# T = 6 yr. Pairs are energy-range [GeV] and N events in that bin.
# This is taken from Number of events

'''
df = pd.read_csv("eff_area_tracks.csv") 
T_tracks = 9.5
tracks = 2*np.pi
tracks_AOm = df.iloc[:, 2]
E_tracks = df.iloc[:, 5]    
'''

# This is taken from the sensitivity plot

E_ic_GeV = np.array([1.5e4, 4e4, 6e4, 8e4, 1e5, 2e5, 4e5, 6e5, 1e6, 2e6, 4e6], dtype=float)
E2Phi_ic = np.array([2.9e-8, 2e-8, 1.8e-8, 1.6e-8, 1.5e-8, 1.2e-8, 8.7e-9, 6.3e-9, 6.15e-9, 4.8e-9, 3.7e-9], dtype=float)

T_ic = 9.5
mu90_ic = 2.3
Omega_ic = 2 * np.pi

E_ic_GeV, A_ic = effective_area_from_sensitivity(
    E_ic_GeV, E2Phi_ic, E_units="GeV", T_years=T_ic, mu90=mu90_ic, DeltaOmega=Omega_ic
)


# -------------------------- Build a master E-grid ---------------------------
# Cover the full range spanned by radio + tambo + cascades
E_min = min(E_grid_radio.min(), E_tambo.min(), IC_casc_edges[0], E_po_GeV.min(), E_tr_GeV.min())
E_max = max(E_grid_radio.max(), E_tambo.max(), IC_casc_edges[-1], E_po_GeV.max(), E_tr_GeV.max())
E_master = np.logspace(np.log10(E_min), np.log10(E_max), 2000)

# -------------------------- Exposures per flavor ----------------------------
# Exposure X(E) = AΩ(E) * T [cm^2·sr·s], where AΩ(E) = A_eff(E)*ΔΩ when needed

def add_exposure_from_Aeff(E_src, Aeff_m2_src, *, DeltaOmega, T_years, accum):
    A_eff = log_interp(E_master, E_src, Aeff_m2_src)    # m^2
    AOm_m2sr = A_eff * float(DeltaOmega)                # m^2·sr
    X_cm2srs = AOm_m2sr * 1e4 * (T_years * SEC_PER_YEAR)
    accum += X_cm2srs
    return accum

def add_exposure_from_AOm(E_src, AOm_m2sr_src, *, T_years, accum):
    AOm = log_interp(E_master, E_src, AOm_m2sr_src)     # m^2·sr
    X_cm2srs = AOm * 1e4 * (T_years * SEC_PER_YEAR)
    accum += X_cm2srs
    return accum

# Initialize flavor exposures
X_e = np.zeros_like(E_master)
X_mu = np.zeros_like(E_master)
X_tau = np.zeros_like(E_master)

# Radio (RNOG + Gen2): split by flavor fractions; GRAND adds to tau
X_e  = add_exposure_from_Aeff(E_grid_radio, Aeff_radio_e,  DeltaOmega=Omega_rnog, T_years=T_rnog, accum=X_e)
X_e  = add_exposure_from_Aeff(E_grid_radio, Aeff_radio_e,  DeltaOmega=Omega_gen2, T_years=T_gen2, accum=X_e)
X_mu = add_exposure_from_Aeff(E_grid_radio, Aeff_radio_mu, DeltaOmega=Omega_rnog, T_years=T_rnog, accum=X_mu)
X_mu = add_exposure_from_Aeff(E_grid_radio, Aeff_radio_mu, DeltaOmega=Omega_gen2, T_years=T_gen2, accum=X_mu)
X_tau= add_exposure_from_Aeff(E_grid_radio, Aeff_radio_tau,DeltaOmega=Omega_rnog, T_years=T_rnog, accum=X_tau)
X_tau= add_exposure_from_Aeff(E_grid_radio, Aeff_radio_tau,DeltaOmega=Omega_gen2, T_years=T_gen2, accum=X_tau)
# GRAND tau-only
X_tau= add_exposure_from_Aeff(E_grid_radio, A_grand_on,    DeltaOmega=Omega_grand,T_years=T_grand, accum=X_tau)


# POEMMA + Trinity (tau-only)
X_tau= add_exposure_from_Aeff(E_po_GeV, A_po, DeltaOmega=Omega_poemma, T_years=T_poemma, accum=X_tau)
X_tau= add_exposure_from_Aeff(E_tr_GeV, A_tr, DeltaOmega=Omega_trinity, T_years=T_trinity, accum=X_tau)

# TAMBO (AΩ): tau and electron
X_tau= add_exposure_from_AOm(E_tambo, AOm_tambo_all, T_years=T_tambo_years, accum=X_tau)
X_e  = add_exposure_from_AOm(E_tambo, AOm_tambo_e,   T_years=T_tambo_years, accum=X_e)

# IceCube Cascades (AΩ per flavor)
X_e  = add_exposure_from_AOm(IC_casc_E_center, IC_casc_AOm_e,   T_years=IC_cascade_T_years, accum=X_e)
X_mu = add_exposure_from_AOm(IC_casc_E_center, IC_casc_AOm_mu,  T_years=IC_cascade_T_years, accum=X_mu)
X_tau= add_exposure_from_AOm(IC_casc_E_center, IC_casc_AOm_tau, T_years=IC_cascade_T_years, accum=X_tau)

#X_mu = add_exposure_from_AOm(E_tracks, tracks_AOm, T_years = T_tracks, accum = X_mu)
X_mu = add_exposure_from_Aeff(E_ic_GeV, A_ic, T_years = T_ic, DeltaOmega=Omega_ic, accum = X_mu)

# ------------------------------ Event counts --------------------------------
flux = FluxModel(phi0_per_flavor=1.68e-18, gamma=2.58, E0=1e5)  

phi_e  = flux.phi(E_master, 'e') 
phi_mu = flux.phi(E_master, 'mu') 
phi_tau= flux.phi(E_master, 'tau')

# Differential expected counts per GeV
n_e_diff  = phi_e  * X_e
n_mu_diff = phi_mu * X_mu
n_tau_diff= phi_tau* X_tau

# Build decade bin edges spanning the master range
log10_min = math.floor(np.log10(E_master[E_master>0].min()))
log10_max = math.ceil(np.log10(E_master.max()))
decade_edges = 10 ** np.arange(log10_min, log10_max + 1)

# Integrate per decade (trapz on master grid restricted to each bin)
N_e_dec  = integrate_binwise(E_master, n_e_diff,  decade_edges)
N_mu_dec = integrate_binwise(E_master, n_mu_diff, decade_edges)
N_tau_dec= integrate_binwise(E_master, n_tau_diff,decade_edges)


def gaussian_errors(*Ns):
    """Return Gaussian σ = sqrt(N) for each array N, after all contributions are summed."""
    return [np.sqrt(np.clip(N, 0.0, None)) for N in Ns]
    
          
err_e, err_mu, err_tau = gaussian_errors(N_e_dec, N_mu_dec, N_tau_dec)
print(N_e_dec, err_e)
print(N_mu_dec, err_mu)
print(N_tau_dec, err_tau)

# ------------------------------ Save results --------------------------------
out = pd.DataFrame({
    'E_low_GeV': decade_edges[:-1],
    'E_high_GeV': decade_edges[1:],
    'N_e': N_e_dec,
    'N_mu': N_mu_dec,
    'N_tau': N_tau_dec,
    'err_e': err_e,
    'err_mu': err_mu,
    'err_tau': err_tau,
})
out.to_csv('nevents_per_decade.csv', index=False)
print("Wrote nevents_per_decade.csv")