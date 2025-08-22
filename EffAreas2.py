# Final EffAreas Plot

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


# ------------------------------ TAMBO (A) ---------------------------------
tambo_angle = 4*np.pi
tambo_E_GeV_base = np.array([3e5, 4e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 1000, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 600, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

E_tambo = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 200)
AOm_tambo_tau = log_interp(E_tambo, tambo_E_GeV_base, tambo_tau_ap_m2sr_base)     
AOm_tambo_all = log_interp(E_tambo, tambo_E_GeV_base, tambo_all_ap_m2sr_base)       
AOm_tambo_e   = np.abs(AOm_tambo_all - AOm_tambo_tau)            

A_tambo_e = AOm_tambo_e / tambo_angle
A_tambo_tau = AOm_tambo_tau / tambo_angle

# -------------------------- All Data (Sensitivities eV Snowmass White Paper) ---------------------------
### GRAND 200k
E_grand   = np.array([6e16,1e17,2e17,4.6e17,4e18,4e19,1e20])
E2Phi_grand = np.array([1.2e-9,8e-10,3e-10,2e-10,4.5e-10,3e-9,7e-9])
T_grand, mu90_grand, Omega_grand = 10, 2.3, 4*np.pi

E_grand, A_grand = effective_area_from_sensitivity(
    E_grand, E2Phi_grand, E_units="eV", T_years=T_grand, mu90=mu90_grand, DeltaOmega=Omega_grand
)

E_grand_interp = np.logspace(np.log10(E_grand[0]), np.log10(E_grand[-1]), 100)
A_grand_interp = log_interp(E_grand_interp, E_grand, A_grand)

### TAMBO
E_tambo2   = np.array([2e14, 1e15,4e15,1e16,4e16,1e17])
E2Phi_tambo = np.array([1e-7,7.5e-9,3.7e-9,2.8e-9,2.3e-9,3e-9])
T_tambo, mu90_tambo, Omega_tambo = 10, 2.3, 4*np.pi

E_tambo2, A_tambo = effective_area_from_sensitivity(
    E_tambo2, E2Phi_tambo, E_units="eV", T_years=T_tambo, mu90=mu90_tambo, DeltaOmega=Omega_tambo
)

E_tambo_interp = np.logspace(np.log10(E_tambo2[0]), np.log10(E_tambo2[-1]), 100)
A_tambo_interp = log_interp(E_tambo_interp, E_tambo2, A_tambo)


### Trinity
E_trinity   = np.array([1e15,2e15,1e16,4e16,1e17,1e18,1e19])
E2Phi_trinity = np.array([1.2e-8,2.8e-9,1e-9,6e-10,5e-10,1.3e-9,8.5e-9])
T_trinity, mu90_trinity, Omega_trinity = 10, 2.3, 4*np.pi

E_trinity, A_trinity = effective_area_from_sensitivity(
    E_trinity, E2Phi_trinity, E_units="eV", T_years=T_trinity, mu90=mu90_trinity, DeltaOmega=Omega_trinity
)

E_trinity_interp = np.logspace(np.log10(E_trinity[0]), np.log10(E_trinity[-1]), 100)
A_trinity_interp = log_interp(E_trinity_interp, E_trinity, A_trinity)


### POEMMA
E_poemma   = np.array([1.5e16,2e16, 3e16,1e17,5e17,3e18,3e19, 1e20])
E2Phi_poemma = np.array([1e-5,1.6e-6, 3e-7,1.1e-7,6e-8,8e-8,3e-7, 1e-6])
T_poemma, mu90_poemma, Omega_poemma = 5, 2.3, 4*np.pi

E_poemma, A_poemma = effective_area_from_sensitivity(
    E_poemma, E2Phi_poemma, E_units="eV", T_years=T_poemma, mu90=mu90_poemma, DeltaOmega=Omega_poemma
)

E_poemma_interp = np.logspace(np.log10(E_poemma[0]), np.log10(E_poemma[-1]), 100)
A_poemma_interp = log_interp(E_poemma_interp, E_poemma, A_poemma)

### IceCube Gen2 Radio
E_gen2   = np.array([3e16,1e17,1e18,1e19,2.5e19])
E2Phi_gen2 = np.array([1e-9,9e-10,4.5e-10,6e-10,9e-10])
T_gen2, mu90_gen2, Omega_gen2 = 10, 2.3, 4*np.pi

E_gen2, A_gen2 = effective_area_from_sensitivity(
    E_gen2, E2Phi_gen2, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
)

E_gen2_interp = np.logspace(np.log10(E_gen2[0]), np.log10(E_gen2[-1]), 100)
A_gen2_interp = log_interp(E_gen2_interp, E_gen2, A_gen2)

### RNO-G
E_rnogo   = np.array([2e16,1e17,1e18,1e19,5e19])
E2Phi_rnogo = np.array([2e-8,9e-9,5.5e-9,6e-9,9e-9])
T_rnogo, mu90_rnogo, Omega_rnogo = 10, 2.3, 4*np.pi

E_rnogo, A_rnogo = effective_area_from_sensitivity(
    E_rnogo, E2Phi_rnogo, E_units="eV", T_years=T_rnogo, mu90=mu90_rnogo, DeltaOmega=Omega_rnogo
)

E_rnogo_interp = np.logspace(np.log10(E_rnogo[0]), np.log10(E_rnogo[-1]), 100)
A_rnogo_interp = log_interp(E_rnogo_interp, E_rnogo, A_rnogo)



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
T_YEARS = 9.5
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

# ---------------------------Radio Scaling ----------------------------

# 1) Build a common log-spaced grid over the overlap
E_low  = max(float(np.min(E_rnogo_interp)), float(np.min(E_gen2_interp)), float(np.min(E_grand_interp)))
E_high = min(float(np.max(E_rnogo_interp)), float(np.max(E_gen2_interp)), float(np.max(E_grand_interp)))
if not (np.isfinite(E_low) and np.isfinite(E_high)) or E_low <= 0 or E_high <= E_low:
    raise ValueError("Invalid or non-overlapping energy ranges.")
npts = 300
E_grid_radio = np.logspace(np.log10(E_low), np.log10(E_high), npts)

# 2) Log–log interpolation of areas onto the common grid
# sort inputs
ord_rno   = np.argsort(E_rnogo_interp)
ord_gen2  = np.argsort(E_gen2_interp)
ord_grand = np.argsort(E_grand_interp)

Ex_rno,   Ay_rno   = np.asarray(E_rnogo_interp)[ord_rno],  np.asarray(A_rnogo_interp)[ord_rno]
Ex_gen2,  Ay_gen2  = np.asarray(E_gen2_interp)[ord_gen2],  np.asarray(A_gen2_interp)[ord_gen2]
Ex_grand, Ay_grand = np.asarray(E_grand_interp)[ord_grand],np.asarray(A_grand_interp)[ord_grand]

if np.any(Ex_rno <= 0) or np.any(Ex_gen2 <= 0) or np.any(Ex_grand <= 0):
    raise ValueError("E arrays must be > 0 for log interpolation.")

def _log_interp_1d(xnew, x, y):
    # pure inline helper (not reusable elsewhere); stays inside the cell
    if np.all(y > 0):
        return 10**np.interp(np.log10(xnew), np.log10(x), np.log10(y))
    else:
        # fallback if some y <= 0: interpolate y vs log10(x)
        return np.interp(np.log10(xnew), np.log10(x), y)

A_rnogo_on  = _log_interp_1d(E_grid_radio, Ex_rno,   Ay_rno)
A_gen2_on   = _log_interp_1d(E_grid_radio, Ex_gen2,  Ay_gen2)
A_grand_on  = _log_interp_1d(E_grid_radio, Ex_grand, Ay_grand)

# 3) Read radioscaling.csv (Energy (GeV), electron fraction, muon fraction, taon/tau fraction)
if not os.path.exists("radioscaling.csv"):
    raise FileNotFoundError("Missing 'radioscaling.csv' in the working directory.")

df = pd.read_csv("radioscaling.csv")
cols = {c.lower(): c for c in df.columns}

def _pick(*names):
    for n in names:
        if n in cols: return cols[n]
    raise KeyError(f"Column not found. Tried: {names}")

c_E   = _pick("energy (gev)", "energy", "e (gev)")
c_fe  = _pick("electron fraction", "e fraction", "frac_e", "electron")
c_fmu = _pick("muon fraction", "mu fraction", "frac_mu", "muon")
c_ft  = _pick("taon fraction", "tau fraction", "tauon fraction", "frac_tau", "tau")

E_csv    = df[c_E].to_numpy(dtype=float)
frac_e_c = df[c_fe].to_numpy(dtype=float)
frac_mu_c= df[c_fmu].to_numpy(dtype=float)
frac_tau_c=df[c_ft].to_numpy(dtype=float)

if np.any(E_csv <= 0):
    raise ValueError("CSV energies must be > 0 (GeV).")

#4
# sort CSV by energy
ord_csv = np.argsort(E_csv)
E_csv       = E_csv[ord_csv]
frac_e_c    = frac_e_c[ord_csv]
frac_mu_c   = frac_mu_c[ord_csv]
frac_tau_c  = frac_tau_c[ord_csv]

def _log_interp_frac(xnew, x, y):
    if np.all(y > 0):
        return 10**np.interp(np.log10(xnew), np.log10(x), np.log10(y))
    else:
        return np.interp(np.log10(xnew), np.log10(x), y)

frac_e  = _log_interp_frac(E_grid_radio, E_csv, frac_e_c)
frac_mu = _log_interp_frac(E_grid_radio, E_csv, frac_mu_c)
frac_tau= _log_interp_frac(E_grid_radio, E_csv, frac_tau_c)

# 5) Scaled radio areas

sum_radio  = A_rnogo_on + A_gen2_on
A_e_radio   = sum_radio * frac_e
A_mu_radio  = sum_radio * frac_mu
A_tau_radio = sum_radio * frac_tau + A_grand_on

# --------------------------- Plot: A_eff -------------------------------
fig, ax_left = plt.subplots(figsize=(9,6), dpi=140)  # Left axis: A_eff (m^2)

def errorline(ax, x, y, label):
    m = (x>0) & (y>0) & np.isfinite(y)
    if not np.any(m):
        return
    x = np.asarray(x[m], dtype=float)
    y = np.asarray(y[m], dtype=float)
    sigma = np.sqrt(y)
    ax.errorbar(x, y, fmt='o-', capsize=3, markersize=1, linewidth=1.2, label=label)

errorline(ax_left,  E_tambo,   A_tambo_e,      "Tambo (e)")
errorline(ax_left, E_tambo,          A_tambo_tau, "TAMBO (τ)")   
errorline(ax_left,  E_tambo_interp,   A_tambo_interp,      "Tambocheck")

errorline(ax_left,  E_grand_interp,   A_grand_interp,      "Grand 200k (τ)")
errorline(ax_left,  E_trinity_interp,   A_trinity_interp,      "Trinity (τ)")
errorline(ax_left,  E_poemma_interp,   A_poemma_interp,      "Poemma (τ)")
errorline(ax_left,  E_gen2_interp,   A_gen2_interp,      "ICGen2")
errorline(ax_left,  E_rnogo_interp,   A_rnogo_interp,      "RNO-G")
errorline(ax_left,  E_grid_radio,   A_e_radio,      "Radio (e)")
errorline(ax_left,  E_grid_radio,   A_tau_radio,      "Radio (τ)")
errorline(ax_left,  E_grid_radio,   A_mu_radio,      "Radio (mu)")

y_lo = np.minimum(A_muon_min_interp, A_muon_max_interp)
y_hi = np.maximum(A_muon_min_interp, A_muon_max_interp)

ax_left.fill_between(
    E_icmuon_interp, y_lo, y_hi,
    alpha=0.25, linewidth=0, label="IC Track (μ)"
)

y_lo2 = np.minimum(A_ic_cascade_emax, A_ic_cascade_emin)
y_hi2 = np.maximum(A_ic_cascade_emax, A_ic_cascade_emin)
y_lo3 = np.minimum(A_ic_cascade_taumax, A_ic_cascade_taumin)
y_hi3 = np.maximum(A_ic_cascade_taumax, A_ic_cascade_taumin)

ax_left.fill_between(
    E_icet_interp, y_lo2, y_hi2,
    alpha=0.25, linewidth=0, label="IC Cascade (e)"
)

ax_left.fill_between(
    E_icet_interp, y_lo3, y_hi3,
    alpha=0.25, linewidth=0, label="IC Cascade (τ)"
)

# Scales, labels, grids
ax_left.set_xscale("log"); ax_left.set_yscale("log")
ax_left.set_xlabel("Energy E (GeV)")
ax_left.set_ylabel("Average Effective Area $A_{\\rm eff}$ (m$^2$)")
ax_left.set_title("Average Effective Area for Diffuse Neutrino Flux (1:1:1 Ratio) assuming 10-year integration")

ax_left.grid(True, which="both", alpha=0.3)

# Legends: one per axis, then combine
h1, l1 = ax_left.get_legend_handles_labels()
ax_left.legend(h1, l1, loc="lower right", fontsize=9, ncol=2)

fig.tight_layout()
out_path = "combined_plot.png"
plt.savefig(out_path)
print(f"Saved: {out_path}")
plt.show()