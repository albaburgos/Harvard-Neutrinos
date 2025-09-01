import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ Config (edit as needed) ------------------------

T_years = 11.3
DeltaOmega = 4*np.pi 

OUTPUT_CSV = "MC_outputs/effareasAll.csv"
PLOT_PNG   = "combined_plot.png"

SEC_PER_YEAR = 365.25 * 24 * 3600.0

# ------------------------ Data ------------------------
'''HESE DATA
# E1: tau double-bang, E2: cascades, E3: tracks
E1 = np.array([4e4,6e4,1e5,2.1e5,3.5e5,5.5e5], dtype=floa
t)
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


# ------------------------ Compute component A_eff IceCube ------------------------
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


### IceCube Gen2 (Visual)
E_gen2vis   = np.array([1e15,3e15,9e15, 1e16,2e16, 3e16, 1e17])
E2Phi_gen2vis = np.array([3e-10,3.5e-10, 4e-10, 5e-10, 7e-10, 1e-9, 1e-9])
T_gen2, mu90_gen2, Omega_gen2 = 10, 2.3, 4*np.pi

E_gen2vis, A_gen2vis = effective_area_from_sensitivity(
    E_gen2vis, E2Phi_gen2vis, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
)

E_gen2vis_interp = np.logspace(np.log10(E_gen2vis[0]), np.log10(E_gen2vis[-1]), 100)
A_gen2vis_interp = log_interp(E_gen2vis_interp, E_gen2vis, A_gen2vis)

### IceCube Gen2 Radio
E_gen2   = np.array([1e17,1e18,1e19,2.5e19])
E2Phi_gen2 = np.array([3e-10,5e-10, 1e-9,9e-10,4.5e-10,6e-10,9e-10])
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


# Ice Cube gen 2 interpolation 

eminIC = min(min(E1), min(E2), min(E3))
emaxIC = max(max(E1), max(E2), max(E3))
masterIC = np.logspace(np.log10(eminIC), np.log10(emaxIC), 300)
IC_mu  = (log_interp(masterIC, E2mu,  A2mu)
        + log_interp(masterIC, E2mu,  A2mu)
        + log_interp(masterIC, E2mu,  A2mu))

IC_e   = (log_interp(masterIC, E2e,   A2e)
        + log_interp(masterIC, E2e,   A2e)
        + log_interp(masterIC, E2e,   A2e))

IC_tau = (log_interp(masterIC, E2tau, A2tau)
        + log_interp(masterIC, E2tau, A2tau)
        + log_interp(masterIC, E2tau, A2tau))

IC_sum = IC_mu + IC_e + IC_tau
eps = 0.0
den = np.where(IC_sum > eps, IC_sum, 1.0)
f_mu  = IC_mu  / den
f_e   = IC_e   / den
f_tau = IC_tau / den

E_gen2vis_interp = np.logspace(np.log10(E_gen2vis[0]), np.log10(E_gen2vis[-1]), 100)
A_gen2vis_interp = log_interp(E_gen2vis_interp, E_gen2vis, A_gen2vis)

# Helper: interpolate in log10(E) *linearly in value* (safer for fractions)
def interp_in_logE(y_src, E_src, E_dst):
    logE_src = np.log10(E_src)
    logE_dst = np.log10(E_dst)
    # Use edge values for extrapolation; assumes E_src is sorted ascending
    return np.interp(logE_dst, logE_src, y_src, left=y_src[0], right=y_src[-1])

# 5) Bring fractions onto the Gen2 grid
f_mu_at_gen2  = interp_in_logE(f_mu,  masterIC, E_gen2vis_interp)
f_e_at_gen2   = interp_in_logE(f_e,   masterIC, E_gen2vis_interp)
f_tau_at_gen2 = interp_in_logE(f_tau, masterIC, E_gen2vis_interp)

# 6) Flavor-split Gen2 effective area
ICgen2mu  = f_mu_at_gen2  * A_gen2vis_interp
ICgen2e   = f_e_at_gen2   * A_gen2vis_interp
ICgen2tau = f_tau_at_gen2 * A_gen2vis_interp

def _plot_pos_loglog(x, y, label):
    x = np.asarray(x)
    y = np.asarray(y)
    m = (x > 0) & (y > 0)
    plt.loglog(x[m], y[m], lw=2, label=label)

plt.figure(figsize=(7,4.5))
_plot_pos_loglog(E_gen2vis_interp, ICgen2mu,  r"$\nu_\mu$")
_plot_pos_loglog(E_gen2vis_interp, ICgen2e,   r"$\nu_e$")
_plot_pos_loglog(E_gen2vis_interp, ICgen2tau, r"$\nu_\tau$")

plt.xlabel("Energy (GeV)")
plt.ylabel(r"Effective area [m$^2$]")
plt.title("IceCube Gen2 — Flavor-split effective area")
plt.grid(True, which="both", ls=":", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
_plot_pos_loglog(masterIC, IC_mu,  r"$\nu_\mu$")
_plot_pos_loglog(masterIC, IC_e,   r"$\nu_e$")
_plot_pos_loglog(masterIC, IC_tau, r"$\nu_\tau$")

plt.xlabel("Energy (GeV)")
plt.ylabel(r"Effective area [m$^2$]")
plt.title("IceCube — Flavor-split effective area")
plt.grid(True, which="both", ls=":", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------ Common energy grid ------------------------
emin = min(min(E1), min(E2), min(E3), min(E_tambo))
emax = max(max(E1), max(E2), max(E3), max(E_tambo), max(E_grid_radio))
master = np.logspace(np.log10(emin), np.log10(emax), 300)

A_mu_master = (log_interp(master, E1mu, A1mu) + log_interp(master, E2mu,  A2mu) + log_interp(master, E3mu,  A3mu))*2 + log_interp(master, E_grid_radio,  A_mu_radio)  +  log_interp(master, E_gen2vis_interp,  ICgen2mu)
A_tau_master = (log_interp(master, E1tau, A1tau) + log_interp(master, E2tau,  A2tau) + log_interp(master, E3tau,  A3tau))*2 + log_interp(master, E_grid_radio,  A_tau_radio) + log_interp(master, E_trinity_interp, A_trinity_interp) + log_interp(master, E_poemma_interp, A_poemma_interp) + 0.1*log_interp(master, E_tambo,  A_tambo_tau) + log_interp(master, E_gen2vis_interp,  ICgen2mu) 
A_e_master = (log_interp(master, E1e, A1e) + log_interp(master, E2e,  A2e) + log_interp(master, E3e,  A3e) )*2 + log_interp(master, E_grid_radio,  A_e_radio)+0.1*log_interp(master, E_tambo,  A_tambo_e)  + log_interp(master, E_gen2vis_interp,  ICgen2mu)

# ------------------------ Plot ------------------------
plt.figure(figsize=(9,6), dpi=140)
plt.loglog(master, A_mu_master,  label="Muon")
plt.loglog(master, A_tau_master, label="Tau")
plt.loglog(master, A_e_master,   label="Electron")
plt.xlabel("Energy E (GeV)")
plt.ylabel(r"Effective Area $A_{\rm eff}$ (cm$^2$)")
plt.title("Effective Area All-Experiment (10-year integration) ")
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
