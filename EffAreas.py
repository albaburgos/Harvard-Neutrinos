#!/usr/bin/env python3
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
    
def interp_loglog(x_new, x, y, eps=1e-300):
    x   = np.asarray(x);     y   = np.asarray(y)
    x_new = np.asarray(x_new)

    # guard against zeros/negatives for log10
    y_safe = np.where(y > 0, y, eps)

    y_new = 10.0 ** np.interp(
        np.log10(x_new),
        np.log10(x),
        np.log10(y_safe)
    )
    return y_new

def effective_area_from_sensitivity(E_pts, E2Phi_pts, *, E_units="GeV", T_years, mu90=2.3, DeltaOmega):
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


def effective_area_from_Nevents(E_edges, N, T_years, DeltaOmega, phi0, gamma, FluxModelCls=FluxModel, E0=1e5):
    E_edges = np.asarray(E_edges, dtype=float)
    N = np.asarray(N, dtype=float)
    nbins = E_edges.size - 1
    E_centers = np.sqrt(E_edges[:-1] * E_edges[1:])
    dlog10E = np.diff(np.log10(E_edges))         
    dE = E_centers * np.log(10.0) * dlog10E   
    T_sec = float(T_years) * SEC_PER_YEAR

    flux = FluxModelCls(phi0, gamma, E0=E0)
    flux_ic = flux.phi(E_centers, 'all')      

    Aeff_cm2 = N / (T_sec * DeltaOmega * flux_ic * dE)
    Aeff_m2 = Aeff_cm2/1e4
    return E_centers, Aeff_m2

# --------------------------- Radio experiments ------------------------------
E_rnog_eV   = np.array([3e16, 1e17, 3e17, 1e18, 4e18, 1e19, 4e19])
E2Phi_rnog  = np.array([2e-8,  9e-9,  6.5e-9, 5.5e-9, 5e-9,  6e-9,  8e-9])
T_rnog      = 5.0; mu90_rnog   = 2.3; Omega_rnog  = 2*np.pi

E_gen2_eV   = E_rnog_eV.copy()
E2Phi_gen2  = np.array([2e-9, 6e-10, 3e-10, 2.2e-10, 2e-10, 4e-10, 8e-10])
T_gen2      = 10.0; mu90_gen2   = 2.3; Omega_gen2  = 2*np.pi

E_grand_GeV   = np.array([4.5e7, 1e8, 5e8, 1e9, 3e9, 1e10, 4e10, 7e10, 1e11])
E2Phi_grand   = np.array([4e-9, 2.7e-9, 7e-10, 7.5e-10, 1.2e-9, 3e-9, 9e-9, 0.5e-8, 2e-8])
T_grand       = 3.0; mu90_grand = 2.44; Omega_grand = 2*np.pi

E_ic_GeV = np.array([1.5e4,4e4,6e4,8e4,1e5,2e5,4e5,6e5,1e6, 2e6,4e6])
E2Phi_ic = np.array([2.9e-8,2e-8,1.8e-8,1.6e-8, 1.5e-8, 1.2e-8, 8.7e-9, 6.3e-9, 6.15e-9, 4.8e-9, 3.7e-9])
T_ic      = 9.5; mu90_ic   = 2.3; Omega_ic  = 4*np.pi

E_ic_GeV = np.array([1.5e4, 4e4, 6e4, 8e4, 1e5, 2e5, 4e5, 6e5, 1e6, 2e6, 4e6], dtype=float)
E2Phi_ic = np.array([2.9e-8, 2e-8, 1.8e-8, 1.6e-8, 1.5e-8, 1.2e-8, 8.7e-9, 6.3e-9, 6.15e-9, 4.8e-9, 3.7e-9], dtype=float)

E_ic_GeV, A_ic = effective_area_from_sensitivity(
    E_ic_GeV, E2Phi_ic, E_units="GeV", T_years=T_ic, mu90=mu90_ic, DeltaOmega=Omega_ic
)


# Convert to A_eff
E_rnog_GeV,  Aeff_rnog  = effective_area_from_sensitivity(
    E_rnog_eV, E2Phi_rnog, E_units="eV", T_years=T_rnog, mu90=mu90_rnog, DeltaOmega=Omega_rnog
)
E_gen2_GeV,  Aeff_gen2  = effective_area_from_sensitivity(
    E_gen2_eV, E2Phi_gen2, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
)
E_grand_GeV, Aeff_grand = effective_area_from_sensitivity(
    E_grand_GeV, E2Phi_grand, E_units="GeV", T_years=T_grand, mu90=mu90_grand, DeltaOmega=Omega_grand
)

# Common log grid (for flavor fractions)
E_low  = min(float(E_rnog_GeV.min()), float(E_gen2_GeV.min()), float(E_grand_GeV.min()))
E_high = max(float(E_rnog_GeV.max()), float(E_gen2_GeV.max()), float(E_grand_GeV.max()))
E_grid_radio = np.logspace(np.log10(E_low), np.log10(E_high), 300)

A_rnog_on  = log_interp(E_grid_radio, E_rnog_GeV,  Aeff_rnog)
A_gen2_on  = log_interp(E_grid_radio, E_gen2_GeV,  Aeff_gen2)
A_grand_on = log_interp(E_grid_radio, E_grand_GeV, Aeff_grand)

# Flavor fractions for radio
fe_g = fmu_g = ft_g = None
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
            print("WARNING: radioscaling.csv missing required columns; defaulting to equal fractions.")
    except Exception as e:
        print(f"WARNING: Could not read radioscaling.csv ({e}); defaulting to equal fractions.")
else:
    print("INFO: radioscaling.csv not found; defaulting to equal fractions.")
if fe_g is None:
    fe_g  = np.full_like(E_grid_radio, 1/3, dtype=float)
    fmu_g = np.full_like(E_grid_radio, 1/3, dtype=float)
    ft_g  = np.full_like(E_grid_radio, 1/3, dtype=float)

def fractions_at(E):
    fe = log_interp(E, E_grid_radio, fe_g)
    fmu = log_interp(E, E_grid_radio, fmu_g)
    ft = log_interp(E, E_grid_radio, ft_g)
    s = fe + fmu + ft
    s[s==0] = 1.0
    return fe/s, fmu/s, ft/s

def radio_points_for(Epts):
    E = np.asarray(Epts, dtype=float)
    A_rnog_p  = log_interp(E, E_rnog_GeV,  Aeff_rnog)
    A_gen2_p  = log_interp(E, E_gen2_GeV,  Aeff_gen2)
    A_grand_p = log_interp(E, E_grand_GeV, Aeff_grand)
    fe, fmu, ft = fractions_at(E)
    A_e   = (A_rnog_p + A_gen2_p) * fe
    A_mu  = (A_rnog_p + A_gen2_p) * fmu
    A_tau = (A_rnog_p + A_gen2_p) * ft + A_grand_p
    return E, A_e, A_mu, A_tau

# Union of radio energies -> sorted points
E_union_radio = np.unique(np.concatenate([E_rnog_GeV, E_gen2_GeV, E_grand_GeV]))
E_pts, A_e_pts, A_mu_pts, A_tau_pts = radio_points_for(E_union_radio)

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
Omega_trinity = 6/36*(np.pi*2)
E_tr_GeV, A_tr = effective_area_from_sensitivity(
    E_trinity_GeV, E2Phi_trinity, E_units="GeV", T_years=T_trinity, mu90=mu90_trinity, DeltaOmega=Omega_trinity
)

# ------------------------------ TAMBO (AΩ) ---------------------------------
tambo_angle = 2*np.pi
tambo_E_GeV_base = np.array([1e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 50, 100, 500, 600, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 50, 100, 500, 700, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

E_tambo = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 200)
AOm_tambo_tau = log_interp(E_tambo, tambo_E_GeV_base, tambo_tau_ap_m2sr_base)     
AOm_tambo_all = log_interp(E_tambo, tambo_E_GeV_base, tambo_all_ap_m2sr_base)       
AOm_tambo_e   = np.abs(AOm_tambo_all - AOm_tambo_tau)            

A_tambo_e = AOm_tambo_e / tambo_angle
A_tambo_tau = AOm_tambo_tau / tambo_angle

# -------------------------- IceCube Cascades (Nevents) ---------------------------

E_min_log10_mu,  E_max_log10_mu  = 2.6, 5.8
E_min_log10_et,  E_max_log10_et  = 2.6, 6.8
step_dex = 0.2
num_mu   = int(round((E_max_log10_mu - E_min_log10_mu) / step_dex)) + 1  # 22 edges
num_etau = int(round((E_max_log10_et - E_min_log10_et) / step_dex)) + 1  # 27 edges
E_edges_mu   = np.logspace(E_min_log10_mu, E_max_log10_mu,   num=num_mu,   base=10.0)
E_edges_etau = np.logspace(E_min_log10_et, E_max_log10_et,   num=num_etau, base=10.0)

N_muons = np.array([1.0, 2.8, 5.0, 6.7, 7.2, 7.2, 6.5, 5.0,
                    4.0, 2.8, 1.8, 1.0, 0.6, 0.4, 0.26, 0.17], dtype=float)

N_etau = np.array([7.0, 20.0, 4e1, 5.5e1, 6e1, 6e1, 5.8e1, 4.9e1, 3.6e1,
                   2.2e1, 1.6e1, 1.0e1, 7.0, 4.0, 3.0, 1.8, 1.0, 0.7, 0.4, 0.36, 0.4], dtype=float)

phi0_casc = 1.68e-18
gamma_casc = 2.53

E_mu,  A_ic_cascade_mu = effective_area_from_Nevents(E_edges_mu,  N_muons, 6, Omega_ic, phi0_casc, gamma_casc)
E_et,  A_ic_cascade_etau  = effective_area_from_Nevents(E_edges_etau, N_etau,  6, Omega_ic, phi0_casc, gamma_casc)

frac_e1 = 127 / (80 + 127)
frac_e2 = (303 - 127) / ((204 - 80) + (303 - 127))

mask_low  = E_et < 1e4
mask_high = ~mask_low

A_ic_cascade_e   = np.empty_like(A_ic_cascade_etau)
A_ic_cascade_tau = np.empty_like(A_ic_cascade_etau)

A_ic_cascade_e[mask_low]   = frac_e1 * A_ic_cascade_etau[mask_low]
A_ic_cascade_e[mask_high]  = frac_e2 * A_ic_cascade_etau[mask_high]

A_ic_cascade_tau[mask_low]  = (1 - frac_e1) * A_ic_cascade_etau[mask_low]
A_ic_cascade_tau[mask_high] = (1 - frac_e2) * A_ic_cascade_etau[mask_high]


E_ic_mu = np.logspace(2.6, 6.4, num=20, base=10.0)
A_ic_mucasc  = log_interp(E_ic_mu, E_mu,       A_ic_cascade_mu)
A_ic_mutrack = log_interp(E_ic_mu, E_ic_GeV,   A_ic)
A_ic_mu = A_ic_mucasc + A_ic_mutrack

# --------------------------- Plot: A_eff -------------------------------
fig, ax_left = plt.subplots(figsize=(9,6), dpi=140)  # Left axis: A_eff (m^2)

# Helper: error bars with lines, safe for log-y
def errorline(ax, x, y, label):
    m = (x>0) & (y>0) & np.isfinite(y)
    if not np.any(m):
        return
    x = np.asarray(x[m], dtype=float)
    y = np.asarray(y[m], dtype=float)
    sigma = np.sqrt(y)
    ax.errorbar(x, y, fmt='o-', capsize=3, markersize=4, linewidth=1.2, label=label)

errorline(ax_left, E_pts,      A_e_pts,   "Radio E")
errorline(ax_left, E_pts,      A_mu_pts,  "Radio Mu")
errorline(ax_left, E_pts,      A_tau_pts, "Radio Tau")
errorline(ax_left, E_tr_GeV,   A_tr,      "Trinity Tau")
errorline(ax_left, E_po_GeV,   A_po,      "Poemma Tau")
errorline(ax_left,  E_tambo,   A_tambo_e,      "Tambo (e)")
errorline(ax_left, E_tambo,          A_tambo_tau, "TAMBO (τ)")   
errorline(ax_left, E_ic_mu, A_ic_mu,  "IC Casc+Track(μ)")
errorline(ax_left, E_et, A_ic_cascade_e,   "IC Casc (e)")
errorline(ax_left, E_et, A_ic_cascade_tau, "IC Casc (τ)")


# Scales, labels, grids
ax_left.set_xscale("log"); ax_left.set_yscale("log")
ax_left.set_xlabel("Energy E (GeV)")
ax_left.set_ylabel("Effective Area $A_{\\rm eff}$ (m$^2$)")
ax_left.set_title("Effective Area and Aperture vs Energy")

ax_left.grid(True, which="both", alpha=0.3)

# Legends: one per axis, then combine
h1, l1 = ax_left.get_legend_handles_labels()
ax_left.legend(h1, l1, loc="lower right", fontsize=9, ncol=2)

fig.tight_layout()
out_path = "combined_plot.png"
plt.savefig(out_path)
print(f"Saved: {out_path}")
plt.show()
