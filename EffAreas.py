#!/usr/bin/env python3
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
T_ic      = 9.5; mu90_ic   = 2.3; Omega_ic  = 2*np.pi

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

# Per-flavor *points* (connect with lines) + Gaussian error bars
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
Omega_trinity = 2*np.pi
E_tr_GeV, A_tr = effective_area_from_sensitivity(
    E_trinity_GeV, E2Phi_trinity, E_units="GeV", T_years=T_trinity, mu90=mu90_trinity, DeltaOmega=Omega_trinity
)

# ------------------------------ TAMBO (AΩ) ---------------------------------
tambo_E_GeV_base = np.array([1e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 50, 100, 500, 600, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 50, 100, 500, 700, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

E_tambo = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 200)
AOm_tambo_tau = log_interp(E_tambo, tambo_E_GeV_base, tambo_tau_ap_m2sr_base)     
AOm_tambo_all = log_interp(E_tambo, tambo_E_GeV_base, tambo_all_ap_m2sr_base)       
AOm_tambo_e   = np.abs(AOm_tambo_all - AOm_tambo_tau)                 

# -------------------------- IceCube Cascades (AΩ) ---------------------------
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


# --------------------------- Plot: A_eff + AΩ -------------------------------
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

# A_eff: points connected by lines + Gaussian error bars
errorline(ax_left, E_pts,      A_e_pts,   "Radio E")
errorline(ax_left, E_pts,      A_mu_pts,  "Radio Mu")
errorline(ax_left, E_pts,      A_tau_pts, "Radio Tau")
errorline(ax_left, E_tr_GeV,   A_tr,      "Trinity Tau")
errorline(ax_left, E_po_GeV,   A_po,      "Poemma Tau")
errorline(ax_left, E_ic_GeV,   A_ic,      "IC Muon Track")
errorline(ax_left,  E_tambo,   AOm_tambo_e,      "Tambo (e)")
errorline(ax_left, E_tambo,          AOm_tambo_all, "TAMBO (τ)")   
errorline(ax_left, IC_casc_E_center, IC_casc_AOm_mu,  "IC Casc (μ)")
errorline(ax_left, IC_casc_E_center, IC_casc_AOm_e,   "IC Casc (e)")
errorline(ax_left, IC_casc_E_center, IC_casc_AOm_tau, "IC Casc (τ)")

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
