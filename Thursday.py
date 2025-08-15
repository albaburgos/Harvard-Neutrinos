
import numpy as np
import matplotlib.pyplot as plt


def effective_area_from_sensitivity(E_pts, E2Phi_pts, *,
                                    E_units="GeV",
                                    T_years,
                                    mu90=2.3,
                                    DeltaOmega=2*np.pi):
    E = np.asarray(E_pts, dtype=float)
    E2Phi = np.asarray(E2Phi_pts, dtype=float)

    # Convert energy to GeV if needed
    if E_units.lower() == "ev":
        E_GeV = E / 1e9
    else:
        E_GeV = E.copy()

    # Sort by energy
    order = np.argsort(E_GeV)
    E_GeV = E_GeV[order]
    E2Phi = E2Phi[order]

    # Build geometric-mean edges to get per-point ΔE
    edges = np.zeros(len(E_GeV) + 1)
    for i in range(1, len(E_GeV)):
        edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
    # Extrapolate endpoints (symmetric in log-space)
    edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
    edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

    # Per-point ΔE
    dlog10E = np.log10(edges[1:]) - np.log10(edges[:-1])
    dE = E_GeV * np.log(10.0) * dlog10E  # GeV

    # Convert E^2 Phi -> Phi (per-energy flux)
    Phi = E2Phi / (E_GeV**2)  # cm^-2 s^-1 sr^-1 GeV^-1

    # Exposure in seconds
    T_sec = T_years * 365.25 * 24 * 3600.0

    # Effective area in cm^2, then to m^2
    Aeff_cm2 = mu90 / (T_sec * DeltaOmega * Phi * dE)
    Aeff_m2 = Aeff_cm2 / 1e4
    return E_GeV, Aeff_m2

# ============================================================
# Datasets
# ============================================================

# --- RNO-G (90% C.L., T=5 yr), energies in eV (E^2 Phi in GeV cm^-2 s^-1 sr^-1)
E_rnog_eV   = [3e16, 1e17, 3e17, 1e18, 4e18, 1e19, 4e19]
E2Phi_rnog  = [2e-8,  9e-9,  6.5e-9, 5.5e-9, 5e-9,  6e-9,  8e-9]
T_rnog      = 5.0
mu90_rnog   = 2.3
Omega_rnog  = 2*np.pi

# --- IceCube-Gen2 Radio (90% C.L., T=10 yr), energies in eV
E_gen2_eV   = E_rnog_eV
E2Phi_gen2  = [2e-9, 6e-10, 3e-10, 2.2e-10, 2e-10, 4e-10, 8e-10]
T_gen2      = 10.0
mu90_gen2   = 2.3
Omega_gen2  = 2*np.pi

# --- POEMMA (90% C.L.), energies in GeV (from log10 grid)
log10E_poemma = np.array([7, 7.2, 7.6, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])
E_poemma_GeV  = 10**log10E_poemma
E2Phi_poemma  = [2.5e1, 4, 1, 0.9, 6.5e-1, 8e-1, 1.2, 2.4, 7]
# 1000 s in years:
T_poemma      = 1000.0 / (365.25*24*3600.0)   # ≈ 3.1688e-5 yr
mu90_poemma   = 2.3
Omega_poemma  = 4*np.pi/12

# --- Trinity (90% C.L., T=3 yr), energies in GeV
E_trinity_GeV = [1e5, 4e5, 1e6, 4e6, 1e7, 7e7, 4e8, 3e9, 1e10, 4e10]
E2Phi_trinity = [2e-7, 3e-8, 1e-8, 2e-9, 1e-9, 5e-10, 7e-10, 3e-9, 8e-9, 4e-8]
T_trinity     = 3.0
mu90_trinity  = 2.3
Omega_trinity = 2*np.pi

# --- GRAND 200k (90% C.L., T=3 yr, μ90=2.44), energies in GeV
E_grand_GeV   = [4.5e7, 1e8, 5e8, 1e9, 3e9, 1e10, 4e10, 7e10, 1e11]
E2Phi_grand   = [4e-9, 2.7e-9, 7e-10, 7.5e-10, 1.2e-9, 3e-9, 9e-9, 0.5e-8, 2e-8]
T_grand       = 3.0
mu90_grand    = 2.44
Omega_grand   = 2*np.pi

# --- RET-N (Radar Echo Telescope - North), energies in eV, TWO curves (band)
RETN_data = [
    (6e19, 3e-9, 2.6e-9),
    (1e19, 1.6e-9, 1.0e-9),
    (1e18, 0.8e-9, 4e-10),
    (4.5e17, 4e-9, 3.5e-10),
    (2e17, 6e-9, 4.5e-10),
    (5e16, 1e-8, 1e-9),
    (2e16, 4e-8, 1.6e-9),
]
E_retn_eV     = [r[0] for r in RETN_data]
E2Phi_retn_hi = [r[1] for r in RETN_data]  # higher curve
E2Phi_retn_lo = [r[2] for r in RETN_data]  # lower curve
T_retn        = 3.0
mu90_retn     = 2.3
# Replace with actual RET-N beam/coverage if narrower than a hemisphere
Omega_retn    = 2*np.pi/12

# --- TAMBO: two lines (all-flavor & tau) from aperture arrays (m^2 sr) → A_eff (m^2)
# Energies in GeV
tambo_E_GeV = np.array([
    1e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9
], dtype=float)

# Apertures (m^2 sr)
tambo_tau_ap_m2sr = np.array([
    1,
    5*10,
    10**2,
    5*10**2,
    6*10**2,
    3*10**3,
    4*10**3,
    3*10**3,
    2*10**3,
    6*10**3,
    2*10**4,
    4*10**4,
    5*10**4
], dtype=float)

tambo_all_ap_m2sr = np.array([
    1,
    5*10,
    10**2,
    5*10**2,
    0.7*10**3,
    0.8*10**3,
    10**3,
    1.2*10**3,
    2*10**3,
    6*10**3,
    2*10**4,
    4*10**4,
    5*10**4
], dtype=float)

# Convert aperture (m^2 sr) → effective area (m^2) by dividing by 2 sr
omega_tambo = 2.0  # sr
tambo_tau_Aeff_m2 = tambo_tau_ap_m2sr / omega_tambo
tambo_all_Aeff_m2 = tambo_all_ap_m2sr / omega_tambo

# --- ICECUBE 
final_rows = np.array([
    [109.999,232.892,160.05588745185227,164.45467039632751,0.0,0.0,4,0,4],
    [232.892,493.079,338.8718850362184,489.2658968675564,0.0,0.0,2,0,2],
    [493.079,1043.947,717.4596454247444,459.2992274601041,0.01189710689723954,0.007494240565190263,3,2,3],
    [1043.947,2210.245,1519.0058054579645,1029.7815972148696,0.09820435879202606,0.06186101341229988,4,2,4],
    [2210.245,4679.532,3216.0398326730965,388.02745820737346,0.11638625576328972,0.07331417685876518,2,1,2],
    [4679.532,9907.507,6809.001104914289,379.07703660299165,0.4670681596373116,0.2942161635510625,4,2,4],
    [9907.507,20976.177,14416.019577565057,89.81852572248584,0.8190008125850179,0.5159060236756019,3,2,3],
    [20976.177,44410.769,30521.60138737994,100.32566788993606,0.42674960446539567,0.268818648482139,2,1,2],
    [44410.769,94026.493,64620.34401411925,124.36472512537776,0.9791502429015384,0.6167875545836463,4,2,4],
    [94026.493,199072.921,136814.21202819556,24.11496663524563,0.9791502429015384,0.6167875545836463,1,0,1],
    [199072.921,421477.252,289663.09342181147,17.537609156303414,1.0301254335306331,0.6488979108854382,3,2,3],
    [421477.252,892351.772,613274.7937750992,9.736213119412295,1.1353159319934158,0.7151596422005769,3,2,3],
    [892351.772,1889287.455,1298425.588274746,8.065149146252075,1.1353159319934158,0.7151596422005769,1,0,1],
    [1889287.455,4000000.0,2749027.067891475,7.562326904682501,1.1353159319934158,2.26869807140475,1,0,1]
], dtype=float)

E_final_GeV = final_rows[:,2]
muon  = final_rows[:,3]
electron   = final_rows[:,4]
tau = final_rows[:,5]

# ============================================================
# Compute A_eff for sensitivity-based experiments
# ============================================================
curves = []

# RNO-G
E_rnog_GeV, Aeff_rnog = effective_area_from_sensitivity(
    E_rnog_eV, E2Phi_rnog, E_units="eV", T_years=T_rnog, mu90=mu90_rnog, DeltaOmega=Omega_rnog
)
curves.append(("RNO-G All Flavor (T=5 yr)", E_rnog_GeV, Aeff_rnog))

# IceCube-Gen2 Radio
E_gen2_GeV, Aeff_gen2 = effective_area_from_sensitivity(
    E_gen2_eV, E2Phi_gen2, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
)
curves.append(("IceCube-Gen2 Radio Taon (T=10 yr)", E_gen2_GeV, Aeff_gen2))

# POEMMA
E_po_GeV, Aeff_po = effective_area_from_sensitivity(
    E_poemma_GeV, E2Phi_poemma, E_units="GeV", T_years=T_poemma, mu90=mu90_poemma, DeltaOmega=Omega_poemma
)
curves.append(("POEMMA Taon (T=5yr)", E_po_GeV, Aeff_po))

# Trinity
E_tr_GeV, Aeff_tr = effective_area_from_sensitivity(
    E_trinity_GeV, E2Phi_trinity, E_units="GeV", T_years=T_trinity, mu90=mu90_trinity, DeltaOmega=Omega_trinity
)
curves.append(("Trinity Taon (T=3 yr)", E_tr_GeV, Aeff_tr))

# GRAND 200k (use μ90=2.44)
E_gr_GeV, Aeff_gr = effective_area_from_sensitivity(
    E_grand_GeV, E2Phi_grand, E_units="GeV", T_years=T_grand, mu90=mu90_grand, DeltaOmega=Omega_grand
)
curves.append((f"GRAND 200k Taon (T=3 yr, μ90={mu90_grand})", E_gr_GeV, Aeff_gr))

# RET-N (band between hi and lo curves)
E_retn_GeV_hi, Aeff_retn_hi = effective_area_from_sensitivity(
    E_retn_eV, E2Phi_retn_hi, E_units="eV", T_years=T_retn, mu90=mu90_retn, DeltaOmega=Omega_retn
)
E_retn_GeV_lo, Aeff_retn_lo = effective_area_from_sensitivity(
    E_retn_eV, E2Phi_retn_lo, E_units="eV", T_years=T_retn, mu90=mu90_retn, DeltaOmega=Omega_retn
)


plt.figure(figsize=(10.5, 7.2))

# Sensitivity-derived curves
for label, E_GeV, Aeff_m2 in curves:
    plt.loglog(E_GeV, Aeff_m2, marker='o', linewidth=1.2, label=label)

# RET-N band
E_band = E_retn_GeV_hi
lo = np.minimum(Aeff_retn_hi, Aeff_retn_lo)
hi = np.maximum(Aeff_retn_hi, Aeff_retn_lo)
plt.fill_between(E_band, lo, hi, alpha=0.20, label='RET-N band all-flavor(T=5 yr)')
plt.loglog(E_retn_GeV_hi, Aeff_retn_hi, linewidth=1.0, color='C6')
plt.loglog(E_retn_GeV_lo, Aeff_retn_lo, linewidth=1.0, color='C6')

plt.loglog(tambo_E_GeV, tambo_all_Aeff_m2, marker='^', linewidth=1.2, label='TAMBO taon (10yr)')
plt.loglog(tambo_E_GeV, tambo_tau_Aeff_m2,  marker='v', linewidth=1.2, label='TAMBO all-flavor (10yr)')

# IceCube
plt.loglog(E_final_GeV, muon, marker='^', linewidth=1.0, label='IceCube Muon (T+C)')
plt.loglog(E_final_GeV, electron,   marker='v', linewidth=1.0, label='IceCube Electron(C)')
plt.loglog(E_final_GeV, tau, marker='x', linewidth=1.0, label='IceCube Taon(T+C)')

plt.xlabel(r'Neutrino energy $E$ [GeV]')
plt.ylabel(r'Effective Area [m$^2$]')
plt.title('All-Flavor Effective Area vs Energy — Combined')
plt.grid(True, which='both', ls='--', alpha=0.45)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig('Thursday1.png', dpi=300)
plt.show()