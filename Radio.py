import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Effective area from sensitivity (90% C.L.)
# A_eff ≈ mu90 / (T * ΔΩ * Phi(E) * ΔE), with Phi = (E^2 Phi)/E^2
# ==========================

# ---- User-provided data: Energy_eV, E2Phi_RNOG, E2Phi_Gen2
#     E2Phi units: GeV cm^-2 s^-1 sr^-1
data = np.array([
    [3e16, 2e-8, 2e-9],
    [1e17, 9e-9, 6e-10],
    [3e17, 6.5e-9, 3e-10],
    [1e18, 5.5e-9, 2.2e-10],
    [4e18, 5e-9, 2e-10],
    [1e19, 6e-9, 4e-10],
    [4e19, 8e-9, 8e-10],
], dtype=float)

# ---- Sort by energy
order = np.argsort(data[:,0])
data = data[order]

E_eV = data[:,0]
E_GeV = E_eV / 1e9
E2phi_rnog = data[:,1]
E2phi_gen2 = data[:,2]

# ---- Constants
mu90 = 2.3                  # 90% C.L., background-free (Feldman–Cousins)
T_rnog_years = 5            # RNO-G exposure
T_gen2_years = 10           # IceCube-Gen2 Radio exposure
DeltaOmega = 2 * np.pi      # sr (assumed hemisphere)

sec_per_year = 365.25 * 24 * 3600.0
T_rnog = T_rnog_years * sec_per_year
T_gen2 = T_gen2_years * sec_per_year

# ---- Per-point ΔE via geometric-mean edges (log spacing)
edges = np.zeros(len(E_GeV) + 1)
for i in range(1, len(E_GeV)):
    edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

log_edges = np.log10(edges)
dlog10E = log_edges[1:] - log_edges[:-1]
dE = E_GeV * np.log(10.0) * dlog10E   # GeV

# ---- Convert E^2 Phi -> Phi (per-energy flux)
Phi_rnog = E2phi_rnog / (E_GeV**2)    # cm^-2 s^-1 sr^-1 GeV^-1
Phi_gen2 = E2phi_gen2 / (E_GeV**2)

# ---- Effective area (cm^2 → m^2)
Aeff_rnog_cm2 = mu90 / (T_rnog * DeltaOmega * Phi_rnog * dE)
Aeff_gen2_cm2 = mu90 / (T_gen2 * DeltaOmega * Phi_gen2 * dE)
Aeff_rnog_m2 = Aeff_rnog_cm2 / 1e4
Aeff_gen2_m2 = Aeff_gen2_cm2 / 1e4

# ---- Plot
plt.figure(figsize=(7,5))
plt.loglog(E_GeV, Aeff_rnog_m2, marker='o', label=f'RNO-G $A_{{eff}}$ (T={T_rnog_years} yr)')
plt.loglog(E_GeV, Aeff_gen2_m2, marker='s', label=f'IceCube-Gen2 Radio $A_{{eff}}$ (T={T_gen2_years} yr)')
plt.xlabel(r'Neutrino energy $E$ [GeV]')
plt.ylabel(r'Effective Area [m$^2$]')
plt.title('Effective Area vs Energy: RNO-G vs IceCube-Gen2 Radio (90% C.L.)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("rnog_gen2_effective_area.png", dpi=300)
plt.show()

