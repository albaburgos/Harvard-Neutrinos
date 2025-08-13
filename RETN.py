"""
RET-N effective area from sensitivity (90% C.L., 5 years)

Inputs:
- Energy given in eV
- Sensitivity as E^2 dN/dE in [GeV cm^-2 s^-1 sr^-1]
- Two sensitivity values per energy -> treat as two curves (band edges)

Constants:
- mu90 = 2.3 (Feldman–Cousins, background-free, 90% C.L.)
- T = 5 years
- ΔΩ = 2π sr (hemisphere)

Output:
- Plot saved as 'rnog_effective_area.png'
- Printed table of E (GeV) and A_eff (m^2) for both band edges
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Input data: (Energy_eV, E2Phi_upper, E2Phi_lower)
# E2Phi in GeV cm^-2 s^-1 sr^-1
# ------------------------------
data = [
    (6e19, 3e-9, 2.6e-9),
    (1e19, 1.6e-9, 1.0e-9),
    (1e18, 0.8e-9, 4e-10),
    (4.5e17, 4e-9, 3.5e-10),
    (2e17, 6e-9, 4.5e-10),
    (5e16, 1e-8, 1e-9),
    (2e16, 4e-8, 1.6e-9),
]

# ---- arrays and sort by energy
E_eV = np.array([row[0] for row in data], dtype=float)
E2Phi_A = np.array([row[1] for row in data], dtype=float)  # band edge A
E2Phi_B = np.array([row[2] for row in data], dtype=float)  # band edge B
order = np.argsort(E_eV)
E_eV, E2Phi_A, E2Phi_B = E_eV[order], E2Phi_A[order], E2Phi_B[order]

# ---- convert energy to GeV
E_GeV = E_eV / 1e9

# ------------------------------
# Analysis constants
# ------------------------------
mu90 = 2.3
T_years = 5
T = T_years * 365.25 * 24 * 3600          # seconds
dOmega = 2 * np.pi                         # sr

# ------------------------------
# Per-point bin widths (ΔE) via geometric-mean edges
# Edge_i between E_i-1 and E_i is sqrt(E_{i-1} * E_i).
# For endpoints, extrapolate using neighbor ratios in log-space.
# ------------------------------
edges = np.zeros(len(E_GeV) + 1)
# interior edges
for i in range(1, len(E_GeV)):
    edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
# endpoints (symmetric in log-space)
edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

# per-bin widths in log10(E) and linear ΔE
log_edges = np.log10(edges)
dlog10E_i = log_edges[1:] - log_edges[:-1]
dE_i = E_GeV * np.log(10.0) * dlog10E_i    # GeV

# ------------------------------
# Convert E^2 Phi -> Phi (per-energy flux)
# ------------------------------
Phi_A = E2Phi_A / (E_GeV ** 2)             # cm^-2 s^-1 sr^-1 GeV^-1
Phi_B = E2Phi_B / (E_GeV ** 2)

# ------------------------------
# Effective area (A_eff ≈ mu90 / (T ΔΩ Φ ΔE))
# Compute in cm^2 then convert to m^2 (1 m^2 = 1e4 cm^2)
# ------------------------------
Aeff_A_cm2 = mu90 / (T * dOmega * Phi_A * dE_i)
Aeff_B_cm2 = mu90 / (T * dOmega * Phi_B * dE_i)
Aeff_A_m2 = Aeff_A_cm2 / 1e4
Aeff_B_m2 = Aeff_B_cm2 / 1e4

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(7,5))
plt.loglog(E_GeV, Aeff_A_m2, marker='o', label='RET-N $A_{\\mathrm{eff}}$ (Lower Limit)')
plt.loglog(E_GeV, Aeff_B_m2, marker='s', label='RET-N $A_{\\mathrm{eff}}$ (Upper Limit)')
plt.xlabel(r'Energy $E_{\mathrm{geo}}$ [GeV]')
plt.ylabel(r'Effective Area [m$^2$]')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.title('RET-N effective area vs. energy (90% C.L., 5 yr)')
plt.tight_layout()
plt.savefig('retn_effective_area.png', dpi=300)
plt.show()

# ------------------------------
# Print table
# ------------------------------
hdr = f"{'E_geo [GeV]':>15} | {'Δlog10E':>9} | {'A_eff_A [m^2]':>14} | {'A_eff_B [m^2]':>14}"
print(hdr)
print("-" * len(hdr))
for e, dlog, aA, aB in zip(E_GeV, dlog10E_i, Aeff_A_m2, Aeff_B_m2):
    print(f"{e:15.3e} | {dlog:9.3f} | {aA:14.3e} | {aB:14.3e}")
