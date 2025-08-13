## Trinity Data

import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Input: Trinity sensitivity data
# ==========================
# Egeo in GeV, E^2 Phi in GeV cm^-2 s^-1 sr^-1
Egeo = np.array([1e5, 4e5, 1e6, 4e6, 1e7, 7e7, 4e8, 3e9, 1e10, 4e10])
E2Phi = np.array([2e-7, 3e-8, 1e-8, 2e-9, 1e-9, 5e-10, 7e-10, 3e-9, 8e-9, 4e-8])

# ==========================
# Constants for Trinity full apparatus
# ==========================
mu90 = 2.3                  # background-free 90% C.L. (Feldman–Cousins, 0 obs)
T_years = 3
T = T_years * 365.25 * 24 * 3600  # seconds
dOmega = 2 * np.pi           # sr (full hemisphere acceptance)
dlog10E = 0.5                # assumed bin width in log10(E)

# ==========================
# Conversion: E^2 Phi -> Phi
# ==========================
Phi = E2Phi / (Egeo**2)  # cm^-2 s^-1 sr^-1 GeV^-1

# ==========================
# Compute ΔE for each bin
# ==========================
dE = Egeo * np.log(10) * dlog10E  # GeV

# ==========================
# Effective area in cm^2 -> m^2
# ==========================
Aeff_cm2 = mu90 / (T * dOmega * Phi * dE)
Aeff_m2 = Aeff_cm2 / 1e4  # 1 m^2 = 10^4 cm^2

# ==========================
# Plot

plt.figure(figsize=(7,5))
plt.loglog(Egeo, Aeff_m2, marker='o', label=r'Trinity Taon $A_{\mathrm{eff}}$')
plt.xlabel(r'Energy $E_{\mathrm{geo}}$ [GeV]')
plt.ylabel(r'Effective Area [m$^2$]')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.title("Trinity Taon effective area vs. energy")
plt.tight_layout()

# Save as PNG
plt.savefig("trinity_taon_effective_area.png", dpi=300)
plt.show()
