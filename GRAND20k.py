import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Input: GRAND 200k sensitivity data
# ==========================
# E_geo in GeV, E^2 Phi in GeV cm^-2 s^-1 sr^-1 (all-flavor, 2π-sorted)
Egeo = np.array([4.5e7, 1e8, 5e8, 1e9, 3e9, 1e10, 4e10, 7e10, 1e11])
E2Phi = np.array([4e-9, 2.7e-9, 7e-10, 7.5e-10, 1.2e-9, 3e-9, 9e-9, 1.5e-8, 2e-8])

# ==========================
# Constants (90% C.L., background-free)
# ==========================
mu90 = 2.44                               # Feldman–Cousins (0 obs), as given
T_years = 3
T = T_years * 365.25 * 24 * 3600             # seconds
dOmega = 2 * np.pi                            # sr (hemisphere)

# ==========================
# Bin width in log10(E)
# (Use average spacing of provided points; replace with exact binning if you know it)
# ==========================
log10E = np.log10(Egeo)
dlog10E = np.diff(log10E).mean()

# ==========================
# Convert E^2 Phi -> Phi (per-energy flux)
# ==========================
Phi = E2Phi / (Egeo**2)                       # cm^-2 s^-1 sr^-1 GeV^-1

# ==========================
# Compute ΔE for log-spaced bins
# ==========================
dE = Egeo * np.log(10.0) * dlog10E            # GeV

# ==========================
# Effective area (cm^2) -> (m^2)
# A_eff ≈ μ90 / (T ΔΩ Φ ΔE)
# ==========================
Aeff_cm2 = mu90 / (T * dOmega * Phi * dE)
Aeff_m2 = Aeff_cm2 / 1e4                      # 1 m^2 = 10^4 cm^2

# ==========================
# Plot
# ==========================
plt.figure(figsize=(7,5))
plt.loglog(Egeo, Aeff_m2, marker='o', label='GRAND 200k All-flavor $A_{\\mathrm{eff}}$')
plt.xlabel(r'Energy $E_{\mathrm{geo}}$ [GeV]')
plt.ylabel(r'Effective Area [m$^2$]')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.title("GRAND 200k effective area vs. energy")
plt.tight_layout()
plt.savefig("grand200k_effective_area.png", dpi=300)
plt.show()

# ==========================
# Print results table
# ==========================
print(f"{'E_geo [GeV]':>15} | {'A_eff [m^2]':>15}")
print("-"*33)
for e, a in zip(Egeo, Aeff_m2):
    print(f"{e:15.2e} | {a:15.3e}")
