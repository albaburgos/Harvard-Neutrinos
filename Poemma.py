import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Input: POEMMA sensitivity data (all-flavor, 2 sr-sorted)
# ==========================
# E_geo in GeV, E^2 Phi in GeV cm^-2 s^-1 sr^-1
Egeo = np.array([2.2e7, 7.0e7, 1.0e8, 4.0e8, 9.0e8, 3.0e9, 1.0e10, 2.0e10])
E2Phi = np.array([1.0e-6, 1.5e-7, 2.0e-7, 7.0e-8, 8.0e-8, 1.2e-7, 3.0e-7, 8.0e-7])

# ==========================
# Constants (90% C.L., background-free)
# ==========================
mu90 = 2.44                     # Feldman–Cousins (0 obs)
T_years = 3                     # <-- change if your exposure differs
T = T_years * 365.25 * 24 * 3600  # seconds
dOmega = 2.0                    # sr (per your "2 sr" note)

# ==========================
# Bin width in log10(E) — keep behavior of your original script
# ==========================
log10E = np.log10(Egeo)
dlog10E = np.diff(log10E).mean()

# ==========================
# Convert E^2 Phi -> Phi (per-energy flux)
# ==========================
Phi = E2Phi / (Egeo**2)         # cm^-2 s^-1 sr^-1 GeV^-1

# ==========================
# Compute ΔE for log-spaced bins
# ==========================
dE = Egeo * np.log(10.0) * dlog10E  # GeV

# ==========================
# Effective area (cm^2) -> (m^2)
# A_eff ≈ μ90 / (T ΔΩ Φ ΔE)
# ==========================
Aeff_cm2 = mu90 / (T * dOmega * Phi * dE)
Aeff_m2  = Aeff_cm2 / 1e4               # 1 m^2 = 10^4 cm^2

# ==========================
# Plot
# ==========================
plt.figure(figsize=(7,5))
plt.loglog(Egeo, Aeff_m2, marker='o', label='POEMMA All-flavor $A_{\\mathrm{eff}}$ (2 sr)')
plt.xlabel(r'Energy $E_{\mathrm{geo}}$ [GeV]')
plt.ylabel(r'Effective Area [m$^2$]')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.title("POEMMA effective area vs. energy")
plt.tight_layout()
plt.savefig("poemma_effective_area.png", dpi=300)
plt.show()

# ==========================
# Print results table
# ==========================
print(f"{'E_geo [GeV]':>15} | {'E^2 Phi':>15} | {'A_eff [m^2]':>15}")
print("-"*52)
for e, f2, a in zip(Egeo, E2Phi, Aeff_m2):
    print(f"{e:15.2e} | {f2:15.3e} | {a:15.3e}")
