import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1) Data (from your message)
# ----------------------------
E_PeV = np.arange(1, 11, dtype=float)
E = E_PeV * 1e6  # GeV

A_ebar = np.array([4.5e1, 1.5e2, 3.2e2, 5.0e2, 6.3e2, 8.5e2, 1.0e3, 1.5e3, 2.0e3, 2.2e3])   # purple → ν̄e
A_tau  = np.array([4.5e1, 1.5e2, 3.2e2, 5.0e2, 6.3e2, 2.8e3, 4.0e3, 3.0e3, 2.0e3, 2.2e3])   # black → ντ

# -----------------------------------------
# 2) Plot (log–log)
# -----------------------------------------
plt.figure(figsize=(8,6))
plt.plot(E, A_ebar, marker='o', label=r'$\bar{\nu}_e$ (purple)')
plt.plot(E, A_tau,  marker='o', label=r'$\nu_\tau$ (black)')
plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'Neutrino energy $E_\nu$ [GeV]')
plt.ylabel(r'Aperture [m$^2$ sr]')
plt.title('Aperture vs Energy (1–10 PeV)')
plt.grid(True, which='both', ls='--', lw=0.6)
plt.legend()
plt.tight_layout()

# Create folder and save plot
save_folder = "plots"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "aperture_vs_energy.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved to: {save_path}")

# ------------------------------------------------------
# 3) Expected counts and relative expected count
# ------------------------------------------------------
phi0   = 1.0
E0     = 1e6
gamma  = 2.0
T      = 1.0

dE = np.empty_like(E)
dE[0]  = (E[1] - E[0]) / 2.0
dE[-1] = (E[-1] - E[-2]) / 2.0
dE[1:-1] = (E[2:] - E[:-2]) / 2.0

phi = phi0 * (E / E0)**(-gamma)

N_ebar = np.sum(phi * A_ebar * dE * T)
N_tau  = np.sum(phi * A_tau  * dE * T)

relative = N_tau / N_ebar if N_ebar > 0 else np.nan

print("=== Expected Counts (arbitrary normalization) ===")
print(f"N(ν̄e)  = {N_ebar:.6e}")
print(f"N(ντ)   = {N_tau:.6e}")
print(f"Relative N(ντ) / N(ν̄e) = {relative:.6f}")
