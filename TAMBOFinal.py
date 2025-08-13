import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Energies in GeV
energies = np.array([
    0.3e5, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6,
    4e6, 1e7, 4e7, 1e8, 1e9
], dtype=float)

# Taon apertures (m^2 sr)
taon = np.array([
    3,
    4.5*10,
    1.5*10**2,
    3.2*10**2,
    5*10**2,
    6.3*10**2,
    8.5*10**2,
    10**3,
    1.5*10**3,
    2*10**3,
    2.2*10**3,
    9*10**3,
    2*10**4,
    4*10**4,
    5*10**4,
    7*10**4
], dtype=float)

# All flavour apertures (m^2 sr)
all_flavour = np.array([
    3,
    4.5*10,
    1.5*10**2,
    3.2*10**2,
    5*10**2,
    6.3*10**3,
    2.8*10**4,
    4*10**4,
    3*10**4,
    2*10**3,
    2.2*10**3,
    9*10**3,
    2*10**4,
    4*10**4,
    5*10**4,
    7*10**4
], dtype=float)

# Electron = All flavour - Taon
electron = all_flavour - taon

# Convert to effective area (m²) by dividing by 2π sr
omega = 2 * np.pi
taon_eff = taon / omega
electron_eff = electron / omega

# Optional: save table to CSV
df = pd.DataFrame({
    "Energy_GeV": energies,
    "Taon_m2_sr": taon,
    "AllFlavour_m2_sr": all_flavour,
    "Electron_m2_sr": electron,
    "Taon_Effective_m2": taon_eff,
    "Electron_Effective_m2": electron_eff
})
df.to_csv("effective_area_table.csv", index=False)

# Save Taon Effective Area PNG
plt.figure()
plt.plot(energies, taon_eff, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy (GeV)")
plt.ylabel("Effective Area (m$^2$)")
plt.title("TAMBO Taon Effective Area vs Energy (Ω = 2π sr)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("taon_effective_area.png", dpi=200)
plt.close()

# Save Electron Effective Area PNG
plt.figure()
plt.plot(energies, electron_eff, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy (GeV)")
plt.ylabel("Effective Area (m$^2$)")
plt.title("TAMBO Electron Effective Area vs Energy (Ω = 2π sr)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("electron_effective_area.png", dpi=200)
plt.close()

print("Saved: taon_effective_area.png, electron_effective_area.png, effective_area_table.csv")
