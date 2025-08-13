import numpy as np
import matplotlib.pyplot as plt

# ----- TAMBO data -----
# Energies in GeV
energies = np.array([
    1e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9
], dtype=float)

# Apertures (m^2 sr)
taon = np.array([
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

all_flavour = np.array([
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

# Convert to effective area by dividing by 2 sr
omega = 2.0  # sr
taon_eff = taon / omega
all_eff = all_flavour / omega

# Plot both on one log-log chart
plt.figure()
plt.plot(energies, taon_eff, marker='o', label='All flavor (effective area)')
plt.plot(energies, all_eff, marker='s', label='Taon signal (effective area)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy (GeV)')
plt.ylabel('Effective Area (m$^2$)')
plt.title('Tambo effective area')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('tambo_effective_area.png', dpi=200)
plt.close()

print('Saved tambo_effective_area.png')
