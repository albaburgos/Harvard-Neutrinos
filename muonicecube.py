#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re, math, pandas as pd
from collections import defaultdict

raw_track_bins = ['10e2 - 9e3', '1.5e2 - 1.2e4', '2e2 - 1.8e4', '2.1e2 - 2.2e4', '3e2 - 3e4', '3.5e2- 4e4', '4.5e2 - 5.6e4', '5.5e2 - 6e4', '7e2 - 7.5e4', '9e2 - 7e4', '1.2e3 - 6e4', '1.5e3 - 5.2e4', '2e3 - 4e4', '2.1e3 - 3e4', '3e3 - 2e4', '3.5e3 - 1.5e4', '4.5e3 - 1e4', '6e3 - 6.5e3', '7e3 - 4.5e3', '9e3 - 3e3', '1.2e4 - 2e3', '1.5e4 - 1.5e3', '2e4 - 1e3', '2e4 - 7e2', '3e4 - 4.5e2', '3.8e4 - 3.5e2', '4.5e4 - 2e2', '6e4 - 1.5e2', '7e4 - 7e1', '9e4 - 6e1', '1e5 - 4e1', '1.5e5 - 3e1', '1.8e5 - 2e1', '2e5 - 1e1', '3e5 - 3.5e0', '3.5e5 - 7e0', '4.5e5 - 4e0', '6e5 - 3e0', '7e5 - 3.4e0', '9e5 - 2e0', '1.2e6 - 1e0', '4.5e6 - 1e0']

def parse_number(s: str) -> float:
    s = s.strip().lower().replace(" ", "")
    return float(s)

# Parse
pairs = []
for entry in raw_track_bins:
    entry = re.sub(r"\s*-\s*", " - ", entry.strip())
    parts = [p.strip() for p in entry.split(" - ") if p.strip()]
    if len(parts) != 2:
        continue
    e_low = parse_number(parts[0])
    N = parse_number(parts[1])
    pairs.append((e_low, N))

# Aggregate duplicate lower edges
agg = defaultdict(float)
for e, n in pairs:
    agg[e] += n

E_low_unique = np.array(sorted(agg.keys()), dtype=float)
N_unique = np.array([agg[e] for e in E_low_unique], dtype=float)

# Bin edges and centers
edges = np.concatenate([E_low_unique, [E_low_unique[-1] * 1.2]])
E_center = np.sqrt(edges[:-1] * edges[1:])
delta_log10E = np.diff(np.log10(edges))

# Parameters
years = 9.0
seconds_per_year = 365.25 * 24 * 3600.0
T = years * seconds_per_year   # seconds
sr = 2*np.pi                     # adjust if needed

# Effective area
A_eff = N_unique * E_center / (sr * T * math.log(10.0) * delta_log10E)

# Save CSV
import pandas as pd
df = pd.DataFrame({
    "E_low": edges[:-1],
    "E_high": edges[1:],
    "E_center": E_center,
    "N": N_unique,
    "delta_log10E": delta_log10E,
    "A_eff": A_eff,
})
df.to_csv("eff_area.csv", index=False)

# Plot
plt.figure(figsize=(8,6))
plt.loglog(E_center, A_eff, marker='o')
plt.xlabel("Energy (same units as input)")
plt.ylabel("Effective Area")
plt.title("Effective Area vs Energy (T = 9 years)")
plt.grid(True, which="both")
plt.savefig("eff_area.png", dpi=200, bbox_inches="tight")
plt.show()