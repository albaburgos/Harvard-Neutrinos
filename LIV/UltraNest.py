#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build three HEALPix NSIDE=1 (12-pixel) maps for μ, e, τ from 10-bin
counts at μ=cos(zenith) centers [-0.9, -0.7, ..., 0.9].
Counts are the sum of DoubleCasc + Casc + Tracks per flavor.

θ = arccos(-μ).  φ is chosen by a strategy:
  - 'uniform' (default): azimuthal symmetry → distribute equally across ring pixels
  - 'zero'              : set φ=0 for each bin and assign to the resulting pixel
  - 'random'            : random φ per bin; use --seed for reproducibility
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# --------------------------
# Input data (10 μ-bins)
# --------------------------
mu_doublecasc = np.array([1e-2, 1.5e-2, 5e-2, 7e-2, 6e-2, 8e-2, 7e-2, 5e-2, 7e-2, 7e-2], dtype=float)
e_doublecasc  = np.array([5e-2, 2e-2, 4e-2, 5e-2, 6e-2, 1e-1, 1e-1, 1.5e-1, 2e-1, 3e-1], dtype=float)
tau_doublecasc= np.array([1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 5e-1, 6e-1, 8e-1, 8e-1, 1e0], dtype=float)

mu_casc = np.array([6e0, 8e0, 1e1, 8e0, 8e0, 8e0, 7e0, 6e0, 4.5e0, 4e0], dtype=float)
e_casc  = np.array([5e1, 6e1, 6e1, 5.5e1, 5e1, 5e1, 5e1, 4e1, 3.5e1, 3e1], dtype=float)
tau_casc= np.array([3.5e1, 4e1, 4.5e1, 4e1, 4e1, 4e1, 3.5e1, 3e1, 2.5e1, 2e1], dtype=float)

mu_tracks  = np.array([4e1, 2e1, 2e1, 1.8e1, 1e1, 6e0, 2e0, 3e0, 3.5e0, 4e0], dtype=float)
e_tracks   = np.array([6e0, 1.5e0, 1.5e0, 1.6e0, 1e0, 1e0, 3e-1, 5e-1, 6e-1, 2e0], dtype=float)
tau_tracks = np.array([1e1, 4e0, 3e0, 3e0, 3e0, 1.7e0, 6e-1, 7e-1, 1e0, 2e0], dtype=float)

MU_CENTERS = np.array([-0.9, -0.7, -0.5, -0.3, -0.1,  0.1,  0.3,  0.5,  0.7,  0.9], dtype=float)

NSIDE = 1
NPIX  = 12

def combine_topologies():
    """Sum DoubleCasc + Casc + Tracks per flavor for the 10 bins."""
    mu_bins  = mu_doublecasc + mu_casc + mu_tracks
    e_bins   = e_doublecasc  + e_casc  + e_tracks
    tau_bins = tau_doublecasc + tau_casc + tau_tracks
    return mu_bins, e_bins, tau_bins

def theta_from_mu(mu):
    """θ (colatitude) from μ = cos(zenith) via θ = arccos(-μ)."""
    mu = np.clip(np.asarray(mu, dtype=float), -1.0, 1.0)
    return np.arccos(mu)


def bin_zone_indices(theta):
    """
    For each θ in the 10 bins, decide which zone it belongs to.
    Returns index arrays for the bins in (north, equatorial, south).
    """
    z = np.degrees(theta)
    north = np.where(z >  120)[0]
    south = np.where(z < 60)[0]
    equ   = np.where((z >= 60) & (z <= 120))[0]
    print (north, south, equ)
    return north, equ, south

def build_map_from_bins(bin_counts, theta_bins=None, nside=1):
    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=float)

    # Accumulators
    m_1 = 0.0
    m_2 = 0.0
    m_3 = 0.0

    # Sum bins into three groups
    for i, count in enumerate(bin_counts):
        c = float(count)
        if i in (0, 1, 2):
            m_1 += c
        elif i in (3, 4, 5, 6):
            m_2 += c
        else:
            m_3 += c

    # Distribute each group's total evenly over pixel groups
    groups = (
        ([0, 1, 2, 3], m_1),
        ([4, 5, 6, 7], m_2),
        ([8, 9, 10, 11], m_3),
    )

    for pix_idxs, total in groups:
        share = total / len(pix_idxs)
        for p in pix_idxs:
            if p < npix:  # guard in case nside is small
                m[p] = share

    return m

def plot_healpix_map(m, title, nest=False):
    plt.figure()
    hp.mollview(m, coord='C', flip='astro', title=title, nest=nest)
    # Overlay pixel boundaries + labels
    th_c, ph_c = hp.pix2ang(NSIDE, np.arange(NPIX), nest=nest)
    for p in range(NPIX):
        b = hp.boundaries(NSIDE, p, step=1, nest=nest)   # shape (3, n)
        th_b, ph_b = hp.vec2ang(b.T)
        lon = np.degrees(ph_b); lat = 90 - np.degrees(th_b)
        hp.projplot(lon, lat, '-', lonlat=True, linewidth=0.8)
        hp.projtext(np.degrees(ph_c[p]), 90 - np.degrees(th_c[p]), str(p), lonlat=True)
        hp.graticule(dpar=30, dmer=60, alpha=0.4)

def main():

    mu_bins, e_bins, tau_bins = combine_topologies()
    theta_bins = theta_from_mu(MU_CENTERS)

    # Build three maps
    mu_map  = build_map_from_bins(mu_bins,  theta_bins)
    e_map   = build_map_from_bins(e_bins,   theta_bins)
    tau_map = build_map_from_bins(tau_bins, theta_bins)

    base_angle_counts1 = mu_map[4] + e_map[4] + tau_map[4]
    print(base_angle_counts1)

    plot_healpix_map(mu_map,  f"μ (all topologies) — counts per NSIDE=1 pixel ]")
    plot_healpix_map(e_map,   f"e (all topologies) — counts per NSIDE=1 pixel]")
    plot_healpix_map(tau_map, f"τ (all topologies) — counts per NSIDE=1 pixel]")
    plt.show()

if __name__ == "__main__":
    main()
