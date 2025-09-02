
## defining the flux calculation

import numpy as np

def neutrino_flux(E,
                  alpha="e",
                  gamma=2.5,
                  f_source=(1/3, 2/3, 0.0),   # (f_e,S, f_mu,S, f_tau,S), sum=1
                  zmax=4.0,
                  nz=400,
                  cosmo=(0.315, 0.685),       # (Omega_m, Omega_L); flat ΛCDM
                  evo=(3.4, 0.3, -3.5, 5000, 9, -10)):  
    """
    Return the diffuse flux Φ_α(E) for flavor α ∈ {"e","mu","tau"}.
    Implements:
      E^2 Φ_α(E) = ∫_0^{zmax} dz [ ρ_src(z) / (h(z) (1+z)^2) ] [E(1+z)]^{2-γ}
                   × ∑_β P_{β→α} f_{β,S}
    using the decohered limit P_{αβ} = Σ_i |U_{αi}|^2 |U_{βi}|^2.

    Args:
      E: scalar or array-like energy (GeV).
      alpha: detected flavor ("e","mu","tau").
      gamma: source spectral index.
      f_source: source flavor fractions (f_e,S, f_mu,S, f_tau,S).
      zmax, nz: redshift upper limit and number of z points for the integral.
      cosmo: (Ω_m, Ω_Λ).
      evo: (a,b,c,B,C,η) parameters of ρ_src(z).

    Returns:
      Φ_α(E) with the same shape as E (arbitrary overall normalization).
    """
    # --- setup ---
    E = np.atleast_1d(np.asarray(E, dtype=float))
    Om, OL = cosmo
    a, b, c, B, C, eta = evo
    alpha = alpha.lower()
    flv_to_i = {"e": 0, "mu": 1, "tau": 2}
    if alpha not in flv_to_i:
        raise ValueError("alpha must be 'e', 'mu', or 'tau'")
    ia = flv_to_i[alpha]
    fS = np.asarray(f_source, dtype=float)

    # --- PMNS (standard) and decohered probabilities ---
    th12, th23, th13, dcp = np.deg2rad([33.44, 49.0, 8.57, 195.0])
    s12, c12 = np.sin(th12), np.cos(th12)
    s23, c23 = np.sin(th23), np.cos(th23)
    s13, c13 = np.sin(th13), np.cos(th13)
    U = np.array([
        [c12*c13,                s12*c13,                s13*np.exp(-1j*dcp)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*dcp),
          c12*c23 - s12*s23*s13*np.exp(1j*dcp),          s23*c13],
        [ s12*s23 - c12*c23*s13*np.exp(1j*dcp),
         -c12*s23 - s12*c23*s13*np.exp(1j*dcp),          c23*c13]
    ], dtype=complex)
    A = np.abs(U)**2
    P = A @ A.T  # rows=detected α, cols=source β
    weight = P[ia, :] @ fS  # ∑_β P_{β→α} f_{β,S}  (P is symmetric here)

    # --- redshift integral (vectorized over E via broadcasting) ---
    z = np.linspace(0.0, zmax, nz)
    h = np.sqrt(Om*(1+z)**3 + OL)                         # h(z)=H(z)/H0
    rho = ((1+z)**(a*eta) + ((1+z)/B)**(b*eta) + ((1+z)/C)**(c*eta))**(1.0/eta)
    kernel = rho / (h * (1.0 + z)**2)                     # ρ_src / [h (1+z)^2]
    Eprime = E[:, None] * (1.0 + z)[None, :]              # shape (nE, nz)
    integrand = kernel[None, :] * (Eprime ** (2.0 - gamma)) * weight
    E2Phi = np.trapz(integrand, z, axis=1)                # shape (nE,)
    Phi = E2Phi / (E**2 + 1e-300)

    return Phi[0] if Phi.size == 1 else Phi
