
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def effective_area_from_sensitivity(E_pts, E2Phi_pts, *,
                                    E_units="GeV",
                                    T_years,
                                    mu90=2.3,
                                    DeltaOmega=2*np.pi):
    E = np.asarray(E_pts, dtype=float)
    E2Phi = np.asarray(E2Phi_pts, dtype=float)

    if E_units.lower() == "ev":
        E_GeV = E / 1e9
    else:
        E_GeV = E.copy()

    order = np.argsort(E_GeV)
    E_GeV = E_GeV[order]
    E2Phi = E2Phi[order]

    edges = np.zeros(len(E_GeV) + 1)
    for i in range(1, len(E_GeV)):
        edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
    edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
    edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

    dlog10E = np.log10(edges[1:]) - np.log10(edges[:-1])
    dE = E_GeV * np.log(10.0) * dlog10E

    Phi = E2Phi / (E_GeV**2)

    T_sec = T_years * 365.25 * 24 * 3600.0

    Aeff_cm2 = mu90 / (T_sec * DeltaOmega * Phi * dE)
    Aeff_m2 = Aeff_cm2 / 1e4
    return E_GeV, Aeff_m2

def log_interp(x, xp, fp):
    xp = np.asarray(xp); fp = np.asarray(fp)
    mask = (xp > 0) & (fp > 0)
    xp = xp[mask]; fp = fp[mask]
    x = np.asarray(x, dtype=float)
    if len(xp) < 2:
        return np.zeros_like(x)
    xlog  = np.log10(x)
    xplog = np.log10(xp)
    fplog = np.log10(fp)
    ylog = np.interp(xlog, xplog, fplog, left=np.nan, right=np.nan)
    y = 10**ylog
    y[np.isnan(y)] = 0.0
    return y

def _trim_trailing_zeros(E, Y):
    """Return E,Y up to the last strictly-positive Y value. Empty if none positive."""
    pos = np.where(Y > 0)[0]
    if pos.size == 0:
        return E[:0], Y[:0]
    last = pos[-1]
    return E[:last+1], Y[:last+1]

def make_radio_plot(icecube_csv_path="radio_all.csv",
                    out_png="radio_all.png"):
    # --- RNO-G (90% C.L., T=5 yr), energies in eV
    E_rnog_eV   = [3e16, 1e17, 3e17, 1e18, 4e18, 1e19, 4e19]
    E2Phi_rnog  = [2e-8,  9e-9,  6.5e-9, 5.5e-9, 5e-9,  6e-9,  8e-9]
    T_rnog      = 5.0; mu90_rnog   = 2.3; Omega_rnog  = 2*np.pi

    # --- IceCube-Gen2 Radio (90% C.L., T=10 yr), energies in eV
    E_gen2_eV   = E_rnog_eV
    E2Phi_gen2  = [2e-9, 6e-10, 3e-10, 2.2e-10, 2e-10, 4e-10, 8e-10]
    T_gen2      = 10.0; mu90_gen2   = 2.3; Omega_gen2  = 2*np.pi

    # --- GRAND 200k (90% C.L., T=3 yr, μ90=2.44), energies in GeV
    E_grand_GeV   = [4.5e7, 1e8, 5e8, 1e9, 3e9, 1e10, 4e10, 7e10, 1e11]
    E2Phi_grand   = [4e-9, 2.7e-9, 7e-10, 7.5e-10, 1.2e-9, 3e-9, 9e-9, 0.5e-8, 2e-8]
    T_grand       = 3.0; mu90_grand = 2.44; Omega_grand = 2*np.pi

    # Effective areas
    E_rnog_GeV,  Aeff_rnog  = effective_area_from_sensitivity(
        E_rnog_eV, E2Phi_rnog, E_units="eV", T_years=T_rnog, mu90=mu90_rnog, DeltaOmega=Omega_rnog
    )
    E_gen2_GeV,  Aeff_gen2  = effective_area_from_sensitivity(
        E_gen2_eV, E2Phi_gen2, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
    )
    E_grand_GeV, Aeff_grand = effective_area_from_sensitivity(
        E_grand_GeV, E2Phi_grand, E_units="GeV", T_years=T_grand, mu90=mu90_grand, DeltaOmega=Omega_grand
    )

    # Common energy grid
    E_low  = min(min(E_rnog_GeV), min(E_gen2_GeV), min(E_grand_GeV))
    E_high = max(max(E_rnog_GeV), max(E_gen2_GeV), max(E_grand_GeV))
    E_grid = np.logspace(np.log10(E_low), np.log10(E_high), 30)

    # Interpolate to grid
    A_rnog_on = log_interp(E_grid, E_rnog_GeV,  Aeff_rnog)
    A_gen2_on = log_interp(E_grid, E_gen2_GeV,  Aeff_gen2)
    A_grand_on= log_interp(E_grid, E_grand_GeV, Aeff_grand)

    # Load flavor fractions
    if not os.path.exists(icecube_csv_path):
        raise FileNotFoundError(f"Scaling CSV not found: {icecube_csv_path}")
    df_frac = pd.read_csv(icecube_csv_path)
    req = {"Energy (GeV)", "electron fraction", "muon fraction", "taon fraction"}
    if not req.issubset(df_frac.columns):
        raise ValueError("CSV must have: Energy (GeV), electron fraction, muon fraction, taon fraction")

    E_scale = df_frac["Energy (GeV)"].to_numpy(float)
    fe  = df_frac["electron fraction"].to_numpy(float)
    fmu = df_frac["muon fraction"].to_numpy(float)
    ft  = df_frac["taon fraction"].to_numpy(float)

    s = fe + fmu + ft
    s[s==0] = 1.0
    fe, fmu, ft = fe/s, fmu/s, ft/s

    logE_scale = np.log10(E_scale)
    def nearest_fracs(E):
        idx = np.abs(logE_scale - np.log10(E)).argmin()
        return fe[idx], fmu[idx], ft[idx]

    fe_g = np.zeros_like(E_grid)
    fmu_g= np.zeros_like(E_grid)
    ft_g = np.zeros_like(E_grid)
    for i,E in enumerate(E_grid):
        e_i, m_i, t_i = nearest_fracs(E)
        fe_g[i], fmu_g[i], ft_g[i] = e_i, m_i, t_i

    # Apply fractions: RNO-G + Gen2 split; GRAND -> tau only
    A_e = A_rnog_on*fe_g + A_gen2_on*fe_g
    A_m = A_rnog_on*fmu_g + A_gen2_on*fmu_g
    A_t = A_rnog_on*ft_g  + A_gen2_on*ft_g + A_grand_on

    # --- Trim trailing zeros for electron and muon curves ---
    E_e_plot, A_e_plot = _trim_trailing_zeros(E_grid, A_e)
    E_m_plot, A_m_plot = _trim_trailing_zeros(E_grid, A_m)
    # tau unchanged
    E_t_plot, A_t_plot = E_grid, A_t

    # Scatter graph connected with lines (log-log)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xscale('log'); ax.set_yscale('log')

    if len(E_e_plot) > 0:
        ax.plot(E_e_plot, A_e_plot, label="Radio Electron Aeff")
        ax.scatter(E_e_plot, A_e_plot)
        ax.scatter([E_e_plot[-1]], [A_e_plot[-1]])
    if len(E_m_plot) > 0:
        ax.plot(E_m_plot, A_m_plot, label="Radio Muon Aeff")
        ax.scatter(E_m_plot, A_m_plot)
        ax.scatter([E_m_plot[-1]], [A_m_plot[-1]])
    ax.plot(E_t_plot, A_t_plot, label="Radio Taon Aeff")
    ax.scatter(E_t_plot, A_t_plot)
    ax.scatter([E_t_plot[-1]], [A_t_plot[-1]])

    ax.set_xlabel("Neutrino energy E [GeV]")
    ax.set_ylabel("Effective Area [m$^2$]")
    ax.set_title("Radio HE and UHE Effective Areas")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    return out_png

if __name__ == "__main__":
    candidates = [
        "radioscaling.csv",
        "/mnt/data/radioscaling.csv",
        "/mnt/data/icecube.csv",
        "/mnt/data/energy_flavor_fractions.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break
    else:
        raise FileNotFoundError("Could not find radioscaling.csv, icecube.csv, or energy_flavor_fractions.csv")
    png = make_radio_plot(csv_path)
    print("Saved plot:", png)
