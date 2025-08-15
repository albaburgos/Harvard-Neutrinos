
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
    """Log-log linear interpolation with safe handling for nonpositive values (returns 0)."""
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
    """Trim array to last strictly-positive point (inclusive)."""
    pos = np.where(Y > 0)[0]
    if pos.size == 0:
        return E[:0], Y[:0]
    last = pos[-1]
    return E[:last+1], Y[:last+1]

def make_plot(icecube_csv_path="friday.csv", out_png="friday.png"):
    # =========================
    # Radio (RNO-G + Gen2 split; GRAND -> tau only)
    # =========================
    E_rnog_eV   = [3e16, 1e17, 3e17, 1e18, 4e18, 1e19, 4e19]
    E2Phi_rnog  = [2e-8,  9e-9,  6.5e-9, 5.5e-9, 5e-9,  6e-9,  8e-9]
    T_rnog      = 5.0; mu90_rnog   = 2.3; Omega_rnog  = 2*np.pi

    E_gen2_eV   = E_rnog_eV
    E2Phi_gen2  = [2e-9, 6e-10, 3e-10, 2.2e-10, 2e-10, 4e-10, 8e-10]
    T_gen2      = 10.0; mu90_gen2   = 2.3; Omega_gen2  = 2*np.pi

    E_grand_GeV   = [4.5e7, 1e8, 5e8, 1e9, 3e9, 1e10, 4e10, 7e10, 1e11]
    E2Phi_grand   = [4e-9, 2.7e-9, 7e-10, 7.5e-10, 1.2e-9, 3e-9, 9e-9, 0.5e-8, 2e-8]
    T_grand       = 3.0; mu90_grand = 2.44; Omega_grand = 2*np.pi

    E_rnog_GeV,  Aeff_rnog  = effective_area_from_sensitivity(
        E_rnog_eV, E2Phi_rnog, E_units="eV", T_years=T_rnog, mu90=mu90_rnog, DeltaOmega=Omega_rnog
    )
    E_gen2_GeV,  Aeff_gen2  = effective_area_from_sensitivity(
        E_gen2_eV, E2Phi_gen2, E_units="eV", T_years=T_gen2, mu90=mu90_gen2, DeltaOmega=Omega_gen2
    )
    E_grand_GeV, Aeff_grand = effective_area_from_sensitivity(
        E_grand_GeV, E2Phi_grand, E_units="GeV", T_years=T_grand, mu90=mu90_grand, DeltaOmega=Omega_grand
    )

    E_low  = min(min(E_rnog_GeV), min(E_gen2_GeV), min(E_grand_GeV))
    E_high = max(max(E_rnog_GeV), max(E_gen2_GeV), max(E_grand_GeV))
    E_grid = np.logspace(np.log10(E_low), np.log10(E_high), 30)

    A_rnog_on = log_interp(E_grid, E_rnog_GeV,  Aeff_rnog)
    A_gen2_on = log_interp(E_grid, E_gen2_GeV,  Aeff_gen2)
    A_grand_on= log_interp(E_grid, E_grand_GeV, Aeff_grand)

    # Fractions CSV with fallbacks
    if not os.path.exists(icecube_csv_path):
        for alt in ["/mnt/data/friday.csv", "/mnt/data/radioscaling.csv", "/mnt/data/icecube.csv", "/mnt/data/energy_flavor_fractions.csv"]:
            if os.path.exists(alt):
                icecube_csv_path = alt
                break
        else:
            raise FileNotFoundError("Scaling CSV not found.")

    df_frac = pd.read_csv(icecube_csv_path)
    req = {"Energy (GeV)", "electron fraction", "muon fraction", "taon fraction"}
    if not req.issubset(df_frac.columns):
        raise ValueError("CSV must have: Energy (GeV), electron fraction, muon fraction, taon fraction")

    E_scale = df_frac["Energy (GeV)"].to_numpy(float)
    fe  = df_frac["electron fraction"].to_numpy(float)
    fmu = df_frac["muon fraction"].to_numpy(float)
    ft  = df_frac["taon fraction"].to_numpy(float)
    s = fe + fmu + ft; s[s==0] = 1.0
    fe, fmu, ft = fe/s, fmu/s, ft/s

    logE_scale = np.log10(E_scale)
    def nearest_fracs(E):
        idx = np.abs(logE_scale - np.log10(E)).argmin()
        return fe[idx], fmu[idx], ft[idx]

    fe_g = np.zeros_like(E_grid); fmu_g = np.zeros_like(E_grid); ft_g = np.zeros_like(E_grid)
    for i,E in enumerate(E_grid):
        e_i,m_i,t_i = nearest_fracs(E); fe_g[i], fmu_g[i], ft_g[i] = e_i, m_i, t_i

    A_e = A_rnog_on*fe_g + A_gen2_on*fe_g
    A_m = A_rnog_on*fmu_g + A_gen2_on*fmu_g
    A_t = A_rnog_on*ft_g  + A_gen2_on*ft_g + A_grand_on

    # Trim zeros for e & mu
    E_e_plot, A_e_plot = _trim_trailing_zeros(E_grid, A_e)
    E_m_plot, A_m_plot = _trim_trailing_zeros(E_grid, A_m)
    E_t_plot, A_t_plot = E_grid, A_t

    # =========================
    # Additional experiments
    # =========================
    # POEMMA
    log10E_poemma = np.array([7, 7.2, 7.6, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])
    E_poemma_GeV  = 10**log10E_poemma
    E2Phi_poemma  = [3e-6, 5e-8, 1e-8, 8e-9, 6e-9, 7e-9, 2e-8, 2.4e-8, 7e-8]
    T_poemma      = 5
    mu90_poemma   = 2.3
    Omega_poemma  = 4*np.pi/12
    E_po_GeV, A_po = effective_area_from_sensitivity(
        E_poemma_GeV, E2Phi_poemma, E_units="GeV", T_years=T_poemma, mu90=mu90_poemma, DeltaOmega=Omega_poemma
    )

    # Trinity (tau-like)
    E_trinity_GeV = [1e5, 4e5, 1e6, 4e6, 1e7, 7e7, 4e8, 3e9, 1e10, 4e10]
    E2Phi_trinity = [2e-7, 3e-8, 1e-8, 2e-9, 1e-9, 5e-10, 7e-10, 3e-9, 8e-9, 4e-8]
    T_trinity     = 3.0
    mu90_trinity  = 2.3
    Omega_trinity = 2*np.pi
    E_tr_GeV, A_tr = effective_area_from_sensitivity(
        E_trinity_GeV, E2Phi_trinity, E_units="GeV", T_years=T_trinity, mu90=mu90_trinity, DeltaOmega=Omega_trinity
    )

    # RET-N (band)
    RETN_data = [
        (6e19, 3e-9, 2.6e-9),
        (1e19, 1.6e-9, 1.0e-9),
        (1e18, 0.8e-9, 4e-10),
        (4.5e17, 4e-9, 3.5e-10),
        (2e17, 6e-9, 4.5e-10),
        (5e16, 1e-8, 1e-9),
        (2e16, 4e-8, 1.6e-9),
    ]
    E_retn_eV     = [r[0] for r in RETN_data]
    E2Phi_retn_hi = [r[1] for r in RETN_data]
    E2Phi_retn_lo = [r[2] for r in RETN_data]
    T_retn        = 3.0
    mu90_retn     = 2.3
    Omega_retn    = 2*np.pi

    E_retn_hi_GeV, A_retn_hi = effective_area_from_sensitivity(
        E_retn_eV, E2Phi_retn_hi, E_units="eV", T_years=T_retn, mu90=mu90_retn, DeltaOmega=Omega_retn
    )
    E_retn_lo_GeV, A_retn_lo = effective_area_from_sensitivity(
        E_retn_eV, E2Phi_retn_lo, E_units="eV", T_years=T_retn, mu90=mu90_retn, DeltaOmega=Omega_retn
    )

    # TAMBO: densify points by log-log interpolation
    tambo_E_GeV_base = np.array([1e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
    tambo_tau_ap_m2sr_base = np.array([1, 50, 100, 500, 600, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
    tambo_all_ap_m2sr_base = np.array([1, 50, 100, 500, 700, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)
    omega_tambo = 2.0
    tambo_tau_Aeff_base = tambo_tau_ap_m2sr_base / omega_tambo
    tambo_all_Aeff_base = tambo_all_ap_m2sr_base / omega_tambo
    # Dense energy sampling (more points)
    tambo_E_GeV = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 40)
    tambo_tau_Aeff_m2 = log_interp(tambo_E_GeV, tambo_E_GeV_base, tambo_tau_Aeff_base)
    tambo_all_Aeff_m2 = log_interp(tambo_E_GeV, tambo_E_GeV_base, tambo_all_Aeff_base)
    tambo_el_Aeff_m2  = np.abs(tambo_all_Aeff_m2 - tambo_tau_Aeff_m2)

    # IceCube (Aeff for mu/e/tau)
    final_rows = np.array([
        [109.999,232.892,160.05588745185227,164.45467039632751,0.0,0.0,4,0,4],
        [232.892,493.079,338.8718850362184,489.2658968675564,0.0,0.0,2,0,2],
        [493.079,1043.947,717.4596454247444,459.2992274601041,0.01189710689723954,0.007494240565190263,3,2,3],
        [1043.947,2210.245,1519.0058054579645,1029.7815972148696,0.09820435879202606,0.06186101341229988,4,2,4],
        [2210.245,4679.532,3216.0398326730965,388.02745820737346,0.11638625576328972,0.07331417685876518,2,1,2],
        [4679.532,9907.507,6809.001104914289,379.07703660299165,0.4670681596373116,0.2942161635510625,4,2,4],
        [9907.507,20976.177,14416.019577565057,89.81852572248584,0.8190008125850179,0.5159060236756019,3,2,3],
        [20976.177,44410.769,30521.60138737994,100.32566788993606,0.42674960446539567,0.268818648482139,2,1,2],
        [44410.769,94026.493,64620.34401411925,124.36472512537776,0.9791502429015384,0.6167875545836463,4,2,4],
        [94026.493,199072.921,136814.21202819556,24.11496663524563,0.9791502429015384,0.6167875545836463,1,0,1],
        [199072.921,421477.252,289663.09342181147,17.537609156303414,1.0301254335306331,0.6488979108854382,3,2,3],
        [421477.252,892351.772,613274.7937750992,9.736213119412295,1.1353159319934158,0.7151596422005769,3,2,3],
        [892351.772,1889287.455,1298425.588274746,8.065149146252075,1.1353159319934158,0.7151596422005769,1,0,1],
        [1889287.455,4000000.0,2749027.067891475,7.562326904682501,1.1353159319934158,2.26869807140475,1,0,1]
    ], dtype=float)
    E_final_GeV = final_rows[:,2]
    ice_mu  = final_rows[:,3]
    ice_el  = final_rows[:,4]
    ice_tau = final_rows[:,5]

    # ========= Plot =========
    fig, ax = plt.subplots(figsize=(11,7.5))
    ax.set_xscale('log'); ax.set_yscale('log')

    # Color scheme
    ORANGE = '#ff7f0e'   # tau family
    BLUE   = '#1f77b4'   # electron family
    GREEN  = '#2ca02c'   # muon family
    PINK_EDGE = '#e91e63'
    PINK_FILL = '#f8bbd0'

    # Markers for tau datasets
    mk_radio_tau = 'o'
    mk_trinity   = 'x'
    mk_poemma    = 's'
    mk_ic_tau    = 'D'

    mk_radio_e   = 'o'
    mk_ic_e      = 'D'
    mk_tambo_e   = '^'

    mk_radio_mu  = 'o'
    mk_ic_mu     = 'D'

    # Radio e (blue)
    if len(E_e_plot) > 0:
        ax.plot(E_e_plot, A_e_plot, color=BLUE, label="Radio Electron o")
        ax.scatter(E_e_plot, A_e_plot, color=BLUE, marker=mk_radio_e, s=18)
        ax.scatter([E_e_plot[-1]], [A_e_plot[-1]], color=BLUE, marker=mk_radio_e, s=24)
    # Radio mu (green)
    if len(E_m_plot) > 0:
        ax.plot(E_m_plot, A_m_plot, color=GREEN, label="Radio Muon o")
        ax.scatter(E_m_plot, A_m_plot, color=GREEN, marker=mk_radio_mu, s=18)
        ax.scatter([E_m_plot[-1]], [A_m_plot[-1]], color=GREEN, marker=mk_radio_mu, s=24)
    # Radio tau (orange)
    ax.plot(E_t_plot, A_t_plot, color=ORANGE, label="Radio Tau o")
    ax.scatter(E_t_plot, A_t_plot, color=ORANGE, marker=mk_radio_tau, s=18)
    ax.scatter([E_t_plot[-1]], [A_t_plot[-1]], color=ORANGE, marker=mk_radio_tau, s=24)

    # POEMMA (orange)
    E_po_GeV, A_po  # already computed
    ax.plot(E_po_GeV, A_po, color=ORANGE, label="POEMMA Tau (square) 5yr")
    ax.scatter(E_po_GeV, A_po, color=ORANGE, marker=mk_poemma, s=20)

    # Trinity (orange)
    E_tr_GeV, A_tr  # already computed
    ax.plot(E_tr_GeV, A_tr, color=ORANGE, label="Trinity Tau x 3yr")
    ax.scatter(E_tr_GeV, A_tr, color=ORANGE, marker=mk_trinity, s=20)

    # RET-N (pink band)
    ax.fill_between(E_retn_hi_GeV, np.minimum(A_retn_hi, A_retn_lo), np.maximum(A_retn_hi, A_retn_lo),
                    color=PINK_FILL, alpha=0.25, label="RET-N band (fill between) 5yr")
    ax.plot(E_retn_hi_GeV, A_retn_hi, color=PINK_EDGE)
    ax.plot(E_retn_lo_GeV, A_retn_lo, color=PINK_EDGE)
    ax.scatter(E_retn_hi_GeV, A_retn_hi, color=PINK_EDGE, s=14)
    ax.scatter(E_retn_lo_GeV, A_retn_lo, color=PINK_EDGE, s=14)

    # TAMBO (more points): Tau (orange), Electron (blue)
    ax.plot(tambo_E_GeV, tambo_all_Aeff_m2, color=ORANGE, label="TAMBO Tau ^ 10yr")
    ax.scatter(tambo_E_GeV, tambo_all_Aeff_m2, color=ORANGE, marker=mk_tambo_e, s=18)
    ax.plot(tambo_E_GeV, tambo_el_Aeff_m2,  color=BLUE,   label="TAMBO Electron ^ 10yr")
    ax.scatter(tambo_E_GeV, tambo_el_Aeff_m2, color=BLUE, marker=mk_tambo_e, s=18)

    # IceCube: delete FIRST tau point
    ax.plot(E_final_GeV, ice_mu, color=GREEN, label="IceCube Muon D")
    ax.scatter(E_final_GeV, ice_mu, color=GREEN, marker=mk_ic_mu, s=16)
    ax.plot(E_final_GeV, ice_el, color=BLUE, label="IceCube Electron D")
    ax.scatter(E_final_GeV, ice_el, color=BLUE, marker=mk_ic_e, s=16)
    ax.plot(E_final_GeV[1:], ice_tau[1:], color=ORANGE, label="IceCube Tau D")
    ax.scatter(E_final_GeV[1:], ice_tau[1:], color=ORANGE, marker=mk_ic_tau, s=16)

    ax.set_xlabel("Neutrino energy E [GeV]")
    ax.set_ylabel("Effective Area [m$^2$]")
    ax.set_title("Combined Radio Curves + POEMMA, Trinity, RET-N, IceCube, TAMBO")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    return out_png

if __name__ == "__main__":
    csv_candidates = [ "radioscaling.csv", "/mnt/data/icecube.csv", "/mnt/data/energy_flavor_fractions.csv"]
    for c in csv_candidates:
        if os.path.exists(c):
            csv_use = c
            break
    else:
        raise FileNotFoundError("No scaling CSV found.")
    print(make_plot(csv_use))
