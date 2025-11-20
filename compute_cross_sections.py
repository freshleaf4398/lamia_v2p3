import numpy as np
import matplotlib.pyplot as plt
import mplcursors

def compute_cross_sections(
    lambda_interp,
    eps_bg,
    luas_ff,
    faceNorm_ff,
    E_farfield_sca,
    H_farfield_sca,
    E_farfield_inc,
    H_farfield_inc,
    E_farfield_total,
    H_farfield_total
):
    """
    Compute scattering, absorption, and extinction cross-sections.

    Parameters
    ----------
    lambda_interp : (N,) array
        Interpolated wavelengths [nm].
    luas_ff : (M,) array
        Area of each far-field face.
    faceNorm_ff : (M,3) array
        Normal vectors for each far-field face.
    E_farfield_sca, H_farfield_sca : (N,M,3) arrays
        Scattered E and H fields for each wavelength and face.
    E_farfield_inc, H_farfield_inc : (N,M,3) arrays
        Incident E and H fields.
    E_farfield_total, H_farfield_total : (N,M,3) arrays
        Total E and H fields.
    imp_bg : float
        Background impedance (default 377 Ω for vacuum).

    Returns
    -------
    C_ext, C_sca, C_abs : (N,) arrays
        Extinction, scattering, and absorption cross-sections.
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """

    h_bar_eV = 6.582119e-16
    eps_0 = 8.854e-12
    mu_0 = 1.257e-6
    
    # Illumination normalization
    imp_bg = np.sqrt(mu_0 / (eps_0 * eps_bg))
    illumination = 1 / (2 * imp_bg)

    # Allocate outputs
    Nw = len(lambda_interp)
    C_sca = np.zeros(Nw)
    C_abs = np.zeros(Nw)
    C_ext = np.zeros(Nw)

    for a in range(Nw):
        E_sca = E_farfield_sca[a]
        H_sca = H_farfield_sca[a]
        E_tot = E_farfield_total[a]
        H_tot = H_farfield_total[a]
        E_inc = E_farfield_inc[a]
        H_inc = H_farfield_inc[a]

        # Poynting vectors
        S_sca = 0.5 * np.real(np.cross(E_sca, np.conj(H_sca)))
        S_abs = 0.5 * np.real(np.cross(E_tot, np.conj(H_tot)))
        S_ext = 0.5 * np.real(
            np.cross(E_inc, np.conj(H_tot)) + np.cross(E_tot, np.conj(H_inc))
        )

        # Integrals over far-field mesh
        sum_sca = luas_ff * np.einsum("ij,ij->i", S_sca, faceNorm_ff)
        sum_abs = luas_ff * np.einsum("ij,ij->i", S_abs, faceNorm_ff)
        sum_ext = luas_ff * np.einsum("ij,ij->i", S_ext, faceNorm_ff)

        # Integrated powers
        P_sca = np.sum(sum_sca)
        P_abs = -np.sum(sum_abs)
        P_ext = -np.sum(sum_ext)

        # Cross-sections
        C_sca[a] = P_sca / illumination
        C_abs[a] = P_abs / illumination
        C_ext[a] = P_ext / illumination

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    line1, = ax.plot(lambda_interp, C_ext, label="Extinction cross section", linewidth=2)
    line2, = ax.plot(lambda_interp, C_sca, label="Scattering cross section", linewidth=2)
    line3, = ax.plot(lambda_interp, C_abs, label="Absorption cross section", linewidth=2)

    ax.set_title("Optical Cross Sections", weight="bold", fontsize=14)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel(r"$C_{\mathrm{sca}}$", fontsize=12)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Add hover cursor
    cursor = mplcursors.cursor([line1, line2, line3], hover=True)

    return {
        "C_ext": C_ext,
        "C_sca": C_sca,
        "C_abs": C_abs,
    }