import numpy as np
import scipy
from scipy.special import lpmv
import matplotlib.pyplot as plt
from tqdm import trange


def multipole_decomp_tmat(
    eps_bg,
    lambda_interp,
    N_multipole,
    titik_tengah_ff,
    faceNorm_ff,
    luas_ff,
    E_farfield_sca,
    H_farfield_sca,
    E_farfield_inc
):
    
    """
    Perform multipole decomposition and compute T-matrix.

    Parameters
    ----------
    eps_bg : float
        Background permittivity.
    lambda_interp : (N,) array
        Wavelengths [nm].
    N_multipole : int
        Maximum multipole order.
    titik_tengah_ff : (M,3) array
        Triangle centers in far-field mesh.
    faceNorm_ff : (M,3) array
        Face normals of far-field mesh.
    E_farfield_sca, H_farfield_sca : (N,M,3) arrays
        Scattered far-field E and H.
    E_farfield_inc : (N,M,3) arrays
        Incident far-field E.
    luas_ff : (M,) array
        Areas of far-field triangles.

    Returns
    -------
    C_sca_cont_E : (N, N_multipole) array
        Electric multipole scattering contributions.
    C_sca_cont_M : (N, N_multipole) array
        Magnetic multipole scattering contributions.
    T_matrix : (N, P, P) array
        T-matrix at each wavelength.
        P = 2 * N_multipole * (2*N_multipole+1) // 2 approx.
    """

    # --- Physical constants ---
    eV_interp = 1240.0 / lambda_interp
    h_bar_eV = 6.582119e-16
    omega = eV_interp / h_bar_eV
    k = 2 * np.pi * np.sqrt(eps_bg) / lambda_interp
    eps_0 = 8.854e-12
    mu_0 = 1.257e-6
    imp_bg = np.sqrt(mu_0 / (eps_0 * eps_bg))
    c = 1 / np.sqrt(eps_0 * mu_0)

    # Coordinates
    titik_x = titik_tengah_ff[:, 0]
    titik_y = titik_tengah_ff[:, 1]
    titik_z = titik_tengah_ff[:, 2]
    jumSegitiga_ff = len(titik_x)
    
    # Spherical coordinate conversion
    r = np.sqrt(titik_x**2 + titik_y**2 + titik_z**2)
    r2 = np.sqrt(titik_x**2 + titik_y**2)
    sin_theta = r2 / r
    cos_theta = titik_z / r
    tan_theta = r2 / titik_z
    sin_phi = titik_y / r2
    cos_phi = titik_x / r2
    tan_phi = sin_phi / cos_phi
    phi = np.arctan2(sin_phi, cos_phi)
    theta = np.arctan2(sin_theta, cos_theta)
    
    # Prepare temp arrays
    temp = np.zeros_like(phi)
    temp2 = np.zeros((len(phi), 3))
    
    # Number of wavelengths
    N_wavelengths = E_farfield_sca.shape[0]
    m_ = np.arange(-N_multipole, N_multipole + 1)
    jum_m = len(m_)
    jum_n = (jum_m - 1) // 2
    a_mn = np.zeros((N_wavelengths, jum_n, jum_m), dtype=complex)
    b_mn = np.zeros((N_wavelengths, jum_n, jum_m), dtype=complex)
    a_mn_i = np.zeros((N_wavelengths, jum_n, jum_m), dtype=complex)
    b_mn_i = np.zeros((N_wavelengths, jum_n, jum_m), dtype=complex)
    
    for hai in trange(N_wavelengths, desc = "Multipole decomposition - wavelength"):
    # for hai in [0]:
        # Electric field (E) components
        E = E_farfield_sca[hai]  # shape (N_points, 3)
        E_i = E_farfield_inc[hai]
        E_sca_rho = sin_theta * cos_phi * E[:, 0] + \
                    sin_theta * sin_phi * E[:, 1] + \
                    cos_theta * E[:, 2]
        E_inc_rho = sin_theta * cos_phi * E_i[:, 0] + \
                sin_theta * sin_phi * E_i[:, 1] + \
                cos_theta * E_i[:, 2]
        E_sca_theta = cos_theta * cos_phi * E[:, 0] + \
                      cos_theta * sin_phi * E[:, 1] - \
                      sin_theta * E[:, 2]
        E_inc_theta = cos_theta * cos_phi * E_i[:, 0] + \
                  cos_theta * sin_phi * E_i[:, 1] - \
                  sin_theta * E_i[:, 2]
        E_sca_phi = -sin_phi * E[:, 0] + cos_phi * E[:, 1]
        E_inc_phi = -sin_phi * E_i[:, 0] + cos_phi * E_i[:, 1]
        E_sca_sph = np.stack([E_sca_rho, E_sca_theta, E_sca_phi], axis=1)
        E_inc_sph = np.stack([E_inc_rho, E_inc_theta, E_inc_phi], axis=1)
    
        for n in range(jum_n):  
            for u in range(jum_m): 
                m = m_[u]
                sum_atas_M = 0.0
                sum_atas_N = 0.0
                sum_atas_M_i = 0.0
                sum_atas_N_i = 0.0
                sum_bawah_M = 0.0
                sum_bawah_N = 0.0
                
                for a in range(len(phi)):
                    var_r = r[a]
                    var_phi = phi[a]
                    var_cos_theta = cos_theta[a]
                    var_sin_theta = sin_theta[a]
                    var_rho = k[hai] * var_r
    
                    n_pyth = n + 1 # remove confusion of order to python 
                    
                    # Compute spherical Hankel function and its derivative
                    f_sphhankel_1 = np.sqrt(np.pi / (2 * var_rho)) * scipy.special.hankel1(n_pyth+0.5, var_rho)
                    f_sphhankel_1_prev = np.sqrt(np.pi / (2 * var_rho)) * scipy.special.hankel1(n_pyth-1+0.5, var_rho)
                    f_sphhankel_1_next = np.sqrt(np.pi / (2 * var_rho)) * scipy.special.hankel1(n_pyth+1+0.5, var_rho)
                    d_sphhankel_1 = f_sphhankel_1 + var_rho * (-f_sphhankel_1 / (2 * var_rho) + (f_sphhankel_1_prev - f_sphhankel_1_next) / 2) # CHECK HERE, NOT SAME AS MATLAB
                        
                    # Associated Legendre polynomial and derivative
                    f_ass_legendre = lpmv(m, n_pyth, var_cos_theta)
                    legendre_next = lpmv(m, n_pyth + 1, var_cos_theta)
                    
                    d_legendre = -1 / var_sin_theta * ((n_pyth + 1) * var_cos_theta * f_ass_legendre + (m - n_pyth - 1) * legendre_next)
        
                    # Construct vec_M and vec_N
                    ert = 1j * m * f_ass_legendre / var_sin_theta * f_sphhankel_1 * np.exp(1j * m * var_phi)
                    ert2 = -d_legendre * f_sphhankel_1 * np.exp(1j * m * var_phi)
                    vec_M = np.array([0, ert[0], ert2[0]])
    
                    ert3 = n_pyth * (n_pyth + 1) * f_ass_legendre * f_sphhankel_1 * np.exp(1j * m * var_phi) / var_rho
                    ert4 = d_legendre * d_sphhankel_1 * np.exp(1j * m * var_phi) / var_rho
                    ert5 = 1j * m * f_ass_legendre / var_sin_theta * d_sphhankel_1 * np.exp(1j * m * var_phi) / var_rho
                    vec_N = np.array([ert3[0], ert4[0], ert5[0]])
        
                    # Project E_sca onto vec_M and vec_N
                    atas_M = np.vdot(E_sca_sph[a,:], np.conj(vec_M))
                    atas_N = np.vdot(E_sca_sph[a,:], np.conj(vec_N))
                    atas_M_i = np.vdot(E_inc_sph[a,:], np.conj(vec_M))
                    atas_N_i = np.vdot(E_inc_sph[a,:], np.conj(vec_N))
                    
                    sum_atas_M += atas_M
                    sum_atas_N += atas_N
                    sum_atas_M_i += atas_M_i
                    sum_atas_N_i += atas_N_i
                    sum_bawah_M += np.linalg.norm(vec_M) ** 2
                    sum_bawah_N += np.linalg.norm(vec_N) ** 2
    
                # Normalization constant
                if (n_pyth + m) < 0 or (n_pyth - m) < 0:
                    continue
                En = (1j)**(n_pyth + 2 * m - 1) * np.sqrt((2 * n_pyth + 1) * scipy.special.factorial(n_pyth - m) /
                                                    scipy.special.factorial(n_pyth + m)) / (2 * np.sqrt(np.pi))
    
                a_mn[hai, n, u] = sum_atas_N / (sum_bawah_N * En)
                b_mn[hai, n, u] = sum_atas_M / (sum_bawah_M * En)
                a_mn_i[hai, n, u] = sum_atas_N_i / (sum_bawah_N * En)
                b_mn_i[hai, n, u] = sum_atas_M_i / (sum_bawah_M * En)
    
    T_matrix = np.zeros((N_wavelengths, 2*jum_n*jum_m, 2*jum_n*jum_m), dtype=complex)
    for hai in trange(N_wavelengths, desc = "T-matrix calculation - wavelength"):
        sca_coef = np.column_stack([a_mn[0,:,:],b_mn[0,:,:]]).flatten().reshape(-1, 1)
        inc_coef = np.column_stack([a_mn_i[0,:,:],b_mn_i[0,:,:]]).flatten().reshape(-1, 1)
        T_matrix[hai,:,:] = sca_coef @ np.linalg.pinv(inc_coef)  

    # --- Bohren scattering ---
    lala_E = np.zeros_like(a_mn, dtype=float)
    lala_M = np.zeros_like(b_mn, dtype=float)
    for i in range(jum_n):
        for j in range(jum_m):
            factor = (i + 1) * (i + 2)
            lala_E[:, i, j] = factor * (np.abs(a_mn[:, i, j]) ** 2)
            lala_M[:, i, j] = factor * (np.abs(b_mn[:, i, j]) ** 2)

    # --- Cross section contributions ---
    lala2 = np.zeros_like(a_mn, dtype=float)
    lala3 = np.zeros_like(a_mn, dtype=float)
    for i in range(jum_n):
        for j in range(jum_m):
            factor = 2*(i+1)+1
            lala2[:,i,j] = factor*(np.abs(a_mn[:,i,j])**2 + np.abs(b_mn[:,i,j])**2)
            lala3[:,i,j] = factor*np.real(a_mn[:,i,j]+b_mn[:,i,j])

    C_sca_bohren_total = 2*np.pi*np.sum(np.sum(lala2,axis=2),axis=1) / (k.flatten()**2)

    C_sca_cont_E = np.sum(lala_E, axis=2) / (k**2)
    C_sca_cont_M = np.sum(lala_M, axis=2) / (k**2)

    # --- Plot
    col = plt.cm.hsv(np.linspace(0,1,jum_n))
    plt.figure(figsize=(10,6))
    plt.plot(lambda_interp*1e9, C_sca_bohren_total, label="c_sca multipole", linewidth=2)
    for i in range(jum_n):
        plt.plot(lambda_interp*1e9, C_sca_cont_E[:,i], label=f"E_{i+1}", linewidth=2)
        plt.plot(lambda_interp*1e9, C_sca_cont_M[:,i], "--", label=f"M_{i+1}", linewidth=2)
    plt.xlabel(r"$\lambda$ (nm)")
    plt.ylabel("Cross section")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "C_sca_cont_E": C_sca_cont_E,
        "C_sca_cont_M": C_sca_cont_M,
        "C_sca_bohren_total": C_sca_bohren_total,
        "T_matrix": T_matrix,
    }