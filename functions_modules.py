import numpy as np
from scipy.constants import mu_0, epsilon_0, c
from scipy.interpolate import interp1d
import pdb

def get_wavelength_parameters(eV_val, n_re_val, n_im_val, eps_bg):
    # Physical constants
    h_bar_eV = 6.582119e-16  # Planck constant [eV·s]
    eps_0 = 8.854e-12        # Vacuum permittivity [F/m]
    mu_0 = 1.257e-6          # Vacuum permeability [H/m]

    omega = eV_val / h_bar_eV
    lamda = 1240e-9 / eV_val  # in meters

    # Material permittivity (complex dielectric function)
    eps_r = (n_re_val**2 - n_im_val**2) + 1j * 2 * n_re_val * n_im_val

    c = 1 / np.sqrt(eps_0 * eps_bg * mu_0)

    # Wavevectors
    k_vec = np.array([0, 0, 1])
    E_0 = np.array([0, 1, 0])  # polarization direction

    k_bg = 2 * np.pi * np.sqrt(eps_bg) / lamda
    kv_bg = k_bg * k_vec

    k = 2 * np.pi * np.sqrt(eps_r) / lamda
    kv = k * k_vec

    # Impedances
    imp_bg = np.sqrt(mu_0 / (eps_0 * eps_bg))
    imp = np.sqrt(mu_0 / (eps_0 * eps_r))

    return {
        'omega': omega,
        'lamda': lamda,
        'eps_r': eps_r,
        'eps_bg': eps_bg,
        'c': c,
        'k': k,
        'kv': kv,
        'k_bg': k_bg,
        'kv_bg': kv_bg,
        'imp': imp,
        'imp_bg': imp_bg,
        'k_vec': k_vec,
        'E_0': E_0,
        'mu_0': mu_0,
        'eps_0': eps_0,
    }

def compute_greens_functions(titik_tengah, k, k_bg):
    """
    Compute Green's function matrices (scalar, smooth, gradients) for source and observation points.
    """
    pi = np.pi
    obs = titik_tengah[np.newaxis, :, :]         # shape: (N, 1, 3)
    source = titik_tengah[:, np.newaxis, :]      # shape: (1, N, 3)
    
    diff = obs - source                          # shape: (N, N, 3)
    
    R = np.linalg.norm(diff, axis=2)             # shape: (N, N)
    logicR = R == 0                               # diagonal terms
    logicR_3D = np.repeat(logicR[:, :, np.newaxis], 3, axis=2)

    # Scalar Green's functions
    greenskalar = np.exp(1j * k * R) / (4 * pi * R)
    greenskalar_bg = np.exp(1j * k_bg * R) / (4 * pi * R)

    # Smooth Green's functions
    greensmooth = ((np.exp(1j * k * R) - 1) / R + (k**2) * R / 2) / (4 * pi)
    greensmooth_bg = ((np.exp(1j * k_bg * R) - 1) / R + (k_bg**2) * R / 2) / (4 * pi)

    # Gradients of scalar Green's functions
    grad_scalar = (
        (1j * k * np.exp(1j * k * R) / (4 * pi * R**2))[..., None] * diff
        - (np.exp(1j * k * R) / (4 * pi * R**3))[..., None] * diff
    )
    grad_scalar_bg = (
        (1j * k_bg * np.exp(1j * k_bg * R) / (4 * pi * R**2))[..., None] * diff
        - (np.exp(1j * k_bg * R) / (4 * pi * R**3))[..., None] * diff
    )

    # Gradients of smooth Green's functions
    grad_smooth = (
        (k**2 / (8 * pi * R))[..., None] * diff +
        (1j * k * np.exp(1j * k * R) / (4 * pi * R**2))[..., None] * diff -
        ((np.exp(1j * k * R) - 1) / (4 * pi * R**3))[..., None] * diff
    )
    grad_smooth_bg = (
        (k_bg**2 / (8 * pi * R))[..., None] * diff +
        (1j * k_bg * np.exp(1j * k_bg * R) / (4 * pi * R**2))[..., None] * diff -
        ((np.exp(1j * k_bg * R) - 1) / (4 * pi * R**3))[..., None] * diff
    )

    # Handle singularities (diagonal elements)
    greensmooth[logicR] = (1j * k) / (4 * pi)
    greensmooth_bg[logicR] = (1j * k_bg) / (4 * pi)

    
    grad_smooth[logicR_3D] = 0
    grad_smooth_bg[logicR_3D] = 0

    return {
        'R': R,
        'greenskalar': greenskalar,
        'greenskalar_bg': greenskalar_bg,
        'greensmooth': greensmooth,
        'greensmooth_bg': greensmooth_bg,
        'gradientgreen': -grad_scalar,  # source-side gradient = negative
        'gradientgreen_bg': -grad_scalar_bg,
        'grad_greensmooth': -grad_smooth,
        'grad_greensmooth_bg': -grad_smooth_bg,
        'logicR': logicR,
        'logicR_3D': logicR_3D
    }

def compute_greens_functions_2d(grid_points_3d, titik_tengah, k, k_bg):
    """
    Compute Green's function matrices (scalar, smooth, gradients) for source and observation points.
    """
    pi = np.pi
    obs1 = grid_points_3d[:, :, 0]  # (51, 51)
    obs2 = grid_points_3d[:, :, 1]  # (51, 51)
    obs3 = grid_points_3d[:, :, 2]  # (51, 51)
    
    source1 = titik_tengah[:, 0].reshape(1, 1, titik_tengah.shape[0])  # (1, 1, 569)
    source2 = titik_tengah[:, 1].reshape(1, 1, titik_tengah.shape[0])
    source3 = titik_tengah[:, 2].reshape(1, 1, titik_tengah.shape[0])
    
    # Use broadcasting instead of bsxfun
    c1 = obs1[:, :, np.newaxis] - source1  # (51, 51, 569)
    c2 = obs2[:, :, np.newaxis] - source2  # (51, 51, 569)
    c3 = obs3[:, :, np.newaxis] - source3  # (51, 51, 569)
    
    R = np.sqrt(c1**2 + c2**2 + c3**2)
    logicR = R == 0                               # diagonal terms
    logicR_3D = np.transpose(np.repeat(logicR[:, :, np.newaxis], 3, axis=2),(0,1,3,2))
    
    # Smooth Green's functions
    greensmooth = ((np.exp(1j * k * R) - 1) / R + (k**2) * R / 2) / (4 * pi)
    greensmooth_bg = ((np.exp(1j * k_bg * R) - 1) / R + (k_bg**2) * R / 2) / (4 * pi)
    
    diff = np.stack((c1, c2, c3), axis=-1)
    
    grad_greensmooth = (
        (k**2 / (8 * pi * R))[..., None] * diff +
        (1j * k * np.exp(1j * k * R) / (4 * pi * R**2))[..., None] * diff -
        ((np.exp(1j * k * R) - 1) / (4 * pi * R**3))[..., None] * diff
    )
    
    grad_greensmooth_bg = (
        (k_bg**2 / (8 * pi * R))[..., None] * diff +
        (1j * k_bg * np.exp(1j * k_bg * R) / (4 * pi * R**2))[..., None] * diff -
        ((np.exp(1j * k_bg * R) - 1) / (4 * pi * R**3))[..., None] * diff
    )
    
    greensmooth[logicR] = (1j * k) / (4 * pi)
    greensmooth_bg[logicR] = (1j * k_bg) / (4 * pi)
    grad_greensmooth[logicR_3D] = 0
    grad_greensmooth_bg[logicR_3D] = 0

    return {
        'R': R,
        'greensmooth': greensmooth,
        'greensmooth_bg': greensmooth_bg,
        'grad_greensmooth': -grad_greensmooth,
        'grad_greensmooth_bg': -grad_greensmooth_bg,
        'logicR': logicR,
        'logicR_3D': logicR_3D
    }

def compute_greens_farfield_functions(titik_tengah_ff, titik_tengah, k, k_bg):
    """
    Compute Green's function matrices (scalar, smooth, gradients) for source and observation points.
    """
    pi = np.pi
    obs = titik_tengah_ff[:, None, :]         # shape: (N, 1, 3)
    source = titik_tengah[None, :, :]      # shape: (1, N, 3)
    diff = obs - source                          # shape: (N, N, 3)
    R = np.linalg.norm(diff, axis=2)             # shape: (N, N)
    
    greenskalar_bg = np.exp(1j * k_bg * R) / (4 * pi * R)
    grad_scalar_bg = (
        (1j * k_bg * np.exp(1j * k_bg * R) / (4 * pi * R**2))[..., None] * diff
        - (np.exp(1j * k_bg * R) / (4 * pi * R**3))[..., None] * diff
    )
    grad_scalar_source_bg = -grad_scalar_bg

    return {
        'R': R,
        'greenskalar_bg': greenskalar_bg,
        'gradientgreen_bg': grad_scalar_bg,
        'gradientgreen_source_bg' : grad_scalar_source_bg 
    }

def compute_incident_fields(
    PanjangSisi, rho_plus, rho_minus,
    luas, SegitigaPlus, SegitigaMinus,
    titik_tengah, kv_bg, k_vec, E_0,
    mu_0, c, eps_bg
):
    """
    Compute incident electric and magnetic field projections for surface integral equations.
    """
    N_edges = PanjangSisi.shape[0]

    # Basis functions and their divergence
    area_plus = luas[SegitigaPlus]
    area_minus = luas[SegitigaMinus]

    f_plus = (PanjangSisi[:, None] * rho_plus) / (2 * area_plus[:, None])
    f_minus = -(PanjangSisi[:, None] * rho_minus) / (2 * area_minus[:, None])
    div_f_plus = PanjangSisi / area_plus
    div_f_minus = -PanjangSisi / area_minus

    # E_inc on triangle centers
    dot_plus = np.einsum("ij,j->i", titik_tengah[SegitigaPlus], kv_bg)
    dot_minus = np.einsum("ij,j->i", titik_tengah[SegitigaMinus], kv_bg)

    EmPlus = np.exp(1j * dot_plus)[:, None] * E_0
    EmMinus = np.exp(1j * dot_minus)[:, None] * E_0

    q1 = (
        np.einsum("ij,ij->i", f_plus, EmPlus) * area_plus +
        np.einsum("ij,ij->i", f_minus, EmMinus) * area_minus
    )

    # H_inc = (k × E) / (μ₀ * c / √ε_bg)
    cross_k_E = np.cross(k_vec, E_0)
    HmPlus = np.exp(1j * dot_plus)[:, None] * cross_k_E / (mu_0 * c / np.sqrt(eps_bg))
    HmMinus = np.exp(1j * dot_minus)[:, None] * cross_k_E / (mu_0 * c / np.sqrt(eps_bg))

    q2 = (
        np.einsum("ij,ij->i", f_plus, HmPlus) * area_plus +
        np.einsum("ij,ij->i", f_minus, HmMinus) * area_minus
    )

    q = np.concatenate([q1, q2])[np.newaxis, :]  # shape: (1, 2*N_edges)
    
    return q, f_plus, f_minus, div_f_plus, div_f_minus

def get_logical_masks(SegitigaPlus, SegitigaMinus, logic_identical, logic_adjacent, logic_touch, logic_far):
    """
    Create boolean masks for different triangle relationship regions.
    """

    # Fetch masks for each pair type
    logic_identical_pp = logic_identical[np.ix_(SegitigaPlus, SegitigaPlus)]
    logic_identical_pm = logic_identical[np.ix_(SegitigaPlus, SegitigaMinus)]
    logic_identical_mp = logic_identical[np.ix_(SegitigaMinus, SegitigaPlus)]
    logic_identical_mm = logic_identical[np.ix_(SegitigaMinus, SegitigaMinus)]

    logic_adjacent_pp = logic_adjacent[np.ix_(SegitigaPlus, SegitigaPlus)]
    logic_adjacent_pm = logic_adjacent[np.ix_(SegitigaPlus, SegitigaMinus)]
    logic_adjacent_mp = logic_adjacent[np.ix_(SegitigaMinus, SegitigaPlus)]
    logic_adjacent_mm = logic_adjacent[np.ix_(SegitigaMinus, SegitigaMinus)]

    logic_touch_pp = logic_touch[np.ix_(SegitigaPlus, SegitigaPlus)]
    logic_touch_pm = logic_touch[np.ix_(SegitigaPlus, SegitigaMinus)]
    logic_touch_mp = logic_touch[np.ix_(SegitigaMinus, SegitigaPlus)]
    logic_touch_mm = logic_touch[np.ix_(SegitigaMinus, SegitigaMinus)]

    logic_far_pp = logic_far[np.ix_(SegitigaPlus, SegitigaPlus)]
    logic_far_pm = logic_far[np.ix_(SegitigaPlus, SegitigaMinus)]
    logic_far_mp = logic_far[np.ix_(SegitigaMinus, SegitigaPlus)]
    logic_far_mm = logic_far[np.ix_(SegitigaMinus, SegitigaMinus)]

    return {
        "identical": (logic_identical_pp, logic_identical_pm, logic_identical_mp, logic_identical_mm),
        "adjacent":  (logic_adjacent_pp, logic_adjacent_pm, logic_adjacent_mp, logic_adjacent_mm),
        "touch":     (logic_touch_pp, logic_touch_pm, logic_touch_mp, logic_touch_mm),
        "far":       (logic_far_pp, logic_far_pm, logic_far_mp, logic_far_mm),
    }

def compute_D_pp_region1(
    K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus, greenskalar_bg,
    greensmooth_bg, f_plus, div_f_plus,
    luas, SegitigaPlus,
    I_1_pp, I_2_pp, k_bg,
    logic_identical_pp, logic_far_pp
):
    """
    Compute plus-plus D-matrix region 1 (background region).
    """

    area = luas[SegitigaPlus]  # (N_edges,)
    N = len(SegitigaPlus)

    # Reshape required arrays
    K21 = K_2_1_plus[SegitigaPlus, :, :] / (4 * np.pi)
    K22 = K_2_2_plus[SegitigaPlus, :, :] * (-k_bg**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth_bg[np.ix_(SegitigaPlus, SegitigaPlus)] *
        area[None,:]
    )[:, :, None] * f_plus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_plus[:, None, :]  # shape (N, 1, 3)
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area[:, None]

    K11 = K_1_1_plus[SegitigaPlus, :] / (4 * np.pi)
    K12 = K_1_2_plus[SegitigaPlus, :] * (-k_bg**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth_bg[np.ix_(SegitigaPlus, SegitigaPlus)] *
        (div_f_plus * area)[None, :]
    )
    div_outer = (
        sum_2 * (div_f_plus * area)[:, None]
    ) / (k_bg**2)

    outer_pp = outer - div_outer

    #### If triangles are identical: use Arcioni singular correction
    # Arcioni Smooth + Singular 
    gs = greensmooth_bg[np.ix_(SegitigaPlus, SegitigaPlus)]
    int_smooth = (
        gs * area[None, :]
    )[:, :, None] * f_plus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area[:, None]

    int_smooth_div = gs * div_f_plus[None, :] * area[None, :]
    div_smooth = (int_smooth_div * div_f_plus[:, None] * area[:, None]) / (k_bg**2)

    outer_smooth = dot_smooth - div_smooth

    # Arcioni singular part
    Cmn = (div_f_plus[:, None] * div_f_plus[None, :]) / 4
    int_arcioni_div = (div_f_plus[:, None] * div_f_plus[None, :]) * I_1_pp / (4 * np.pi)
    int_arcioni = Cmn * I_2_pp / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k_bg**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_pp)
    outer_pp[rows, cols] = jum[rows, cols]

    #### If triangles are far: use Green scalar
    gs = greenskalar_bg[np.ix_(SegitigaPlus, SegitigaPlus)]
    int_gs = (gs * area[None, :])[:, :, None] * f_plus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area[:, None]

    int_gs_div = gs * div_f_plus[None, :] * area[None, :]
    div_gs = (int_gs_div * div_f_plus[:, None] * area[:, None]) / (k_bg**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_pp)
    outer_pp[rows, cols] = far_term[rows, cols]

    return outer_pp

def compute_D_pp_region2(
    K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus, greenskalar,
    greensmooth, f_plus, div_f_plus,
    luas, SegitigaPlus,
    I_1_pp, I_2_pp, k,
    logic_identical_pp, logic_far_pp
):
    """
    Compute plus-plus D-matrix region 2 (internal region).
    """

    area = luas[SegitigaPlus]
    N = len(SegitigaPlus)

    # Smooth field contribution
    K21 = K_2_1_plus[SegitigaPlus, :, :] / (4 * np.pi)
    K22 = K_2_2_plus[SegitigaPlus, :, :] * (-k**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth[np.ix_(SegitigaPlus, SegitigaPlus)] * area[None, :]
    )[:, :, None] * f_plus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_plus[:, None, :]  # shape (N, 1, 3)
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area[:, None]

    K11 = K_1_1_plus[SegitigaPlus, :] / (4 * np.pi)
    K12 = K_1_2_plus[SegitigaPlus, :] * (-k**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth[np.ix_(SegitigaPlus, SegitigaPlus)] * div_f_plus[None, :] * area[None, :]
    )
    div_outer = (sum_2 * (div_f_plus * area)[:, None]) / (k**2)

    outer_pp = outer - div_outer

    #### For identical regions, use Arcioni correction
    gs = greensmooth[np.ix_(SegitigaPlus, SegitigaPlus)]
    int_smooth = (
        gs * area[None, :]
    )[:, :, None] * f_plus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area[:, None]

    int_smooth_div = gs * div_f_plus[None, :] * area[None, :]
    div_smooth = (int_smooth_div * div_f_plus[:, None] * area[:, None]) / (k**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_plus[:, None] * div_f_plus[None, :]) / 4
    int_arcioni_div = (div_f_plus[:, None] * div_f_plus[None, :]) * I_1_pp / (4 * np.pi)
    int_arcioni = Cmn * I_2_pp / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_pp)
    outer_pp[rows, cols] = jum[rows, cols]

    # Far-field scalar Green version (same structure)
    gs = greenskalar[np.ix_(SegitigaPlus, SegitigaPlus)]
    int_gs = (gs * area[None, :])[:, :, None] * f_plus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area[:, None]

    int_gs_div = gs * div_f_plus[None, :] * area[None, :]
    div_gs = (int_gs_div * div_f_plus[:, None] * area[:, None]) / (k**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_pp)
    outer_pp[rows, cols] = far_term[rows, cols]

    return outer_pp

def compute_D_pm_region1(
    K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus, greenskalar_bg,
    greensmooth_bg, f_plus, div_f_plus, f_minus, div_f_minus,
    luas, SegitigaPlus, SegitigaMinus,
    I_1_pm, I_2_pm, k_bg,
    logic_identical_pm, logic_far_pm
):
    """
    Compute plus-minus D-matrix region 1 (background).
    """

    area_plus = luas[SegitigaPlus]
    area_minus = luas[SegitigaMinus]
    N = len(SegitigaPlus)

    # Smooth part
    K21 = K_2_1_minus[SegitigaPlus, :, :] / (4 * np.pi)
    K22 = K_2_2_minus[SegitigaPlus, :, :] * (-k_bg**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth_bg[np.ix_(SegitigaPlus, SegitigaMinus)] * area_minus[None, :]
    )[:, :, None] * f_minus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_plus[:, None, :]
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area_plus[:, None]

    K11 = K_1_1_minus[SegitigaPlus, :] / (4 * np.pi)
    K12 = K_1_2_minus[SegitigaPlus, :] * (-k_bg**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth_bg[np.ix_(SegitigaPlus, SegitigaMinus)] * div_f_minus[None, :] * area_minus[None, :]
    )
    div_outer = (sum_2 * (div_f_plus * area_plus)[:, None]) / (k_bg**2)

    outer_pm = outer - div_outer

    # Arcioni + Smooth correction (for identical regions)
    gs = greensmooth_bg[np.ix_(SegitigaPlus, SegitigaMinus)]
    int_smooth = (gs * area_minus[None, :])[:, :, None] * f_minus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area_plus[:, None]
    
    int_smooth_div = gs * div_f_minus[None, :] * area_minus[None, :]
    div_smooth = (int_smooth_div * div_f_plus[:, None] * area_plus[:, None]) / (k_bg**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_plus[:, None] * div_f_minus[None, :]) / 4
    int_arcioni_div = (div_f_plus[:, None] * div_f_minus[None, :]) * I_1_pm / (4 * np.pi)
    int_arcioni = Cmn * I_2_pm / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k_bg**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_pm)
    outer_pm[rows, cols] = jum[rows, cols]

    # Far-field uses scalar Green function (same structure)
    gs = greenskalar_bg[np.ix_(SegitigaPlus, SegitigaMinus)]
    int_gs = (gs * area_minus[None, :])[:, :, None] * f_minus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area_plus[:, None]

    int_gs_div = gs * div_f_minus[None, :] * area_minus[None, :]
    div_gs = (int_gs_div * div_f_plus[:, None] * area_plus[:, None]) / (k_bg**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_pm)
    outer_pm[rows, cols] = far_term[rows, cols]

    return outer_pm

def compute_D_pm_region2(
    K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus, greenskalar,
    greensmooth, f_plus, div_f_plus, f_minus, div_f_minus,
    luas, SegitigaPlus, SegitigaMinus,
    I_1_pm, I_2_pm, k,
    logic_identical_pm, logic_far_pm
):
    """
    Compute plus-minus D-matrix region 2 (internal).
    """

    area_plus = luas[SegitigaPlus]
    area_minus = luas[SegitigaMinus]
    N = len(SegitigaPlus)

    # Smooth field contribution
    K21 = K_2_1_minus[SegitigaPlus, :, :] / (4 * np.pi)
    K22 = K_2_2_minus[SegitigaPlus, :, :] * (-k**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth[np.ix_(SegitigaPlus, SegitigaMinus)] * area_minus[None, :]
    )[:, :, None] * f_minus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_plus[:, None, :]
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area_plus[:, None]

    K11 = K_1_1_minus[SegitigaPlus, :] / (4 * np.pi)
    K12 = K_1_2_minus[SegitigaPlus, :] * (-k**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth[np.ix_(SegitigaPlus, SegitigaMinus)] * div_f_minus[None, :] * area_minus[None, :]
    )
    div_outer = (sum_2 * (div_f_plus * area_plus)[:, None]) / (k**2)

    outer_pm = outer - div_outer

    # Arcioni + smooth correction for identical
    gs = greensmooth[np.ix_(SegitigaPlus, SegitigaMinus)]
    int_smooth = (gs * area_minus[None, :])[:, :, None] * f_minus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area_plus[:, None]

    int_smooth_div = gs * div_f_minus[None, :] * area_minus[None, :]
    div_smooth = (int_smooth_div * div_f_plus[:, None] * area_plus[:, None]) / (k**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_plus[:, None] * div_f_minus[None, :]) / 4
    int_arcioni_div = (div_f_plus[:, None] * div_f_minus[None, :]) * I_1_pm / (4 * np.pi)
    int_arcioni = Cmn * I_2_pm / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_pm)
    outer_pm[rows, cols] = jum[rows, cols]

    # Far-field scalar Green version (same structure)
    gs = greenskalar[np.ix_(SegitigaPlus, SegitigaMinus)]
    int_gs = (gs * area_minus[None, :])[:, :, None] * f_minus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area_plus[:, None]

    int_gs_div = gs * div_f_minus[None, :] * area_minus[None, :]
    div_gs = (int_gs_div * div_f_plus[:, None] * area_plus[:, None]) / (k**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_pm)
    outer_pm[rows, cols] = far_term[rows, cols]

    return outer_pm

def compute_D_mp_region1(
    K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus, greenskalar_bg,
    greensmooth_bg, f_plus, div_f_plus, f_minus, div_f_minus,
    luas, SegitigaPlus, SegitigaMinus,
    I_1_mp, I_2_mp, k_bg,
    logic_identical_mp, logic_far_mp
):
    """
    Compute minus–plus D-matrix region 1 (external).
    """

    area_plus = luas[SegitigaPlus]
    area_minus = luas[SegitigaMinus]
    N = len(SegitigaMinus)

    # Matrix terms
    K21 = K_2_1_plus[SegitigaMinus, :, :] / (4 * np.pi)
    K22 = K_2_2_plus[SegitigaMinus, :, :] * (-k_bg**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth_bg[np.ix_(SegitigaMinus, SegitigaPlus)] * area_plus[None, :]
    )[:, :, None] * f_plus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_minus[:, None, :]
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area_minus[:, None]

    K11 = K_1_1_plus[SegitigaMinus, :] / (4 * np.pi)
    K12 = K_1_2_plus[SegitigaMinus, :] * (-k_bg**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth_bg[np.ix_(SegitigaMinus, SegitigaPlus)] * div_f_plus[None, :] * area_plus[None, :]
    )
    div_outer = (sum_2 * (div_f_minus * area_minus)[:, None]) / (k_bg**2)

    outer_mp = outer - div_outer

    # Arcioni + smooth correction for identical
    gs = greensmooth_bg[np.ix_(SegitigaMinus, SegitigaPlus)]
    int_smooth = (gs * area_plus[None, :])[:, :, None] * f_plus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area_minus[:,None]

    int_smooth_div = gs * div_f_plus[None, :] * area_plus[None, :]
    div_smooth = (int_smooth_div * div_f_minus[:, None] * area_minus[:, None]) / (k_bg**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_minus[:, None] * div_f_plus[None, :]) / 4
    int_arcioni_div = (div_f_minus[:, None] * div_f_plus[None, :]) * I_1_mp / (4 * np.pi)
    int_arcioni = Cmn * I_2_mp / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k_bg**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_mp)
    outer_mp[rows, cols] = jum[rows, cols]

    # Far-field scalar Green version
    gs = greenskalar_bg[np.ix_(SegitigaMinus, SegitigaPlus)]
    int_gs = (gs * area_plus[None, :])[:, :, None] * f_plus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area_minus[:, None]

    int_gs_div = gs * div_f_plus[None, :] * area_plus[None, :]
    div_gs = (int_gs_div * div_f_minus[:, None] * area_minus[:, None]) / (k_bg**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_mp)
    outer_mp[rows, cols] = far_term[rows, cols]

    return outer_mp

def compute_D_mp_region2(
    K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus, greenskalar,
    greensmooth, f_plus, div_f_plus, f_minus, div_f_minus,
    luas, SegitigaPlus, SegitigaMinus,
    I_1_mp, I_2_mp, k,
    logic_identical_mp, logic_far_mp
):
    """
    Compute minus–plus D-matrix region 2 (internal).
    """

    area_plus = luas[SegitigaPlus]
    area_minus = luas[SegitigaMinus]
    N = len(SegitigaMinus)

    # Matrix terms
    K21 = K_2_1_plus[SegitigaMinus, :, :] / (4 * np.pi)
    K22 = K_2_2_plus[SegitigaMinus, :, :] * (-k**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth[np.ix_(SegitigaMinus, SegitigaPlus)] * area_plus[None, :]
    )[:, :, None] * f_plus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_minus[:, None, :]
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area_minus[:, None]

    K11 = K_1_1_plus[SegitigaMinus, :] / (4 * np.pi)
    K12 = K_1_2_plus[SegitigaMinus, :] * (-k**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth[np.ix_(SegitigaMinus, SegitigaPlus)] * div_f_plus[None, :] * area_plus[None, :]
    )
    div_outer = (sum_2 * (div_f_minus * area_minus)[:, None]) / (k**2)

    outer_mp = outer - div_outer

    # Arcioni + smooth correction
    gs = greensmooth[np.ix_(SegitigaMinus, SegitigaPlus)]
    int_smooth = (gs * area_plus[None, :])[:, :, None] * f_plus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area_minus[:, None]

    int_smooth_div = gs * div_f_plus[None, :] * area_plus[None, :]
    div_smooth = (int_smooth_div * div_f_minus[:, None] * area_minus[:, None]) / (k**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_minus[:, None] * div_f_plus[None, :]) / 4
    int_arcioni_div = (div_f_minus[:, None] * div_f_plus[None, :]) * I_1_mp / (4 * np.pi)
    int_arcioni = Cmn * I_2_mp / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_mp)
    outer_mp[rows, cols] = jum[rows, cols]

    # Far-field scalar Green version (same structure)
    gs = greenskalar[np.ix_(SegitigaMinus, SegitigaPlus)]
    int_gs = (gs * area_plus[None, :])[:, :, None] * f_plus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area_minus[:, None]

    int_gs_div = gs * div_f_plus[None, :] * area_plus[None, :]
    div_gs = (int_gs_div * div_f_minus[:, None] * area_minus[:, None]) / (k**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_mp)
    outer_mp[rows, cols] = far_term[rows, cols]

    return outer_mp

def compute_D_mm_region1(
    K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus, greenskalar_bg,
    greensmooth_bg, f_minus, div_f_minus,
    luas, SegitigaMinus,
    I_1_mm, I_2_mm, k_bg,
    logic_identical_mm, logic_far_mm
):
    """
    Compute minus–minus D-matrix region 1 (external).
    """

    area = luas[SegitigaMinus]
    N = len(SegitigaMinus)

    # Matrix terms
    K21 = K_2_1_minus[SegitigaMinus, :, :] / (4 * np.pi)
    K22 = K_2_2_minus[SegitigaMinus, :, :] * (-k_bg**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth_bg[np.ix_(SegitigaMinus, SegitigaMinus)] * area[None, :]
    )[:, :, None] * f_minus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_minus[:, None, :]
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area[:, None]

    K11 = K_1_1_minus[SegitigaMinus, :] / (4 * np.pi)
    K12 = K_1_2_minus[SegitigaMinus, :] * (-k_bg**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth_bg[np.ix_(SegitigaMinus, SegitigaMinus)] * div_f_minus[None, :] * area[None, :]
    )
    div_outer = (sum_2 * (div_f_minus * area)[:, None]) / (k_bg**2)

    outer_mm = outer - div_outer

    # Arcioni + smooth correction
    gs = greensmooth_bg[np.ix_(SegitigaMinus, SegitigaMinus)]
    int_smooth = (gs * area[None, :])[:, :, None] * f_minus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area[:, None]

    int_smooth_div = gs * div_f_minus[None, :] * area[None, :]
    div_smooth = (int_smooth_div * div_f_minus[:, None] * area[:, None]) / (k_bg**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_minus[:, None] * div_f_minus[None, :]) / 4
    int_arcioni_div = (div_f_minus[:, None] * div_f_minus[None, :]) * I_1_mm / (4 * np.pi)
    int_arcioni = Cmn * I_2_mm / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k_bg**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_mm)
    outer_mm[rows, cols] = jum[rows, cols]

    # Far-field scalar Green version
    gs = greenskalar_bg[np.ix_(SegitigaMinus, SegitigaMinus)]
    int_gs = (gs * area[None, :])[:, :, None] * f_minus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area[:, None]

    int_gs_div = gs * div_f_minus[None, :] * area[None, :]
    div_gs = (int_gs_div * div_f_minus[:, None] * area[:, None]) / (k_bg**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_mm)
    outer_mm[rows, cols] = far_term[rows, cols]

    return outer_mm

def compute_D_mm_region2(
    K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus, greenskalar,
    greensmooth, f_minus, div_f_minus,
    luas, SegitigaMinus,
    I_1_mm, I_2_mm, k,
    logic_identical_mm, logic_far_mm
):
    """
    Compute minus–minus D-matrix region 2 (internal).
    """

    area = luas[SegitigaMinus]
    N = len(SegitigaMinus)

    # Matrix terms
    K21 = K_2_1_minus[SegitigaMinus, :, :] / (4 * np.pi)
    K22 = K_2_2_minus[SegitigaMinus, :, :] * (-k**2 / (8 * np.pi))
    greensmooth_term = (
        greensmooth[np.ix_(SegitigaMinus, SegitigaMinus)] * area[None, :]
    )[:, :, None] * f_minus[None, :, :]

    sum_1 = K21 + K22 + greensmooth_term
    f_outer = f_minus[:, None, :]
    outer = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), sum_1) * area[:, None]

    K11 = K_1_1_minus[SegitigaMinus, :] / (4 * np.pi)
    K12 = K_1_2_minus[SegitigaMinus, :] * (-k**2 / (8 * np.pi))
    sum_2 = K11 + K12 + (
        greensmooth[np.ix_(SegitigaMinus, SegitigaMinus)] * div_f_minus[None, :] * area[None, :]
    )
    div_outer = (sum_2 * (div_f_minus * area)[:, None]) / (k**2)

    outer_mm = outer - div_outer

    # Arcioni + smooth correction
    gs = greensmooth[np.ix_(SegitigaMinus, SegitigaMinus)]
    int_smooth = (gs * area[None, :])[:, :, None] * f_minus[None, :, :]
    dot_smooth = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_smooth) * area[:, None]

    int_smooth_div = gs * div_f_minus[None, :] * area[None, :]
    div_smooth = (int_smooth_div * div_f_minus[:, None] * area[:, None]) / (k**2)

    outer_smooth = dot_smooth - div_smooth

    Cmn = (div_f_minus[:, None] * div_f_minus[None, :]) / 4
    int_arcioni_div = (div_f_minus[:, None] * div_f_minus[None, :]) * I_1_mm / (4 * np.pi)
    int_arcioni = Cmn * I_2_mm / (4 * np.pi)
    outer_arcioni = int_arcioni - int_arcioni_div / (k**2)

    jum = outer_smooth + outer_arcioni
    rows, cols = np.where(logic_identical_mm)
    outer_mm[rows, cols] = jum[rows, cols]

    # Far-field scalar Green version (same structure)
    gs = greenskalar[np.ix_(SegitigaMinus, SegitigaMinus)]
    int_gs = (gs * area[None, :])[:, :, None] * f_minus[None, :, :]
    dot_gs = np.einsum("nij,nij->ni", f_outer.repeat(N, axis=1), int_gs) * area[:, None]

    int_gs_div = gs * div_f_minus[None, :] * area[None, :]
    div_gs = (int_gs_div * div_f_minus[:, None] * area[:, None]) / (k**2)

    far_term = dot_gs - div_gs
    rows, cols = np.where(logic_far_mm)
    outer_mm[rows, cols] = far_term[rows, cols]

    return outer_mm

def bm_block(K_4_1, K_4_2, grad_smooth, grad_scalar, f_a, f_b, area_a, area_b,
             logic_identical, logic_far, k_val, seg_a, seg_b, jump_size):
    # Base contribution
    term1 = K_4_1[seg_a] / (4 * np.pi)
    term2 = -(k_val**2) * K_4_2[seg_a] / (8 * np.pi)
    term3 = np.cross(grad_smooth[np.ix_(seg_a, seg_b)], f_b[None, :, :]) * area_b[None, :, None]
    outer = np.einsum("ijk,ijk->ij", f_a[:, None, :], term1 + term2 + term3) * area_a[:, None]

    # Identical fix
    rows, cols = np.where(logic_identical)
    outer[rows, cols] = 0

    # Far field correction
    term_far = np.cross(grad_scalar[np.ix_(seg_a, seg_b)], f_b[None,:,:]) * area_b[None, :, None]
    dot_far = np.einsum("ijk,ijk->ij", f_a[:,None,:], term_far) * area_a[:, None]
    rows, cols = np.where(logic_far)
    outer[rows, cols] = dot_far[rows, cols]

    return outer

def compute_bm_region1(params):
    f_plus = params["f_plus"]
    f_minus = params["f_minus"]
    area = params["area"]
    k_bg = params["k_bg"]
    jumSisi = params["jumSisi"]

    # Unpack everything else needed
    segP = params["SegitigaPlus"]
    segM = params["SegitigaMinus"]
    K41p = params["K_4_1_plus"]
    K42p = params["K_4_2_plus"]
    K41m = params["K_4_1_minus"]
    K42m = params["K_4_2_minus"]
    grad_smooth_bg = params["grad_greensmooth_bg"]
    grad_scalar_bg = params["gradientgreen_bg"]
    logic_ident_pp = params["logic_identical_pp"] 
    logic_ident_pm = params["logic_identical_pm"] 
    logic_ident_mp = params["logic_identical_mp"] 
    logic_ident_mm = params["logic_identical_mm"] 
    logic_faar_pp = params["logic_far_pp"] 
    logic_faar_pm = params["logic_far_pm"] 
    logic_faar_mp = params["logic_far_mp"] 
    logic_faar_mm = params["logic_far_mm"] 

    grad_scalar_source_bg = -grad_scalar_bg
    grad_smooth_source_bg = -grad_smooth_bg
    
    outer_pp = bm_block(K41p, K42p, grad_smooth_source_bg, grad_scalar_source_bg,
                        f_plus, f_plus, area[segP], area[segP],
                        logic_ident_pp, logic_faar_pp, k_bg, segP, segP, jumSisi)

    outer_pm = bm_block(K41m, K42m, grad_smooth_source_bg, grad_scalar_source_bg,
                        f_plus, f_minus, area[segP], area[segM],
                        logic_ident_pm, logic_faar_pm, k_bg, segP, segM, jumSisi)

    outer_mp = bm_block(K41p, K42p, grad_smooth_source_bg, grad_scalar_source_bg,
                        f_minus, f_plus, area[segM], area[segP],
                        logic_ident_mp, logic_faar_mp, k_bg, segM, segP, jumSisi)

    outer_mm = bm_block(K41m, K42m, grad_smooth_source_bg, grad_scalar_source_bg,
                        f_minus, f_minus, area[segM], area[segM],
                        logic_ident_mm, logic_faar_mm, k_bg, segM, segM, jumSisi)

    outer = outer_pp + outer_pm + outer_mp + outer_mm
    return outer

def compute_bm_region2(params):
    f_plus = params["f_plus"]
    f_minus = params["f_minus"]
    area = params["area"]
    k = params["k"]
    jumSisi = params["jumSisi"]
    segP = params["SegitigaPlus"]
    segM = params["SegitigaMinus"]
    K41p = params["K_4_1_plus"]
    K42p = params["K_4_2_plus"]
    K41m = params["K_4_1_minus"]
    K42m = params["K_4_2_minus"]
    grad_smooth = params["grad_greensmooth"]
    grad_scalar = params["gradientgreen"]
    logic_ident_pp = params["logic_identical_pp"] 
    logic_ident_pm = params["logic_identical_pm"] 
    logic_ident_mp = params["logic_identical_mp"] 
    logic_ident_mm = params["logic_identical_mm"] 
    logic_faar_pp = params["logic_far_pp"] 
    logic_faar_pm = params["logic_far_pm"] 
    logic_faar_mp = params["logic_far_mp"] 
    logic_faar_mm = params["logic_far_mm"] 

    grad_scalar_source = -grad_scalar
    grad_smooth_source = -grad_smooth

    outer_pp = bm_block(K41p, K42p, grad_smooth_source, grad_scalar_source,
                        f_plus, f_plus, area[segP], area[segP],
                        logic_ident_pp, logic_faar_pp, k, segP, segP, jumSisi)

    outer_pm = bm_block(K41m, K42m, grad_smooth_source, grad_scalar_source,
                        f_plus, f_minus, area[segP], area[segM],
                        logic_ident_pm, logic_faar_pm, k, segP, segM, jumSisi)

    outer_mp = bm_block(K41p, K42p, grad_smooth_source, grad_scalar_source,
                        f_minus, f_plus, area[segM], area[segP],
                        logic_ident_mp, logic_faar_mp, k, segM, segP, jumSisi)

    outer_mm = bm_block(K41m, K42m, grad_smooth_source, grad_scalar_source,
                        f_minus, f_minus, area[segM], area[segM],
                        logic_ident_mm, logic_faar_mm, k, segM, segM, jumSisi)
    outer = outer_pp + outer_pm + outer_mp + outer_mm
    return outer

import numpy as np
import scipy
from scipy.special import lpmv
from tqdm.notebook import trange
import matplotlib.pyplot as plt

import numpy as np
import scipy
from scipy.special import lpmv
from tqdm.notebook import trange
import matplotlib.pyplot as plt

def compute_multipole_decomposition(
    E_farfield_sca,
    H_farfield_sca,
    titik_tengah_ff,
    faceNorm_ff,
    luas_ff,
    wl_interp,
    N_multipole,
    eps_bg,
    visualize=True
):
    # Physical constants
    h_bar_eV = 6.582119e-16
    eps_0 = 8.854e-12
    mu_0 = 1.257e-6
    c = 1 / np.sqrt(eps_0 * mu_0)
    imp_bg = np.sqrt(mu_0 / (eps_0 * eps_bg))

    # Derived quantities
    eV_interp = 1240. / wl_interp
    omega = eV_interp / h_bar_eV
    lamda = 1240e-9 / eV_interp
    k = 2 * np.pi * np.sqrt(eps_bg) / lamda

    # Geometry
    titik_x, titik_y, titik_z = titik_tengah_ff[:, 0], titik_tengah_ff[:, 1], titik_tengah_ff[:, 2]
    r = np.sqrt(titik_x**2 + titik_y**2 + titik_z**2)
    r2 = np.sqrt(titik_x**2 + titik_y**2)
    sin_theta = r2 / r
    cos_theta = titik_z / r
    sin_phi = titik_y / r2
    cos_phi = titik_x / r2
    phi = np.arctan2(sin_phi, cos_phi)
    theta = np.arctan2(sin_theta, cos_theta)
    jumSegitiga_ff = len(titik_x)

    # Initialize parameters
    N_wavelengths = E_farfield_sca.shape[0]
    m_ = np.arange(-N_multipole, N_multipole + 1)
    jum_m = len(m_)
    jum_n = (jum_m - 1) // 2
    a_mn = np.zeros((N_wavelengths, jum_n, jum_m), dtype=complex)
    b_mn = np.zeros_like(a_mn)
    c_sca_num = np.zeros(N_wavelengths)

    for hai in trange(N_wavelengths, desc="Multipole decomposition - wavelength"):
        # Spherical field components
        E = E_farfield_sca[hai]
        E_sca_rho = sin_theta * cos_phi * E[:, 0] + sin_theta * sin_phi * E[:, 1] + cos_theta * E[:, 2]
        E_sca_theta = cos_theta * cos_phi * E[:, 0] + cos_theta * sin_phi * E[:, 1] - sin_theta * E[:, 2]
        E_sca_phi = -sin_phi * E[:, 0] + cos_phi * E[:, 1]
        E_sca_sph = np.stack([E_sca_rho, E_sca_theta, E_sca_phi], axis=1)

        H = H_farfield_sca[hai]
        H_sca_rho = sin_theta * cos_phi * H[:, 0] + sin_theta * sin_phi * H[:, 1] + cos_theta * H[:, 2]
        H_sca_theta = cos_theta * cos_phi * H[:, 0] + cos_theta * sin_phi * H[:, 1] - sin_theta * H[:, 2]
        H_sca_phi = -sin_phi * H[:, 0] + cos_phi * H[:, 1]
        H_sca_sph = np.stack([H_sca_rho, H_sca_theta, H_sca_phi], axis=1)

        for n in range(jum_n):
            n_pyth = n + 1
            for u in range(jum_m):
                m = m_[u]
                sum_atas_M = 0.0
                sum_atas_N = 0.0
                sum_bawah_M = 0.0
                sum_bawah_N = 0.0

                for a in range(jumSegitiga_ff):
                    var_rho = k[hai] * r[a]
                    hankel = np.sqrt(np.pi / (2 * var_rho)) * scipy.special.hankel1(n_pyth + 0.5, var_rho)
                    hankel_prev = np.sqrt(np.pi / (2 * var_rho)) * scipy.special.hankel1(n_pyth - 1 + 0.5, var_rho)
                    hankel_next = np.sqrt(np.pi / (2 * var_rho)) * scipy.special.hankel1(n_pyth + 1 + 0.5, var_rho)
                    d_hankel = hankel + var_rho * (-hankel / (2 * var_rho) + (hankel_prev - hankel_next) / 2)

                    leg = lpmv(m, n_pyth, cos_theta[a])
                    leg_next = lpmv(m, n_pyth + 1, cos_theta[a])
                    d_leg = -1 / sin_theta[a] * ((n_pyth + 1) * cos_theta[a] * leg + (m - n_pyth - 1) * leg_next)

                    exp_term = np.exp(1j * m * phi[a])
                    vec_M = np.array([
                        0,
                        1j * m * leg / sin_theta[a] * hankel * exp_term,
                        -d_leg * hankel * exp_term
                    ])
                    vec_N = np.array([
                        n_pyth * (n_pyth + 1) * leg * hankel * exp_term / var_rho,
                        d_leg * d_hankel * exp_term / var_rho,
                        1j * m * leg / sin_theta[a] * d_hankel * exp_term / var_rho
                    ])

                    sum_atas_M += np.vdot(E_sca_sph[a, :], np.conj(vec_M))
                    sum_atas_N += np.vdot(E_sca_sph[a, :], np.conj(vec_N))
                    sum_bawah_M += np.linalg.norm(vec_M)**2
                    sum_bawah_N += np.linalg.norm(vec_N)**2

                if (n_pyth + m) < 0 or (n_pyth - m) < 0:
                    continue
                En = (1j)**(n_pyth + 2 * m - 1) * np.sqrt((2 * n_pyth + 1) * scipy.special.factorial(n_pyth - m) /
                                                        scipy.special.factorial(n_pyth + m)) / (2 * np.sqrt(np.pi))

                b_mn[hai, n, u] = sum_atas_M / (sum_bawah_M * En)
                a_mn[hai, n, u] = sum_atas_N / (sum_bawah_N * En)

        # SIE-based scattering cross section
        sum_sca = np.zeros(jumSegitiga_ff)
        for b in range(jumSegitiga_ff):
            E_tmp = E_farfield_sca[hai, b, :]
            H_tmp = H_farfield_sca[hai, b, :]
            cross_EH = np.cross(E_tmp, np.conj(H_tmp)).real / 2
            sum_sca[b] = luas_ff[b] * np.dot(faceNorm_ff[b, :], cross_EH)
        P_sca = np.sum(sum_sca)
        illumination = 1 / (2 * imp_bg)
        c_sca_num[hai] = P_sca / illumination

    # Multipole-based decomposition
    lala2 = np.zeros_like(a_mn, dtype=float)
    lala3 = np.zeros_like(a_mn, dtype=float)
    for i in range(jum_n):
        for j in range(jum_m):
            factor = 2 * (i + 1) + 1
            lala2[:, i, j] = factor * (np.abs(a_mn[:, i, j])**2 + np.abs(b_mn[:, i, j])**2)
            lala3[:, i, j] = factor * np.real(a_mn[:, i, j] + b_mn[:, i, j])

    c_sca_bohren_total = 2 * np.pi * np.sum(np.sum(lala2, axis=2), axis=1) / (k.flatten()**2)
    c_ext_bohren_total = 2 * np.pi * np.sum(np.sum(lala3, axis=2), axis=1) / (k.flatten()**2)
    c_abs_bohren_total = c_ext_bohren_total - c_sca_bohren_total

    # Contributions per mode
    lala_E = np.zeros_like(a_mn, dtype=float)
    lala_M = np.zeros_like(b_mn, dtype=float)
    for i in range(jum_n):
        for j in range(jum_m):
            factor = (i + 1) * (i + 2)
            lala_E[:, i, j] = factor * (np.abs(a_mn[:, i, j])**2)
            lala_M[:, i, j] = factor * (np.abs(b_mn[:, i, j])**2)

    c_sca_cont_E = np.sum(lala_E, axis=2) / (k**2)
    c_sca_cont_M = np.sum(lala_M, axis=2) / (k**2)

    if visualize:
        lamda2 = lamda.flatten() * 1e9
        col = plt.cm.hsv(np.linspace(0, 1, jum_n))
        plt.figure(figsize=(10, 6))
        plt.plot(lamda2, c_sca_bohren_total, label='c_sca decomp', linewidth=2)
        plt.plot(lamda2, c_sca_num.flatten(), '--', label='c_sca SIE', linewidth=2)
        for i in range(jum_n):
            plt.plot(lamda2, c_sca_cont_E[:, i], color=col[i], label=f'E_{i+1}', linewidth=2)
            plt.plot(lamda2, c_sca_cont_M[:, i], '--', color=col[i], label=f'M_{i+1}', linewidth=2)
        plt.xlabel(r'$\lambda$ (nm)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "a_mn": a_mn,
        "b_mn": b_mn,
        "c_sca_num": c_sca_num,
        "c_sca_bohren_total": c_sca_bohren_total,
        "c_ext_bohren_total": c_ext_bohren_total,
        "c_abs_bohren_total": c_abs_bohren_total,
        "c_sca_cont_E": c_sca_cont_E,
        "c_sca_cont_M": c_sca_cont_M,
        "lamda": lamda.flatten()
    }

import numpy as np
import matplotlib.pyplot as plt

def compute_optical_cross_sections(
    E_farfield_sca,
    H_farfield_sca,
    E_farfield_inc,
    H_farfield_inc,
    E_farfield_total,
    H_farfield_total,
    luas_ff,
    faceNorm_ff,
    eV_interp,
    imp_bg,
    radius_nm=350
):
    """
    Compute extinction, scattering, absorption cross sections and efficiencies.
    
    Parameters
    ----------
    E_farfield_sca : ndarray, shape (N_wl, N_pts, 3)
    H_farfield_sca : ndarray, shape (N_wl, N_pts, 3)
    E_farfield_inc : ndarray, shape (N_wl, N_pts, 3)
    H_farfield_inc : ndarray, shape (N_wl, N_pts, 3)
    E_farfield_total : ndarray, shape (N_wl, N_pts, 3)
    H_farfield_total : ndarray, shape (N_wl, N_pts, 3)
    luas_ff : ndarray, shape (N_pts,)
    faceNorm_ff : ndarray, shape (N_pts, 3)
    eV_interp : ndarray, shape (N_wl,)
    imp_bg : float
    radius_nm : float, default 350
    
    Returns
    -------
    view_lambda : ndarray
    C_ext, C_sca, C_abs : ndarray
    Q_ext, Q_sca, Q_abs : ndarray
    """
    view_lambda = 1240. / eV_interp.flatten()  # in nm
    view_sizeparam = 2 * np.pi * 80. / view_lambda
    illumination = 1 / (2 * imp_bg)
    radius = radius_nm * 1e-9
    rr = np.pi * radius**2

    C_sca = np.zeros_like(view_lambda)
    C_abs = np.zeros_like(view_lambda)
    C_ext = np.zeros_like(view_lambda)
    Q_sca = np.zeros_like(view_lambda)
    Q_abs = np.zeros_like(view_lambda)
    Q_ext = np.zeros_like(view_lambda)

    for a in range(len(view_lambda)):
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

        # Integrate over triangles
        sum_sca = luas_ff * np.einsum('ij,ij->i', S_sca, faceNorm_ff)
        sum_abs = luas_ff * np.einsum('ij,ij->i', S_abs, faceNorm_ff)
        sum_ext = luas_ff * np.einsum('ij,ij->i', S_ext, faceNorm_ff)

        # Total power
        P_sca = np.sum(sum_sca)
        P_abs = -np.sum(sum_abs)
        P_ext = -np.sum(sum_ext)

        # Cross sections
        C_sca[a] = P_sca / illumination
        C_abs[a] = P_abs / illumination
        C_ext[a] = P_ext / illumination

        # Efficiencies
        Q_sca[a] = C_sca[a] / rr
        Q_abs[a] = C_abs[a] / rr
        Q_ext[a] = C_ext[a] / rr

    return view_lambda, C_ext, C_sca, C_abs, Q_ext, Q_sca, Q_abs