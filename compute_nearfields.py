from functions_modules import *
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm.notebook import trange
import warnings
warnings.filterwarnings("ignore")

def compute_nearfields(
    lambda_interp, x1, y1_re, y1_im, eps_bg,
    geo, elem_int_2d, grid_points_3d, cur_mat,
    funcs_module=None,   
):
    
    eV_interp = 1240.0 / lambda_interp  # Convert nm to eV
    c = 299792458.0
    
    gm = {"get_wavelength_parameters": get_wavelength_parameters,
      "compute_greens_functions": compute_greens_functions,
      "compute_greens_functions_2d": compute_greens_functions_2d,
      "compute_greens_farfield_functions": compute_greens_farfield_functions,
      "compute_incident_fields": compute_incident_fields,
      "get_logical_masks": get_logical_masks,
      "compute_D_pp_region1": compute_D_pp_region1,
      "compute_D_pp_region2": compute_D_pp_region2,
      "compute_D_pm_region1": compute_D_pm_region1,
      "compute_D_pm_region2": compute_D_pm_region2,
      "compute_D_mp_region1": compute_D_mp_region1,
      "compute_D_mp_region2": compute_D_mp_region2,
      "compute_D_mm_region1": compute_D_mm_region1,
      "compute_D_mm_region2": compute_D_mm_region2,
      "bm_block": bm_block,
      "compute_bm_region1": compute_bm_region1,
      "compute_bm_region2": compute_bm_region2,
}

    # unpack geometry & arrays needed
    titik_tengah = geo["titik_tengah"]
    titik_tengah_ff = geo["titik_tengah_ff"]
    jumSegitiga = int(geo["jumSegitiga"])
    jumSegitiga_ff = int(geo["jumSegitiga_ff"])
    jumSisi = int(geo["jumSisi"])
    luas = geo["luas"]
    PanjangSisi = geo["PanjangSisi"]
    rho_plus = geo["rho_plus"]
    rho_minus = geo["rho_minus"]
    SegitigaPlus = geo["SegitigaPlus"]
    SegitigaMinus = geo["SegitigaMinus"]
    logic_identical = geo.get("logic_identical")
    logic_adjacent  = geo.get("logic_adjacent")
    logic_touch     = geo.get("logic_touch")
    logic_far       = geo.get("logic_far")
    
    interp_re = PchipInterpolator(x1, y1_re)
    interp_im = PchipInterpolator(x1, y1_im)
    n_re = interp_re(eV_interp.flatten())
    n_im = interp_im(eV_interp.flatten())

    params = gm["get_wavelength_parameters"](eV_interp[0], n_re[0], n_im[0], eps_bg)
    omega = params['omega']
    kv = params['kv']
    kv_bg = params['kv_bg']
    k = params['k']
    k_bg = params['k_bg']
    E_0 = params['E_0']
    k_vec = params['k_vec']
    mu_0 = params['mu_0']
    eps_0 = params['eps_0']
    imp = params['imp']
    imp_bg = params['imp_bg']

    greens = gm["compute_greens_functions_2d"](grid_points_3d, titik_tengah, k=params['k'], k_bg=params['k_bg'])

    q, f_plus, f_minus, div_f_plus, div_f_minus = gm["compute_incident_fields"](
                PanjangSisi, rho_plus, rho_minus,
                luas, SegitigaPlus, SegitigaMinus,
                titik_tengah, kv_bg, k_vec, E_0,
                mu_0, c, eps_bg
            )

    K_4_1_plus = elem_int_2d["K_4_1_plus"]
    K_4_2_plus = elem_int_2d["K_4_2_plus"]
    K_4_1_minus = elem_int_2d["K_4_1_minus"]
    K_4_2_minus = elem_int_2d["K_4_2_minus"]
    K_3_1_plus = elem_int_2d["K_3_1_plus"]
    K_3_2_plus = elem_int_2d["K_3_2_plus"]
    K_3_1_minus = elem_int_2d["K_3_1_minus"]
    K_3_2_minus = elem_int_2d["K_3_2_minus"]
    K_2_1_plus = elem_int_2d["K_2_1_plus"]
    K_2_2_plus = elem_int_2d["K_2_2_plus"]
    K_2_1_minus = elem_int_2d["K_2_1_minus"]
    K_2_2_minus = elem_int_2d["K_2_2_minus"]

    grad_greensmooth_bg = greens['grad_greensmooth_bg']
    greensmooth_bg = greens['greensmooth_bg']
    grad_greensmooth = greens['grad_greensmooth']
    greensmooth = greens['greensmooth']

    alpha = cur_mat[:jumSisi]
    beta = cur_mat[jumSisi:2*jumSisi]
    alpha_exp = alpha.reshape(1, 1, jumSisi, 1)
    beta_exp  = beta.reshape(1, 1, jumSisi, 1)

    coeff_div_luasP = (div_f_plus * luas[SegitigaPlus]).reshape(1, 1, jumSisi, 1)  # (1,1,jumSisi,1)
    part1_plus = grad_greensmooth_bg[:, :, SegitigaPlus, :] * coeff_div_luasP / (k_bg**2)
    coeff_divP = div_f_plus.reshape(1, 1, jumSisi, 1)
    part2_plus = (K_3_1_plus / (4*np.pi) - (k_bg**2) * K_3_2_plus / (8*np.pi)) * coeff_divP / (k_bg**2)
    greens_exp_P = greensmooth_bg[:, :, SegitigaPlus][..., None]                     # (jum_y, jum_x, jumSisi, 1)
    f_plus_exp = f_plus.reshape(1, 1, jumSisi, 3)                            # (1,1,jumSisi,3)
    luasP_exp = luas[SegitigaPlus].reshape(1, 1, jumSisi, 1)                         # (1,1,jumSisi,1)
    part3_plus = greens_exp_P * f_plus_exp * luasP_exp                       # (jum_y, jum_x, jumSisi, 3)
    part4_plus = (K_2_1_plus / (4*np.pi) - (k_bg**2) * K_2_2_plus / (8*np.pi))
    part5_plus = np.cross(-grad_greensmooth_bg[:, :, SegitigaPlus, :], f_plus_exp) * luasP_exp
    part6_plus = (K_4_1_plus / (4*np.pi) - (k_bg**2) * K_4_2_plus / (8*np.pi))

    coeff_div_luasP = (div_f_minus * luas[SegitigaMinus]).reshape(1, 1, jumSisi, 1)  # (1,1,jumSisi,1)
    part1_minus = grad_greensmooth_bg[:, :, SegitigaMinus, :] * coeff_div_luasP / (k_bg**2)
    coeff_divP = div_f_minus.reshape(1, 1, jumSisi, 1)
    part2_minus = (K_3_1_minus / (4*np.pi) - (k_bg**2) * K_3_2_minus / (8*np.pi)) * coeff_divP / (k_bg**2)
    greens_exp_P = greensmooth_bg[:, :, SegitigaMinus][..., None]                     # (jum_y, jum_x, jumSisi, 1)
    f_minus_exp = f_minus.reshape(1, 1, jumSisi, 3)                            # (1,1,jumSisi,3)
    luasP_exp = luas[SegitigaMinus].reshape(1, 1, jumSisi, 1)                         # (1,1,jumSisi,1)
    part3_minus = greens_exp_P * f_minus_exp * luasP_exp                       # (jum_y, jum_x, jumSisi, 3)
    part4_minus = (K_2_1_minus / (4*np.pi) - (k_bg**2) * K_2_2_minus / (8*np.pi))
    part5_minus = np.cross(-grad_greensmooth_bg[:, :, SegitigaMinus, :], f_minus_exp) * luasP_exp
    part6_minus = (K_4_1_minus / (4*np.pi) - (k_bg**2) * K_4_2_minus / (8*np.pi))

    sum_alpha = part1_plus + part1_minus + part2_plus + part2_minus + part3_plus + part3_minus + part4_plus + part4_minus
    sum_beta = part5_plus + part5_minus + part6_plus + part6_minus

    # inner = -alpha * omega * mu_0 * sum_alpha / (1j) + beta * sum_beta
    inner_bg = - alpha_exp * (omega * mu_0) * sum_alpha / (1j) + beta_exp * sum_beta  # (jum_y,jum_x,jumSisi,3)

    # Sum over triangle index (axis=2)
    E_sca = np.sum(inner_bg, axis=2)                 # (jum_y, jum_x, 3)
    mag_E = np.linalg.norm(E_sca, axis=2)            # (jum_y, jum_x)  (real non-negative)

    #Scatterer

    coeff_div_luasP = (div_f_plus * luas[SegitigaPlus]).reshape(1, 1, jumSisi, 1)  # (1,1,jumSisi,1)
    part1_plus = grad_greensmooth[:, :, SegitigaPlus, :] * coeff_div_luasP / (k**2)
    coeff_divP = div_f_plus.reshape(1, 1, jumSisi, 1)
    part2_plus = (K_3_1_plus / (4*np.pi) - (k**2) * K_3_2_plus / (8*np.pi)) * coeff_divP / (k**2)
    greens_exp_P = greensmooth[:, :, SegitigaPlus][..., None]                     # (jum_y, jum_x, jumSisi, 1)
    f_plus_exp = f_plus.reshape(1, 1, jumSisi, 3)                            # (1,1,jumSisi,3)
    luasP_exp = luas[SegitigaPlus].reshape(1, 1, jumSisi, 1)                         # (1,1,jumSisi,1)
    part3_plus = greens_exp_P * f_plus_exp * luasP_exp                       # (jum_y, jum_x, jumSisi, 3)
    part4_plus = (K_2_1_plus / (4*np.pi) - (k**2) * K_2_2_plus / (8*np.pi))
    part5_plus = np.cross(-grad_greensmooth[:, :, SegitigaPlus, :], f_plus_exp) * luasP_exp
    part6_plus = (K_4_1_plus / (4*np.pi) - (k**2) * K_4_2_plus / (8*np.pi))

    coeff_div_luasP = (div_f_minus * luas[SegitigaMinus]).reshape(1, 1, jumSisi, 1)  # (1,1,jumSisi,1)
    part1_minus = grad_greensmooth[:, :, SegitigaMinus, :] * coeff_div_luasP / (k**2)
    coeff_divP = div_f_minus.reshape(1, 1, jumSisi, 1)
    part2_minus = (K_3_1_minus / (4*np.pi) - (k**2) * K_3_2_minus / (8*np.pi)) * coeff_divP / (k**2)
    greens_exp_P = greensmooth[:, :, SegitigaMinus][..., None]                     # (jum_y, jum_x, jumSisi, 1)
    f_minus_exp = f_minus.reshape(1, 1, jumSisi, 3)                            # (1,1,jumSisi,3)
    luasP_exp = luas[SegitigaMinus].reshape(1, 1, jumSisi, 1)                         # (1,1,jumSisi,1)
    part3_minus = greens_exp_P * f_minus_exp * luasP_exp                       # (jum_y, jum_x, jumSisi, 3)
    part4_minus = (K_2_1_minus / (4*np.pi) - (k**2) * K_2_2_minus / (8*np.pi))
    part5_minus = np.cross(-grad_greensmooth[:, :, SegitigaMinus, :], f_minus_exp) * luasP_exp
    part6_minus = (K_4_1_minus / (4*np.pi) - (k**2) * K_4_2_minus / (8*np.pi))
    
    sum_alpha = part1_plus + part1_minus + part2_plus + part2_minus + part3_plus + part3_minus + part4_plus + part4_minus
    sum_beta = part5_plus + part5_minus + part6_plus + part6_minus

    # inner = -alpha * omega * mu_0 * sum_alpha / (1j) + beta * sum_beta
    inner = alpha_exp * (omega * mu_0) * sum_alpha / (1j) - beta_exp * sum_beta  # (jum_y,jum_x,jumSisi,3)

    # Sum over triangle index (axis=2)
    E_sca_material = np.sum(inner, axis=2)                 # (jum_y, jum_x, 3)
    mag_E_material = np.linalg.norm(E_sca_material, axis=2)            # (jum_y, jum_x)  (real non-negative)
    
    return {
        "E_sca_material": E_sca_material,
        "mag_E_material": mag_E_material,
        "E_sca": E_sca,
        "mag_E": mag_E
    }
        