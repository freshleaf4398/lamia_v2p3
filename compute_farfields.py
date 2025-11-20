import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm.notebook import trange
import warnings
warnings.filterwarnings("ignore")
from functions_modules import *

def compute_farfields(
    lambda_interp, x1, y1_re, y1_im, eps_bg,
    geo, integrals, K_mats,
    funcs_module=None,    # optional module object that contains helper functions; if None, expects them in globals()
):
    """
    Compute far-field (incident, scattered, total) E and H over wavelength sweep.

    Parameters
    ----------
    lamda_interp : array_like
        wavelengths (Nw,1) used for interpolation (nm).
    eV_interp : array_like
        energies corresponding to lambdas (Nw,1).
    x1, y1_re, y1_im : array_like
        material data arrays for interpolation (from J_C_weaver_Au).
    eps_bg : float
        background permittivity (e.g. 1.0)
    geo : dict
        geometry dict returned by preprocess_geometry(...) (contains titik_tengah, titik_tengah_ff, jumSegitiga_ff, jumSegitiga, luas, PanjangSisi, rho_plus, rho_minus, SegitigaPlus, SegitigaMinus, logic_* etc.)
    integrals : dict
        dictionary returned by compute_integrals_block(...) (contains I_1_pp, I_2_pp, I_1_pm, ... etc.)
    K_mats : dict
        dictionary returned by element_integrals(...) (contains K_4_1_plus, K_2_1_plus, ... etc.)
    funcs_module : module or None
        module where helper functions like get_wavelength_parameters, compute_greens_functions, compute_incident_fields, get_logical_masks,
        compute_D_pp_region1, compute_bm_region1, etc. live. If None, the functions are looked up in globals().

    Returns
    -------
    dict with keys:
        E_farfield_sca, E_farfield_inc, E_farfield_total,
        H_farfield_sca, H_farfield_inc, H_farfield_total
        shapes: (Nw, jumSegitiga_ff, 3)
    """
    eV_interp = 1240.0 / lambda_interp  # Convert nm to eV
    c = 299792458.0
    
    # helper lookup: either from provided module or from globals
    gm = {"get_wavelength_parameters": get_wavelength_parameters,
     "compute_greens_functions": compute_greens_functions,
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

    interp_re = PchipInterpolator(x1, y1_re)
    interp_im = PchipInterpolator(x1, y1_im)

    eV_interp_arr = np.atleast_1d(eV_interp)
    n_re = interp_re(eV_interp_arr.flatten())
    n_im = interp_im(eV_interp_arr.flatten())

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
    # integrals
    I_1_pp = integrals["I_1_pp"]
    I_1_pm = integrals["I_1_pm"]
    I_1_mp = integrals["I_1_mp"]
    I_1_mm = integrals["I_1_mm"]
    I_2_pp = integrals["I_2_pp"]
    I_2_pm = integrals["I_2_pm"]
    I_2_mp = integrals["I_2_mp"]
    I_2_mm = integrals["I_2_mm"]

    # K matrices
    K_4_1_plus = K_mats["K_4_1_plus"]
    K_4_2_plus = K_mats["K_4_2_plus"]
    K_4_1_minus = K_mats["K_4_1_minus"]
    K_4_2_minus = K_mats["K_4_2_minus"]

    K_2_1_plus = K_mats["K_2_1_plus"]
    K_2_2_plus = K_mats["K_2_2_plus"]
    K_2_1_minus = K_mats["K_2_1_minus"]
    K_2_2_minus = K_mats["K_2_2_minus"]

    K_1_1_plus = K_mats.get("K_1_1_plus")
    K_1_1_minus= K_mats.get("K_1_1_minus")
    K_1_2_plus = K_mats.get("K_1_2_plus")
    K_1_2_minus= K_mats.get("K_1_2_minus")

    # prepare output arrays
    lambda_interp = np.atleast_1d(lambda_interp)
    Nw = len(lambda_interp)
    E_farfield_inc = np.zeros((Nw, jumSegitiga_ff, 3), dtype=np.complex128)
    H_farfield_inc = np.zeros((Nw, jumSegitiga_ff, 3), dtype=np.complex128)
    E_farfield_sca = np.zeros((Nw, jumSegitiga_ff, 3), dtype=np.complex128)
    H_farfield_sca = np.zeros((Nw, jumSegitiga_ff, 3), dtype=np.complex128)
    E_farfield_total = np.zeros((Nw, jumSegitiga_ff, 3), dtype=np.complex128)
    H_farfield_total = np.zeros((Nw, jumSegitiga_ff, 3), dtype=np.complex128)

    # iterate wavelengths
    for h in trange(Nw, desc="Far-field loop - wavelength"):
        # wavelength-dependent params (user-supplied function)
        params = gm["get_wavelength_parameters"](
            eV_interp[h], n_re[h], n_im[h], eps_bg
        )

        # unpack relevant params
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

        # greens functions (helper)
        greens = gm["compute_greens_functions"](titik_tengah, k=params['k'], k_bg=params['k_bg'])
        greens_ff = gm["compute_greens_farfield_functions"](titik_tengah_ff, titik_tengah, k=params['k'], k_bg=params['k_bg'])

        # compute incident field contributions (helper)
        q, f_plus, f_minus, div_f_plus, div_f_minus = gm["compute_incident_fields"](
            PanjangSisi, rho_plus, rho_minus,
            luas, SegitigaPlus, SegitigaMinus,
            titik_tengah, kv_bg, k_vec, E_0,
            mu_0, c, eps_bg
        )

        # logical masks
        logic_masks = gm["get_logical_masks"](
            SegitigaPlus, SegitigaMinus,
            logic_identical, logic_adjacent,
            logic_touch, logic_far
        )

        # compute outer region contributions using your helper compute_D_* functions
        outer_pp_region1 = gm["compute_D_pp_region1"](
            K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus,
            greens['greenskalar_bg'], greens['greensmooth_bg'],
            f_plus, div_f_plus,
            luas, SegitigaPlus,
            I_1_pp, I_2_pp, k_bg,
            logic_masks["identical"][0], logic_masks["far"][0]
        )
        am_1 = omega * mu_0 * outer_pp_region1 / (1j)

        outer_pp_region2 = gm["compute_D_pp_region2"](
            K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus,
            greens['greenskalar'], greens['greensmooth'],
            f_plus, div_f_plus,
            luas, SegitigaPlus,
            I_1_pp, I_2_pp, k,
            logic_masks["identical"][0], logic_masks["far"][0]
        )
        am_2 = omega * mu_0 * outer_pp_region2 / (1j)

        outer_pm_region1 = gm["compute_D_pm_region1"](
            K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus,
            greens['greenskalar_bg'], greens['greensmooth_bg'],
            f_plus, div_f_plus, f_minus, div_f_minus,
            luas, SegitigaPlus, SegitigaMinus,
            I_1_pm, I_2_pm, k_bg,
            logic_masks["identical"][1], logic_masks["far"][1]
        )
        am_3 = omega * mu_0 * outer_pm_region1 / (1j)

        outer_pm_region2 = gm["compute_D_pm_region2"](
            K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus,
            greens['greenskalar'], greens['greensmooth'],
            f_plus, div_f_plus, f_minus, div_f_minus,
            luas, SegitigaPlus, SegitigaMinus,
            I_1_pm, I_2_pm, k,
            logic_masks["identical"][1], logic_masks["far"][1]
        )
        am_4 = omega * mu_0 * outer_pm_region2 / (1j)

        outer_mp_region1 = gm["compute_D_mp_region1"](
            K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus,
            greens['greenskalar_bg'], greens['greensmooth_bg'],
            f_plus, div_f_plus, f_minus, div_f_minus,
            luas, SegitigaPlus, SegitigaMinus,
            I_1_mp, I_2_mp, k_bg,
            logic_masks["identical"][2], logic_masks["far"][2]
        )
        am_5 = omega * mu_0 * outer_mp_region1 / (1j)

        outer_mp_region2 = gm["compute_D_mp_region2"](
            K_2_1_plus, K_2_2_plus, K_1_1_plus, K_1_2_plus,
            greens['greenskalar'], greens['greensmooth'],
            f_plus, div_f_plus, f_minus, div_f_minus,
            luas, SegitigaPlus, SegitigaMinus,
            I_1_mp, I_2_mp, k,
            logic_masks["identical"][2], logic_masks["far"][2]
        )
        am_6 = omega * mu_0 * outer_mp_region2 / (1j)

        outer_mm_region1 = gm["compute_D_mm_region1"](
            K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus,
            greens['greenskalar_bg'], greens['greensmooth_bg'],
            f_minus, div_f_minus,
            luas, SegitigaMinus,
            I_1_mm, I_2_mm, k_bg,
            logic_masks["identical"][3], logic_masks["far"][3]
        )
        am_7 = omega * mu_0 * outer_mm_region1 / (1j)

        outer_mm_region2 = gm["compute_D_mm_region2"](
            K_2_1_minus, K_2_2_minus, K_1_1_minus, K_1_2_minus,
            greens['greenskalar'], greens['greensmooth'],
            f_minus, div_f_minus,
            luas, SegitigaMinus,
            I_1_mm, I_2_mm, k,
            logic_masks["identical"][3], logic_masks["far"][3]
        )
        am_8 = omega * mu_0 * outer_mm_region2 / (1j)

        am_odd = am_1 + am_3 + am_5 + am_7
        am_even = am_2 + am_4 + am_6 + am_8

        # attach many params into params dict for bm/assembly
        params.update({
            "f_plus": f_plus,
            "f_minus": f_minus,
            "div_f_plus": div_f_plus,
            "div_f_minus": div_f_minus,
            "area": luas,
            "SegitigaPlus": SegitigaPlus,
            "SegitigaMinus": SegitigaMinus,
            "jumSisi": jumSisi,

            "gradientgreen": greens.get("gradientgreen"),
            "grad_greensmooth": greens.get("grad_greensmooth"),
            "gradientgreen_bg": greens.get("gradientgreen_bg"),
            "grad_greensmooth_bg": greens.get("grad_greensmooth_bg"),

            "K_4_1_plus": K_4_1_plus,
            "K_4_2_plus": K_4_2_plus,
            "K_4_1_minus": K_4_1_minus,
            "K_4_2_minus": K_4_2_minus,

            "k_bg": params["k_bg"],
            "omega": params["omega"],
            "mu_0": mu_0,
        })

        # masks (precomputed)
        params.update({
            "logic_identical_pp": logic_masks["identical"][0],
            "logic_identical_pm": logic_masks["identical"][1],
            "logic_identical_mp": logic_masks["identical"][2],
            "logic_identical_mm": logic_masks["identical"][3],

            "logic_far_pp": logic_masks["far"][0],
            "logic_far_pm": logic_masks["far"][1],
            "logic_far_mp": logic_masks["far"][2],
            "logic_far_mm": logic_masks["far"][3],
        })

        # compute bm (helper functions)
        bm_1 = gm["compute_bm_region1"](params)
        bm_2 = gm["compute_bm_region2"](params)

        # Assemble and solve linear system
        A = np.block([
            [am_odd + am_even, -(bm_1 + bm_2)],
            [bm_1 + bm_2, am_odd / (imp_bg**2) + am_even / (imp**2)]
        ])
        cur_mat = np.linalg.solve(A, q.T)

        # farfield assembly using greens_ff
        greenskalar_bg_ff = greens_ff['greenskalar_bg']
        gradientgreen_bg_ff = greens_ff['gradientgreen_bg']
        gradientgreen_source_bg_ff = greens_ff['gradientgreen_source_bg']

        for a in range(jumSegitiga_ff):
            sum_1 = gradientgreen_bg_ff[a, SegitigaPlus, :] * (div_f_plus * luas[SegitigaPlus])[:, None]
            sum_2 = greenskalar_bg_ff[a, SegitigaPlus][:, None] * f_plus * luas[SegitigaPlus][:, None]
            inner_plus_a = sum_1 / (k_bg**2) + sum_2

            sum_1 = gradientgreen_bg_ff[a, SegitigaMinus, :] * (div_f_minus * luas[SegitigaMinus])[:, None]
            sum_2 = greenskalar_bg_ff[a, SegitigaMinus][:, None] * f_minus * luas[SegitigaMinus][:, None]
            inner_minus_a = sum_1 / (k_bg**2) + sum_2

            inner_plus_b = np.cross(gradientgreen_source_bg_ff[a, SegitigaPlus, :], f_plus) * luas[SegitigaPlus][:, None]
            inner_minus_b = np.cross(gradientgreen_source_bg_ff[a, SegitigaMinus, :], f_minus) * luas[SegitigaMinus][:, None]

            cur_J = cur_mat[:jumSisi]
            cur_M = cur_mat[jumSisi:2*jumSisi]

            inner_E = -cur_J * omega * mu_0 * (inner_plus_a + inner_minus_a) / 1j + \
                       cur_M * (inner_plus_b + inner_minus_b)
            inner_H = -cur_M * omega * params['eps_0'] * eps_bg * (inner_plus_a + inner_minus_a) / 1j - \
                       cur_J * (inner_plus_b + inner_minus_b)

            E_farfield_sca[h, a, :] = np.sum(inner_E, axis=0)
            H_farfield_sca[h, a, :] = np.sum(inner_H, axis=0)

            phase = np.exp(1j * np.dot(titik_tengah_ff[a, :], params['kv_bg']))
            E_farfield_inc[h, a, :] = E_0 * phase
            H_farfield_inc[h, a, :] = np.cross(params['k_vec'], E_0) * phase / (mu_0 * c / np.sqrt(eps_bg))

            E_farfield_total[h, a, :] = E_farfield_sca[h, a, :] + E_farfield_inc[h, a, :]
            H_farfield_total[h, a, :] = H_farfield_sca[h, a, :] + H_farfield_inc[h, a, :]

    return {
        "E_farfield_sca": E_farfield_sca,
        "E_farfield_inc": E_farfield_inc,
        "E_farfield_total": E_farfield_total,
        "H_farfield_sca": H_farfield_sca,
        "H_farfield_inc": H_farfield_inc,
        "H_farfield_total": H_farfield_total,
        "cur_mat": cur_mat
    }