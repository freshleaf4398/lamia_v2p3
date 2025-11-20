import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm.notebook import trange
import warnings
warnings.filterwarnings("ignore")
from element_integrals_2d import element_integrals_2d
from compute_nearfields import compute_nearfields
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_nearfields_full(lamda_interp, x1, y1_re, y1_im, eps_bg, geo, X, Y, far, mask, mask_x, mask_y):

    PanjangSisi = geo['PanjangSisi']
    rho_plus = geo['rho_plus']
    rho_minus = geo['rho_minus']
    SegitigaPlus = geo['SegitigaPlus']
    SegitigaMinus = geo['SegitigaMinus']
    luas = geo['luas']
    jumSegitiga = geo['jumSegitiga']
    jumSisi = geo['jumSisi']
    faceNorm = geo['faceNorm']
    FreeVertex_plus = geo['FreeVertex_plus']
    FreeVertex_minus = geo['FreeVertex_minus']
    titik_tengah = geo['titik_tengah']
    p = geo['p']
    t = geo['t']
    cur_mat = far['cur_mat']
    
    p = p.T
    grid_points_3d = np.stack((X,Y,np.zeros((X.shape[0],Y.shape[0]))),axis=2)

    elem_int_2d = element_integrals_2d(SegitigaPlus,SegitigaMinus,FreeVertex_plus,FreeVertex_minus,titik_tengah,PanjangSisi,jumSegitiga,jumSisi,faceNorm,rho_plus,rho_minus,luas,p,t,grid_points_3d)

    nearfield = compute_nearfields(lamda_interp, x1, y1_re, y1_im, eps_bg, geo, elem_int_2d, grid_points_3d, cur_mat)

    E_sca = nearfield['E_sca']
    E_sca_material = nearfield['E_sca_material']
    mag_E = nearfield['mag_E']
    mag_E_material = nearfield['mag_E_material']

    E_sca[mask, :] = E_sca_material[mask, :]
    mag_E[mask] = mag_E_material[mask]

    fig, ax = plt.subplots()  # 2D "surface" plot with shading
    c = ax.pcolormesh(X, Y, mag_E, shading='auto', cmap='inferno')  # equivalent to surf + shading interp
    fig.colorbar(c)  # colorbar
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')  # axis square

    plt.plot(mask_x, mask_y, 'w', linewidth=2)  # 'k' = black

    plt.show()
    
    return {
        "E_sca": E_sca,
        "E_sca_material": E_sca_material,
        "mag_E": mag_E,
        "mag_E_material": mag_E_material
    }
    