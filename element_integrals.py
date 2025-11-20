def element_integrals(geo):

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
    
    #titik_tengah = obs_point
    
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    
    def normalize(v): #function for norm like matlab
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / norm
    
    # Restart core variables needed for singularity processing
    f_plus = (PanjangSisi[:, None] * rho_plus) / (2 * luas[SegitigaPlus][:, None])
    f_minus = -(PanjangSisi[:, None] * rho_minus) / (2 * luas[SegitigaMinus][:, None])
    
    div_f_plus = PanjangSisi / luas[SegitigaPlus]
    div_f_minus = -PanjangSisi / luas[SegitigaMinus]
    
    # Prepare triangle centers and obs/source grids
    x, y = np.meshgrid(np.arange(titik_tengah.shape[0]), np.arange(titik_tengah.shape[0]), indexing='ij')
    obs_point_1 = titik_tengah[y, 0].reshape(titik_tengah.shape[0], titik_tengah.shape[0]).T
    obs_point_2 = titik_tengah[y, 1].reshape(titik_tengah.shape[0], titik_tengah.shape[0]).T
    obs_point_3 = titik_tengah[y, 2].reshape(titik_tengah.shape[0], titik_tengah.shape[0]).T
    obs_point = np.stack([obs_point_1, obs_point_2, obs_point_3], axis=-1)
    
    # Vertices and edges
    source_tri = t[:jumSegitiga]
    q1 = np.transpose(p[source_tri[:, 0]][x],(1,0,2))
    q2 = np.transpose(p[source_tri[:, 1]][x],(1,0,2))
    q3 = np.transpose(p[source_tri[:, 2]][x],(1,0,2))
    
    vec_sisi1 = q2 - q1
    vec_sisi2 = q3 - q2
    vec_sisi3 = q1 - q3
    
    source_point1 = (q2 + q1) / 2
    source_point2 = (q3 + q2) / 2
    source_point3 = (q1 + q3) / 2
    
    # Compute edge lengths from obs to triangle edges
    R_sisi1_plus = np.linalg.norm(obs_point - q2, axis=2) #check here
    R_sisi2_plus = np.linalg.norm(obs_point - q3, axis=2)
    R_sisi3_plus = np.linalg.norm(obs_point - q1, axis=2)
    R_sisi1_minus = R_sisi3_plus
    R_sisi2_minus = R_sisi1_plus
    R_sisi3_minus = R_sisi2_plus
    
    # Normalize edge vectors
    unit_edge_sisi1 = normalize(vec_sisi1)
    unit_edge_sisi2 = normalize(vec_sisi2)
    unit_edge_sisi3 = normalize(vec_sisi3)
    
    S_sisi1_plus = np.einsum('ijk,ijk->ij', q2 - obs_point, unit_edge_sisi1)
    S_sisi2_plus = np.einsum('ijk,ijk->ij', q3 - obs_point, unit_edge_sisi2)
    S_sisi3_plus = np.einsum('ijk,ijk->ij', q1 - obs_point, unit_edge_sisi3)
    S_sisi1_minus = np.einsum('ijk,ijk->ij', q1 - obs_point, unit_edge_sisi1)
    S_sisi2_minus = np.einsum('ijk,ijk->ij', q2 - obs_point, unit_edge_sisi2)
    S_sisi3_minus = np.einsum('ijk,ijk->ij', q3 - obs_point, unit_edge_sisi3)
    
    # Face normals and their unit M vectors
    faceNorm_1 = faceNorm[x, 0].reshape(jumSegitiga, jumSegitiga)
    faceNorm_2 = faceNorm[x, 1].reshape(jumSegitiga, jumSegitiga)
    faceNorm_3 = faceNorm[x, 2].reshape(jumSegitiga, jumSegitiga)
    faceNorm_m = np.transpose(np.stack([faceNorm_1, faceNorm_2, faceNorm_3], axis=-1),(1,0,2))
    
    unit_M_sisi1 = normalize(np.cross(unit_edge_sisi1, faceNorm_m))
    unit_M_sisi2 = normalize(np.cross(unit_edge_sisi2, faceNorm_m))
    unit_M_sisi3 = normalize(np.cross(unit_edge_sisi3, faceNorm_m))
    
    W_0 = np.einsum('ijk,ijk->ij', faceNorm_m, obs_point - source_point1)
    rho = obs_point - W_0[:,:,None] * faceNorm_m
    R0_sisi1_kuad = R_sisi1_plus**2-S_sisi1_plus**2
    R0_sisi2_kuad = R_sisi2_plus**2-S_sisi2_plus**2
    R0_sisi3_kuad = R_sisi3_plus**2-S_sisi3_plus**2
    
    T_0_1 = np.einsum('ijk,ijk->ij', obs_point - source_point1, unit_M_sisi1)
    T_0_2 = np.einsum('ijk,ijk->ij', obs_point - source_point2, unit_M_sisi2)
    T_0_3 = np.einsum('ijk,ijk->ij', obs_point - source_point3, unit_M_sisi3)
    a_1 = normalize(q1 - obs_point)
    a_2 = normalize(q2 - obs_point)
    a_3 = normalize(q3 - obs_point)
    X = 1 + np.einsum('ijk,ijk->ij', a_1, a_2) + np.einsum('ijk,ijk->ij', a_1, a_3) + np.einsum('ijk,ijk->ij', a_2, a_3)
    Y = np.abs(np.einsum('ijk,ijk->ij', a_1, np.cross(a_2, a_3)))
    
    AngleExcess = np.zeros((jumSegitiga,jumSegitiga))
    temp = 2 * np.atan2(Y, X)
    AngleExcess[W_0 > 0] = temp[W_0 > 0]
    temp = -2.*np.atan2(Y, X)
    AngleExcess[W_0 < 0] = temp[W_0 < 0]
    AngleExcess = AngleExcess
    
    basic_line_1 = np.log((R_sisi1_minus - S_sisi1_minus) / (R_sisi1_plus - S_sisi1_plus))
    temp = np.log((R_sisi1_plus + S_sisi1_plus) / (R_sisi1_minus + S_sisi1_minus))
    logic_RS_1 = np.abs(R_sisi1_minus + S_sisi1_minus) > np.abs(R_sisi1_plus - S_sisi1_plus)
    basic_line_1[logic_RS_1] = temp[logic_RS_1]
    
    basic_line_2 = np.log((R_sisi2_minus - S_sisi2_minus) / (R_sisi2_plus - S_sisi2_plus))
    temp = np.log((R_sisi2_plus + S_sisi2_plus) / (R_sisi2_minus + S_sisi2_minus))
    logic_RS_2 = np.abs(R_sisi2_minus + S_sisi2_minus) > np.abs(R_sisi2_plus - S_sisi2_plus)
    basic_line_2[logic_RS_2] = temp[logic_RS_2]
    
    basic_line_3 = np.log((R_sisi3_minus - S_sisi3_minus) / (R_sisi3_plus - S_sisi3_plus))
    temp = np.log((R_sisi3_plus + S_sisi3_plus) / (R_sisi3_minus + S_sisi3_minus))
    logic_RS_3 = np.abs(R_sisi3_minus + S_sisi3_minus) > np.abs(R_sisi3_plus - S_sisi3_plus)
    basic_line_3[logic_RS_3] = temp[logic_RS_3]
    
    adv1_line_1 = R0_sisi1_kuad * basic_line_1 / 2 + (R_sisi1_plus * S_sisi1_plus - R_sisi1_minus * S_sisi1_minus) / 2
    adv1_line_2 = R0_sisi2_kuad * basic_line_2 / 2 + (R_sisi2_plus * S_sisi2_plus - R_sisi2_minus * S_sisi2_minus) / 2
    adv1_line_3 = R0_sisi3_kuad * basic_line_3 / 2 + (R_sisi3_plus * S_sisi3_plus - R_sisi3_minus * S_sisi3_minus) / 2
    
    adv2_line_1 = R0_sisi1_kuad * adv1_line_1 * 3 / 4 + (R_sisi1_plus**3 * S_sisi1_plus - R_sisi1_minus**3 * S_sisi1_minus) / 4
    adv2_line_2 = R0_sisi2_kuad * adv1_line_2 * 3 / 4 + (R_sisi2_plus**3 * S_sisi2_plus - R_sisi2_minus**3 * S_sisi2_minus) / 4
    adv2_line_3 = R0_sisi3_kuad * adv1_line_3 * 3 / 4 + (R_sisi3_plus**3 * S_sisi3_plus - R_sisi3_minus**3 * S_sisi3_minus) / 4
    
    basic_surf = AngleExcess / W_0
    adv1_surf = -W_0**2 * basic_surf - (T_0_1 * basic_line_1 + T_0_2 * basic_line_2 + T_0_3 * basic_line_3)
    adv2_surf = W_0**2 * adv1_surf / 3 - (T_0_1 * adv1_line_1 + T_0_2 * adv1_line_2 + T_0_3 * adv1_line_3) / 3
    temp = -(T_0_1 * basic_line_1 + T_0_2 * basic_line_2 + T_0_3 * basic_line_3)
    adv1_surf[W_0 == 0] = temp[W_0 == 0]
    temp = -(T_0_1 * adv1_line_1 + T_0_2 * adv1_line_2 + T_0_3 * adv1_line_3) / 3
    adv2_surf[W_0 == 0] = temp[W_0 == 0]
    
    basic_I_nm = (unit_M_sisi1 * basic_line_1[:, :, np.newaxis] +
        unit_M_sisi2 * basic_line_2[:, :, np.newaxis] +
        unit_M_sisi3 * basic_line_3[:, :, np.newaxis]
    )
    adv1_I_nm = (
        unit_M_sisi1 * adv1_line_1[:, :, np.newaxis] +
        unit_M_sisi2 * adv1_line_2[:, :, np.newaxis] +
        unit_M_sisi3 * adv1_line_3[:, :, np.newaxis]
    )
    adv2_I_nm = (
        unit_M_sisi1 * adv2_line_1[:, :, np.newaxis] +
        unit_M_sisi2 * adv2_line_2[:, :, np.newaxis] +
        unit_M_sisi3 * adv2_line_3[:, :, np.newaxis]
    )
    
    #
    x, y = np.meshgrid(np.arange(jumSegitiga), np.arange(jumSisi), indexing='ij')
    
    # K_1 terms
    K_1_1_plus  = div_f_plus.T[np.newaxis, :]  * adv1_surf[:, SegitigaPlus]
    K_1_1_minus = div_f_minus.T[np.newaxis, :] * adv1_surf[:, SegitigaMinus]
    K_1_2_plus  = div_f_plus.T[np.newaxis, :]  * adv2_surf[:, SegitigaPlus]
    K_1_2_minus = div_f_minus.T[np.newaxis, :] * adv2_surf[:, SegitigaMinus]
    
    # Free vertices reshaped
    FreeVertex_plus_ = FreeVertex_plus[x,:]
    FreeVertex_minus_ = FreeVertex_minus[x,:]
    
    # K_2 terms
    # Reshape div_f for broadcasting correctly over (jumSegitiga, jumSisi, 3)
    div_f_plus_ = np.tile(div_f_plus.T[np.newaxis, :, np.newaxis], (jumSegitiga, 1, 3))
    div_f_minus_ = np.tile(div_f_minus.T[np.newaxis, :, np.newaxis], (jumSegitiga, 1, 3))
    
    # Expand adv1_surf and adv2_surf terms to match shape (jumSegitiga, jumSisi, 3)
    adv1_surf_plus = np.tile(adv1_surf[:, SegitigaPlus][:, :, np.newaxis], (1, 1, 3))
    adv1_surf_minus = np.tile(adv1_surf[:, SegitigaMinus][:, :, np.newaxis], (1, 1, 3))
    adv2_surf_plus = np.tile(adv2_surf[:, SegitigaPlus][:, :, np.newaxis], (1, 1, 3))
    adv2_surf_minus = np.tile(adv2_surf[:, SegitigaMinus][:, :, np.newaxis], (1, 1, 3))
    
    # Now compute K_2_* terms correctly
    K_2_1_plus = div_f_plus_ * (
        (rho[:, SegitigaPlus, :] - FreeVertex_plus) * adv1_surf_plus +
        adv1_I_nm[:, SegitigaPlus, :]
    ) / 2
    
    K_2_1_minus = div_f_minus_ * (
        (rho[:, SegitigaMinus, :] - FreeVertex_minus) * adv1_surf_minus +
        adv1_I_nm[:, SegitigaMinus, :]
    ) / 2
    
    K_2_2_plus = div_f_plus_ * (
        (rho[:, SegitigaPlus, :] - FreeVertex_plus) * adv2_surf_plus +
        adv2_I_nm[:, SegitigaPlus, :] / 3
    ) / 2
    
    K_2_2_minus = div_f_minus_ * (
        (rho[:, SegitigaMinus, :] - FreeVertex_minus) * adv2_surf_minus +
        adv2_I_nm[:, SegitigaMinus, :] / 3
    ) / 2
    
    # K_3 terms
    K_3_1_plus = basic_I_nm[:, SegitigaPlus, :] - (
         (-W_0[:, SegitigaPlus] * basic_surf[:, SegitigaPlus])[:, :, np.newaxis] *
         faceNorm_m[:, SegitigaPlus, :]
    )
    K_3_2_plus = adv1_I_nm[:, SegitigaPlus, :] - (
         (W_0[:, SegitigaPlus] * adv1_surf[:, SegitigaPlus])[:, :, np.newaxis] *
         faceNorm_m[:, SegitigaPlus, :]
    )
    K_3_1_minus = basic_I_nm[:, SegitigaMinus, :] - (
         (-W_0[:, SegitigaMinus] * basic_surf[:, SegitigaMinus])[:, :, np.newaxis] *
         faceNorm_m[:, SegitigaMinus, :]
    )
    K_3_2_minus = adv1_I_nm[:, SegitigaMinus, :] - (
         (W_0[:, SegitigaMinus] * adv1_surf[:, SegitigaMinus])[:, :, np.newaxis] *
     faceNorm_m[:, SegitigaMinus, :]
    )
    
    logic_W_plus = np.tile(W_0[:, SegitigaPlus][:, :, np.newaxis],(1,1,3)) == 0
    logic_W_minus = np.tile(W_0[:, SegitigaMinus][:, :, np.newaxis],(1,1,3)) == 0
    
    K_3_1_plus[logic_W_plus] = (
         basic_I_nm[:, SegitigaPlus, :] +
         AngleExcess[:, SegitigaPlus][:, :, np.newaxis] * faceNorm_m[:, SegitigaPlus, :]
    )[logic_W_plus]
    K_3_2_plus[logic_W_plus] = 0
    
    K_3_1_minus[logic_W_minus] = (
         basic_I_nm[:, SegitigaMinus, :] +
         AngleExcess[:, SegitigaMinus][:, :, np.newaxis] * faceNorm_m[:, SegitigaMinus, :]
    )[logic_W_minus]
    K_3_2_minus[logic_W_minus] = 0
    
    # K_4 terms
    logic_K_4_plus = (x == SegitigaPlus[y])
    logic_K_4_minus = (x == SegitigaMinus[y])
    
    obs_point_x = titik_tengah[x, 0]
    obs_point_y = titik_tengah[x, 1]
    obs_point_z = titik_tengah[x, 2]
    obs_point = np.stack([obs_point_x, obs_point_y, obs_point_z], axis=2)  # shape (jumSegitiga, jumSisi, 3)
    
    K_4_1_plus = div_f_plus.T[np.newaxis, :, np.newaxis] * (
        np.cross(K_3_1_plus, obs_point - FreeVertex_plus) / 2
    )
    K_4_2_plus = div_f_plus.T[np.newaxis, :, np.newaxis] * (
        np.cross(K_3_2_plus, obs_point - FreeVertex_plus) / 2
    )
    K_4_1_minus = div_f_minus.T[np.newaxis, :, np.newaxis] * (
        np.cross(K_3_1_minus, obs_point - FreeVertex_minus) / 2
    )
    K_4_2_minus = div_f_minus.T[np.newaxis, :, np.newaxis] * (
        np.cross(K_3_2_minus, obs_point - FreeVertex_minus) / 2
    )
    
    K_4_1_plus[logic_K_4_plus] = 0
    K_4_2_plus[logic_K_4_plus] = 0
    K_4_1_minus[logic_K_4_minus] = 0
    K_4_2_minus[logic_K_4_minus] = 0
    
    return {
        "K_4_1_plus": K_4_1_plus,
        "K_4_1_minus": K_4_1_minus,
        "K_4_2_plus": K_4_2_plus,
        "K_4_2_minus": K_4_2_minus,
        "K_2_1_plus": K_2_1_plus,
        "K_2_1_minus": K_2_1_minus,
        "K_2_2_plus": K_2_2_plus,
        "K_2_2_minus": K_2_2_minus,
        "K_1_1_plus": K_1_1_plus,
        "K_1_1_minus": K_1_1_minus,
        "K_1_2_plus": K_1_2_plus,
        "K_1_2_minus": K_1_2_minus,
        "unit_M_sisi1": unit_M_sisi1,
        "unit_M_sisi2": unit_M_sisi2,
        "unit_M_sisi3": unit_M_sisi3,
    }