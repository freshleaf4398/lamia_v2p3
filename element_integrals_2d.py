import numpy as np
from tqdm.notebook import trange

def element_integrals_2d(SegitigaPlus,SegitigaMinus,FreeVertex_plus,FreeVertex_minus,titik_tengah,PanjangSisi,
                     jumSegitiga,jumSisi,faceNorm,rho_plus,rho_minus,luas,p,t,grid_points_3d):
                     

    f_plus = (PanjangSisi[:, None] * rho_plus) / (2 * luas[SegitigaPlus][:, None])
    f_minus = -(PanjangSisi[:, None] * rho_minus) / (2 * luas[SegitigaMinus][:, None])
    div_f_plus = PanjangSisi / luas[SegitigaPlus]
    div_f_minus = -PanjangSisi / luas[SegitigaMinus]
    
    p = p.T
    
    R_sisi1_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    R_sisi2_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    R_sisi3_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    R_sisi1_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    R_sisi2_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    R_sisi3_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    S_sisi1_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    S_sisi2_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    S_sisi3_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    S_sisi1_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    S_sisi2_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    S_sisi3_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    unit_M_sisi1 = np.zeros((jumSegitiga,3), dtype=np.double)
    unit_M_sisi2 = np.zeros((jumSegitiga,3), dtype=np.double)
    unit_M_sisi3 = np.zeros((jumSegitiga,3), dtype=np.double)
    W_0 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    rho = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga, 3), dtype=np.double)
    AngleExcess = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    basic_line_1 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv1_line_1 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv2_line_1 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    basic_line_2 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv1_line_2 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv2_line_2 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    basic_line_3 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv1_line_3 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv2_line_3 = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    basic_surf = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv1_surf = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    adv2_surf = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga), dtype=np.double)
    basic_I_nm = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga, 3), dtype=np.double)
    adv1_I_nm = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga, 3), dtype=np.double)
    adv2_I_nm = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSegitiga, 3), dtype=np.double)

    for a in trange(grid_points_3d.shape[0], desc="2D element integrals - Variables"):
        for b in range(grid_points_3d.shape[1]):
            obs_point = grid_points_3d[a, b, :].reshape(3)

            for m in range(jumSegitiga):
                source_tri = t[m, :].astype(int)
                q1 = p[source_tri[0], :]
                q2 = p[source_tri[1], :]
                q3 = p[source_tri[2], :]

                vec_sisi1 = q2 - q1
                vec_sisi2 = q3 - q2
                vec_sisi3 = q1 - q3

                source_point1 = (q2 + q1) / 2
                source_point2 = (q3 + q2) / 2
                source_point3 = (q1 + q3) / 2

                R_sisi1_plus[a,b,m]  = np.linalg.norm(obs_point - q2)
                R_sisi2_plus[a,b,m]  = np.linalg.norm(obs_point - q3)
                R_sisi3_plus[a,b,m]  = np.linalg.norm(obs_point - q1)
                R_sisi1_minus[a,b,m] = np.linalg.norm(obs_point - q1)
                R_sisi2_minus[a,b,m] = np.linalg.norm(obs_point - q2)
                R_sisi3_minus[a,b,m] = np.linalg.norm(obs_point - q3)

                unit_edge_sisi1 = vec_sisi1 / np.linalg.norm(vec_sisi1)
                unit_edge_sisi2 = vec_sisi2 / np.linalg.norm(vec_sisi2)
                unit_edge_sisi3 = vec_sisi3 / np.linalg.norm(vec_sisi3)

                S_sisi1_plus[a,b,m]  = np.dot(q2 - obs_point, unit_edge_sisi1)
                S_sisi2_plus[a,b,m]  = np.dot(q3 - obs_point, unit_edge_sisi2)
                S_sisi3_plus[a,b,m]  = np.dot(q1 - obs_point, unit_edge_sisi3)
                S_sisi1_minus[a,b,m] = np.dot(q1 - obs_point, unit_edge_sisi1)
                S_sisi2_minus[a,b,m] = np.dot(q2 - obs_point, unit_edge_sisi2)
                S_sisi3_minus[a,b,m] = np.dot(q3 - obs_point, unit_edge_sisi3)

                cross1 = np.cross(unit_edge_sisi1, faceNorm[m, :])
                cross2 = np.cross(unit_edge_sisi2, faceNorm[m, :])
                cross3 = np.cross(unit_edge_sisi3, faceNorm[m, :])
                unit_M_sisi1[m, :] = cross1 / np.linalg.norm(cross1)
                unit_M_sisi2[m, :] = cross2 / np.linalg.norm(cross2)
                unit_M_sisi3[m, :] = cross3 / np.linalg.norm(cross3)

                temp = np.dot(faceNorm[m, :], obs_point - q1)
                W_0[a,b,m] = 0 if temp < 1e-15 else temp

                rho[a,b,m,:] = obs_point - W_0[a,b,m] * faceNorm[m, :]

                T_0_1 = np.dot(obs_point - source_point1, unit_M_sisi1[m, :])
                T_0_2 = np.dot(obs_point - source_point2, unit_M_sisi2[m, :])
                T_0_3 = np.dot(obs_point - source_point3, unit_M_sisi3[m, :])

                R0_sisi1_kuad = R_sisi1_plus[a,b,m]**2 - S_sisi1_plus[a,b,m]**2
                R0_sisi2_kuad = R_sisi2_plus[a,b,m]**2 - S_sisi2_plus[a,b,m]**2
                R0_sisi3_kuad = R_sisi3_plus[a,b,m]**2 - S_sisi3_plus[a,b,m]**2

                a_1 = (q1 - obs_point) / np.linalg.norm(q1 - obs_point)
                a_2 = (q2 - obs_point) / np.linalg.norm(q2 - obs_point)
                a_3 = (q3 - obs_point) / np.linalg.norm(q3 - obs_point)

                x = 1 + np.dot(a_1,a_2) + np.dot(a_1,a_3) + np.dot(a_2,a_3)
                y = abs(np.dot(a_1, np.cross(a_2, a_3)))

                if W_0[a,b,m] >= 0:
                    AngleExcess[a,b,m] = 2 * np.arctan2(y, x)
                else:
                    AngleExcess[a,b,m] = -2 * np.arctan2(y, x)

                # --- basic_line terms ---
                if abs(R_sisi1_minus[a,b,m] + S_sisi1_minus[a,b,m]) > abs(R_sisi1_plus[a,b,m] - S_sisi1_plus[a,b,m]):
                    basic_line_1[a,b,m] = np.log((R_sisi1_plus[a,b,m] + S_sisi1_plus[a,b,m]) /
                                                 (R_sisi1_minus[a,b,m] + S_sisi1_minus[a,b,m]))
                else:
                    basic_line_1[a,b,m] = np.log((R_sisi1_minus[a,b,m] - S_sisi1_minus[a,b,m]) /
                                                 (R_sisi1_plus[a,b,m] - S_sisi1_plus[a,b,m]))

                if abs(R_sisi2_minus[a,b,m] + S_sisi2_minus[a,b,m]) > abs(R_sisi2_plus[a,b,m] - S_sisi2_plus[a,b,m]):
                    basic_line_2[a,b,m] = np.log((R_sisi2_plus[a,b,m] + S_sisi2_plus[a,b,m]) /
                                                 (R_sisi2_minus[a,b,m] + S_sisi2_minus[a,b,m]))
                else:
                    basic_line_2[a,b,m] = np.log((R_sisi2_minus[a,b,m] - S_sisi2_minus[a,b,m]) /
                                                 (R_sisi2_plus[a,b,m] - S_sisi2_plus[a,b,m]))

                if abs(R_sisi3_minus[a,b,m] + S_sisi3_minus[a,b,m]) > abs(R_sisi3_plus[a,b,m] - S_sisi3_plus[a,b,m]):
                    basic_line_3[a,b,m] = np.log((R_sisi3_plus[a,b,m] + S_sisi3_plus[a,b,m]) /
                                                 (R_sisi3_minus[a,b,m] + S_sisi3_minus[a,b,m]))
                else:
                    basic_line_3[a,b,m] = np.log((R_sisi3_minus[a,b,m] - S_sisi3_minus[a,b,m]) /
                                                 (R_sisi3_plus[a,b,m] - S_sisi3_plus[a,b,m]))

                adv1_line_1[a,b,m] = 0.5 * R0_sisi1_kuad * basic_line_1[a,b,m] + \
                    0.5 * (R_sisi1_plus[a,b,m]*S_sisi1_plus[a,b,m] - R_sisi1_minus[a,b,m]*S_sisi1_minus[a,b,m])
                adv1_line_2[a,b,m] = 0.5 * R0_sisi2_kuad * basic_line_2[a,b,m] + \
                    0.5 * (R_sisi2_plus[a,b,m]*S_sisi2_plus[a,b,m] - R_sisi2_minus[a,b,m]*S_sisi2_minus[a,b,m])
                adv1_line_3[a,b,m] = 0.5 * R0_sisi3_kuad * basic_line_3[a,b,m] + \
                    0.5 * (R_sisi3_plus[a,b,m]*S_sisi3_plus[a,b,m] - R_sisi3_minus[a,b,m]*S_sisi3_minus[a,b,m])

                adv2_line_1[a,b,m] = 0.75 * R0_sisi1_kuad * adv1_line_1[a,b,m] + \
                    0.25 * (S_sisi1_plus[a,b,m]*R_sisi1_plus[a,b,m]**3 - S_sisi1_minus[a,b,m]*R_sisi1_minus[a,b,m]**3)
                adv2_line_2[a,b,m] = 0.75 * R0_sisi2_kuad * adv1_line_2[a,b,m] + \
                    0.25 * (S_sisi2_plus[a,b,m]*R_sisi2_plus[a,b,m]**3 - S_sisi2_minus[a,b,m]*R_sisi2_minus[a,b,m]**3)
                adv2_line_3[a,b,m] = 0.75 * R0_sisi3_kuad * adv1_line_3[a,b,m] + \
                    0.25 * (S_sisi3_plus[a,b,m]*R_sisi3_plus[a,b,m]**3 - S_sisi3_minus[a,b,m]*R_sisi3_minus[a,b,m]**3)

                basic_surf[a,b,m] = AngleExcess[a,b,m] / W_0[a,b,m] if W_0[a,b,m] != 0 else 0

                if W_0[a,b,m] == 0:
                    adv1_surf[a,b,m] = -(T_0_1*basic_line_1[a,b,m] + T_0_2*basic_line_2[a,b,m] + T_0_3*basic_line_3[a,b,m])
                    adv2_surf[a,b,m] = -(T_0_1*adv1_line_1[a,b,m] + T_0_2*adv1_line_2[a,b,m] + T_0_3*adv1_line_3[a,b,m]) / 3
                else:
                    adv1_surf[a,b,m] = -W_0[a,b,m]**2 * basic_surf[a,b,m] - (
                        T_0_1*basic_line_1[a,b,m] + T_0_2*basic_line_2[a,b,m] + T_0_3*basic_line_3[a,b,m])
                    adv2_surf[a,b,m] = W_0[a,b,m]**2 * adv1_surf[a,b,m] / 3 - (
                        T_0_1*adv1_line_1[a,b,m] + T_0_2*adv1_line_2[a,b,m] + T_0_3*adv1_line_3[a,b,m]) / 3

                basic_I_nm[a,b,m,:] = (unit_M_sisi1[m,:]*basic_line_1[a,b,m] +
                                       unit_M_sisi2[m,:]*basic_line_2[a,b,m] +
                                       unit_M_sisi3[m,:]*basic_line_3[a,b,m])
                adv1_I_nm[a,b,m,:] = (unit_M_sisi1[m,:]*adv1_line_1[a,b,m] +
                                       unit_M_sisi2[m,:]*adv1_line_2[a,b,m] +
                                       unit_M_sisi3[m,:]*adv1_line_3[a,b,m])
                adv2_I_nm[a,b,m,:] = (unit_M_sisi1[m,:]*adv2_line_1[a,b,m] +
                                       unit_M_sisi2[m,:]*adv2_line_2[a,b,m] +
                                       unit_M_sisi3[m,:]*adv2_line_3[a,b,m])
    
    K_1_1_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi), dtype=np.double)
    K_1_1_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi), dtype=np.double)
    K_1_2_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi), dtype=np.double)
    K_1_2_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi), dtype=np.double)
    K_2_1_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_2_1_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_2_2_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_2_2_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_3_1_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_3_1_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_3_2_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_3_2_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_4_1_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_4_1_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_4_2_plus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)
    K_4_2_minus = np.zeros((grid_points_3d.shape[0], grid_points_3d.shape[1], jumSisi, 3), dtype=np.double)

    # Loop 1: K_1 matrices
    for a in trange(grid_points_3d.shape[0], desc="2D element integrals - K_1 matrices"):
        for b in range(grid_points_3d.shape[1]):
            for m in range(jumSisi):
                K_1_1_plus[a,b,m]  = div_f_plus[m] * adv1_surf[a,b,SegitigaPlus[m]]
                K_1_1_minus[a,b,m] = div_f_minus[m] * adv1_surf[a,b,SegitigaMinus[m]]
                K_1_2_plus[a,b,m]  = div_f_plus[m] * adv2_surf[a,b,SegitigaPlus[m]]
                K_1_2_minus[a,b,m] = div_f_minus[m] * adv2_surf[a,b,SegitigaMinus[m]]

    # Loop 2: K_2 matrices
    for a in trange(grid_points_3d.shape[0], desc="2D element integrals - K_2 matrices"):
        for b in range(grid_points_3d.shape[1]):
            for m in range(jumSisi):
                sc = rho[a,b,SegitigaPlus[m],:] - FreeVertex_plus[m,:]
                sc = sc * adv1_surf[a,b,SegitigaPlus[m]]
                sc = sc + adv1_I_nm[a,b,SegitigaPlus[m],:]
                K_2_1_plus[a,b,m,:] = div_f_plus[m] * sc / 2

                sc = rho[a,b,SegitigaMinus[m],:] - FreeVertex_minus[m,:]
                sc = sc * adv1_surf[a,b,SegitigaMinus[m]]
                sc = sc + adv1_I_nm[a,b,SegitigaMinus[m],:]
                K_2_1_minus[a,b,m,:] = div_f_minus[m] * sc / 2

                sc = rho[a,b,SegitigaPlus[m],:] - FreeVertex_plus[m,:]
                sc = sc * adv2_surf[a,b,SegitigaPlus[m]]
                sc = sc + adv2_I_nm[a,b,SegitigaPlus[m],:] / 3
                K_2_2_plus[a,b,m,:] = div_f_plus[m] * sc / 2

                sc = rho[a,b,SegitigaMinus[m],:] - FreeVertex_minus[m,:]
                sc = sc * adv2_surf[a,b,SegitigaMinus[m]]
                sc = sc + adv2_I_nm[a,b,SegitigaMinus[m],:] / 3
                K_2_2_minus[a,b,m,:] = div_f_minus[m] * sc / 2

    # Loop 3: K_3 and K_4 matrices
    for a in trange(grid_points_3d.shape[0], desc="2D element integrals - K_3 and K_4 matrices"):
        for b in range(grid_points_3d.shape[1]):
            obs_point = grid_points_3d[a, b, :].reshape(3)
            for m in range(jumSisi):
                # --- Plus ---
                if W_0[a,b,SegitigaPlus[m]] == 0:
                    temp = -AngleExcess[a,b,SegitigaPlus[m]] * faceNorm[SegitigaPlus[m],:]
                    K_3_1_plus[a,b,m,:] = basic_I_nm[a,b,SegitigaPlus[m],:] - temp
                    temp = 0
                    K_3_2_plus[a,b,m,:] = adv1_I_nm[a,b,SegitigaPlus[m],:] - temp
                else:
                    temp = -W_0[a,b,SegitigaPlus[m]] * basic_surf[a,b,SegitigaPlus[m]] * faceNorm[SegitigaPlus[m],:]
                    K_3_1_plus[a,b,m,:] = basic_I_nm[a,b,SegitigaPlus[m],:] - temp
                    temp = W_0[a,b,SegitigaPlus[m]] * adv1_surf[a,b,SegitigaPlus[m]] * faceNorm[SegitigaPlus[m],:]
                    K_3_2_plus[a,b,m,:] = adv1_I_nm[a,b,SegitigaPlus[m],:] - temp

                if np.allclose(grid_points_3d[a,b,:], titik_tengah[SegitigaPlus[m],:]):
                    K_4_1_plus[a,b,m,:] = np.zeros(3)
                    K_4_2_plus[a,b,m,:] = np.zeros(3)
                else:
                    K_4_1_plus[a,b,m,:] = div_f_plus[m] * np.cross(K_3_1_plus[a,b,m,:], obs_point - FreeVertex_plus[m,:]) / 2
                    K_4_2_plus[a,b,m,:] = div_f_plus[m] * np.cross(K_3_2_plus[a,b,m,:], obs_point - FreeVertex_plus[m,:]) / 2

                # --- Minus ---
                if W_0[a,b,SegitigaMinus[m]] == 0:
                    temp = -AngleExcess[a,b,SegitigaMinus[m]] * faceNorm[SegitigaMinus[m],:]
                    K_3_1_minus[a,b,m,:] = basic_I_nm[a,b,SegitigaMinus[m],:] - temp
                    temp = 0
                    K_3_2_minus[a,b,m,:] = adv1_I_nm[a,b,SegitigaMinus[m],:] - temp
                else:
                    temp = -W_0[a,b,SegitigaMinus[m]] * basic_surf[a,b,SegitigaMinus[m]] * faceNorm[SegitigaMinus[m],:]
                    K_3_1_minus[a,b,m,:] = basic_I_nm[a,b,SegitigaMinus[m],:] - temp
                    temp = W_0[a,b,SegitigaMinus[m]] * adv1_surf[a,b,SegitigaMinus[m]] * faceNorm[SegitigaMinus[m],:]
                    K_3_2_minus[a,b,m,:] = adv1_I_nm[a,b,SegitigaMinus[m],:] - temp

                if np.allclose(grid_points_3d[a,b,:], titik_tengah[SegitigaMinus[m],:]):
                    K_4_1_minus[a,b,m,:] = np.zeros(3)
                    K_4_2_minus[a,b,m,:] = np.zeros(3)
                else:
                    K_4_1_minus[a,b,m,:] = div_f_minus[m] * np.cross(K_3_1_minus[a,b,m,:], obs_point - FreeVertex_minus[m,:]) / 2
                    K_4_2_minus[a,b,m,:] = div_f_minus[m] * np.cross(K_3_2_minus[a,b,m,:], obs_point - FreeVertex_minus[m,:]) / 2

    
    return {
        "K_4_1_plus": K_4_1_plus,
        "K_4_1_minus": K_4_1_minus,
        "K_4_2_plus": K_4_2_plus,
        "K_4_2_minus": K_4_2_minus,
        "K_3_1_plus": K_3_1_plus,
        "K_3_1_minus": K_3_1_minus,
        "K_3_2_plus": K_3_2_plus,
        "K_3_2_minus": K_3_2_minus,
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