def triangulation(p, t, p_ff, t_ff, r=20e-6, theta_step=0.025):
    """
    Preprocess mesh geometry and far-field setup.

    Parameters
    ----------
    p : np.ndarray
        Node coordinates (3, N).
    t : np.ndarray
        Triangles (M, 3).
    p_ff : np.ndarray
        Far-field node coordinates (3, N_ff).
    t_ff : np.ndarray
        Far-field triangles (M_ff, 3).
    r : float
        Observer circle radius.
    theta_step : float
        Step size for observer angular sampling.

    Returns
    -------
    dict
        Dictionary containing all computed geometry quantities.
    """
    import meshio
    import numpy as np
    import pandas as pd
    from numpy.linalg import norm
    from scipy.io import savemat, loadmat
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt
    from math import factorial
    
    # === Main mesh ===
    jumPoint = p.shape[1]
    jumSegitiga = t.shape[0]

    pts = p[:, t.T]  # shape: (3, 3, N)
    v1 = pts[:, 2, :] - pts[:, 1, :]
    v2 = pts[:, 0, :] - pts[:, 2, :]
    v3 = pts[:, 1, :] - pts[:, 0, :]

    cross_prod = np.cross(v1.T, v2.T)
    luas = norm(cross_prod, axis=1) / 2
    keliling = norm(v1.T, axis=1) + norm(v2.T, axis=1) + norm(v3.T, axis=1)
    faceNorm = cross_prod / np.linalg.norm(cross_prod, axis=1)[:, None]
    titik_tengah = np.mean(pts, axis=1).T  # (N, 3)

    # === Define sides, plus-minus triangles ===
    sisi = []
    SegitigaPlus = []
    SegitigaMinus = []

    for i in range(jumSegitiga):
        N = t[i]
        for j in range(i + 1, jumSegitiga):
            M = t[j]
            shared = np.isin(N, M)
            if np.sum(shared) == 2:
                edge_nodes = list(set(M) & set(N))
                sisi.append(edge_nodes)
                SegitigaPlus.append(i)
                SegitigaMinus.append(j)

    sisi = np.array(sisi)
    SegitigaPlus = np.array(SegitigaPlus)
    SegitigaMinus = np.array(SegitigaMinus)
    jumSisi = len(sisi)

    # === Side lengths ===
    PanjangSisi = np.array([norm(p[:, e[0]] - p[:, e[1]]) for e in sisi])

    # === Free vertices and rho vectors ===
    sisi_set = [set(pair) for pair in sisi]
    t_set = [set(tri) for tri in t]

    free_idx_plus = [list(t_set[SegitigaPlus[i]] - sisi_set[i])[0] for i in range(jumSisi)]
    free_idx_minus = [list(t_set[SegitigaMinus[i]] - sisi_set[i])[0] for i in range(jumSisi)]

    FreeVertex_plus = p[:, free_idx_plus].T
    FreeVertex_minus = p[:, free_idx_minus].T
    rho_plus = titik_tengah[SegitigaPlus] - FreeVertex_plus
    rho_minus = titik_tengah[SegitigaMinus] - FreeVertex_minus

    # === Logic matrices ===
    logic_identical = np.eye(jumSegitiga, dtype=int)
    logic_far = np.zeros((jumSegitiga, jumSegitiga), dtype=int)
    logic_adjacent = np.zeros((jumSegitiga, jumSegitiga), dtype=int)
    logic_touch = np.zeros((jumSegitiga, jumSegitiga), dtype=int)

    jar_center = np.linalg.norm(titik_tengah[:, None, :] - titik_tengah[None, :, :], axis=2)

    for i in range(jumSegitiga):
        obs = t[i]
        for j in range(jumSegitiga):
            source = t[j]

            jar_center[i, j] = np.linalg.norm(titik_tengah[i] - titik_tengah[j])

            avg_perimeter = 0.5 * (keliling[i] + keliling[j])
            if jar_center[i, j] > avg_perimeter:
                logic_far[i, j] = True

            shared_vertices = np.intersect1d(obs, source).size
            logic_adjacent[i, j] = (shared_vertices == 2)
            logic_touch[i, j] = (shared_vertices == 1)

    # === Far-field triangles ===
    pts_ff = p_ff[:, t_ff.T]  # shape: (3, 3, N)
    v1 = pts_ff[:, 2, :] - pts_ff[:, 1, :]
    v2 = pts_ff[:, 0, :] - pts_ff[:, 2, :]
    v3 = pts_ff[:, 1, :] - pts_ff[:, 0, :]
    cross_ff = np.cross(v1.T, v2.T)
    luas_ff = norm(cross_ff, axis=1) / 2
    keliling_ff = norm(v1.T, axis=1) + norm(v2.T, axis=1) + norm(v3.T, axis=1)
    faceNorm_ff = cross_ff / np.linalg.norm(cross_ff, axis=1)[:, None]
    titik_tengah_ff = np.mean(pts_ff, axis=1).T
    jumSegitiga_ff = t_ff.shape[0]

    return dict(
        p=p.T, t=t,
        PanjangSisi=PanjangSisi,
        SegitigaMinus=SegitigaMinus,
        SegitigaPlus=SegitigaPlus,
        jumSegitiga=jumSegitiga,
        jumSisi=jumSisi,
        titik_tengah=titik_tengah,
        rho_plus=rho_plus,
        rho_minus=rho_minus,
        FreeVertex_plus=FreeVertex_plus,
        FreeVertex_minus=FreeVertex_minus,
        luas=luas,
        sisi=sisi,
        keliling=keliling,
        jar_center=jar_center,
        faceNorm=faceNorm,
        logic_touch=logic_touch,
        logic_identical=logic_identical,
        logic_far=logic_far,
        logic_adjacent=logic_adjacent,
        luas_ff=luas_ff,
        faceNorm_ff=faceNorm_ff,
        titik_tengah_ff=titik_tengah_ff,
        jumSegitiga_ff=jumSegitiga_ff,
    )