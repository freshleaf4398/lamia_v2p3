def special_integrals(geo):
    """
    Compute the I1 and I2 integrals using Arcioni-like method.
    Requires geometry dict from triangulation().
    """
    import numpy as np
    
    # Unpack geometry
    SegitigaPlus = geo["SegitigaPlus"]
    SegitigaMinus = geo["SegitigaMinus"]
    FreeVertex_plus = geo["FreeVertex_plus"]
    FreeVertex_minus = geo["FreeVertex_minus"]
    PanjangSisi = geo["PanjangSisi"]
    keliling = geo["keliling"]
    luas = geo["luas"]
    #p = geo["p"].T   # back to (N,3)
    p = geo["p"] 
    sisi = geo["sisi"]
    jumSisi = geo["jumSisi"]
    logic_identical = geo["logic_identical"]

    # Precompute logic matrices
    logic_identical_pp = logic_identical[SegitigaPlus[:, None], SegitigaPlus[None, :]]
    logic_identical_pm = logic_identical[SegitigaPlus[:, None], SegitigaMinus[None, :]]
    logic_identical_mp = logic_identical[SegitigaMinus[:, None], SegitigaPlus[None, :]]
    logic_identical_mm = logic_identical[SegitigaMinus[:, None], SegitigaMinus[None, :]]

    # Initialize integrals
    I_1_pp = np.zeros((jumSisi, jumSisi))
    I_1_pm = np.zeros((jumSisi, jumSisi))
    I_1_mp = np.zeros((jumSisi, jumSisi))
    I_1_mm = np.zeros((jumSisi, jumSisi))
    I_2_pp = np.zeros((jumSisi, jumSisi))
    I_2_pm = np.zeros((jumSisi, jumSisi))
    I_2_mp = np.zeros((jumSisi, jumSisi))
    I_2_mm = np.zeros((jumSisi, jumSisi))

    def compute_integrals(a, b, seg_idx, FV, PanjangSisi, keliling, luas, p, sisi):
        if a == b:
            edge1 = PanjangSisi[a]
            vec2 = FV[a, :] - p[sisi[a, 0], :]
            vec3 = FV[a, :] - p[sisi[a, 1], :]
            edge2 = np.linalg.norm(vec2).item()
            edge3 = np.linalg.norm(vec3).item()
        else:
            edge1 = PanjangSisi[a]
            edge2 = PanjangSisi[b]
            edge3 = keliling[seg_idx[a]] - edge1 - edge2
            edge1, edge3 = edge3, edge1

        half_kel = keliling[seg_idx[a]] / 2
        L = luas[seg_idx[a]]

        log_terms = (
            np.log(1 - edge1 / half_kel).item() / edge1 +
            np.log(1 - edge2 / half_kel).item() / edge2 +
            np.log(1 - edge3 / half_kel).item() / edge3
        )
        I1 = -4 / 3 * L**2 * log_terms

        if a == b:
            p1 = edge1 * (10 + 3 * (edge3**2 - edge1**2) / edge2**2 - 3 * (edge1**2 - edge2**2) / edge3**2)
            p2 = edge2 * (5 - 3 * (edge1**2 - edge2**2) / edge3**2 - 2 * (edge2**2 - edge3**2) / edge1**2)
            p3 = edge3 * (5 + 3 * (edge3**2 - edge1**2) / edge2**2 + 2 * (edge2**2 - edge3**2) / edge1**2)
            p4 = 2 * np.log(1 - edge1 / half_kel).item() / edge1 * (edge1**2 - 3 * edge2**2 - 3 * edge3**2 - 8 * L**2 / edge1**2)
            p5 = 4 * np.log(1 - edge2 / half_kel).item() / edge2 * (edge1**2 - 2 * edge2**2 - 4 * edge3**2 + 6 * L**2 / edge2**2)
            p6 = 4 * np.log(1 - edge3 / half_kel).item() / edge3 * (edge1**2 - 4 * edge2**2 - 2 * edge3**2 + 6 * L**2 / edge3**2)
            I2 = (p1 - p2 - p3 + p4 + p5 + p6) * L**2 / 30
        else:
            p1 = edge1 * (-10 + (edge3**2 - edge1**2) / edge2**2 - (edge1**2 - edge2**2) / edge3**2)
            p2 = edge2 * (5 + (edge1**2 - edge2**2) / edge3**2 - 6 * (edge2**2 - edge3**2) / edge1**2)
            p3 = edge3 * (5 - (edge3**2 - edge1**2) / edge2**2 + 6 * (edge2**2 - edge3**2) / edge1**2)
            p4 = 12 * np.log(1 - edge1 / half_kel).item() / edge1 * (2 * edge1**2 - edge2**2 - edge3**2 + 4 * L**2 / edge1**2)
            p5 = 2 * np.log(1 - edge2 / half_kel).item() / edge2 * (9 * edge1**2 - 3 * edge2**2 - edge3**2 + 4 * L**2 / edge2**2)
            p6 = 2 * np.log(1 - edge3 / half_kel).item() / edge3 * (9 * edge1**2 - edge2**2 - 3 * edge3**2 + 4 * L**2 / edge3**2)
            I2 = (p1 + p2 + p3 + p4 + p5 + p6) * L**2 / 60

        return I1, I2

    # === Loop for computation ===
    for a in range(jumSisi):
        for b in range(jumSisi):
            if logic_identical_pp[a, b]:
                I_1_pp[a, b], I_2_pp[a, b] = compute_integrals(a, b, SegitigaPlus, FreeVertex_plus, PanjangSisi, keliling, luas, p, sisi)
            if logic_identical_pm[a, b]:
                I_1_pm[a, b], I_2_pm[a, b] = compute_integrals(a, b, SegitigaPlus, FreeVertex_plus, PanjangSisi, keliling, luas, p, sisi)
            if logic_identical_mp[a, b]:
                I_1_mp[a, b], I_2_mp[a, b] = compute_integrals(a, b, SegitigaMinus, FreeVertex_minus, PanjangSisi, keliling, luas, p, sisi)
            if logic_identical_mm[a, b]:
                I_1_mm[a, b], I_2_mm[a, b] = compute_integrals(a, b, SegitigaMinus, FreeVertex_minus, PanjangSisi, keliling, luas, p, sisi)

    return {
        "I_1_pp": I_1_pp,
        "I_1_pm": I_1_pm,
        "I_1_mp": I_1_mp,
        "I_1_mm": I_1_mm,
        "I_2_pp": I_2_pp,
        "I_2_pm": I_2_pm,
        "I_2_mp": I_2_mp,
        "I_2_mm": I_2_mm,
    }