import numpy as np

def sta_hel2em(T):
    """The method converts from a basis of electric and magnetic multipoles
    to a basis of pure helicity multipoles and viceversa. (The
    transformation is exactly the same).
    INPUTS:
    - T: [N x N] matrix containing the T-matrix of an object. The ordering
         of the different entries of the matrix in terms of multipole
         degree and order must be the ones given by the conventions of
         the basis 'electric-magnetic' or 'helicity' of the class clsT.
    OUTPUTS:
    - new_T: [N x N] matrix containing the input T-matrix
                 but expressed in a different basis of vector spherical
                 wave functions. If the input T-matrix is expressed in the
                 basis of electric and magnetic multipoles the returned
                 T-matrix will be expressed in a basis of pure helicity
                 vector spherical multipoles and viceversa."""
    if T.ndim == 2:
        l_VSH = int(T.shape[0]/2)
        Trans = np.eye(2*l_VSH)
        Trans[:l_VSH,l_VSH:] = np.eye(l_VSH)
        Trans[l_VSH:,:l_VSH] = np.eye(l_VSH)
        Trans[l_VSH:,l_VSH:] = -np.eye(l_VSH)
        new_T = 1/2*Trans@T@Trans
    elif T.ndim == 3:
        new_T = np.zeros((T.shape[0],T.shape[1],T.shape[2]), dtype=complex)
        for a in range(T.shape[0]):
            T_ = T[a,:,:]
            l_VSH = int(T_.shape[0]/2)
            Trans = np.eye(2*l_VSH)
            Trans[:l_VSH,l_VSH:] = np.eye(l_VSH)
            Trans[l_VSH:,:l_VSH] = np.eye(l_VSH)
            Trans[l_VSH:,l_VSH:] = -np.eye(l_VSH)
            new_T[a,:,:] = 1/2*Trans@T_@Trans
    else:
        raise ValueError(f"Expected 2D or 3D array, got {T.ndim}D instead.")
    return new_T