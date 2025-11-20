import numpy as np
import scipy
from scipy import special
import math
import warnings
import scipy.sparse as sparse
import scipy.sparse.linalg

def make_mn(N):
    N = int(N)
    mn = np.zeros([N*(N+2),2],int)
    index = 0
    for n in range(1,N+1):
        for m in range(-n,n+1):
            mn[index,:] = [m,n]
            index = index+1
    return mn

def sphericalBesselFunction(n,J,x):
        """spherical_bessel_function computes the values of the spherical 
        Bessel function of first or second kind (J = 1 or 2) or the 
        spherical Hankel functions of first or second kind (J =3 or 4) for 
        the points specified by vector x. 
        INPUTS:
        - n:   order of the bessel function j_n, y_n or h_n^(1,2).
        - J:   this input indicates if the user wants to obtain the bessel
                         function of the first kind, sqrt(pi/2/x)J_(n+0.5), second kind, 
                    sqrt(pi/2/x)Y_(n+0.5), or the Hankel functions of first or secon
                    kind, sqrt(pi/2/x)H_(n+0.5)^(1,2). The values of J are 1,2,3 or 4
                    respectively.
        - x:   points where the user wants to evaluate the function.
        OUTPUTS:
        bessel_values: [length(x) x 1] array with the values of the desired
                    function in the points specified by the user."""
         
        if J == 1:
                bessel_value = special.jv(n+0.5,x)
        elif J == 2:
                bessel_value = special.yv(n+0.5,x)
        elif J == 3:
                bessel_value = special.hankel1(n+0.5,x)
        elif J == 4:
                bessel_value = special.hankel2(n+0.5,x)
        bessel_value = np.sqrt(np.pi/2/x)*bessel_value;
        return bessel_value 

def associatedLegendrePolynomial(n,m,x):
    """This function computes the values of the associated legendre
                polynomial P_nm (x) = (-1)^mP_n^m (x) for the given points x.
    INPUTS:
      - n:   index n of the associated Legendre polynomial P_n^m (x).
      - m:   index m of the associated Legendre polynomial P_n^m (x).   
      - x:   array with the points where the user wants to evaluate the
             specified associated Legendre polynomial.
      OUTPUTS:
      - legendre_value:  [length(x) x 1] array containing the values of
                         the associated legendre polynomial evaluated at
                         the points specified by the user."""
    if np.isnan(m):
        x = np.squeeze(x)
        ms = np.arange(-n,n+1)
        legendre_value = np.zeros([x.size,ms.size])
        for i_m in range(ms.size):
            legendre_value[:,i_m] = special.lpmv(ms[i_m],n,x)
    else:
        if abs(m) > n:
            legendre_value = np.zeros(x.shape)
        else:
            legendre_value = special.lpmv(m,n,x) #(1-2*(m%2))*
    return legendre_value

def wigner3j(n1,n2,n,m1,m2,m):
    comp = np.array([n1,n2,n,m1,m2,m])
    condition = (m1+m2!=-m) or np.any(np.abs(comp[3:6])>np.abs(comp[0:3])) or (n>(n1+n2)) or (n < abs(n1-n2))
    if condition:
        value = 0
    else:
        if not (m1 or m2 or m): #Special case useful in the addition theorem of vector spherical harmonics:
            N = n1+n2+n
            if np.mod(N,2) == 1:
                value = 0
            else:
                g = np.int64(N/2)
                factor1 = math.factorial(2*(g-n1))*math.factorial(2*(g-n2))*math.factorial(2*(g-n))/math.factorial(2*g+1)
                factor1 = np.sqrt(factor1)
                factor2 = math.factorial(g)/(math.factorial(g-n1)*math.factorial(g-n2)*math.factorial(g-n))
                value = (-1.)**g*factor1*factor2
        else: #general_case:
            factor = (-1.)**(n1-n2-m)
            triangle_coefficient = math.factorial(-n1+n2+n)*math.factorial(n1-n2+n)*math.factorial(n1+n2-n)/math.factorial(n1+n2+n+1)
            factor1 = math.factorial(n1+m1)*math.factorial(n1-m1)*math.factorial(n2+m2)*math.factorial(n2-m2)*math.factorial(n+m)*math.factorial(n-m)
            factor = factor*np.sqrt(triangle_coefficient*factor1)

            c = np.array([n1+n2-n,n1-m1,n2+m2,n-n2+m1,n-n1-m2,0])
            index1 = np.min(np.append([0],c[3:5]))
            index2 = np.min(c[:3])
            value = 0
            for t in range(-index1,index2+1):
                term = c+np.array([-t,-t,-t,t,t,t])
                value = value+(-1.)**t/np.prod(scipy.special.factorial(term))
            value = factor*value
    return value

def addition_theorem_p_values(n1_max,n3_max):
    mn = make_mn(n3_max)
    uv = make_mn(n1_max)
    size_mn = mn.shape[0]
    size_uv = uv.shape[0]
    AB = [[np.zeros([2,n1_max+n3_max]) for j_ab in range(size_mn)] for i_ab in range(size_uv)]
    for i_mn in range(size_mn):
        m = mn[i_mn,0]
        n = mn[i_mn,1]
        for i_uv in range(size_uv):
            u = uv[i_uv,0]
            v = uv[i_uv,1]
            factor_a5_a6 = (-1.)**(m-u)*np.sqrt(math.factorial(n+m)*math.factorial(v-u)/(math.factorial(n-m)*math.factorial(v+u)))
            factor_a3_b3 = 1j**(v-n)*(2*v+1)/(2*v*(v+1))
            p_values = np.arange(np.max([np.abs(n-v),np.abs(u-m)]),n+v+1)
            p_vector_A = np.zeros([1,len(p_values)],dtype=complex)
            p_vector_B = np.zeros([1,len(p_values)],dtype=complex)
            for i_p in range(len(p_values)):
                p = p_values[i_p]
                factor_a5_a6_p = (2*p+1)*np.sqrt(math.factorial(p-m+u)/math.factorial(p+m-u))*wigner3j(n,v,p,m,-u,-(m-u))
                a5_p = factor_a5_a6_p*wigner3j(n,v,p,0,0,0)
                a6_p = factor_a5_a6_p*wigner3j(n,v,p-1,0,0,0)
                factor_a3_p =  (1j)**p*(n*(n+1)+v*(v+1)-p*(p+1))
                factor_b3_p =  -(1j)**p*np.sqrt((n+v+1+p)*(n+v+1-p)*(p+n-v)*(p-n+v))
                p_vector_A[0,i_p] = a5_p*factor_a3_p
                p_vector_B[0,i_p] = a6_p*factor_b3_p
            gamma_mn = np.sqrt((2*n+1)*math.factorial(n-m)/(4*np.pi*n*(n+1)*math.factorial(n+m)))
            gamma_uv = np.sqrt((2*v+1)*math.factorial(v-u)/(4*np.pi*v*(v+1)*math.factorial(v+u)))
            Coef_A = gamma_mn/gamma_uv*(-1.)**u
            Coef_B = -Coef_A
            factor = factor_a5_a6*factor_a3_b3
            AB[i_uv][i_mn] = factor*np.vstack((Coef_A*p_vector_A,Coef_B*p_vector_B))
    return AB

def addition_theorem(max_n1,max_n3,pos2orig,k_mod,ReciprocalValues=False,CartesianInput=True,change_of_type=True):
    if CartesianInput:
        R = np.sqrt(np.sum(pos2orig**2,1))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='invalid value encountered in true_divide')
            cos_theta = pos2orig[:,2]/R
        indices = R ==0
        cos_theta[indices] = 1
        phi = np.arctan2(pos2orig[:,1],pos2orig[:,0])
    else:
        R = pos2orig[:,0]
        indices = R==0  #It might be needed later
        cos_theta = np.cos(pos2orig[:,1])
        phi = pos2orig[:,2]
    num_points = R.shape[0]
    
    AB_p_coefs = addition_theorem_p_values(max_n1,max_n3)
    
    #Calculating the indices of the VSH:
    mn_pairs = make_mn(max_n3)
    uv_pairs = make_mn(max_n1)
    num_mn_pairs = mn_pairs.shape[0]
    num_uv_pairs = uv_pairs.shape[0]
    
    #Computing the coordinate-dependent part of the translation
    #coefficients (bessel and legendre polynomials):
    p = np.arange(0,max_n1+max_n3+1)
    num_p = len(p)
    h_values = np.zeros([num_p,num_points],dtype=complex)
    num_p_mu_pairs = p[-1]*(p[-1]+2)+1
    P_values = np.zeros([num_p_mu_pairs,num_points])
    index = np.array([-1])
    x = k_mod*R
    
    if np.any(indices):
    	x[indices] = 1e-30
    for i_p in range(num_p):
    	if change_of_type:
    		h_values[i_p,:] = sphericalBesselFunction(p[i_p],3,x)
    	else:
    		h_values[i_p,:] = sphericalBesselFunction(p[i_p],1,x)
    	index = index[-1]+np.arange(1,2*p[i_p]+2)
    	P_values[index,:] = associatedLegendrePolynomial(p[i_p],np.nan,cos_theta).T
    #Combining the coordiante-dependent and the non coordinate-dependent
    #parts:
    A = np.zeros([num_uv_pairs,num_mn_pairs,num_points],dtype=complex)
    B = np.zeros([num_uv_pairs,num_mn_pairs,num_points],dtype=complex)
    
    if ReciprocalValues:
    	A_reciprocal = np.zeros([num_uv_pairs,num_mn_pairs,num_points],dtype=complex)
    	B_reciprocal = np.zeros([num_uv_pairs,num_mn_pairs,num_points],dtype=complex)
    for i_mn in range(num_mn_pairs):
    	n = mn_pairs[i_mn,1]
    	m = mn_pairs[i_mn,0]
    	for i_uv in range(num_uv_pairs):
    		v = uv_pairs[i_uv,1]
    		u = uv_pairs[i_uv,0]
    		#index_mn = m+(n-1)*(n+1)+(n+1)
    		p_coefs = AB_p_coefs[i_uv][i_mn]
    		p_values = np.arange(np.max([np.abs(n-v),np.abs(u-m)]),n+v+1)
    		not_zero = np.logical_or((p_coefs[0,:]!= 0), (p_coefs[1,:] != 0))
    		p_coefs = p_coefs[:,not_zero]
    		p_values = p_values[not_zero]
    		P_upper_index = m-u
    		A_uvmn = 0
    		B_uvmn = 0
    		for i_p in range(np.sum(not_zero)):
    			p = p_values[i_p]
    			P_index = P_upper_index+(p-1)*(p+1)+(p+1)
    			factor = h_values[p,:]*P_values[P_index,:]
    			if p_coefs[0,i_p] !=0:
    				A_uvmn = A_uvmn + p_coefs[0,i_p]*factor
    			if p_coefs[1,i_p] !=0:
    				B_uvmn = B_uvmn + p_coefs[1,i_p]*factor
    		exp_factor = np.exp(1j*(m-u)*phi.T)
    		A[i_uv,i_mn,:] = A_uvmn*exp_factor
    		B[i_uv,i_mn,:] = B_uvmn*exp_factor
    		
    		if ReciprocalValues: #Computing reciprocal values:
    			A_uvmn_reciprocal = 0
    			B_uvmn_reciprocal = 0
    			for i_p in range(np.sum(not_zero)):
    				p = p_values[i_p]
    				P_index = P_upper_index+(p-1)*(p+1)+(p+1)
    				factor = h_values[p,:]*P_values[P_index,:]
    				minus_factor = (-1.)**(p+P_upper_index)
    				if p_coefs[0,i_p] !=0:
    					A_uvmn_reciprocal = A_uvmn_reciprocal + minus_factor*p_coefs[0,i_p]*factor
    				if p_coefs[1,i_p] !=0:
    					B_uvmn_reciprocal = B_uvmn_reciprocal + minus_factor*p_coefs[1,i_p]*factor
    			exp_factor = exp_factor*(-1.)**P_upper_index
    			A_reciprocal[i_uv,i_mn,:] = A_uvmn_reciprocal*exp_factor
    			B_reciprocal[i_uv,i_mn,:] = B_uvmn_reciprocal*exp_factor
    
    if ReciprocalValues:
    	return A,B,A_reciprocal,B_reciprocal
    else:
    	return A,B

def sta_relative_positions_matrix(scatterer_positions):
    """Calculates the relative positions between scatterer objects.
    INPUTS:
    - scatterer_positions:
    OUTPUS:
    - Oi_Oj:   Option 1: [num_scatterers(num_scatterers-1) x 3] matrix containing the relative position between all the scatterers. """

    n_scatterers = scatterer_positions.shape[0]
    r_3D = np.transpose(scatterer_positions[:,:,None],[0,2,1]) #moving the coordinates to the third dimmension. 
    r_3D = np.tile(r_3D,(1,n_scatterers,1))
    O1_O2_vectors = r_3D-np.transpose(r_3D,[1,0,2])
    x = O1_O2_vectors[:,:,0]
    y = O1_O2_vectors[:,:,1]
    z = O1_O2_vectors[:,:,2]
    upper_triangular = np.triu(np.ones([scatterer_positions.shape[0]]*2,dtype=bool),1)
    Oi_Oj = np.c_[x[upper_triangular],y[upper_triangular],z[upper_triangular]]
    return Oi_Oj

def assemble_coupling_matrix(T_append_dummy,max_N,positions,k_mod,which_T=None):
    # size of "T_append_dummy = N_sph x max_N x max_N
    size_S = T_append_dummy[0].T.shape
    all_the_same = len(T_append_dummy)==1
    
    num_scatterers = positions.shape[0]
    O_O2_vectors = sta_relative_positions_matrix(positions)
    A,B,A_reciprocal,B_reciprocal = addition_theorem(max_N,max_N,O_O2_vectors,k_mod,True)
    illumination_dims = A.shape[0]*2
    scatter_dims = A.shape[0]*2
    M_coupling = np.eye(num_scatterers*illumination_dims,dtype='complex') #Here I am assuming illumination_dims = scatter_dims
    row_index = np.zeros([num_scatterers*illumination_dims**2]) #Here I am assuming illumination_dims = scatter_dims
    col_index = np.zeros([num_scatterers*illumination_dims**2]) 
    values = np.zeros([num_scatterers*illumination_dims**2],dtype='complex')
    
    for i_scatterer in range(num_scatterers):
        if all_the_same:
            S_scat = T_append_dummy[0].T
        else:
            if which_T == None:
                S_scat = T_append_dummy[i_scatterer].T
            else:
                S_scat = T_append_dummy[which_T[i_scatterer]].T
        rows = i_scatterer*illumination_dims+np.arange(illumination_dims)
        indice = np.concatenate([np.arange(i_scatterer),np.arange(i_scatterer+1,num_scatterers)])
        for j_scatterer in indice:
            if j_scatterer > i_scatterer:
                position = int(i_scatterer*num_scatterers+j_scatterer-i_scatterer-(i_scatterer+1)*(i_scatterer)/2-1) #Bufff, check this
                M = np.block([[A[:,:,position],B[:,:,position]],[B[:,:,position],A[:,:,position]]])
                columns = j_scatterer*scatter_dims+np.arange(scatter_dims)
                M_coupling[np.ix_(rows,columns)] = M_coupling[np.ix_(rows,columns)]-S_scat@M
            else:
                position = int(j_scatterer*num_scatterers+i_scatterer-j_scatterer-(j_scatterer+1)*j_scatterer/2-1) #Also, check this
                M = np.block([[A_reciprocal[:,:,position],B_reciprocal[:,:,position]],[B_reciprocal[:,:,position],A_reciprocal[:,:,position]]])
                columns = j_scatterer*scatter_dims+np.arange(scatter_dims)
                M_coupling[np.ix_(rows,columns)] = M_coupling[np.ix_(rows,columns)]-S_scat@M
        pos = i_scatterer*scatter_dims**2+np.arange(scatter_dims**2)
        columns = i_scatterer*scatter_dims+np.arange(scatter_dims)
        row_index[pos] = np.reshape(np.tile(columns,(scatter_dims,1)),(1,-1),'F').copy().flatten()
        col_index[pos] = np.tile(columns,(1,scatter_dims))
        values[pos] = S_scat.flatten() #Not the non 'F' order, col_index and row_index are interchanged with respect to the matlab code
    
    M_illumination = sparse.csc_matrix((values,(row_index,col_index)),shape=(num_scatterers*illumination_dims,num_scatterers*illumination_dims),dtype=complex)
    return M_coupling, M_illumination

def Tmat_global(T_mat_append,max_N,positions,k_mod):
    origin = np.array([0,0,0])
    pos2orig = origin-positions
    [reg_A,reg_B] = addition_theorem(max_N,max_N,pos2orig,k_mod,change_of_type=False)
    W = np.reshape(np.block([[[reg_A],[reg_B]],[[reg_B],[reg_A]]]),[reg_A.shape[0]*2,-1],order='F')
    M,M_illu = assemble_coupling_matrix(T_mat_append,max_N,positions,k_mod,which_T=None)
    rhs = M_illu@W.T.conj()
    T_aux = W@scipy.linalg.solve(M,rhs)
    
    T_corr = T_aux
    a1 = int(T_mat_append.shape[1]/2-1)
    a2 = int(T_mat_append.shape[1])
    T_corr[0:a1,a1+1:a2] = -T_corr[0:a1,a1+1:a2]
    T_corr[a1+1:a2,0:a1] = -T_corr[a1+1:a2,0:a1]
    return T_corr