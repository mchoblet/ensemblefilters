import numpy as np

def ETKF(Xf, HXf, Y, R):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 7, see section 5.4.
    Errors: Calculation of W1 prime, divide by square root of eigenvalues. The mathematical formula in the paper has an error already.
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_y) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    # number of ensemble members
    Ne=np.shape(Xf)[1]

    #Obs error matrix
    #Rmat=np.diag(R)
    Rmat_inv=np.diag(1/R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    C=Rmat_inv @ HXp
    A1=(Ne-1)*np.identity(Ne)
    A2=A1 + (HXp.T @ C)

    #eigenvalue decomposition of A2, A2 is symmetric
    eigs, ev = np.linalg.eigh(A2) 

    #compute perturbations
    Wp1 = np.diag(np.sqrt(1/eigs)) @ ev .T
    Wp = ev @ Wp1 * np.sqrt(Ne-1)

    #differing from pseudocode
    d=Y-mY
    D1 = Rmat_inv @ d
    D2 = HXp.T @ D1
    wm=ev @ np.diag(1/eigs) @ ev.T @ D2  #/ np.sqrt(Ne-1) 

    #adding pert and mean (!row-major formulation in Python!)
    W=Wp + wm[:,None]

    #final adding up (most costly operation)
    Xa=mX[:,None] + Xfp @ W

    return Xa
    
    
