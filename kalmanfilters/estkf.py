import numpy as np

def ESTKF(Xf, HXf, Y, R):
    """
    Error-subspace transform Kalman Filter
    
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 12, see section 5.9
    Errors: 
        5th line: A instead of L (A needs to be created)
        Last line: W_A instead of W'
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_y) 
    - R: Measurement Error (assumed uncorrelated) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    
    #Obs error matrix
    Rmat=np.diag(R)
    Rmat_inv=np.diag(1/R)
    #Mean of prior ensemble for each state vector variable 
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    
    #Mean of model values in observation space
    mY = np.mean(HXf, axis=1)
    d=Y-mY

    """
    Create projection matrix:
    - create matrix of shape Ne x Ne-1 filled with off diagonal values
    - fill diagonal with diagonal values
    - replace values of last row
    """

    sqr_ne=-1/np.sqrt(Ne)
    off_diag=-1/(Ne*(-sqr_ne+1))
    diag=1+off_diag

    A=np.ones((Ne,Ne-1))*off_diag
    np.fill_diagonal(A,diag)
    A[-1,:]=sqr_ne

    #error in pseudocode, replace L by A
    HL=HXf @ A
    B1=Rmat_inv @ HL
    C1=(Ne-1)*np.identity(Ne-1)
    C2=C1+HL.T @ B1
    
    #EVD of C2, assumed symmetric
    eigs,U=np.linalg.eigh(C2)
    
    d1=B1.T @ d
    d2=U.T @ d1
    d3=d2/eigs
    T=U @ np.diag(1/np.sqrt(eigs)) @ U.T
    
    #mean weight
    wm=U @ d3
    #perturbation weight
    Wp=T @ A.T*np.sqrt((Ne-1))
    #total weight matrix + projection matrix transform
    W=wm[:,None]+Wp
    Wa = A @ W

    #Analysis ensemble
    Xa = mX[:,None] + Xfp @ Wa

    return Xa
