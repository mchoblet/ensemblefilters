import numpy as np

def EnSRF(Xf, HXf, Y, R):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 9, see section 5.6. Pseudocode has some errors, eg. in step 7 it should be sqrt(Lambda).
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_y) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    #Obs error matrix
    Rmat=np.diag(R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    #Gram matrix of perturbations
    I1=HXp @ HXp.T
    Ny=np.shape(Y)[0]
    Ne=np.shape(Xf)[1]

    I2=I1+(Ne-1)*Rmat
    #compute eigenvalues and eigenvectors (use that matrix is symmetric and real)
    eigs, ev = np.linalg.eigh(I2) 

    #Error in Pseudocode: Square Root + multiplication order (important!)
    G1=ev @ np.diag(np.sqrt(1/eigs)) 
    G2=HXp.T @ G1

    U,s,Vh=np.linalg.svd(G2)
    #Compute  sqrt of matrix, Problem of imaginary values?? (singular values are small)
    rad=(np.ones(Ne)-np.square(s)).astype(complex)
    rad=np.sqrt(rad)
    A=np.diag(rad)

    W1p=U @ A
    W2p=W1p@U.T

    d=Y-mY

    w1=ev.T @ d
    w2=np.diag(1/eigs).T @ w1
    w3=ev @ w2
    w4=HXp.T @ w3
    W=W2p+w4[:,None]
    Xa=mX[:,None]+Xfp @ W

    return Xa



