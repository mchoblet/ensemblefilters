import numpy as np

def ETKF_livings(Xf, HXf, Y, R):
    """
    Adaption of the ETKF proposed by David Livings (2005)
    
    Implementation adapted from
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    
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
    Ny=np.shape(Y)[0]

    #Obs error matrix
    Rmat=np.diag(R)
    Rmat_inv=np.diag(1/R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]
    
    #Scaling of perturbations proposed by Livings (2005), numerical stability
    S_hat=np.diag(1/np.sqrt(R)) @ HXp/np.sqrt(Ne-1)
    
    #svd of S_hat transposed
    U,s,Vh=np.linalg.svd(S_hat.T)
    
    C=Rmat_inv @ HXp
    #recreate singular value matrix
    Sig=np.zeros((Ne,Ny))
    np.fill_diagonal(Sig,s)
    
    #perturbation weight
    mat=np.diag(1/np.sqrt(1+np.square(s)))
    Wp1=mat @ U.T
    Wp=U @ Wp1
    
    #innovation
    d=Y-mY
    #mean weight
    D = np.diag(1/np.sqrt(R)) @ d
    D2= Vh @ D
    D3 = np.diag(1/(1+np.square(s))) @ Sig @ D2
    wm= U @ D3 / np.sqrt(Ne-1)

    #adding pert and mean (!row-major formulation in Python!)
    W=Wp + wm[:,None]

    #final adding up (most costly operation)
    Xa=mX[:,None] + Xfp @ W
    
    return Xa
