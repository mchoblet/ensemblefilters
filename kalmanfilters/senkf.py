def SEnKF(Xf, HXf, Y, R):
    """
    Stochastic Ensemble Kalman Filter
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    
    Changes: The pseudocode is not consistent with the description in 5.1, where the obs-from-model are perturbed, but in the pseudocode it's the other way round.
    Hence the 8th line D= ... is confusing if we would generate Y as described in the text.
    Last line needs to have 1/(Ne-1) (+always better to do that on the smaller matrix)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    
    
    """
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    Ny=np.shape(R)[0]
    #Obs error matrix
    Rmat=np.diag(R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    HPH=HXp@HXp.T /(Ne-1)

    A=HPH + Rmat

    rng = np.random.default_rng(seed=42)
    Y_p=rng.standard_normal((Ny, Ne))*np.sqrt(R)[:,None]

    D= Y[:,None]+Y_p - HXf
    
    #solve linear system for getting inverse
    C=np.linalg.solve(A,D)
    
    E=HXp.T @ C
    
    Xa=Xf+Xfp@(E/(Ne-1))
    
    return Xa
