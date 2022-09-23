def SEnKF_loc(Xf, HXf, Y, R,PH_loc, HPH_loc):
    """
    Stochastic Ensemble Kalman Filter that can do localisation. Changed the order of calculations
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    
    for the calculation of PH_loc/HPH_loc look at the function in ensrf_direct_loc.py
    
    Changes: The pseudocode is not consistent with the description in 5.1, where the obs-from-model are perturbed, but in the pseudocode it's the other way round.
    Hence the 8th line D= ... is confusing if we would generate Y as described in the text.
    Last line needs to have 1/(Ne-1)
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

    #Hadamard product for localisation
    HPH=HPH_loc * (HXp@HXp.T /(Ne-1))

    A=HPH + Rmat

    rng = np.random.default_rng(seed=42)
    Y_p=rng.standard_normal((Ny, Ne))*np.sqrt(R)[:,None]

    D= Y[:,None]+Y_p - HXf
    
    #solve linear system for getting inverse
    C=np.linalg.solve(A,D)
    
    Pb=PH_loc*(Xfp @ HXp.T/(Ne-1)) 
    
    Xa=Xf + Pb @ C
    
    return Xa
