import numpy as np
def EnSRF_serial(Xf, HXf, Y, R):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 10, see section 5.7.
    Errors: Line 1 must be inside of loop, in HPH^T the divisor Ne-1 is missing.
    This version uses the appended state vector approach, which also updates the precalculated observations from the model.
    
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_y) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """

    # augmented state vector with Ye appended
    Xfn = np.append(Xf, HXf, axis=0)
    
    # number of state variables
    Nx= np.shape(Xf)[0]
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    #Number of measurements
    Ny=np.shape(Y)[0]
    for i in range(Ny):
        #ensemble mean and perturbations
        mX = np.mean(Xfn, axis=1)
        Xfp=np.subtract(Xfn,mX[:,None])
        
        #get obs from model
        HX=Xfn[Nx+i,:]
        #ensemble mean for obs
        mY=np.mean(HX)
        #remove mean
        HXp=(HX-mY)[None]

        HP=HXp @ Xfp.T /(Ne-1)
        
        #Variance at location (here divisor is missing in reference!)
        HPHT=HXp @ HXp.T/(Ne-1)

        ##Localize HP ?
        
        #compute scalar
        sig=R[i]
        F=HPHT + sig
        K=(HP/F)

        #compute factors for final calc
        d=Y[i]-mY
        a1=1+np.sqrt(sig/F)
        a2=1/a1
        
        #final calcs
        mXa=mX+np.squeeze((K*d))
        Xfp=Xfp-a2*K.T @ HXp
        Xfn=Xfp+mXa[:,None]
        
    return Xfn[:Nx,:]
    
