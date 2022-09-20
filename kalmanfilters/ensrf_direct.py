import numpy as np
import scipy

def ENSRF_direct(Xf, HXf, Y, R):
    """
    direct calculation of Ensemble Square Root Filter from Whitaker and Hamill
    As for instance done in Steiger 2018: "A reconstruction of global hydroclimate and dynamical variables over the Common Era".
    
    In comparison to the code for that paper [1], the matrix multiplications are performed  consequently from left to right and 
    the kalman gain is not explicitely computed, because this would be inefficient when we are just interested in the posterior ensemble.
    One could also avoid computing the matrix inverses and solve linear systems instead (one could even use Cholesky decomposition
    because the covariance matrices are positive definite), but as the number of observations is small the speed up is insignificant.
    When using many observations (>1000) one should consider doing it. Here, the main computation effort comes from the matrix square root
    (potentially numerically unstable) and unavoidable matrix - matrix multiplications.
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)
    Output:
    - Analysis ensemble (N_x, N_e)
    
    [1] https://github.com/njsteiger/PHYDA-v1/blob/master/M_update.m
    """
    Ne=np.shape(Xf)[1]

    #Obs error matrix, assumption that it's diagonal
    Rmat=np.diag(R)
    Rsqr=np.diag(np.sqrt(R)) 

    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]
    #innovation
    d=Y-mY

    #compute matrix products directly
    #BHT=(Xfp @ HXp.T)/(Ne-1) #avoid this, it's inefficient to compute it here
    HPHT=(HXp @ HXp.T)/(Ne-1)

    #second Kalman gain factor
    HPHTR=HPHT+Rmat
    #inverse of term
    HPHTR_inv=np.linalg.inv(HPHTR)
    #matrix square root of denominator
    HPHTR_sqr=scipy.linalg.sqrtm(HPHTR)

    #Kalman gain for mean
    xa_m=mX + (Xfp @ (HXp.T /(Ne-1) @ (HPHTR_inv @ d)))

    #Perturbation Kalman gain
    #inverse of square root calculated via previous inverse: sqrt(A)^(-1)=sqrt(A) @ A^(-1)
    HPHTR_sqr_inv=HPHTR_sqr @ HPHTR_inv
    fac2=HPHTR_sqr + Rsqr
    factor=np.linalg.inv(fac2)

    #right to left multiplication!
    pert = (Xfp @ (HXp.T/(Ne-1) @ (HPHTR_sqr_inv.T @ (factor @ HXp))))
    Xap=Xfp-pert
    
    return Xap+xa_m[:,None]
