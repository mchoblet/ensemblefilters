import numpy as np
import scipy.linalg

def ENSRF_direct(Xf, HXf, Y, R):
    """
    direct calculation of Ensemble Square Root Filter from Whitaker and Hamill
    as for instance done in Steiger 2018: "A reconstruction
    of global hydroclimate and dynamical variables over the Common Era".
    
    Issue: Matrix square roots/inverse give imaginary parts (small for my test data)

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

    #compute matrix products directly, do not calculate B separately (huge!)
    BHT=(Xfp @ HXp.T)/(Ne-1)
    HBHT=(HXp @ HXp.T)/(Ne-1)

    #second Kalman gain factor
    HBHTR=HBHT+Rmat
    #inverse of factor
    HBHTR_inv=np.linalg.inv(HBHTR)
    #matrix square root of denominator
    HBHTR_sqr=scipy.linalg.sqrtm(HBHTR)

    #Kalman gain for mean
    xa_m=mX + BHT @ (HBHTR_inv @ d)

    #Perturbation Kalman gain
    #inverse of square root calculated via previous inverse: sqrt(A)^(-1)=sqrt(A) @ A^(-1)
    HBHTR_sqr_inv=HBHTR_sqr @ HBHTR_inv
    fac2=HBHTR_sqr + Rsqr
    factor=np.linalg.inv(fac2)

    # use brackets for right to left matrix multiplication
    pert = BHT @ (HBHTR_sqr_inv.T @ (factor @ HXp))
    Xap=Xfp-pert
    
    return Xap+xa_m[:,None]
