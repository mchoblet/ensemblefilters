#localized version of the direct kalman solver
import numpy as np
import scipy

def ENSRF_direct_loc(Xf, HXf, Y, R,PH_loc, HPH_loc):
    """
    direct calculation of Ensemble Square Root Filter from Whitaker and Hamill
    applying localization matrices to PH^T and HPH^T as in Tierney 2020: 
    https://www.nature.com/articles/s41586-020-2617-x#Sec7 (Data Assimilation section).
    This is less efficient than without localization, becaue PH needs to be explicitely calculated for the entry-wise hadamard product
    (https://en.wikipedia.org/wiki/Hadamard_product_(matrices)). 
    However, this is still better than using the serial EnSRF formulation (At least an order of magnitude faster).
    It is important to not compute the Kalman gains explicitely.
    As commented in the docstring ENSRF_direct, avoiding inverting matrices could be done, but the speed up is insignificant
    in comparison to the rest.
    
    I propose to compute PH_loc and HPH_loc once for all possible proxy locations, and here only select the 
    relevant columns (for PH_loc) and the relvant rows and columns for HPH_loc using fancy indexing:
    PH_loc -> PH_loc[:,[column_indices]]
    HPH_loc -> HPH_loc[[row_indices]][:,[column_indices]],
    given which proxies are available at one timestep.

    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector (N_y)
    - PH_loc: Matrix for localization of PH^T (N_x * N_y)
    - HPH_loc: Matrix for localization of HPH^T (N_y * N_y)
    
    Output:
    - Analysis ensemble (N_x, N_e)
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
    #entry wise product of covariance localization matrices
    PHT= PH_loc * (Xfp @ HXp.T/(Ne-1))
    HPHT= HPH_loc * (HXp @ HXp.T/(Ne-1))
    
    #second Kalman gain factor
    HPHTR=HPHT+Rmat
    #inverse of factor
    HPHTR_inv=np.linalg.inv(HPHTR)
    #matrix square root of denominator
    HPHTR_sqr=scipy.linalg.sqrtm(HPHTR)

    #Kalman gain for mean
    xa_m=mX + PHT @ (HPHTR_inv @ d)

    #Perturbation Kalman gain
    #inverse of square root calculated via previous inverse: sqrt(A)^(-1)=sqrt(A) @ A^(-1)
    HPHTR_sqr_inv=HPHTR_sqr @ HPHTR_inv
    fac2=HPHTR_sqr + Rsqr
    factor=np.linalg.inv(fac2)

    # right to left multiplication!
    pert = PHT @ (HPHTR_sqr_inv.T @ (factor @ HXp))
    Xap=Xfp-pert
    
    return Xap+xa_m[:,None]
