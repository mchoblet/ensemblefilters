#localized version of the direct kalman solver
import numpy as np
import scipy


def covariance_loc(model_data,proxy_lat,proxy_lon, cov_len):    
    """
    Function that returns the matrices needed for the Covariance Localization in the direct EnSRF solver by Hadamard (element-wise) product.
    These are the terms called W_loc and Y_loc here: https://www.nature.com/articles/s41586-020-2617-x#Sec7 (Data Assimilation section).
    The idea is to compute these matrices once in the beginning for all available proxy locations, and later in the DA loop one only selects
    the relevant columns of W_loc / rows and columns of Y_loc for the localized simultaneous Kalman Filter Solver.
    
    Input:
       - model_data from which the grid point locations are extracted. Here I use the stack function, which I also use when constructing the
       prior vector. In brings all gridpoints in a vector form (xarray-DataArray such that stack can be applied, N_x grid points)
       - proxy_lat, proxy_lon are the latitudes and longitudes of the proxy locations (np.arrays, length = N_y). Make sure they have the same ordering as
       the entries of your Observations-from-Model (HXf) in the Kalman Filter.
       - cov_len: Radius for Gaspari Cohn function [float, in km ]
       
    Ouput:
        - PH_loc: Matrix for localization of PH^T (N_x * N_y)
        - HPH_loc: Matrix for localization of HPH^T (N_y * N_y)
    """
    from haversine import haversine_vector, Unit
    
    #bring coordinates of model (field) and proxy (individual locations) into the right form
    #the method we use are different due to this different structures (field and individual locations)
    loc=np.array([[lat,lon] for lat,lon in zip(proxy_lat,proxy_lon)])
    stacked=model_data.stack(z=('lat','lon')).transpose('z','time')
    coords=[list(z) for z in stacked.z.values]
    
    #model-proxy distances
    dists_mp=haversine_vector(loc,coords, Unit.KILOMETERS,comb=True)
    dists_mp_shape=dists_mp.shape
    
    #proxy-proxy distances
    dists_pp=haversine_vector(loc,loc, Unit.KILOMETERS,comb=True)
    dists_pp_shape=dists_pp.shape
    
    def gaspari_cohn(dists,cov_len):
        """
        Gaspari Cohn decorrelation function https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.49712555417 page 26
        dists: need to be a 1-D array with all the distances (Reshapeto your needs afterwards)
        cov_len: radius given in km

        """
        dists = np.abs(dists)
        array = np.zeros_like(dists)
        r = dists/cov_len
        #first the short distances
        i=np.where(r<=1.)[0]
        array[i]=-0.25*(r[i])**5+0.5*r[i]**4+0.625*r[i]**3-5./3.*r[i]**2+1.
        #then the long ones
        i=np.where((r>1) & (r<=2))[0]
        array[i]=1./12.*r[i]**5-0.5*r[i]**4+0.625*r[i]**3+5./3.*r[i]**2.-5.*r[i]+4.-2./(3.*r[i])

        array[array < 0.0] = 0.0
        return array
    
    #flatten distances, apply to Gaspari Cohn and reshape
    PH_loc=gaspari_cohn(dists_mp.reshape(-1),cov_len).reshape(dists_mp_shape)
    HPH_loc=gaspari_cohn(dists_pp.reshape(-1),cov_len).reshape(dists_pp_shape)
    
    return PH_loc, HPH_loc



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
