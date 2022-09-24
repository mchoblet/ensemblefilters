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
