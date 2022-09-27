# A collection of ensemble square root kalman filters implemented in Python

This repository offers Python code for a variety of Ensemble Kalman Filters as presented in the overview paper by Vetra-Carvalho et al. (2018) [1].The authors present a variety of data-assimilation methods using a unified mathematical notation. It's really a pleasant to read paper that makes the math more understandable than the separate papers for different methods.

I also added the possibility of localization, for that a function creating the distance decorrelation matrices is also included (cov_loc.py).

For the implementation of the algorithms I followed the Fortran-like pseudocode given by authors in the appendix and indicated in the comments where I deviated from it (unfortunately there are some errors, but at least that makes you think about what one is actually implementing). The jupyter notebook shows that the output (posterior mean + covariance) from all functions is equal for my test data, but of course strictly speaking this is not a proof.

I hope to have time to implement other methods mentioned in the paper one day.

## Content of repository:
* Folder kalmanfilters: Separate file for each Kalman Filter
* Folder testdata: data from a general circulation climate model which can be assimilated with the functions (you could also just generate random vectors)
* kalman_filters_tests-notebook: Simple script to check that the output of the different functions is equal (posterior mean and covariance matrix)

## Dependencies
The functions work on pure numpy arrays.

* numpy 
* scipy (only the EnSRF_direct function needs it for matrix square root calculation)

## Input variables and dimension conventions
* Note that the observation operator  H  is only implemented implicitely in these functions, the observations from the model  Hx  need to be precalculated. The observation uncertainties are assumed to be uncorrelated, hence the matrix R is diagonal (algorithms are written for diagonal R).

**Variables**
* Xf: Prior ensemble ( Nx  *  N_e )
* HX: Observations from model ( Ny  *  Ne )
* Y: Observations ( N_y  * 1) 
* R: Observation error (uncorrelated, R is assumed diagonal) ( Ny  * 1)

**Dimensions**
*  Ne:  Ensemble Size 
*  Nx:  State Vector length
*  Ny:  Number of measurements

I usually work with climate fields as [xarrays](https://docs.xarray.dev/en/stable/), which you can easily bring into the right shape using methods like '.stack(z=('lat','lon')), 'swap_dims' for getting the dimensions in the right order, and '.values' to convert to numpy arrays. Although the algorithms here work on pure numpy arrays, using xarray for the pre- and postprocessing is really an asset.

## Ensemble Kalman Filters implemented

* EnSRF: Ensemble Square Root Filter
    * simultaneous solver
    * serialized solver
    * direct solving of square root filter
    * direct solving with covariance localization (requires prior/measurement with latitudes/longitudes, see cov_loc.py)
* ETKF: Ensemble Transform Kalman Filter:
    * Square Root Formulation by Hunt
    * Adaptation by David Livings 
* ESTKF: Error-subspace transform Kalman Filter 
* Stochastic EnKF (the Burgers 1998 update), also with localization.


## Test Data
As I work on a paleoclimate Data Assimilation project the test-data is from a past-millenium climate simulation. Of course you can also easily generate some random test data.

* Y: Measurements (293 * 1) (Actualizly synthesized observations generated from model with additional noise from the prior)
* R: Measurement errors (293 * 1)
* Xf: Forecast from model (55296 * 100) (The number of rows is given by the number of gridpoints of the climate model. Prior contains temperature values (K))
* HXf: Observations from model (293 * 100)

For this type of test data, the speed is dominated by the last operation (multiplication of perturbation matrix with weight matrix). You will see how much faster an optimized variant like the ETKF or ESTKF is in comparison to the serialized EnSRF. In my discipline people have also simply used the direct solving for K and K-tilde, which doesn't require much fancy mathematics. When the matrices are multiplied efficiently, it is only a factor 2-3 slower than the optimized variants.

# Contact
If you find errors, ways to optimize the code etc.  feel free to open an issue or contact me via mchoblet -AT- iup.uni-heidelberg.de

# Literature
[1] Sanita Vetra-Carvalho et al. State-of-the-art stochastic data assimilation methods for high-dimensional non-Gaussian problems. Tellus A: Dynamic Meteorology and Oceanography, 70(1):1445364, 2018. https://doi.org/10.1080/16000870.2018.1445364
The authors have implemented most of the functions for the sangema project in Fortran and in julia language. I have not checked their code in detail: https://sourceforge.net/projects/sangoma/, https://github.com/Alexander-Barth/DataAssim.jlts

