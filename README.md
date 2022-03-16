# Collection of ensemble square root kalman filters implemented in Python

This repository offers Python Code for a variety of Ensemble Kalman Filters
as presented in "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018).[1]
Besides giving a comprehensive overview of the different methods using a consistent mathematical framework the authors also deliver Fortran-like pseudocode for each presented algorithm. The pseudocode there has some errors and possible misunderstandings for a python implementation, I tried to indicate this specifically for each algorithm at the beginning of the functions.

I provide a test script and test data to test the correctness of the functions.

The original authors have implemented some of the functions for the sangema project in Fortran [2] and in julia language [3].

## Input variables and dimension conventions
* Note that the observation operator $H$ is not implemented explicitely in these functions, the observations from the model Hx are precalculated 

**Dimensions**
* $N_e$ Ensemble Size (e.g. 100)
* $N_x$ State Vector length (e.g. number of gridboxes if one variable assimilated, 55000)
* $N_y$ Number of measurements

**Variables**
* Xf: Prior ensemble (Nx * Ne)
* HX: Observations from model (Ny * Ne)
* Y: Observations (Ny * 1000) (its a timeseries for different locations)
* R: Observation error (uncorrelated, R is assumed diagonal) (Ny * 1)

## Ensemble Kalman Filters
Implemented so far:

* EnSRF: Ensemble Square Root Filter  (Whitaker and Hamill 2002)
    * simultaneous solving
    * serialized solving
    * direct solving of K and K-tilde (numerical errors can be significant)
* ETKF: Ensemble Transform Kalman Filter:
    * Square Root Formulation as in Hunt 2007
    * Adaptation by Livings 2005
* ESTKF: Error-subspace transform Kalman Filter 
* more to come...

In order to test the correctness of the implementation the output is compared for some test data in the filters_test Jupyter Notebook.

## Test Data
As I work on paleoclimate DA project the test-data is from a past-millenium climate simulation [4]

* Y Measurement timeseries (293 * 1000)
* R Measurement errors (293 * 1)
* Xf Forecast from model (55496 * 100)
* HXf Observations from model (293 * 100)

# Contact
If you find ways to optimize the code ot errors feel free to open an issue or contact me via mchoblet -AT- iup.uni-heidelberg.de

# Literature
[1] Sanita Vetra-Carvalho et al. State-of-the-art stochastic data assimilation methods for high-dimensional non-Gaussian problems. Tellus A: Dynamic Meteorology and Oceanography, 70(1):1445364, 2018. https://doi.org/10.1080/16000870.2018.1445364
[2] https://sourceforge.net/projects/sangoma/
#[3] https://github.com/Alexander-Barth/DataAssim.jlts

