# dfdjaxGP

A python package for fitting Gaussian processes in Jax and performing prediction on (nearly) arbitrary derivatives. 

### Functionality and syntax are currently shown in the example jupyter notebooks. Further documentation will come in the future. 

## JaxGPwDerivativesPackageTest_MLE.ipynb

This notebook demonstrates using the package to fit GP parameters by maximizing the marginal likelihood of observations of the input-output pairs of the function. Then (nearly) arbitrary derivatives can be predicted, with uncertainty, from the GP.

## JaxGPwDerivativesPackageTest_Bayes.ipynb

This notebook demonstrates using the package to fit GP parameters using observations of the input-output pairs of the function and doing full Bayesian inference through sampling using numpyro. Then (nearly) arbitrary derivatives can be predicted, with uncertainty, from the GP.

## JaxGPDerivativeInference.ipynb

This notebook demonstrates combining the package with numpyro to do inference on a function from observations of the function and/or its derivatives. Predictions can then be made of (nearly) arbitrary derivatives. 


The derivative limitations are largely driven by computational cost and numerical stability. For instance, making predictions on the 10th derivative of a function based on the GP requires jax to take 10 derivatives of the covariance function and for the resulting covariance matrix to retain numerical positive definiteness.

-------

© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
