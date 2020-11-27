import numpy as np
import theano.tensor as tt
import specgp as sgp
import exoplanet as xo
import pymc3 as pm

def delta_bic(gp, mean, nparams, t):
    """
        Estimates the change in BIC between 
        a model with a zero mean and with 
        the provided mean function mu. The 
        computation assumes that the maximum 
        likelihood GP hyperparameters are the 
        same with and without the model. In the 
        transit case this is accurate if the 
        noise is sufficiently well-constrained 
        and the out-of-transit baseline is much 
        larger than the in-transit portion of the 
        time series, or in the case that the 
        transit depth is small so that the presence 
        of the transit dip does not alter the 
        inference of the noise parameters.
    """
    
    if (isinstance(gp.kernel, sgp.terms.KronTerm) 
        or isinstance(gp.kernel, sgp.terms.KronTermSum)):
        mean = sgp.means.KronMean(mean)
    else:
        mean = xo.gp.means.Constant(mean)
    mu = mean(t)[:, None]
    return tt.sum(mu * gp.apply_inverse(mu)) - nparams*tt.log(t.size)
   