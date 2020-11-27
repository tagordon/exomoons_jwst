import numpy as np
import theano.tensor as tt
import pymc3 as pm
import sys

# need exoplanet version 0.3.3 for the gp submodule. 
# Should upgrade specgp to use celerite2 eventually. 
import exoplanet as xo
import specgp as sgp
from specgp.distributions import MvUniform

from astropy.table import Table
from types import SimpleNamespace
filename = sys.argv[1]
data = Table.read(filename)
flux = np.array([data.columns[i] for i in range(len(data.columns)-1)], dtype='float64')
t = np.array(data.columns[-1], dtype='float64')
vars = SimpleNamespace(**data.meta)

wn_est = np.std(flux.T/np.mean(flux, axis=1), axis=0)
obs = np.reshape(np.array([f/np.mean(f) for f in flux]).T, (np.shape(flux.T)[1]*len(t)))

nb = np.shape(flux)[0]
mu = {'logS0':np.log(vars.S0), 'logw':np.log(vars.W0), 'alpha':np.zeros(nb-1), 
      'logsig':np.log(wn_est), 'mean':np.ones(nb), 'u':[0.5, 0.5], 
      'logrp':np.log(vars.RP), 't0p':vars.T0P, 'logrm':np.log(vars.RM), 't0m':vars.T0M}
sig = {'logS0':10, 'logw':10, 'logQ':None, 'alpha':np.ones(nb-1), 
       'logsig':np.ones(nb)*5, 'mean':np.ones(nb)*0.1, 
       't0p':2, 't0m':2, 'logrm':1, 'logrp':1}

with pm.Model() as model:
    logS0 = pm.Normal("logS0", mu=mu["logS0"], 
                        sd=sig["logS0"])
    logw = pm.Normal("logw", mu=mu["logw"], 
                       sd=sig["logw"])
    alpha =  pm.MvNormal("alpha", mu=mu["alpha"], 
                          chol=np.diag(sig["alpha"]), shape=np.shape(flux)[0]-1)
    logsig = pm.MvNormal("logsig", mu=mu["logsig"], 
                         chol=np.diag(sig["logsig"]), shape=np.shape(flux)[0])
    mean = pm.MvNormal("mean", mu=mu["mean"], 
                       chol=np.diag(sig["mean"]), shape=np.shape(flux)[0])
    u1 = pm.Uniform("u1", lower=0, upper=1, testval=0.5)
    u2 = pm.Uniform("u2", lower=0, upper=1, testval=0.5)
    u = [u1, u2]
    logrp = pm.Normal("logrp", mu=mu['logrp'], sd=sig['logrp'],
                           transform=None)
    t0p = pm.Normal("t0p", mu=mu['t0p'], sd=sig['t0p'], testval=mu['t0p'], 
                         transform=None)
    logrm = pm.Normal("logrm", mu=mu['logrm'], sd=sig['logrm'],
                           transform=None)
    t0m = pm.Normal("t0m", mu=mu['t0m'], sd=sig['t0m'], testval=mu['t0m'], 
                         transform=None)
        
    orbit = xo.orbits.KeplerianOrbit(period=5.0*60*60)
    lcp = (xo.LimbDarkLightCurve(u)
              .get_light_curve(orbit=orbit, r=np.exp(logrp), t=t/(60*60)-t0p, texp=np.mean(np.diff(t))/(60*60)))
    lcm = (xo.LimbDarkLightCurve(u)
              .get_light_curve(orbit=orbit, r=np.exp(logrm), t=t/(60*60)-t0m, texp=np.mean(np.diff(t))/(60*60)))
    mean = mean[:, None] + lcm.T[0]
    
    term = xo.gp.terms.SHOTerm(
            log_S0 = logS0,
            log_w0 = logw,
            log_Q = -np.log(np.sqrt(2))
        )
        
    a = tt.exp(tt.concatenate([[0.0], alpha]))
        
    kernel = sgp.terms.KronTerm(term, alpha=a)
        
    yerr = tt.exp(2 * logsig)
    yerr = yerr[:, None] * tt.ones(len(t))
        
    gp = xo.gp.GP(kernel, t, yerr[0], J=2, mean=sgp.means.KronMean(mean))
    marg = gp.marginal("gp", observed = obs.T)
        
    trace = pm.sample(
        tune=50,
        draws=50,
        start=model.test_point,
        cores=2,
        chains=2,
        step=xo.get_dense_nuts_step(target_accept=0.9)
    )

outdir = '../traces/trace_{0}_{1}'.format(vars.NAME.replace(" ", ""), vars.RM)
pm.save_trace(trace, outdir, overwrite=True)
