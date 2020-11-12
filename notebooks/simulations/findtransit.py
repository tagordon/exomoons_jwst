import numpy as np
import theano.tensor as tt
import pymc3 as pm
import sys

# need exoplanet version 0.3.3 for the gp submodule. 
# Should upgrade specgp to use celerite2 eventually. 
import exoplanet as xo
import specgp as sgp
from specgp.distributions import MvUniform
import os
os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())

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

def getmodel(holds={}, mu={}, sig={}, transform=False, nterms=1):
    
    params = ['logS0', 'logw', 'alpha', 
             'logsig', 'mean', 'u', 'logrp', 'logrm', 't0p', 't0m']
    for p in params:
        if p not in holds:
            holds[p] = None
        if p not in mu:
            mu[p] = None
        if p not in sig:
            sig[p] = None
    
    with pm.Model() as model:
        logS0 = pm.Normal("logS0", mu=mu["logS0"], 
                        sd=sig["logS0"], observed=holds['logS0'])
        logw = pm.Normal("logw", mu=mu["logw"], 
                       sd=sig["logw"], observed=holds['logw'])
        
        if np.shape(flux)[0] > 1:
            alpha =  pm.MvNormal("alpha", mu=mu["alpha"], 
                          chol=np.diag(sig["alpha"]), shape=np.shape(flux)[0]-1, observed=holds['alpha'])
        logsig = pm.MvNormal("logsig", mu=mu["logsig"], 
                         chol=np.diag(sig["logsig"]), shape=np.shape(flux)[0], observed=holds['logsig'])
        mean = pm.MvNormal("mean", mu=mu["mean"], 
                       chol=np.diag(sig["mean"]), shape=np.shape(flux)[0], observed=holds['mean'])
        u = sgp.distributions.MvUniform("u", lower=[0, 0], upper=[1, 1], 
                                        testval=[0.5, 0.5], observed=holds['u'])
        
        if transform:
            logrp = pm.Uniform("logrp", lower=-20.0, upper=0.0, testval=mu['logrp'], 
                               observed=holds['logrp'])
            logrm = pm.Uniform("logrm", lower=-20.0, upper=0.0, testval=mu['logrm'], 
                               observed=holds['logrm'])
            t0p = pm.Uniform("t0p", lower=t[0], upper=t[-1], testval=mu['t0p'], 
                             observed=holds['t0p'])
            t0m = pm.Uniform("t0m", lower=t[0], upper=t[-1], testval=mu['t0m'],
                             observed=holds['t0m'])
        else:
            logrp = pm.Uniform("logrp", lower=-20.0, upper=0.0, testval=mu['logrp'], 
                               transform=None, observed=holds['logrp'])
            logrm = pm.Uniform("logrm", lower=-20.0, upper=0.0, testval=mu['logrm'], 
                               transform=None, observed=holds['logrm'])
            t0p = pm.Uniform("t0p", lower=t[0], upper=t[-1], testval=mu['t0p'], 
                               transform=None, observed=holds['t0p'])
            t0m = pm.Uniform("t0m", lower=t[0], upper=t[-1], testval=mu['t0m'],
                               transform=None, observed=holds['t0m'])
        
        orbit = xo.orbits.KeplerianOrbit(period=5.0*60*60)
        lcp = (xo.LimbDarkLightCurve(u)
              .get_light_curve(orbit=orbit, r=np.exp(logrp), t=t/(60*60)-t0p, texp=np.mean(np.diff(t))/(60*60)))
        lcm = (xo.LimbDarkLightCurve(u)
              .get_light_curve(orbit=orbit, r=np.exp(logrm), t=t/(60*60)-t0m, texp=np.mean(np.diff(t))/(60*60)))
        mean = mean[:, None] + lcp.T[0] + lcm.T[0]
    
        term = xo.gp.terms.SHOTerm(
                log_S0 = logS0,
                log_w0 = logw,
                log_Q = -np.log(np.sqrt(2))
            )
        
        if np.shape(flux)[0] > 1:
            a = tt.exp(tt.concatenate([[0.0], alpha]))
            kernel = sgp.terms.KronTerm(term, alpha=a)
        else:
            kernel = term
        
        yerr = tt.exp(2 * logsig)
        yerr = yerr[:, None] * tt.ones(len(t))
        
        if np.shape(flux)[0] > 1:
            gp = xo.gp.GP(kernel, t, yerr, J=2, mean=sgp.means.KronMean(mean))
        else:
            gp = xo.gp.GP(kernel, t, yerr[0], J=2, mean=mean)
        marg = gp.marginal("gp", observed = obs.T) 
    return model

m = getmodel(mu=mu, sig=sig)
with m:
    map_soln = xo.optimize(start=m.test_point, verbose=True)
    
import copy
holds = copy.deepcopy(map_soln)
logrpstart = holds.pop('logrp')
t0pstart = holds.pop('t0p')
astart = holds.pop('alpha')
meanstart = holds.pop('mean')
logsigstart = holds.pop('logsig')
holds.pop('logrm')
holds_notransit = copy.deepcopy(holds)
holds_notransit['logrm'] = np.log(0.000001)

start = {
    "alpha": astart,
    "mean": meanstart,
    "logsig": logsigstart,
    "logrm": np.log(0.01),
    "logrp": logrpstart,
    "t0p": t0pstart
}
start_notransit = {
    "alpha": astart,
    "mean": meanstart, 
    "logsig": logsigstart,
    "logrp": logrpstart,
    "t0p": t0pstart
}

if len(sys.argv) == 3:
    t0grid = np.linspace(t.min(), t.max(), int(sys.argv[2]))
else:
    t0grid = np.linspace(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))

def loglike(t0):
    print("t0={0}".format(t0))
    h = copy.deepcopy(holds)
    h['t0m'] = t0
    m = getmodel(holds=h, mu=mu, sig=sig)
    with m:
        newmap_soln = xo.optimize(start=start, verbose=False)
        ll = m.logp(newmap_soln)
        #r[i] = np.exp(newmap_soln['logrm'])
    return ll

from multiprocessing import Pool
pool = Pool(processes=28)
ll = pool.map(loglike, t0grid)

#ll = np.zeros_like(t0grid)
#r = np.zeros_like(t0grid)


#for i, t0 in enumerate(t0grid):
#    print('\r{0}/{1}'.format(i+1, len(t0grid)), end='')
#    holds['t0m'] = t0
#    m = getmodel(holds=holds, mu=mu, sig=sig)
#    with m:
#        newmap_soln = xo.optimize(start=start, verbose=False)
#        print(newmap_soln)
#        ll[i] = m.logp(newmap_soln)
#        r[i] = np.exp(newmap_soln['logrm'])
m_notransit = getmodel(holds=holds_notransit, mu=mu, sig=sig)
with m_notransit:
    newmap_soln_notransit = xo.optimize(start=start_notransit, verbose=False)
    ll_notransit = m_notransit.logp(newmap_soln_notransit)
        
np.savetxt('xi2_{0}.txt'.format(vars.NAME.replace(" ", "")), np.array([ll-ll_notransit, t0grid]).T)
