import pickle5 as pickle
import matplotlib.pyplot as pl
import sys
import numpy as np

red = '#FE4365'
blue = '#00A9FF'
yellow = '#ECA25C'
green = '#3F9778'
darkblue = '#005D7F'
colors = [red, green, blue, yellow, darkblue]

fullpath = sys.argv[1]
result = pickle.load(open(fullpath, 'rb'))
exp_time = result['exp_time'].value
nhours = 20

filename = fullpath.split('/')[-1]
parts = filename.split('_')
instrument = parts[2]
mode = parts[3]
filt = parts[4]
disp = parts[5]
name = parts[6]
filt_disp = '{0}/{1}'.format(filt, disp)

wl = result['noise_dic']['All noise']['wl']
means = result['noise_dic']['All noise']['signal_mean_stack']
stds = result['noise_dic']['All noise']['signal_std_stack']

t = np.arange(0, nhours*60*60, cycle_time)

import celerite2
from celerite2 import terms

S0 = float(sys.argv[2])
w0 = 886
term = terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))
gp = celerite2.GaussianProcess(term, mean=0.0)
gp.compute(t/(60*60*24), yerr = 0)
fk = (gp.dot_tril(np.random.randn(len(t))) + 1)

import sys
sys.path.append('./')
import generate_noise

factors, data, wl = generate_noise.variability_factors(fk, wl, cold_temp=int(sys.argv[3]), hot_temp=int(sys.argv[4]), effective_temp=int(sys.argv[3]), spec_path='..')

shot_noise = np.random.randn(len(wl), len(t))*stds[:, None]
noisy_lc = means[None, :]*factors.T + shot_noise.T
ppm = (np.sqrt(np.sum(stds ** 2)) / np.sum(means)) / np.sqrt(60 / np.mean(np.diff(t)))

# equal white-noise bins
nbins = np.int(sys.argv[5])
total_wn = np.sqrt(np.sum(stds ** 2))
wn_per_bin = total_wn / np.sqrt(nbins)

inds = [0]
j = 0
for i in range(nbins):
    sum_wn_squared = 0
    sum_mean = 0
    while (np.sqrt(sum_wn_squared) < wn_per_bin) & (j < len(stds)):
        sum_wn_squared += stds[j] ** 2
        j += 1
    inds.append(j)
    
lcs = np.zeros((len(t), len(inds)-1))
for i in range(len(inds)-1):
    lcs[:, i] = np.sum(noisy_lc[:, inds[i]:inds[i+1]], axis=1)
    
import exoplanet as xo
orbit = xo.orbits.KeplerianOrbit(period=5.0*60*60)
u = [0.3, 0.2]
rp = 0.0203
rm = 0.01
t0p = 10
t0m = 5
planet = (
    xo.LimbDarkLightCurve(u)
    .get_light_curve(orbit=orbit, r=rp, t=t/(60*60) - t0p, texp=np.mean(np.diff(t))/(60*60))
    .eval()
).T[0]
moon = (
    xo.LimbDarkLightCurve(u)
    .get_light_curve(orbit=orbit, r=rm , t=t/(60*60) - t0m, texp=np.mean(np.diff(t))/(60*60))
    .eval()
).T[0]

from astropy.table import table

data = lcs + np.mean(lcs, axis=0)*(moon[:, None] + planet[:, None])
data = table.QTable(data, names=["bin{0}".format(i) for i in range(nbins)])
data.add_column(t, name='time')
data.meta['name'] = name
data.meta['S0'] = S0
data.meta['w0'] = w0
data.meta['t0p'] = t0p
data.meta['t0m'] = t0m
data.meta['rp'] = rp
data.meta['rm'] = rm
data.meta['ppm'] = ppm
data.meta['filter'] = filt_disp
data.meta['mode'] = mode
data.write(name.replace(" ", "") + ".fits", format='fits', overwrite=True)
