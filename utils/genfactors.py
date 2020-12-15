import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import quad

spec_num = lambda x: str(np.int(np.round(x/100)))

def variability_factors(wl, cold_temp, hot_temp, effective_temp, spec_path):
    
    spec_cold = fits.open(spec_path + '/BT-Settl_M-0.0a+0.0/lte0' 
                          + spec_num(cold_temp) 
                          + '.0-4.5-0.0a+0.0.BT-Settl.spec.fits.gz')
    spec_hot = fits.open(spec_path + '/BT-Settl_M-0.0a+0.0/lte0' 
                         + spec_num(hot_temp) 
                         + '.0-4.5-0.0a+0.0.BT-Settl.spec.fits.gz')
    spec_mean = fits.open(spec_path + '/BT-Settl_M-0.0a+0.0/lte0' 
                          + spec_num(effective_temp) 
                          + '.0-4.5-0.0a+0.0.BT-Settl.spec.fits.gz')
    
    
    wlc = spec_cold[1].data.field('wavelength')
    wlh = spec_hot[1].data.field('wavelength')
    wlm = spec_mean[1].data.field('wavelength')

    fc = spec_cold[1].data.field('flux')
    fh = spec_hot[1].data.field('flux')
    fm = spec_mean[1].data.field('flux')

    interp_spec_hot = interp1d(wlh, fh)
    interp_spec_cold = interp1d(wlc, fc)
    interp_spec_mean = interp1d(wlm, fm)
    
    kep_band = np.loadtxt('/Users/tgordon/research/exomoons_jwst/data/kep.dat').T
    kep_interp = interp1d(kep_band[0]/1e3, kep_band[1])

    int_hot_kep = lambda x: kep_interp(x)*interp_spec_hot(x)
    int_cold_kep = lambda x: kep_interp(x)*interp_spec_cold(x)
    int_mean_kep = lambda x: kep_interp(x)*interp_spec_mean(x)

    flux_hot_kep = quad(int_hot_kep, np.min(kep_band[0])/1e3, np.max(kep_band[0])/1e3)
    flux_cold_kep = quad(int_cold_kep, np.min(kep_band[0])/1e3, np.max(kep_band[0])/1e3)
    flux_mean_kep = quad(int_mean_kep, np.min(kep_band[0])/1e3, np.max(kep_band[0])/1e3)
    
    FC = flux_cold_kep[0]
    FH = flux_hot_kep[0]
    Fmu = flux_mean_kep[0]
    RH, RC = FH / Fmu, FC / Fmu
     
    st = np.where(np.isclose(wlm, 0.6))[0][0]
    end = np.where(np.isclose(wlm, 5.3))[0][1]

    fc_norm = interp_spec_cold(wl) / interp_spec_mean(wl)
    fh_norm = interp_spec_hot(wl) / interp_spec_mean(wl)
    
    return (fh_norm - fc_norm)

def equal_wn_bins(alpha, stds, nbins=2):
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

    alphas = np.zeros(len(inds)-1)
    for i in range(len(inds)-1):
        alphas[i] = np.mean(alpha[inds[i]:inds[i+1]])
    return alphas