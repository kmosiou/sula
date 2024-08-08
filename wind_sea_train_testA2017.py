import xarray as xr
import numpy as np
import pickle
import dill
from scipy.stats import circmean
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

z0 = 0.0002
grav=9.8

ds = xr.open_dataset('2017_A_Sulafjord_specwind_new.nc')
ds['cp'] = 9.81/(2*np.pi*ds['frequency']) # phase speed
ds['u10'] = dsa['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
ds['pdir'] = ds['SPEC'].integrate('frequency').idxmax(dim='direction')

# Drennan et al., 2003. On the wave age dependence of wind stress over pure wind seas
# u10*cos(thetad) > 0.83*cp & thetad [winddir - pdir] < 45 
angdif = np.abs(ds['pdir']-ds['WindDirection'])
thetadd = angdif.where(~np.isnan(angdif), drop=True)
thetad3 = xr.where(thetadd > 180, 360 - thetadd, thetadd)
test2 = thetad3.where(thetad3<45, drop=True)

ds['a'] = 1.2*(ds['u10']/ds['cp'])*np.cos(np.deg2rad(thetad3))
ds['spec_windsea'] = ds['SPEC'].where(ds['a']>1,0)
ds['spec_swell'] = ds['SPEC'].where(ds['a']<=1,0)
ds['hspec_windsea'] = ds['spec_windsea'].integrate('direction')

# Load effective fetch
filepath = 'axeff_c.pickle'
with open(filepath, 'rb') as file:
    axeff = pickle.load(file)

wd2 = ds['WindDirection'].where((ds['WindDirection'] >= 120) & (ds['WindDirection'] <= 160), drop=True)


