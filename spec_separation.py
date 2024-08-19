import xarray as xr
import numpy as np
import pickle
import dill
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
ncfile = '2016_A_Sulafjord_specwind_new.nc'
ds = xr.open_dataset(ncfile)

beta=1.2 
grav=9.8
z0=0.0002
ang_diff_threshold=45

ds['u10'] = ds['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
ds['cp'] = grav / (2 * np.pi * ds['frequency'])
angdif = np.abs(ds['pdir'] - ds['WindDirection'])
thetadd = angdif.where(~np.isnan(angdif), drop=True)
thetad3 = xr.where(thetadd > 180, 360 - thetadd, thetadd)
test2 = thetad3.where(thetad3 < ang_diff_threshold, drop=True)
ds['a'] = beta * (ds['u10'] / ds['cp']) * np.cos(np.deg2rad(test2))
ds['WS'] = ds['SPEC'].where(ds['a'] > 1, 0)
ds['SW'] = ds['SPEC'].where(ds['a'] <= 1, 0)
windsea = ds['WS'].integrate('direction')
swell = ds['SW'].integrate('direction')


