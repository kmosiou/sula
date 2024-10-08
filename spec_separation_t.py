import xarray as xr
import numpy as np
from aux_funcs import angdiff
beta=1.2 
#grav=9.8
z0=0.0002

ncfile = '2016_A_Sulafjord_specwind_new.nc'
ds = xr.open_dataset(ncfile)

ds['u10'] = ds['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
ds['cp'] = grav / (2 * np.pi * ds['frequency'])
dth = angdif(np.rad2deg(ds['direction']), ds['WindDirection'])
ds['a'] = beta * (ds['u10'] / ds['cp']) * np.cos(np.deg2rad(dth))
ds['WS'] = ds['SPEC'].where(ds['a'] > 1, 0)
ds['windsea'] = ds['WS'].integrate('direction')
ds['SW'] = ds['SPEC'].where(ds['a'] <= 1, 0)
ds['swell'] = ds['SW'].integrate('direction')
