import xarray as xr
import numpy as np

beta=1.2 
grav=9.8
z0=0.0002

ds = xr.open_dataset('2016_A_Sulafjord_specwind_new.nc')
ds['u10'] = ds['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
ds['cp'] = grav / (2 * np.pi * ds['frequency'])
ds['a'] = beta * (ds['u10'] / ds['cp']) * np.cos(ds['direction']-np.deg2rad(ds['WindDirection']))
ds['WS'] = ds['SPEC'].where(ds['a'] > 1, 0)
ds['windsea'] = ds['WS'].integrate('direction')
ds['SW'] = ds['SPEC'].where(ds['a'] <= 1, 0)
ds['swell'] = ds['SW'].integrate('direction')
