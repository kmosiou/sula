#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
from scipy.stats import circmean

def angdif(ang1, ang2):
# ang* in degrees
    dtheta = np.abs(ang1 - ang2)
    dtheta = dtheta.where(~np.isnan(dtheta), drop=True)
    dtheta = xr.where(dtheta > 180, 360 - dtheta, dtheta)

    return dtheta

def process_wind_direction(winddir, interval='1h', max_deviation=15):
    # Convert wind direction to radians
    wdc = np.deg2rad(winddir)
    
    # Resample the wind direction data to the specified interval and calculate the mean
    hourlywind = wdc.resample(time=interval)
    hourlywindmean = hourlywind.reduce(circmean, high=360, low=0)
    
    # Reindex to match the original timestamps, using forward fill for missing values
    hourlywindmeann = hourlywindmean.reindex(time=wdc.time, method='ffill')
    
    # Calculate wind deviation
    winddev = np.abs(winddir - np.rad2deg(hourlywindmeann))
    winddev = winddev % 360
    winddev = winddev.where(~np.isnan(winddev), drop=True)
    winddev = xr.where(winddev > 180, 360 - winddev, winddev)
    
    # Filter deviations within the specified maximum deviation limit
    limwinddir = winddev.where(winddev <= max_deviation, drop=True)
    
    return limwinddir
