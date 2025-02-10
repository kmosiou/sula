#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
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

def process_dataset(file_path, z0, beta, grav=9.81):
    """
    Process a NetCDF dataset to calculate specific wind and wave-related variables.

    Parameters:
    - file_path (str): Path to the NetCDF file.
    - z0 (float): Roughness length for logarithmic wind profile.
    - beta (float): Proportionality constant.
    - grav (float, optional): Acceleration due to gravity (default is 9.81 m/s²).

    Returns:
    - xr.Dataset: The processed dataset with added variables.
    """
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    # Calculate 10m wind speed
    ds['u10'] = ds['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
    
    # Calculate phase speed
    ds['cp'] = grav / (2 * np.pi * ds['frequency'])
    
    dth = angdif(ds['direction'], ds['WindDirection'])
    
    # Calculate a
    ds['a'] = beta * (ds['u10'] / ds['cp']) * np.cos(np.deg2rad(dth))
    
    # Calculate WS with the condition
    ds['WS'] = ds['SPEC'].where(ds['a'] > 1, 0)
    ds['SW'] = ds['SPEC'].where(ds['a'] <= 1, 0)
    
    # Calculate wp (peak wave frequency) and Hspec
    ds['fp_ws'] = ds['WS'].integrate('direction').idxmax(dim='frequency')
    ds['fp_sw'] = ds['SW'].integrate('direction').idxmax(dim='frequency')
    ds['fp'] = ds['SPEC'].integrate('direction').idxmax(dim='frequency')
    ds['wp_ws'] = 2 * np.pi * ds['fp_ws']
    ds['Ef'] = ds['SPEC'].integrate('direction')
    ds['Ef_ws'] = ds['WS'].integrate('direction')
    ds['Ef_sw'] = ds['SW'].integrate('direction')
  
    return ds

def aarnes_windsea(file_path, z0, beta, grav=9.81):
    """
    Partitioning of the directional spectrum to extract the windsea part.
    Methodology according to: a) Bidlot 2001 & b) Aarnes & Krogstad 2001
    Parameters:
    - file_path (str): Path to the NetCDF file.
    - z0 (float): Roughness length for logarithmic wind profile.
    - beta (float): 1.2 according to Bidlot 2001.
    - grav (float, optional): Acceleration due to gravity (default is 9.81 m/s²).

    Returns:
    - xr.Dataset: The processed dataset with added variables.
    """
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    # Calculate 10m wind speed
    ds['u10'] = ds['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
    
    # Calculate phase speed
    ds['cp'] = grav / (2 * np.pi * ds['frequency'])
    
    ds['dd'] = angdif(ds['direction'] * (180/np.pi), ds['WindDirection'])
    
    # Calculate a
    ds['a'] = beta * (ds['u10'] / ds['cp']) * np.cos(np.deg2rad(ds['dd']))
    
    # Calculate the windsea part of the spectrum
    ds['WS'] = ds['SPEC'].where(ds['a'] > 1, 0)
    ds['WS2'] = ds['WS'].where(ds['dd'] < 90, 0)
    ds['Ef'] = ds['SPEC'].integrate('direction')
    ds['Ef_ws'] = ds['WS2'].integrate('direction')
    
    # Calculate integrated parameters
    ds['fp_ws'] = ds['WS2'].integrate('direction').idxmax(dim='frequency')
    ds['wp_ws'] = 2 * np.pi * ds['fp_ws']
    ds['pdir_ws'] = ds['WS2'].integrate('frequency').idxmax(dim='direction')
    
  
    return ds

def process_wind_data(wind_data, z0=0.0002, grav=9.8, window_size=9, ws_threshold=2.5, dir_threshold=15):
    """
    Process wind data to compute rolling averages and apply filtering.

    Parameters:
        wind_data (xarray.DataArray or pandas.DataFrame): The input wind data.
        z0 (float): Roughness length (default 0.0002).
        grav (float): Gravitational acceleration (default 9.8).
        window_size (int): Rolling window size (default 9).
        ws_threshold (float): Wind speed difference threshold for filtering (default 2.5).
        dir_threshold (float): Wind direction difference threshold for filtering (default 15).

    Returns:
        wind_f (xarray.DataArray): Filtered wind speed data.
        dir_f (xarray.DataArray): Filtered wind direction data.
    """

    # Convert to DataFrame and compute u and v components of the wind if data is xarray
    wind = wind_data.to_dataframe()

    wind['u'] = -wind['WindSpeed'] * np.sin(np.deg2rad(wind['WindDirection']))
    wind['v'] = -wind['WindSpeed'] * np.cos(np.deg2rad(wind['WindDirection']))
    
    # Compute 1.5 hour rolling averages for u, v
    wind['u15'] = wind['u'].rolling(window=window_size).mean()
    wind['v15'] = wind['v'].rolling(window=window_size).mean()
    wind['ws15'] = np.sqrt(wind['u15']**2 + wind['v15']**2)
    
    # Compute 1.5 hour rolling direction
    dperr = 180 / np.pi #deg to rad
    wind['dir15'] = ((np.arctan2(wind['u15'], wind['v15']) * dperr) + 180) % 360 # direction from
    
    
    # Convert DataFrame to xarray and compute differences
    wind2 = wind.to_xarray()
    wind2['difws'] = np.abs(wind2['WindSpeed'] - wind2['ws15'])
    wind2['difphi'] = np.abs(wind2['WindDirection'] - wind2['dir15'])
    
    # Adjust angular differences to be within 180 degrees
    dtheta = wind2.difphi
    dtheta = xr.where(dtheta > 180, 360 - dtheta, dtheta)
    
    # Apply filters based on thresholds
    wind_filt = wind2['WindSpeed'].where(wind2.difws <= ws_threshold, drop=True)
    dir_filt = wind2['WindDirection'].where(dtheta <= dir_threshold, drop=True)
    
    # Find common time indices between the two filtered datasets
    ind = np.intersect1d(wind_filt.time.values, dir_filt.time.values)
    
    # Return the filtered data
    dir_f = dir_filt.sel(time=ind)
    wind_f = wind_filt.sel(time=ind)
    
    return wind_f, dir_f

def angdif_v2(angle1, angle2):

    """
              angle1, angle2: in degrees
    Returns:
              smallest differece between angle1, angle2: dif

    """
    
    dif = np.zeros(len(angle1))

    for i in range(len(dif)):
        if (angle1[i] > angle2[i]):
           if ((angle1[i] - angle2[i]) > 180):
               dif[i] = (360 - angle1[i]) + angle2[i]
           else:
               dif[i] = angle1[i] - angle2[i]
        else:
            if ((angle2[i] - angle1[i]) > 180):
                dif[i] = (360 - angle2[i]) + angle1[i]
            else:
                dif[i] = angle2[i] - angle1[i]

    return dif

