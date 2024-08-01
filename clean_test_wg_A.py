# coding: utf-8

import xarray as xr
import numpy as np
import pickle
import matplotlib.pyplot as plt
import dill
from scipy.stats import circmean
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 26})

# Load effective fetch
filepath = 'axeff_c.pickle'
with open(filepath, 'rb') as file:
    axeff = pickle.load(file)

# Load windsea_spec2d
filepath = 'awindsea.pickle'
with open(filepath, 'rb') as file:
    aws = pickle.load(file)

# Load wind
dill.load_session('wswd_f.pkl')

# Constants
z0 = 0.0002
grav = 9.8

# Calculate windspeed @ 10m
u10a = ws * (np.log(10 / z0)) / np.log(4.1 / z0)
u10 = u10a.where(u10a > 8, drop=True)  # Filter windspeed

# Filter wind direction
wdd = wd.where((wd >= 90) & (wd <= 210), drop=True)

# Get intersection of times
indud = np.intersect1d(u10['time'].values, wdd['time'].values)
ud412 = wdd['WindDirection'].sel(time=indud)
udr = np.round(ud412 / 10).values * 10

# Create a dictionary for axeff mapping
axeff_map = {90: axeff[0], 100: axeff[1], 110: axeff[2], 120: axeff[3], 130: axeff[4], 
             140: axeff[5], 150: axeff[6], 160: axeff[7], 170: axeff[8], 180: axeff[9], 
             190: axeff[10], 200: axeff[11], 210: axeff[12]}

# Map the rounded wind directions to axeff values
xx = np.array([axeff_map.get(int(dir), axeff[12]) for dir in udr])

# Calculate dimensionless fetch and energy
u10b = u10.sel(time=indud)
afetch = (grav * xx) / u10b['WindSpeed'] ** 2
windsea = aws['h_SPEC_windsea'].sel(time=indud)
intS = windsea.integrate('frequency')
ndewindsea = (grav ** 2 * intS) / u10b['WindSpeed'] ** 4

# Filter non-NaN values
fc = afetch.where(~np.isnan(afetch), drop=True)
ec = ndewindsea.where(~np.isnan(ndewindsea), drop=True)
cf = afetch.sel(time=ec['time'].values)
u10c = u10a.sel(time=ec['time'].values)

# Resample and adjust windspeed data
three_hour_means = u10c.resample(time='3H').mean(skipna=True)
aligned_means = three_hour_means.reindex_like(u10c, method='ffill')
adjusted_data_array = np.abs(u10c - aligned_means)

# Resample and adjust wind direction data
wdc = wdd.sel(time=ec['time'].values)
resampled = wdc.resample(time='3H')
three_hour_circular_means = resampled.reduce(circmean, high=360, low=0)
aligned_circular_means = three_hour_circular_means.reindex(time=wdc.time, method='ffill')
adjusted_circdata_array = np.abs(wdc['WindDirection'] - aligned_circular_means)

# Filter conditions |u - umean| <= 2.5 & |theta - thetamean| <= 15
cond_wd = adjusted_circdata_array.where(adjusted_circdata_array <= 15, drop=True)
cond_ws = adjusted_data_array.where(adjusted_data_array <= 2.5, drop=True)
cond_both = np.intersect1d(cond_wd.time.values, cond_ws.time.values)

# Select data based on conditions
cf2 = cf.sel(time=cond_both)
ec2 = ec.sel(time=cond_both)
u10c2 = u10c.sel(time=cond_both)

# Define models
kce2 = 5.2e-7 * np.power(cf2, 0.9)
hasselman2 = 1.6e-7 * cf2
fir_ord_e_hwang2 = 6.191e-7 * np.power(cf2, 0.8106)

# Fit the model to the data
def model(x, a, b):
    return a * np.power(x, b)

popt, pcov = curve_fit(model, cf2, ec2)
a, b = popt

# Plot results
fig, ax = plt.subplots()
points = ax.scatter(cf2, ec2, c=u10c2['WindSpeed'], cmap="viridis", lw=0, s=100)
ax.plot(cf2, kce2, 'g-.', lw=4, label='Kahma & Calkoen, 1992')
ax.plot(cf2, hasselman2, 'b:', lw=4, label='Hasselman, 1973')
ax.plot(cf2, fir_ord_e_hwang2, 'k--', lw=4, label='1st order Hwang & Wang, 2004')
ax.plot(cf2, model(cf2, *popt), 'r-', lw=4, label='fit')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('dimensionless fetch X', fontweight='bold')
ax.set_ylabel('dimensionless energy E', fontweight='bold')
ax.set_title('A , 90<=winddir<=210, wspd>8m/s, 66<=fetch<=234')
ax.legend()
cbar = plt.colorbar(points)
cbar.ax.set_ylabel('WindSpeed m/s')
ax.grid(which='major', color='#808080', linewidth=0.9)
ax.grid(which='minor', color='#C0C0C0', linestyle=':', linewidth=0.6)
ax.minorticks_on()
plt.show()
