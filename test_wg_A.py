# coding: utf-8

import xarray as xr
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import dill
plt.rcParams.update({'font.size': 26})

filepath = 'axeff_c.pickle' #load effective fetch
with open(filepath, 'rb') as file:
    axeff = pickle.load(file)

# previously
#filepath = 'a16_windsea.pickle'
#with open(filepath, 'rb') as file:
#    a16 = pickle.load(file)
#    
#filepath = 'a17_wsean.pickle'
#with open(filepath, 'rb') as file:
#    a17 = pickle.load(file)
#    
#filepath = 'a18_wsea.pickle'
#with open(filepath, 'rb') as file:
#    a18 = pickle.load(file)
#    
#filepath = 'a19_wsean.pickle'
#with open(filepath, 'rb') as file:
#    a19 = pickle.load(file)
#    
#filepath = 'a20_wsean.pickle'
#with open(filepath, 'rb') as file:
#    a20 = pickle.load(file)
#    
#filepath = 'a21_wsean.pickle'
#with open(filepath, 'rb') as file:
#    a21 = pickle.load(file)
#    
#filepath = 'a22_windsea.pickle'
#with open(filepath, 'rb') as file:
#    a22 = pickle.load(file)
#    
#filepath = 'a23_windsea.pickle'
#with open(filepath, 'rb') as file:
#    a23 = pickle.load(file)
#    
#filepath = 'a24_windsea.pickle'
#with open(filepath, 'rb') as file:
#    a24 = pickle.load(file)
#
#aws = xr.merge([a16, a17, a18, a19, a20, a21, a22, a23, a24])

filepath = 'awindsea.pickle' #load windsea_spec2d
with open(filepath, 'rb') as file:
    aws = pickle.load(file)
    

dill.load_session('wswd_f.pkl') # load wind
z0=0.0002
grav=9.8
u10a = ws* (np.log(10/z0))/np.log(4.1/z0) # Calculate windspeed @ 10m
u10 = u10a.where(u10a>8, drop=True) # Filter windspeed 
wdd = wd.where((wd >= 90) & (wd <=210), drop=True) # Filter wind direction
indud = np.intersect1d(u10['time'].values, wdd['time'].values) 
ud412 = wdd['WindDirection'].sel(time=indud)
udr = np.round(ud412 / 10).values * 10

#xx = np.empty(len(udr))
#for i in range(len(udr)):
#    if udr[i]==90:
#        xx[i]=axeff[0]
#    elif udr[i]==100:
#        xx[i]=axeff[1]
#    elif udr[i]==110:
#        xx[i]=axeff[2]
#    elif udr[i]==120:
#        xx[i]=axeff[3]
#    elif udr[i]==130:
#        xx[i]=axeff[4]  
#    elif udr[i]==140:
#        xx[i]=axeff[5]    
#    elif udr[i]==150:
#        xx[i]=axeff[6]
#    elif udr[i]==160:
#        xx[i]=axeff[7]  
#    elif udr[i]==170:
#        xx[i]=axeff[8]    
#    elif udr[i]==180:
#        xx[i]=axeff[9] 
#    elif udr[i]==190:
#        xx[i]=axeff[10]  
#    elif udr[i]==200:
#       xx[i]=axeff[11]  
#    else:
#        xx[i]=axeff[12]

# Create a dictionary for axeff mapping
axeff_map = {90: axeff[0], 100: axeff[1], 110: axeff[2], 120: axeff[3], 130: axeff[4], 
             140: axeff[5], 150: axeff[6], 160: axeff[7], 170: axeff[8], 180: axeff[9], 
             190: axeff[10], 200: axeff[11], 210: axeff[12]}

# Map the rounded wind directions to axeff values
xx = np.array([axeff_map.get(int(dir), axeff[12]) for dir in udr])

u10b = u10.sel(time=indud)
afetch = (grav * xx)/u10b['WindSpeed']**2 # Calculate dimensionless fetch
windsea = aws['h_SPEC_windsea'].sel(time=indud)
intS = windsea.integrate('frequency')
ndewindsea = (grav**2 * intS)/u10b['WindSpeed']**4 # Calculate dimensionless energy
fc = afetch.where(~np.isnan(afetch), drop=True) 
ec = ndewindsea.where(~np.isnan(ndewindsea), drop=True)
cf = afetch.sel(time=ec['time'].values)
u10c = u10a.sel(time=ec['time'].values)

three_hour_means = u10c.resample(time='3H').mean(skipna=True)
# Use broadcasting to align the original data with the 3-hour means
aligned_means = three_hour_means.reindex_like(u10c, method='ffill')
# Subtract the 3-hour means from the original data
adjusted_data_array = np.abs(u10c - aligned_means)

from scipy.stats import circmean
# Function to calculate the circular mean for each group
def circular_mean(group):
    return circmean(group, high=360, low=0)
    
wdc = wdd.sel(time=ec['time'].values)
# Resample data into 3-hour intervals
resampled = wdc.resample(time='3h')
# Calculate the circular mean for each resampled group
three_hour_circular_means = resampled.reduce(circmean, high=360, low=0)
# Use reindexing to align the means with the original data points
aligned_circular_means = three_hour_circular_means.reindex(time=wdc.time, method='ffill')
# Subtract the aligned circular means from the original data
adjusted_circdata_array = np.abs(wdc['WindDirection'] - aligned_circular_means)

cond_wd = adjusted_circdata_array.where(adjusted_circdata_array.WindDirection <= 15, drop=True)
cond_ws = adjusted_data_array.where(adjusted_data_array.WindSpeed <= 2.5, drop=True)
cond_both = np.intersect1d(cond_wd.time.values, cond_ws.time.values)

cf2 = cf.sel(time=cond_both)
ec2 = ec.sel(time=cond_both)
u10c2 = u10c.sel(time=cond_both)
kce2= 5.2*pow(10,-7)*pow(cf2,0.9)
hasselman2 = 1.6*pow(10,-7)*cf2
fir_ord_e_hwang2 = 6.191*pow(10,-7)*pow(cf2,0.8106)

from scipy.optimize import curve_fit
def model(x, a, b):
    return a * np.power(x, b)
    
popt, pcov = curve_fit(model, cf2, ec2)
a,b = popt

fig, ax = plt.subplots()
points = plt.scatter(cf2, ec2, c=u10c2['WindSpeed'],cmap="viridis", lw=0, s=100)
line2 = ax.plot(cf2, kce2, 'g-.',lw=4, label='Kahma & Calkoen, 1992')
line3 = ax.plot(cf2, hasselman2, 'b:',lw=4, label='Hasselman, 1973')
line4 = ax.plot(cf2, fir_ord_e_hwang2, 'k--',lw=4, label='1st order Hwang & Wang, 2004')
line5 = ax.plot(cf2, model(cf2,*popt), 'r-', lw=4, label= 'fit')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('dimensionless fetch X', fontweight='bold')
ax.set_ylabel('dimensionless energy E', fontweight='bold')
ax.set_title('A , 90<=winddir<=210, wspd>8m/s, 66<=fetch<=234')
ax.legend()
cbar = plt.colorbar()
cbar.ax.set_ylabel('WindSpeed m/s')
ax.grid(which='major', color='#808080', linewidth=0.9)
ax.grid(which='minor', color='#C0C0C0', linestyle=':', linewidth=0.6)
ax.minorticks_on()
plt.show()
