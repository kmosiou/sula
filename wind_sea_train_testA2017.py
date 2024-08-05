import xarray as xr
import numpy as np
import pickle
import dill
from scipy.stats import circmean
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 26})

z0 = 0.0002
grav=9.8

dsa17 = xr.open_dataset('2017_A_Sulafjord_specwind_new.nc')
dsa17['cp'] = 9.81/(2*np.pi*dsa17['frequency']) # phase speed
dsa17['u10'] = dsa17['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)

thetad = np.abs(dsa17['WindDirection']-dsa17['pdir'])
for i in range(len(thetad)):
    if thetad[i] > 180:
        thetad[i] = 360 - thetad[i]
    else:
        thetad[i] = thetad[i]

cond2 = thetad.where(thetad<45, drop=True) % Drennan et al., 2003
dsa17['A'] = dsa17['u10']*np.cos(np.deg2rad(thetad))
dsa17['B'] = 0.83*dsa17['cp']
dsa17['WS'] = dsa17.where(dsa17['A']>dsa17['B'], drop=True) % Drennan et al., 2003
dsa17['WS'] = dsa17['SPEC'].where(dsa17['A']>dsa17['B'], drop=True)
dsa17['WS_1d'] = dsa17['WS'].integrate('direction')
u10 = dsa17['u10'].sel(time=dsa17['WS_1d'].time.values)

# Load effective fetch
filepath = 'axeff_c.pickle'
with open(filepath, 'rb') as file:
    axeff = pickle.load(file)
dill.load_session('wswd_f.pkl')

wd1 = dsa17['WindDirection'].sel(time=dsa17['WS_1d'].time.values)
wd2 = wd.where((wd1 >= 90) & (wd1 <= 210), drop=True)

indud = np.intersect1d(u10['time'].values, wd2['time'].values)
udr = np.round(wd2.WindDirection / 10).values * 10

# Create a dictionary for axeff mapping
axeff_map = {90: axeff[0], 100: axeff[1], 110: axeff[2], 120: axeff[3], 130: axeff[4], 
             140: axeff[5], 150: axeff[6], 160: axeff[7], 170: axeff[8], 180: axeff[9], 
             190: axeff[10], 200: axeff[11], 210: axeff[12]}

# Map the rounded wind directions to axeff values
xx = np.array([axeff_map.get(int(dir), axeff[12]) for dir in udr])
u10b = u10.sel(time=indud)
afetch = (grav * xx) / u10b ** 2
windsea = dsa17['WS_1d'].sel(time=indud)
intS = windsea.integrate('frequency')
ndewindsea = (grav ** 2 * intS) / u10b ** 4

# Filter non-NaN values
fc = afetch.where(~np.isnan(afetch), drop=True)
ec = ndewindsea.where(~np.isnan(ndewindsea), drop=True)
cf = afetch.sel(time=ec['time'].values)
u10c = u10b.sel(time=ec['time'].values)

# Define models
kce = 5.2e-7 * np.power(cf, 0.9)
hasselman = 1.6e-7 * cf
fir_ord_e_hwang = 6.191e-7 * np.power(cf, 0.8106)

# Fit the model to the data
def model(x, a, b):
    return a * np.power(x, b)
# Fit the model to the data
def model(x, a, b):
    return a * np.power(x, b)

popt, pcov = curve_fit(model, cf, ec)
a, b = popt
