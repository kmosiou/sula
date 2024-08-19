import xarray as xr
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
grav=9.8
z0=0.0002


m0 = hspec.integrate('frequency')

#Load wind
filepath = 'lws.pickle'
with open(filepath, 'rb') as file:
    ws = pickle.load(file)
    
filepath = 'lwd.pickle'
with open(filepath, 'rb') as file:
    wd = pickle.load(file)

u10 = ws.WindSpeed * (np.log(10/z0)) / np.log(4.1/z0)
u10a = u10.where(u10 > 5, drop=True)
wdd = wd['WindDirection'].where((wd['WindDirection']>=120) & (wd['WindDirection']<=160), drop=True)

indud = np.intersect1d(u10a.time.values, wdd.time.values)
u10b = u10a.sel(time=indud)
m0b = m0.sel(time=indud)
wd2 = wdd.sel(time=indud)

#Load fetch
filepath = 'lxeff_c.pickle'
with open(filepath, 'rb') as file:
     axeff = pickle.load(file)
     
udr = np.round(wd2 / 10).values * 10
axeff_map = {120: axeff[3], 130: axeff[4], 140: axeff[5], 150: axeff[6], 160: axeff[7]}
xx = np.array([axeff_map.get(int(dir), axeff[7]) for dir in udr])

ndewindsea2 = (grav ** 2 * m0b) / u10b ** 4
afetch2 = (grav * xx) / u10b ** 2

# Step 1: Replace zeros with NaN
no_zeros = ndewindsea2.where(ndewindsea2 != 0, np.nan)

# Step 2: Drop NaN values
e = no_zeros.dropna(dim='time', how='any')

u10c = u10b.sel(time=e['time'].values)
cf = afetch2.sel(time=e['time'].values)

def model(x, a, b):
    return a * np.power(x, b)

popt3, _ = curve_fit(model, cf, e.WS)
a3, b3 = popt3

kce3 = 5.2e-7 * np.power(cf, 0.9)
hasselman3 = 1.6e-7 * cf
fir_ord_e_hwang3 = 6.191e-7 * np.power(cf, 0.8106)

fig, ax = plt.subplots()
points = ax.scatter(cf, e.WS, c=u10c, cmap="gist_rainbow", lw=0, s=100)
ax.plot(cf, kce3, 'g-.', lw=4, label='Kahma & Calkoen, 1992')
ax.plot(cf, hasselman3, 'b:', lw=4, label='Hasselman, 1973')
ax.plot(cf, fir_ord_e_hwang3, 'k--', lw=4, label='1st order Hwang & Wang, 2004')
ax.plot(cf, model(cf, *popt3), 'r-', lw=4, label='fit')
    
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('dimensionless fetch X', fontweight='bold')
ax.set_ylabel('dimensionless energy E', fontweight='bold')
ax.set_title(f'L4 120<=winddir<=160, wspd>5m/s, a={a3:.3e}, b={b3:.3f}')
ax.legend()
   
cbar = plt.colorbar(points)
cbar.ax.set_ylabel('WindSpeed m/s')
ax.grid(which='major', color='#808080', linewidth=1.5)
ax.grid(which='minor', color='#C0C0C0', linestyle=':', linewidth=1)
ax.minorticks_on()
plt.show()
