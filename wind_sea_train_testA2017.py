import xarray as xr
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(dataset_path, pickle_path):
    ds = xr.open_dataset(dataset_path)
    with open(pickle_path, 'rb') as file:
        axeff = pickle.load(file)
    return ds, axeff

def calculate_parameters(ds, beta=1.2, grav=9.8, z0=0.0002):
    ds['u10'] = ds['WindSpeed'] * (np.log(10 / z0)) / np.log(4.1 / z0)
    ds['cp'] = grav / (2 * np.pi * ds['frequency'])
    return ds

def filter_data(ds, wind_speed_threshold=10, wind_dir_min=120, wind_dir_max=160, ang_diff_threshold=45):
    u10 = ds['u10'].where(ds['u10'] > wind_speed_threshold, drop=True)
    wdd = ds['WindDirection'].where((ds['WindDirection'] >= wind_dir_min) & (ds['WindDirection'] <= wind_dir_max), drop=True)
    indud = np.intersect1d(u10['time'].values, wdd['time'].values)
    
    angdif = np.abs(ds['pdir'] - ds['WindDirection'])
    thetadd = angdif.where(~np.isnan(angdif), drop=True)
    thetad3 = xr.where(thetadd > 180, 360 - thetadd, thetadd)
    test2 = thetad3.where(thetad3 < ang_diff_threshold, drop=True)
    
    return indud, test2, wdd

def calculate_wave_parameters(ds, indud, thetad3, test2, beta=1.2, grav=9.8):
    ds['a'] = beta * (ds['u10'] / ds['cp']) * np.cos(np.deg2rad(thetad3))
    ds['WS'] = ds['SPEC'].where(ds['a'] > 1, 0)
    hspec = ds['WS'].integrate('direction')
    ee = hspec.sel(time=test2['time'].values)
    indud2 = np.intersect1d(indud, ee['time'].values)
    ee2 = hspec.sel(time=indud2)
    anemos = ds['u10'].sel(time=indud2)
    m0 = ee2.integrate('frequency')
    ndewindsea2 = (grav ** 2 * m0) / anemos ** 4
    
    return ndewindsea2, anemos, indud2

def fetch_values(ds, wdd, indud2, axeff):
    udr = np.round(wdd.sel(time=indud2) / 10).values * 10
    axeff_map = {120: axeff[3], 130: axeff[4], 140: axeff[5], 150: axeff[6], 160: axeff[7]}
    xx = np.array([axeff_map.get(int(dir), axeff[7]) for dir in udr])
    return (9.8 * xx) / ds['u10'].sel(time=indud2) ** 2

def fit_model(afetch2, ndewindsea2):
    def model(x, a, b):
        return a * np.power(x, b)
    
    popt3, _ = curve_fit(model, afetch2, ndewindsea2)
    return popt3

def plot_results(afetch2, ndewindsea2, anemos, popt3):
    a3, b3 = popt3
    plt.rcParams.update({'font.size': 22})
    
    kce3 = 5.2e-7 * np.power(afetch2, 0.9)
    hasselman3 = 1.6e-7 * afetch2
    fir_ord_e_hwang3 = 6.191e-7 * np.power(afetch2, 0.8106)
    
    fig, ax = plt.subplots()
    points = ax.scatter(afetch2, ndewindsea2, c=anemos, cmap="gist_rainbow", lw=0, s=100)
    ax.plot(afetch2, kce3, 'g-.', lw=4, label='Kahma & Calkoen, 1992')
    ax.plot(afetch2, hasselman3, 'b:', lw=4, label='Hasselman, 1973')
    ax.plot(afetch2, fir_ord_e_hwang3, 'k--', lw=4, label='1st order Hwang & Wang, 2004')
    ax.plot(afetch2, a3 * np.power(afetch2, b3), 'r-', lw=4, label='fit')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('dimensionless fetch X', fontweight='bold')
    ax.set_ylabel('dimensionless energy E', fontweight='bold')
    ax.set_title(f'A 2018, 120<=winddir<=160, wspd>10m/s, a={a3:.3e}, b={b3:.3f}')
    ax.legend()
    
    cbar = plt.colorbar(points)
    cbar.ax.set_ylabel('WindSpeed m/s')
    ax.grid(which='major', color='#808080', linewidth=1.5)
    ax.grid(which='minor', color='#C0C0C0', linestyle=':', linewidth=1)
    ax.minorticks_on()
    plt.show()

def main():
    dataset_path = '2018_A_Sulafjord_specwind_n2.nc'
    
    ds, axeff = load_data(dataset_path, pickle_path)
    ds = calculate_parameters(ds)
    
    indud, test2, wdd = filter_data(ds)
    ndewindsea2, anemos, indud2 = calculate_wave_parameters(ds, indud, test2)
    
    afetch2 = fetch_values(ds, wdd, indud2, axeff)
    
    popt3 = fit_model(afetch2, ndewindsea2)
    
    plot_results(afetch2, ndewindsea2, anemos, popt3)

if __name__ == "__main__":
    main()
