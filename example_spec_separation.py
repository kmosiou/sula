import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dsa17 = xr.open_dataset('2017_B_Sulafjord_specwind_new.nc', drop_variables={'Hs','Tp','pdir','h_SPEC','WindGust','Latitude','Longitude'})
beta = 1.3
grav=9.8
z0=0.0002

dsa17['cp'] = 9.81/(2*np.pi*dsa17['frequency']) # phase speed
dsa17['A']  = beta*(dsa17['WindSpeed']/dsa17['cp'])*np.cos(dsa17['direction']-np.deg2rad(dsa17['WindDirection']))
dsa17['SPEC_windsea'] = dsa17['SPEC'].where(dsa17['A']>1,0)
dsa17['h_SPEC_windsea'] = dsa17['SPEC_windsea'].integrate('direction')
b17 = dsa17['h_SPEC_windsea']

filepath = 'b17_wsea.pickle'
with open(filepath, 'wb') as file:
    pickle.dump(b17_wsea,file)
