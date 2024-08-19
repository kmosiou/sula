import numpy as np
import pickle

filepath = 'fetch_ext.pickle'

with open(filepath, 'rb') as file:
    a_fetch_ext = pickle.load(file)
angles = np.arange(90,220,10)
x = len(angles)
y = len(a_fetch_ext)
ang = [90, 102, 114, 120, 132, 144, 150, 162, 174, 180, 192, 204, 210]
axeff = np.zeros(x)

for i in range(x):
    loc = np.argmin(np.abs(a_fetch_ext.index - angles[i]))
    temp = a_fetch_ext.iloc[loc-4:loc+5,0]
    dirs = np.zeros(len(temp))
    for j in range(len(temp)):
        dirs[j] = np.deg2rad(temp.index[j]-ang[i])
    A = np.empty(len(temp))
    B = np.empty(len(temp))
    for u in range(len(dirs)):
        A[u] = temp.values[u]*(np.cos(dirs[u])**2)
        B[u] = np.cos(dirs[u])
    axeff[i] = np.sum(A)/np.abs(np.sum(B))
del A, B, dirs, temp, loc

