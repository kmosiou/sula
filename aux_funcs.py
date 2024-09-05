#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np

def angdif(ang1, ang2):
# ang* in degrees
    dtheta = np.abs(ang1 - ang2)
    dtheta = dtheta.where(~np.isnan(dtheta), drop=True)
    dtheta = xr.where(dtheta > 180, 360 - dtheta, dtheta)

    return dtheta
