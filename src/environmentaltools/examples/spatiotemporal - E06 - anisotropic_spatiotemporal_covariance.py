#!/usr/bin/env python3
"""
Converted from Jupyter notebook: spatiotemporal - E06 - anisotropic-spatiotemporal-covariance.ipynb

This file was automatically converted from a Jupyter notebook.
Markdown cells are preserved as comments, code cells as executable Python.
"""

# ------------------------------------------------------------
# Markdown Cell 1
# ------------------------------------------------------------
# # spatiotemporal - E06 - anisotropic-spatiotemporal-covariance
#
# #### Revised on: 2025-10-30
#
# ------------------------------------------------------------
# Code Cell 2
# ------------------------------------------------------------


import numpy as np
import pandas as pd
from environmentaltools.common.utils import save
import os
from environmentaltools.graphics import spatiotemporal as figures
from environmentaltools.spatiotemporal import covariance
from scipy.optimize import minimize

tstr = ["20_Jan_2017", "26_Jan_2017"]
# tstr = ['19_Feb_2017', '24_Feb_2017']

if tstr[0] == "20_Jan_2017":
    smax, ns, tmax, nt, nd = 9, 5, 72, 8, 6
else:
    smax, ns, tmax, nt, nd = 4, 5, 72, 8, 4

path = os.path.join("..", "ProcessedData", tstr[0] + "-" + tstr[1])

dfh = pd.read_csv(
    os.path.join(path, "Hard_data.txt"), sep=" ", names=["x", "y", "t", "h"]
)
dfs = pd.read_csv(
    os.path.join(path, "Soft_data.txt"), sep=" ", names=["x", "y", "t", "h", "s"]
)

slag = smax * (1 - np.log(ns + 1 - (np.arange(ns) + 1)) / np.log(ns))
tlag = tmax * (1 - np.log(nt + 1 - (np.arange(nt) + 1)) / np.log(nt))
dlag, dlagtol = np.arange(nd) * 180.0 / nd, 90.0 / nd

empcovang, pairsnostd, covdist, covdistd, covdistt = covariance.compute_spatiotemporal_covariance(
    dfh, dfs, slag, tlag, [dlag, dlagtol]
)
save.to_npy(
    [empcovang, pairsnostd, covdist, covdistd, covdistt],
    ["empcovang", "covangpairs", "covdist", "covdistd", "covdistt"],
    path,
)

# D = [covdistx, covdisty]
# par0 = [np.amax(empcovst), 50, 50, -0.2]
# family = 'exponentialST'
# res = minimize(covST.fit, par0, method='SLSQP', args=(expcovst, D, family), options = {'disp': True})
#
figures.covariance_comparison(covdists, covdistt, expcovst, tlag, res, family, '3d')
# ut.save([family, res], ['family', 'param'], path)
slag = smax * (1 - np.log(ns + 1 - (np.arange(ns) + 1)) / np.log(ns))
figures.anisotropic_spatiotemporal_covariance(covdist, covdistd, covdistt, empcovang, slag, ["polar", "covariance"])

