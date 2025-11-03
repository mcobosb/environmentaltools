"""
Spatiotemporal Covariance Module
=================================

This module provides functions for computing and modeling spatiotemporal covariance
structures used in Bayesian Maximum Entropy estimation.

Functions include empirical covariance calculation, directional covariance analysis,
theoretical covariance models, and model fitting.
"""

import warnings as warn

import numpy as np
from environmentaltools.spatiotemporal.bme import (
    find_pairs_by_distance,
    create_spatiotemporal_matrix,
)


def compute_spatiotemporal_covariance(dfh, dfs, slag, tlag):
    """
    Compute empirical spatiotemporal covariance.

    Calculates the covariance function for spatiotemporal data at specified
    spatial and temporal lag distances.

    Parameters
    ----------
    dfh : pd.DataFrame
        Hard data with columns ['x', 'y', 't', 'h'].
    dfs : pd.DataFrame
        Soft data with columns ['x', 'y', 't', 'h'].
    slag : np.ndarray
        Vector of spatial lag distances at which to compute covariance.
    tlag : np.ndarray
        Vector of temporal lag distances at which to compute covariance.

    Returns
    -------
    empcovst : np.ndarray
        Matrix of empirical spatiotemporal covariance values, shape (n_spatial, n_temporal).
    pairsnost : np.ndarray
        Number of valid data pairs at each spatiotemporal lag, shape (n_spatial, n_temporal).
    covdists : np.ndarray
        Spatial distance grid for covariance values.
    covdistt : np.ndarray
        Temporal distance grid for covariance values.

    Notes
    -----
    The covariance at lag (0,0) is set to the variance of the data.
    Computes sample covariance: Cov(X,Y) = E[XY] - E[X]E[Y].
    """
    slagtol = np.hstack((0, np.diff(slag)))
    tlagtol = np.hstack((0, np.diff(tlag)))

    dfz, coups, coupt = create_spatiotemporal_matrix(dfh, dfs)

    ns, nt = len(slag), len(tlag)
    idxpairs = find_pairs_by_distance(coups, slag, slagtol)
    idtpairs = find_pairs_by_distance(coupt, tlag, tlagtol)

    # Computing covariance
    empcovst = np.ones((ns, nt)) * np.nan
    pairsnost = np.zeros((ns, nt))
    xmean, ymean, xymean = (
        np.ones((ns, nt)) * np.nan,
        np.ones((ns, nt)) * np.nan,
        np.ones((ns, nt)) * np.nan,
    )
    for inds in range(ns):
        # Test if we have data for this spatial lag
        if np.shape(idxpairs[inds])[1] != 0:
            for indt in range(0, nt):
                # Test if we have data for this temporal lag
                if np.shape(idtpairs[indt])[1] != 0:
                    xcol = dfz.loc[idtpairs[indt][0], idxpairs[inds][0]].values
                    xrow = dfz.loc[idtpairs[indt][1], idxpairs[inds][1]].values
                    idxValid = ~np.isnan(xcol) & ~np.isnan(xrow)
                    pairsnost[inds, indt] = np.sum(idxValid)
                    if pairsnost[inds, indt] != 0:
                        xcol = xcol[idxValid]
                        xrow = xrow[idxValid]
                        xmean[inds, indt] = np.mean(xcol)
                        ymean[inds, indt] = np.mean(xrow)
                        xymean[inds, indt] = np.mean(xcol * xrow)
                        empcovst[inds, indt] = (
                            xymean[inds, indt] - xmean[inds, indt] * ymean[inds, indt]
                        )

    empcovst[0, 0] = dfz.stack().var()
    covdists = np.tile(slag, (nt, 1)).T
    covdistt = np.tile(tlag, (ns, 1))
    return empcovst, pairsnost, covdists, covdistt


def compute_directional_covariance(dfh, dfs, slag, tlag, dinfo):
    """
    Compute spatiotemporal covariance for multiple directional bins.

    Calculates anisotropic (direction-dependent) covariance structures by
    analyzing data pairs grouped by their spatial direction.

    Parameters
    ----------
    dfh : pd.DataFrame
        Hard data with columns ['x', 'y', 't', 'h'].
    dfs : pd.DataFrame
        Soft data with columns ['x', 'y', 't', 'h'].
    slag : np.ndarray
        Vector of spatial lag distances.
    tlag : np.ndarray
        Vector of temporal lag distances.
    dinfo : list
        Directional information [dlag, dlagtol] where:
        - dlag: array of directional angles (degrees)
        - dlagtol: angular tolerance for binning

    Returns
    -------
    empcovst : np.ndarray
        3D array of empirical covariance, shape (n_spatial, n_directions, n_temporal).
    pairsnost : np.ndarray
        3D array of pair counts, shape (n_spatial, n_temporal, n_directions).
    covdist : np.ndarray
        Spatial distance grid.
    covdistd : np.ndarray
        Directional angle grid.
    covdistt : np.ndarray
        Temporal distance vector.

    Notes
    -----
    Useful for detecting and modeling spatial anisotropy in spatiotemporal fields.
    Directions are typically binned (e.g., 0°, 45°, 90°, 135°) to capture
    orientation-dependent correlation structures.
    """
    slagtol = np.hstack((0, np.diff(slag)))
    tlagtol = np.hstack((0, np.diff(tlag)))
    dlag, dlagtol = dinfo[0], dinfo[1]

    dfz, coups, coupt = create_spatiotemporal_matrix(dfh, dfs)

    ns, nt, nd = len(slag), len(tlag), len(dlag)
    idxpairs = find_pairs_by_distance(coups, slag, slagtol, [dlag, dlagtol])
    idtpairs = find_pairs_by_distance(coupt, tlag, tlagtol)

    # Computing directional covariance
    empcovst = np.ones((ns, nd, nt)) * np.nan
    pairsnost = np.zeros((ns, nt, nd))
    xmean, ymean, xymean = (
        np.ones((ns, nt, nd)) * np.nan,
        np.ones((ns, nt, nd)) * np.nan,
        np.ones((ns, nt, nd)) * np.nan,
    )
    for indd in range(nd):
        print("indd: " + str(indd))
        for inds in range(ns):
            print("inds: " + str(inds))
            # Test if we have data for this spatial lag
            if np.shape(idxpairs[inds])[1] > 1:
                for indt in range(nt):
                    print("indt: " + str(indt))
                    # Test if we have data for this temporal lag
                    if np.shape(idtpairs[indt])[1] > 1:
                        xcol = dfz.loc[
                            idtpairs[indt][0], idxpairs[inds][indd][0]
                        ].values
                        xrow = dfz.loc[
                            idtpairs[indt][1], idxpairs[inds][indd][1]
                        ].values
                        idxValid = ~np.isnan(xcol) & ~np.isnan(xrow)
                        pairsnost[inds, indt, indd] = np.sum(idxValid)
                        if pairsnost[inds, indt, indd] != 0:
                            xcol = xcol[idxValid]
                            xrow = xrow[idxValid]
                            xmean[inds, indt, indd] = np.mean(xcol)
                            ymean[inds, indt, indd] = np.mean(xrow)
                            xymean[inds, indt, indd] = np.mean(xcol * xrow)
                            empcovst[inds, indd, indt] = (
                                xymean[inds, indt, indd]
                                - xmean[inds, indt, indd] * ymean[inds, indt, indd]
                            )

    empcovst[0, 0, 0] = dfz.stack().var()
    covdist, covdistd, covdistt = np.tile(slag, (nd, 1)), np.tile(dlag, (ns, 1)).T, tlag
    return empcovst, pairsnost, covdist, covdistd, covdistt


def calculate_theoretical_covariance(name, D, param):
    """
    Compute theoretical spatiotemporal covariance from parametric models.

    Evaluates covariance functions from theoretical models at specified
    spatiotemporal separation distances.

    Parameters
    ----------
    name : str
        Covariance model family. Supported models:
        - 'exponentialST': Separable exponential model
        - 'exponentialSTC': Non-separable exponential model with interaction term
    D : list of np.ndarray
        Separation distances [spatial_distances, temporal_distances].
    param : array-like
        Model parameters. Length depends on model:
        - 'exponentialST': [sill, spatial_range, temporal_range, nugget]
        - 'exponentialSTC': [sill, spatial_range, temporal_range, interaction_range, nugget]

    Returns
    -------
    res : np.ndarray
        Theoretical covariance values at the specified separations.

    Raises
    ------
    Warning
        If requested model is not implemented.

    Notes
    -----
    **Exponential ST Model (separable)**:
        C(d,t) = sill * exp(-d/range_s - t/range_t) - nugget

    **Exponential STC Model (non-separable with interaction)**:
        C(d,t) = sill * exp(-√d/range_s - t/range_t - √d*t/range_st) - nugget

    The nugget effect represents measurement error or micro-scale variability.
    """
    d, t = np.asarray(D[0]), np.asarray(D[1])
    if name == "exponentialST":
        res = param[0] * np.exp(-d / param[1] - t / param[2]) - param[3]
    elif name == "exponentialSTC":
        res = (
            param[0]
            * np.exp(-(d ** 0.5) / param[1] - t / param[2] - d ** 0.5 * t / param[3])
            - param[4]
        )
    else:
        warn.warn("Sorry! This model is not implemented yet.")
    return res


def fit_covariance_model(param, empcovst, dist, name):
    """
    Compute sum of squared errors between theoretical and empirical covariance.

    Objective function for least squares fitting of covariance model parameters
    to empirical covariance estimates.

    Parameters
    ----------
    param : array-like
        Covariance model parameters to optimize.
    empcovst : np.ndarray
        Empirical spatiotemporal covariance matrix.
    dist : list of np.ndarray
        Spatiotemporal separation distances [spatial, temporal].
    name : str
        Covariance model family name (e.g., 'exponentialST', 'exponentialSTC').

    Returns
    -------
    float
        Sum of squared errors between theoretical and empirical covariance.

    Notes
    -----
    This function is typically used with scipy.optimize.minimize or similar
    optimization routines to find optimal model parameters.

    The objective is: SSE = Σ(C_empirical - C_theoretical)²
    """
    cov = calculate_theoretical_covariance(name, dist, param)
    return np.sum(np.sum(((empcovst - cov) ** 2), axis=1), axis=0)
