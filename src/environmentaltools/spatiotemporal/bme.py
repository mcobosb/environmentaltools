"""
Bayesian Maximum Entropy (BME) Module
======================================

This module provides functions for Bayesian Maximum Entropy spatiotemporal estimation
and analysis, including moment computation, local mean estimation, and covariance modeling.
"""

import itertools as it
import os

import numpy as np
import pandas as pd
import scipy.stats as st


def compute_bme_moments(dfk, dfh, dfs, covmodel, covparam, nmax, dmax, order, options, path, name):
    """
    Compute Bayesian Maximum Entropy moments for spatiotemporal estimation.

    This function calculates statistical moments at estimation points using BME theory,
    incorporating both hard (exact) and soft (probabilistic) data with spatiotemporal
    covariance modeling.

    Parameters
    ----------
    dfk : pd.DataFrame
        Estimation points with columns ['x', 'y', 't'] for spatial and temporal coordinates.
    dfh : pd.DataFrame
        Hard data (exact observations) with columns ['x', 'y', 't', 'h'].
    dfs : pd.DataFrame
        Soft data (probabilistic observations) with columns ['x', 'y', 't', 'h', 's'].
    covmodel : str
        Covariance model type (e.g., 'exponential', 'gaussian', 'spherical').
    covparam : array-like
        Parameters for the covariance model.
    nmax : list of int
        Maximum number of data points [hard_max, soft_max] to use in local neighborhoods.
    dmax : list of float
        Maximum distances [spatial_max, temporal_max, space_time_ratio] for data selection.
    order : list of int
        Polynomial regression orders [spatial_order, temporal_order] for local mean.
    options : list
        BME options [max_integration_intervals, num_moments, percentile].
    path : str
        Directory path for saving/loading cached results.
    name : str
        Base filename for cached results.

    Returns
    -------
    np.ndarray
        Array of shape (n_estimation_points, n_moments+1) containing estimated moments
        at each estimation point. Columns include probability, mean, variance, and skewness.

    Notes
    -----
    Results are cached to disk using NumPy's .npy format. If a cached file exists,
    it will be loaded instead of recomputing.
    
    The BME approach combines prior knowledge (covariance model) with hard and soft
    data to produce optimal spatiotemporal estimates.
    """
    # Import locally to avoid circular import
    from environmentaltools.spatiotemporal import covariance
    
    if os.path.isfile(os.path.join(path, name + ".npy")):
        moments = np.load(os.path.join(path, name + ".npy"))
    else:
        # no of elements
        nk = np.shape(dfk)[0]

        moments = np.zeros([nk, int(options[1]) + 1]) * np.nan

        # Main loop starts here
        for i in range(len(dfk)):
            print(
                "Point no. "
                + str(i + 1).zfill(5)
                + " of "
                + str(len(dfk))
                + ". A "
                + str(np.round((i + 1) / float(len(dfk)) * 100, decimals=2)).zfill(5)
                + "% completed."
            )

            ck0 = pd.DataFrame([dfk.values[i]], index=[0], columns=["x", "y", "t"])
            # print(ck0)

        # Select the local neighbourhood for all variables
        chl, zhl, dh, sumnh, _ = select_neighbours(
            ck0, dfh.loc[:, ["x", "y", "t"]], dfh.loc[:, ["h"]], nmax[0], dmax
        )
        csl, zsl, ds, sumns, indexs = select_neighbours(
            ck0, dfs.loc[:, ["x", "y", "t"]], dfs.loc[:, ["h"]], nmax[1], dmax
        )
        
        # Calculate the covariance matrices
        k_kk = covariance.calculate_theoretical_covariance(covmodel, [0, 0], covparam)
        k_hh = covariance.calculate_theoretical_covariance(
            covmodel,
            [
                coordinates_to_distance(chl.loc[:, ["x", "y"]].values),
                coordinates_to_distance(chl.loc[:, "t"].values),
            ],
            covparam,
        )
        k_ss = covariance.calculate_theoretical_covariance(
            covmodel,
            [
                coordinates_to_distance(csl.loc[:, ["x", "y"]].values),
                coordinates_to_distance(csl.loc[:, "t"].values),
            ],
            covparam,
        )
        k_kh = covariance.calculate_theoretical_covariance(covmodel, [dh[0], dh[1]], covparam)
        k_ks = covariance.calculate_theoretical_covariance(covmodel, [ds[0], ds[1]], covparam)
        k_sh = covariance.calculate_theoretical_covariance(
            covmodel,
            [
                coordinates_to_distance(
                    csl.loc[:, ["x", "y"]].values, chl.loc[:, ["x", "y"]].values
                ),
                coordinates_to_distance(csl.loc[:, "t"].values, chl.loc[:, "t"].values),
            ],
            covparam,
        )

        if k_hh.size == 0:
            khs = k_ss
        else:
            khs = np.vstack((np.hstack((k_hh, k_sh)), np.hstack((k_sh.T, k_ss))))

        vsl = pd.DataFrame(dfs.iloc[indexs].s)
        mkest, mhest, msest, _ = estimate_local_mean_bme(
            ck0, chl, csl, zhl, zsl, vsl, khs, order
        )
        if sumnh > 0:  # Subtract the mean from the hard data
            zhl.h = zhl.h - mhest
        if sumns > 0:  # Subtract the mean from the soft data
            zsl.h = zsl.h - msest

        # Calculate the statistical moments
        BsIFh = np.dot(k_sh.T, np.linalg.inv(k_hh))
        KsIFh = k_ss - np.dot(BsIFh, k_sh)

        kk_hs = np.hstack((k_kh, k_ks))
        BkIFhs = np.dot(kk_hs, np.linalg.inv(khs))
        KkIFhs = k_kk - np.dot(BkIFhs, kk_hs.T)
        moments[i, 0], moments[i, 1], moments[i, 2], moments[i, 3] = calculate_moments(
            zhl.h.values,
            zsl.h.values,
            vsl.values,
            [sumnh, sumns],
            BsIFh,
            KsIFh,
            BkIFhs,
            KkIFhs,
            options,
        )

        moments[i, 1] = moments[i, 1]
        moments[i, 2] = moments[i, 2] ** 2
        np.save(os.path.join(path, name + ".npy"), moments)

    return moments


def estimate_local_mean_bme(ck, ch, cs, zh, ms, vs, khs, order):
    """
    Estimate local mean for Bayesian Maximum Entropy.

    Computes local mean estimates at estimation points and data locations using
    polynomial regression weighted by covariance structure.

    Parameters
    ----------
    ck : pd.DataFrame
        Coordinates of estimation point with columns ['x', 'y', 't'].
    ch : pd.DataFrame
        Coordinates of hard data locations with columns ['x', 'y', 't'].
    cs : pd.DataFrame
        Coordinates of soft data locations with columns ['x', 'y', 't'].
    zh : pd.DataFrame
        Values of hard data.
    ms : pd.DataFrame
        Mean values of soft data.
    vs : pd.DataFrame
        Variance values of soft data.
    khs : np.ndarray
        Combined covariance matrix for hard and soft data.
    order : list of int
        Polynomial regression orders [spatial_order, temporal_order].

    Returns
    -------
    mkest : float
        Estimated mean at the estimation point.
    mhest : np.ndarray
        Estimated means at hard data locations.
    msest : np.ndarray
        Estimated means at soft data locations.
    vkest : float
        Variance of the estimated mean at the estimation point.

    Notes
    -----
    Uses generalized least squares regression with covariance weighting to
    estimate spatiotemporal trends in the data.
    """
    nh = np.shape(zh)[0]
    c = pd.concat([ch, cs], sort=False)

    ind_diag = np.arange(nh, len(c))
    khs[ind_diag, ind_diag] = np.diag(vs)
    z = np.vstack((zh, ms))

    best, vbest, mbest = estimate_bme_regression(c, z, order, khs)
    mhest = mbest[:nh, 0]
    msest = mbest[nh:, 0]

    ck = np.vstack((ck.x.values, ck.y.values, ck.t.values)).T
    x = create_design_matrix(ck, order)

    mkest = np.dot(x, best)
    vkest = np.dot(np.dot(x, vbest), x.T)

    return mkest, mhest, msest, vkest


def calculate_moments(zh, zs, vs, nhs, BsIFh, KsIFh, BkIFhs, KkIFhs, options):
    """
    Calculate moments of the BME posterior probability density function.

    Computes statistical moments (mean, variance, skewness) of the posterior
    distribution using numerical integration over soft data intervals.

    Parameters
    ----------
    zh : np.ndarray
        Hard data values.
    zs : np.ndarray
        Soft data mean values.
    vs : np.ndarray
        Soft data variance values.
    nhs : list of int
        Number of hard and soft data points [n_hard, n_soft].
    BsIFh : np.ndarray
        BME coefficient matrix for soft data given hard data.
    KsIFh : np.ndarray
        BME covariance matrix for soft data given hard data.
    BkIFhs : np.ndarray
        BME coefficient matrix for estimation point given all data.
    KkIFhs : float
        BME covariance at estimation point given all data.
    options : list
        Integration options [max_intervals, num_moments, percentile].

    Returns
    -------
    tuple of float
        (probability, mean, std_dev, skewness) - Four moments of the posterior PDF.

    Notes
    -----
    Uses trapezoidal rule for numerical integration over soft data probability
    intervals defined by the specified percentile bounds.
    """
    nh, ns = nhs[0], nhs[1]
    maxpts, nmom, perc = options[0], options[1], options[2]

    Bs = np.ones(2)
    moments = np.zeros(4)
    if zh.size == 0:
        msIFh = zs
    else:
        msIFh = np.dot(BsIFh, zh)
    Bs[1] = np.dot(BkIFhs[:nh], zh)

    per = np.tile([1 - perc, perc], (ns, 1))
    zper = []
    for i, j in enumerate(zs):
        zper.append(st.norm.ppf(per[i], loc=j, scale=np.sqrt(vs[i])))
    zper = np.asarray(zper)

    As = np.zeros([ns, 2])
    As[:, 1] = BkIFhs[nh:]
    p = np.array([1, 1])
    val = integrate_moment_vector(msIFh, vs, KsIFh, As, Bs, p, maxpts, zper)

    moments[0] = val[0]
    moments[1] = val[1] / val[0]
    p = np.array([2, 3])
    As[:, 0] = BkIFhs[nh:]
    Bs[0] = Bs[1] - moments[1]
    val = integrate_moment_vector(msIFh, vs, KsIFh, As, Bs, p, maxpts, zper)

    moments[2] = np.sqrt(KkIFhs + val[0] / moments[0])
    moments[3] = (val[1] / moments[0]) / moments[2] ** 3

    return moments


def integrate_moment_vector(ms, vs, ks, As, Bs, p, maxpts, zp):
    """
    Compute moment integrals using the trapezoidal rule.

    Performs numerical integration for BME moment calculations over soft data
    probability intervals.

    Parameters
    ----------
    ms : np.ndarray
        Mean values of soft data.
    vs : np.ndarray
        Variance values of soft data.
    ks : np.ndarray
        Covariance matrix for soft data.
    As : np.ndarray
        Coefficient matrix for moment calculation.
    Bs : np.ndarray
        Constant terms for moment calculation.
    p : np.ndarray
        Powers for moment orders [power1, power2].
    maxpts : int
        Maximum number of integration intervals.
    zp : np.ndarray
        Percentile bounds for integration intervals.

    Returns
    -------
    np.ndarray
        Array containing integrated moment values.

    Notes
    -----
    Uses trapezoidal integration combined with multivariate normal distributions
    to evaluate moments of the posterior distribution.
    """
    maxpts = int(maxpts)
    xs = np.asarray([np.linspace(zp[i, 0], zp[i, 1], maxpts) for i in range(len(zp))])

    # def is_pos_def(x):
    #     return np.all(np.linalg.eigvals(x) > 0)

    # if is_pos_def(ks):
    stn = st.multivariate_normal(cov=ks)
    # else:
    #     stn = st.multivariate_normal(cov=ks*0+np.diag(ks))
    nms_c = stn.pdf(xs.T)
    Bs = np.tile(Bs, (maxpts, 1)).T
    if np.all(nms_c == 0):
        nms_c = 1
        if p[0]:
            Bs[:, 1] = Bs[:, 1] * 0
        elif p[0] == 2:
            Bs = np.zeros(np.shape(Bs))

    stn = st.norm(loc=ms, scale=np.sqrt(np.ravel(vs)))
    ns = stn.pdf(xs.T)

    p = np.tile(p, (maxpts, 1)).T
    dxs = np.abs(xs[:, 1] - xs[:, 0])
    I = np.zeros([2, maxpts])
    for i in range(len(zp)):
        I += (
            (np.dot(As.T, xs) + Bs) ** p
            * np.tile(ns[:, i], (2, 1))
            * np.tile(nms_c, (2, 1))
            * dxs[i]
        )
    results = np.sum(I, axis=1)

    return results


def apply_data_smoothing(dfh, dfs, dfk, nmax, dmax, path):
    """
    Apply spatial smoothing to hard and soft data.

    Performs kernel smoothing on spatiotemporal data to reduce noise and normalize
    values for BME analysis.

    Parameters
    ----------
    dfh : pd.DataFrame
        Hard data with columns ['x', 'y', 't', 'h'].
    dfs : pd.DataFrame
        Soft data with columns ['x', 'y', 't', 'h'].
    dfk : pd.DataFrame
        Estimation points with columns ['x', 'y', 't'].
    nmax : list of int
        Maximum number of neighbors [hard_max, soft_max].
    dmax : list of float
        Maximum distances [spatial_max, temporal_max, space_time_ratio].
    path : str
        Directory path for caching smoothed results.

    Returns
    -------
    zh : np.ndarray
        Smoothed hard data values.
    zs : np.ndarray
        Smoothed soft data values.
    zk : np.ndarray
        Smoothed values at estimation points.
    dfh : pd.DataFrame
        Normalized hard data.
    dfs : pd.DataFrame
        Normalized soft data.

    Notes
    -----
    Results are cached to disk. Smoothing uses exponential distance weighting.
    Data is standardized by subtracting smooth trend and dividing by standard deviation.
    """

    if os.path.isfile(path + "/Smooth_h.npy"):
        zh = np.load(path + "/Smooth_h.npy")
        zs = np.load(path + "/Smooth_s.npy")
        zk = np.load(path + "/Smooth_k.npy")
    else:
        zh = smooth_data(
            dfh.loc[:, ["x", "y", "t"]],
            dfh.loc[:, ["x", "y", "t"]],
            dfh.loc[:, ["h"]],
            0.2,
            nmax[0],
            dmax,
        )
        np.save(path + "/Smooth_h.npy", zh)
        zs = smooth_data(
            dfs.loc[:, ["x", "y", "t"]],
            dfs.loc[:, ["x", "y", "t"]],
            dfs.loc[:, ["h"]],
            0.2,
            nmax[0],
            dmax,
        )
        np.save(path + "/Smooth_s.npy", zs)

        dfhs = pd.concat([dfh, dfs], sort=False)
        dfhs.reset_index(drop=True, inplace=True)

        zk = smooth_data(
            dfk.loc[:, ["x", "y", "t"]],
            dfhs.loc[:, ["x", "y", "t"]],
            dfhs.loc[:, ["h"]],
            0.2,
            nmax[0],
            dmax,
        )
        np.save(path + "/Smooth_k.npy", zk)

    dfh.h = (dfh.h - zh) / (dfh.h - zh).std()
    dfs.h = (dfs.h - zs) / (dfs.h - zs).std()
    dfhs = pd.concat([dfh, dfs], sort=False)
    dfhs.reset_index(drop=True, inplace=True)

    return zh, zs, zk, dfh, dfs


def create_spatiotemporal_matrix(dfh, dfs):
    """
    Create spatiotemporal data matrix.

    Organizes hard and soft data into a matrix format where columns represent
    unique spatial locations and rows represent time points.

    Parameters
    ----------
    dfh : pd.DataFrame
        Hard data with columns ['x', 'y', 't', 'h'].
    dfs : pd.DataFrame
        Soft data with columns ['x', 'y', 't', 'h'].

    Returns
    -------
    dfz : pd.DataFrame
        Matrix of values where index is time and columns are spatial locations.
    coups : np.ndarray
        Array of unique spatial coordinate pairs [x, y].
    coupt : np.ndarray
        Array of unique time points.

    Notes
    -----
    Missing data are represented as NaN values in the resulting matrix.
    """

    cPt = dfh.append(dfs)
    cPt.sort_index(sort=False, inplace=True)
    coupt = cPt.t.unique()
    coups = np.unique((cPt.x, cPt.y), axis=1).T
    dfz = pd.DataFrame(np.nan, index=coupt, columns=(np.arange(len(coups))))

    for i, j in enumerate(coups[:, 0]):
        for t in coupt:
            dfz.loc[t, i] = cPt[
                ((cPt.t == t) & (cPt.x == j) & (cPt.y == coups[i, 1]))
            ].h.values[0]

    return dfz, coups, coupt


def coordinates_to_distance(pi, *args):
    """
    Compute distances between coordinate points.

    Calculates pairwise distances for spatial (2D) or temporal (1D) coordinates,
    or distances from points to a reference location.

    Parameters
    ----------
    pi : np.ndarray
        Coordinate array. Shape (n,) for temporal or (n, 2) for spatial coordinates.
    *args : tuple, optional
        Reference coordinates for computing distances to a single point.
        If provided, computes distances from all points in `pi` to this reference.

    Returns
    -------
    dist : np.ndarray or pd.DataFrame
        Distance matrix or array. If no args provided, returns symmetric distance
        matrix between all points. If args provided, returns distances to reference.

    Notes
    -----
    - For spatial data (2D): Euclidean distance
    - For temporal data (1D): Absolute difference
    - Distance matrices are symmetric with zeros on diagonal
    """
    if args:
        ncoor = len(np.shape(pi))
        pi0 = args[0]
        nl = np.shape(pi0)[0]
        if nl == 1:
            if ncoor == 1:
                dist = np.abs(pi - pi0)
            else:
                dist = np.sqrt(
                    (pi[:, 0] - pi0[0, 0]) ** 2 + (pi[:, 1] - pi0[0, 1]) ** 2
                )
        else:
            dist = []
            if ncoor == 1:
                for m in range(nl):
                    dist.append(np.abs(pi - pi0[m]))
            else:
                for m in range(nl):
                    dist.append(
                        np.sqrt(
                            (pi[:, 0] - pi0[m, 0]) ** 2 + (pi[:, 1] - pi0[m, 1]) ** 2
                        )
                    )
    else:
        ncoor = len(np.shape(pi))
        df = [[] for ii in range(ncoor)]
        if ncoor == 1:
            pi = pi[:, np.newaxis]

        for m in range(ncoor):
            df[m] = pd.DataFrame(
                0, index=np.arange(len(pi)), columns=np.arange(len(pi))
            )
            dx = []
            for a, b in it.combinations_with_replacement(pi[:, m], 2):
                dx.append(a - b)
            dx = np.array(dx)

            k, l = len(pi), 0
            for i, j in enumerate(df[m].index):
                df[m].loc[j, i:] = dx[l : l + k]
                df[m].loc[i:, j] = dx[l : l + k]
                l += k
                k -= 1

        if ncoor == 1:
            dist = pd.DataFrame(np.abs(df[0]))
        elif ncoor == 2:
            dist = pd.DataFrame(np.sqrt(df[0] ** 2.0 + df[1] ** 2.0))
        else:
            dist = 0

    return dist


def coordinates_to_distance_angle(pi):
    """
    Compute distances and angles between spatial coordinate points.

    Calculates pairwise Euclidean distances and directional angles for 2D
    spatial coordinates.

    Parameters
    ----------
    pi : np.ndarray
        Spatial coordinate array of shape (n, 2) with columns [x, y].

    Returns
    -------
    dist : pd.DataFrame
        Symmetric matrix of Euclidean distances between all point pairs.
    ang : pd.DataFrame
        Matrix of angles in degrees (0-180) representing direction from
        each point to every other point.

    Notes
    -----
    Angles are computed using complex number representation and converted
    to degrees in the range [0, 180).
    """
    df = [[] for ii in range(2)]

    for m in range(2):
        df[m] = pd.DataFrame(0, index=np.arange(len(pi)), columns=np.arange(len(pi)))
        dx = []
        for a, b in it.combinations_with_replacement(pi[:, m], 2):
            dx.append(a - b)
        dx = np.array(dx)

        k, l = len(pi), 0
        for i, j in enumerate(df[m].index):
            df[m].loc[j, i:] = dx[l : l + k]
            df[m].loc[i:, j] = dx[l : l + k]
            l += k
            k -= 1

    dist = pd.DataFrame(np.sqrt(df[0] ** 2.0 + df[1] ** 2.0))
    ang = pd.DataFrame(np.angle(df[0] + 1j * df[1], deg=True))
    ang[ang < 0] += 180
    return dist, ang


def find_pairs_by_distance(pi, plag, plagtol, *args):
    """
    Find pairs of points separated by specified distance intervals.

    Identifies point pairs whose separation falls within given distance lag
    tolerances, optionally considering directional angles for spatial data.

    Parameters
    ----------
    pi : np.ndarray
        Coordinate array (1D for temporal, 2D for spatial).
    plag : array-like
        Array of target distance lags.
    plagtol : array-like
        Array of distance tolerance values for each lag.
    *args : tuple, optional
        For directional analysis: [dlag, dlagtol] where dlag is array of
        target angles and dlagtol is angular tolerance.

    Returns
    -------
    idxpairs : list
        Nested list of index tuples identifying point pairs meeting distance
        criteria. Structure depends on whether directional analysis is included.

    Notes
    -----
    Without directional args: Returns list of index arrays for each distance lag.
    With directional args: Returns 2D list organized by [distance_lag][angle_bin].
    """
    if not args:
        nr = len(plag)
        dist = coordinates_to_distance(pi)

        idxpairs = [[] for i in range(0, nr)]
        for ir in range(0, nr):
            idxpairs[ir] = np.where(
                (plag[ir] - plagtol[ir] <= dist) & (dist <= plag[ir] + plagtol[ir])
            )
    else:
        dlag, dlagtol = args[0][0], args[0][1]
        nr, nd = len(plag), len(dlag)

        dist, ang = coordinates_to_distance_angle(pi)
        idxpairs = [[[] for j in range(nd)] for i in range(nr)]
        db = 2 * plag[1] * np.sin(dlagtol * np.pi / 360)
        daI2 = np.arctan(db / (2 * dist)) * 180 / np.pi
        for i in range(nd):
            idxpairs[0][i] = np.where(
                (plag[0] - plagtol[0] <= dist) & (dist <= plag[0] + plagtol[0])
            )
            idxpairs[1][i] = np.where(
                (plag[1] - plagtol[1] <= dist)
                & (dist <= plag[1] + plagtol[1])
                & (dlag[1] - dlagtol / 2 <= ang)
                & (ang <= dlag[1] + dlagtol / 2)
            )

        for id in range(nd):
            for ir in range(2, nr):
                idxpairs[ir][id] = np.where(
                    (plag[ir] - plagtol[ir] <= dist)
                    & (dist <= plag[ir] + plagtol[ir])
                    & (dlag[id] - daI2 <= ang)
                    & (ang <= dlag[id] + daI2)
                )

    return idxpairs


def select_neighbours(c0, c, z, nmax, dmax):
    """
    Select neighboring data points for local BME estimation.

    Finds data points within specified spatiotemporal distance constraints,
    limiting to a maximum number of nearest neighbors.

    Parameters
    ----------
    c0 : pd.DataFrame
        Estimation point coordinates with columns ['x', 'y', 't'].
    c : pd.DataFrame
        Data point coordinates with columns ['x', 'y', 't'].
    z : pd.DataFrame
        Data values at coordinates `c`.
    nmax : int
        Maximum number of neighbors to select.
    dmax : list of float
        Maximum distances [spatial_max, temporal_max, space_time_ratio].

    Returns
    -------
    csub : pd.DataFrame
        Coordinates of selected neighbor points.
    zsub : pd.DataFrame
        Values at selected neighbor points.
    dsub : list of np.ndarray
        Distances [spatial, temporal, combined] for selected neighbors.
    nsub : int
        Number of selected neighbors.
    index : np.ndarray
        Indices of selected neighbors in original data.

    Notes
    -----
    Combined distance uses weighted sum: ds + (space_time_ratio * dt).
    If more than nmax neighbors meet distance criteria, selects nearest ones.
    """
    # Computing distances
    ds = coordinates_to_distance(c.loc[:, ["x", "y"]].values, c0.loc[:, ["x", "y"]].values)
    dt = coordinates_to_distance(c.loc[:, "t"].values, c0.loc[:, "t"].values)
    index = np.where((ds <= dmax[0]) & (dt <= dmax[1]))[0]

    # Check if more data meet the conditions than allowed; select nearest
    nsub = len(index)
    dp = ds + dmax[2] * dt
    if nsub > nmax:
        di = dp[index]
        dis, indexi = np.sort(di), np.argsort(di)
        indexi = indexi[:nmax]
        nsub = nmax
        index = index[indexi]

    dsub = [ds[index], dt[index], dp[index]]

    if np.shape(z)[1] == 1:
        zsub = z.iloc[index]
    else:
        zsub = z.iloc[index, :]

    csub = c.iloc[index, :]

    return csub, pd.DataFrame(zsub), dsub, nsub, index


def estimate_bme_regression(c, z, order, k):
    """
    Compute parameter estimates for linear regression with covariance weighting.

    Performs generalized least squares regression incorporating spatiotemporal
    covariance structure.

    Parameters
    ----------
    c : pd.DataFrame
        Spatiotemporal coordinates with columns ['x', 'y', 't'].
    z : np.ndarray or pd.DataFrame
        Data values at coordinates.
    order : list of int
        Polynomial orders [spatial_order, temporal_order].
    k : np.ndarray
        Covariance matrix for the data.

    Returns
    -------
    best : np.ndarray
        Estimated regression coefficients.
    vbest : np.ndarray
        Covariance matrix of coefficient estimates.
    zest : np.ndarray
        Regression-fitted values at input coordinates.

    Notes
    -----
    Uses the design matrix approach with polynomial terms up to specified orders.
    Accounts for spatiotemporal correlation through the covariance matrix.
    """
    c = np.vstack((c.x.values, c.y.values, c.t.values)).T
    x = create_design_matrix(c, order)

    xtinvk = np.dot(x.T, np.linalg.inv(k))
    vbest = np.linalg.inv(np.dot(xtinvk, x))
    best = np.dot(np.dot(vbest, xtinvk), z)
    zest = np.dot(x, best)

    return best, vbest, zest


def create_design_matrix(c, order):
    """
    Create design matrix for polynomial regression.

    Constructs matrix of polynomial terms for spatiotemporal regression,
    with separate spatial (x, y) and temporal (t) terms.

    Parameters
    ----------
    c : np.ndarray
        Coordinate matrix of shape (n, 3) with columns [x, y, t].
    order : list of int
        Polynomial orders [spatial_order, temporal_order].

    Returns
    -------
    x : np.ndarray
        Design matrix of shape (n, n_terms) where n_terms = 1 + 2*spatial_order + temporal_order.
        Column 0 is constant term, followed by spatial terms (x, y) up to specified order,
        then temporal terms up to specified order.

    Notes
    -----
    Spatial terms alternate between x and y coordinates: [1, x, y, x², y², x³, y³, ...].
    Temporal terms follow: [t, t², t³, ...].
    """
    n, nd = np.shape(c)
    x = np.ones([n, 1 + order[0] * 2 + order[1]])

    for i in range(order[0]):
        x[:, 1 + i * 2 : i * 2 + 3] = c[:, 0:2] ** (i + 1)

    for i in range(order[1]):
        x[:, 1 + order[0] * 2 + i] = c[:, 2] ** (i + 1)

    return x


def smooth_data(ck, chs, zhs, v, nmax, dmax):
    """
    Apply spatial smoothing to data using exponential distance weighting.

    Performs kernel smoothing on spatiotemporal data to reduce noise.

    Parameters
    ----------
    ck : pd.DataFrame
        Estimation point coordinates with columns ['x', 'y', 't'].
    chs : pd.DataFrame
        Data point coordinates with columns ['x', 'y', 't'].
    zhs : pd.DataFrame
        Data values.
    v : float
        Bandwidth parameter for exponential kernel.
    nmax : int
        Maximum number of neighbors to use.
    dmax : list of float
        Maximum distances [spatial_max, temporal_max, space_time_ratio].

    Returns
    -------
    zk : np.ndarray
        Smoothed values at estimation points.

    Notes
    -----
    Uses exponential kernel: weight = exp(-distance / v).
    Weights are normalized to sum to 1 for each estimation point.
    """

    zk = np.zeros(len(ck))
    for i in range(len(ck)):
        print(
            "Point no. "
            + str(i + 1).zfill(5)
            + " of "
            + str(len(ck))
            + ". A "
            + str(np.round((i + 1) / float(len(ck)) * 100, decimals=2)).zfill(5)
            + "% completed."
        )
        ck0 = ck[ck.index == i]
        chl, zhl, dhl, sumnh, _ = select_neighbours(ck0, chs, zhs, nmax, dmax)
        if sumnh > 0:
            lam = np.exp(-dhl[2] / v)
            lam = lam / np.sum(lam)
            zk[i] = np.dot(lam.T, zhl)
    return zk


def perform_cross_validation(
    dfh, dfs, zh, covmodel, covparam, nmax, dmax, order, option, path, name, k
):
    """
    Perform k-fold cross-validation for BME model evaluation.

    Assesses BME model performance by splitting hard data into training and
    validation sets.

    Parameters
    ----------
    dfh : pd.DataFrame
        Hard data with columns ['x', 'y', 't', 'h'].
    dfs : pd.DataFrame
        Soft data with columns ['x', 'y', 't', 'h', 's'].
    zh : np.ndarray
        Smoothed hard data values.
    covmodel : str
        Covariance model type.
    covparam : array-like
        Covariance model parameters.
    nmax : list of int
        Maximum neighbors [hard_max, soft_max].
    dmax : list of float
        Maximum distances [spatial_max, temporal_max, space_time_ratio].
    order : list of int
        Regression orders [spatial_order, temporal_order].
    option : list
        BME options [max_intervals, num_moments, percentile].
    path : str
        Directory for caching results.
    name : str
        Base filename for results.
    k : int
        Number of cross-validation folds.

    Returns
    -------
    e_mda : list
        Mean absolute errors for each fold.
    e_mse : list
        Root mean squared errors for each fold.

    Notes
    -----
    Results are cached to disk. Uses random subsampling without replacement
    for each fold.
    """
    if os.path.isfile(os.path.join(path, "e_mda.npy")):
        e_mda = np.load(os.path.join(path, "e_mda.npy"))
        e_mse = np.load(os.path.join(path, "e_mse.npy"))
    else:
        lh = len(dfh)
        npoints = int(lh / k)
        e_mda, e_mse = [], []
        for i in range(k):
            indx = np.arange(lh)
            ind = np.random.choice(lh, npoints, replace=False)
            dfk = dfh.loc[ind, ["x", "y", "t"]]
            indx = [ii for ii in indx if ii not in ind]
            dfhm = dfh.loc[indx]
            namei = name + str(k) + "_" + str(i)
            moments = compute_bme_moments(
                dfk,
                dfhm,
                dfs,
                covmodel,
                covparam,
                nmax,
                dmax,
                order,
                option,
                path,
                namei,
            )
            moments[:, 1] = moments[:, 1] * moments[:, 2] + zh[ind]
            e_mda.append(np.sum(np.abs(zh[ind] - moments[:, 1])) / npoints)
            e_mse.append(np.sqrt(np.sum((zh[ind] - moments[:, 1]) ** 2) / npoints))
        np.save(os.path.join(path, "e_mda.npy"), e_mda)
        np.save(os.path.join(path, "e_mse.npy"), e_mse)
    return e_mda, e_mse

