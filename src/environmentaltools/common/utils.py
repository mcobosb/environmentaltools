from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import Rbf
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from environmentaltools.common import save
from environmentaltools.temporal import analysis
from environmentaltools.temporal import core


def max_moving_window(data: pd.DataFrame, duration: int) -> pd.DataFrame:
    """Select peaks from time series using a moving window.

    Identifies peak values that occur at the center of a moving window,
    effectively filtering out values that are not local maxima within
    the specified duration.

    Args:
        data (pd.DataFrame): Time series data.
        duration (int): Duration of the moving window (in index units).

    Returns:
        pd.DataFrame: DataFrame containing only the detected peaks with
            their original timestamps.
    """

    n = len(data)
    id_ = []  # List to store indices of detected peaks
    # Slide a window of length 'duration' across the data
    for k in range(0, n - duration):
        # Find the index of the maximum value within the current window
        idx = data.iloc[k : k + duration + 1].idxmax()
        # Only select the peak if it is at the center of the window
        if idx == data.index[k + int(duration / 2)]:
            id_.append(idx)

    if id_ is []:
        for k in range(0, n - duration):
            # Find the index of the minimum value within the current window
            idx = data.iloc[k : k + duration + 1].idxmin()
            # Only select the peak if it is at the center of the window
            if idx == data.index[k + int(duration / 2)]:
                id_.append(idx)
    
    # Create a DataFrame with the selected peaks
    results = pd.DataFrame(data.loc[id_], index=id_)
    return results


def gaps(data: pd.DataFrame, variables: str | list, file_name: str = "gaps", buoy: bool = False):
    """Create summary table of data gaps for time series variables.

    Analyzes time series to identify gaps, sampling frequency, and data quality
    metrics. Saves results to Excel file.

    Args:
        data (pd.DataFrame): Time series data with datetime index.
        variables (str or list): Variable name(s) to analyze for gaps.
        file_name (str): Output filename (without extension) for the gap summary
            table. Defaults to "gaps".
        buoy (bool): If True, includes quality control metrics assuming 'Qc_e'
            column exists. Defaults to False.

    Returns:
        pd.DataFrame: Summary table with columns including cadency, accuracy,
            period, number of years, gap percentage, median gap, and maximum gap.
    """

    if not isinstance(variables, list):
        variables = [variables]  # Ensure variables is a list

    # Define columns for the output table depending on whether it's buoy data
    if not buoy:
        columns_ = [
            "Cadency (min)",
            "Accuracy*",
            "Period",
            "No. years",
            "Gaps (%)",
            "Med. gap (hr)",
            "Max. gap (d)",
        ]
    else:
        columns_ = [
            "Cadency (min)",
            "Accuracy*",
            "Period",
            "No. years",
            "Gaps (%)",
            "Med. gap (hr)",
            "Max. gap (d)",
            "Quality data (%)",
        ]

    # Initialize the output DataFrame
    tbl_gaps = pd.DataFrame(
        0,
        columns=columns_,
        index=variables,
    )
    tbl_gaps.index.name = "var"

    for i in variables:
        dt_nan = data[i].dropna()  # Remove NaNs for the variable
        if buoy:
            # Count good quality data points if buoy
            quality = np.sum(data.loc[dt_nan.index, "Qc_e"] <= 2)

        # Calculate time differences (in hours) between consecutive non-NaN samples
        dt0 = (dt_nan.index[1:] - dt_nan.index[:-1]).total_seconds() / 3600
        # Identify gaps as intervals significantly larger than the median
        dt = dt0[dt0 > np.median(dt0) + 0.1].values
        if dt.size == 0:
            dt = 0  # No gaps found
        # Calculate the most common sampling interval (accuracy)
        acc = st.mode(np.diff(dt_nan.sort_values().unique()))[0]

        # Fill the summary table for this variable
        tbl_gaps.loc[i, "Cadency (min)"] = np.round(st.mode(dt0)[0]*60, decimals=2)
        tbl_gaps.loc[i, "Accuracy*"] = np.round(acc, decimals=2)
        tbl_gaps.loc[i, "Period"] = str(dt_nan.index[0]) + "-" + str(dt_nan.index[-1])
        tbl_gaps.loc[i, "No. years"] = dt_nan.index[-1].year - dt_nan.index[0].year
        tbl_gaps.loc[i, "Gaps (%)"] = np.round(
            np.sum(dt) / data[i].shape[0] * 100, decimals=2
        )
        tbl_gaps.loc[i, "Med. gap (hr)"] = np.round(np.median(dt), decimals=2)
        tbl_gaps.loc[i, "Max. gap (d)"] = np.round(np.max(dt)/24, decimals=2)

        if buoy:
            # Add percentage of good quality data for buoys
            tbl_gaps.loc[i, "Quality data (%)"] = np.round(
                quality / len(dt_nan) * 100, decimals=2
            )

    # Save the table to Excel file
    save.to_xlsx(tbl_gaps, file_name)

    return tbl_gaps


def nonstationary_ecdf(
    data: pd.DataFrame,
    variable: str,
    wlen: float = 14 / 365.25,
    equal_windows: bool = False,
    pemp: list = None,
):

    """Computes empirical percentiles using a moving window.

    Args:
        data (pd.DataFrame): Time series.
        variable (str): Name of the variable.
        wlen (float): Length of window in years (default 14 days).
        equal_windows (bool): If True, use equal window size.
        pemp (list, optional): Empirical percentiles to use.

    Returns:
        pd.DataFrame: Values of the given non-stationary percentiles.
        list: Chosen empirical percentiles.
    """

    timestep = 1 / 365.25  # Default timestep: daily in years
    if equal_windows:
        timestep = wlen  # Use window length as timestep if requested

    # Choose default percentiles if not provided, with special cases for some variables
    if pemp is None:
        pemp = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        if variable.lower().startswith("d") | (variable == "Wd"):
            pemp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        if (variable == "Hs") | (variable == "Hm0"):
            pemp = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995])

    # Create result DataFrame: rows are unique normalized times, columns are percentiles
    res = pd.DataFrame(0, index=data.n.unique(), columns=pemp)

    # For each time index, compute percentiles in a moving window
    for i in res.index:
        if i >= (1 - wlen):
            # Handle window at the end of the time series (wrap-around)
            final_offset = i + wlen - 1
            mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
                data["n"] <= final_offset
            )
        elif i <= wlen:
            # Handle window at the start of the time series (wrap-around)
            initial_offset = i - wlen
            mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
                data["n"] >= 1 + initial_offset
            )
        else:
            # Standard window in the middle
            mask = (data["n"] >= i - wlen) & (data["n"] <= i + wlen)
        # Compute percentiles for the window
        res.loc[i, pemp] = data[variable].loc[mask].quantile(q=pemp).values

    return res, pemp


def best_params(data: pd.DataFrame, bins: int, distrib: str, tail: bool = False):
    """Computes the best parameters of a simple probability model based on the RMSE of the PDF.

    Args:
        data (pd.DataFrame): Raw time series.
        bins (int): Number of bins for the histogram.
        distrib (str): Name of the probability model.
        tail (bool, optional): If True, fit only the tail. Defaults to False.

    Returns:
        list: The estimated parameters.
    """

    dif_, sser = 1e2, 1e3
    nlen = int(len(data) / 200)

    data = data.sort_values(ascending=True).values
    while (dif_ > 1) & (sser > 30) & (0.95 * nlen < len(data)):
        # Fit the distribution using the statistical_fit module
        results = fit_(data, bins, distrib)
        sse, params = results[0], results[1:]
        dif_, sser = np.abs(sser - sse), sse

        if tail:
            data = data[int(nlen / 4) :]
        else:
            data = data[nlen:-nlen]
    return params



def fit_(data: pd.DataFrame, bins: int, model: str):
    """Fits a simple probability model and computes the sse with the empirical pdf

    Args:
        * data (pd.DataFrame): raw time series
        * bins (int): no. of bins for the histogram
        * model (string): name of the probability model

    Returns:
        * results (np.array): the parameters computed
    """

    y, x = np.histogram(data, bins=bins, density=True)
    xq = np.diff(x)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    yq = np.cumsum(xq * y)

    if model is st.genpareto:
        params = model.fit(data, 0.01, loc=np.min(data))
    else:
        params = model.fit(data)
    cdf = model.cdf(x, loc=params[-2], scale=params[-1], *params[:-2])
    sse = np.sum((yq - cdf) ** 2)
    if np.isnan(sse):
        sse = 1e10

    results = np.zeros(len(params) + 1)
    results[: int(len(params) + 1)] = np.append(sse, params)
    return results


def ecdf(df: pd.DataFrame, variable: str, num_percentiles: int | bool = False) -> pd.DataFrame:
    """Compute the empirical cumulative distribution function (ECDF).

    Calculates non-exceedance probabilities for the variable values. Can
    optionally interpolate to a specified number of percentiles.

    Args:
        df (pd.DataFrame): Raw time series data.
        variable (str): Name of the variable column to analyze.
        num_percentiles (int | bool, optional): Number of empirical percentiles
            to interpolate. If False, returns all data points. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with variable values and their non-exceedance
            probabilities. Index represents probability values.
    """
    dfs = df[variable].sort_values().to_frame()
    dfs.index = np.arange(1, len(dfs) + 1) / (len(dfs) + 1)
    if not isinstance(num_percentiles, bool):
        percentiles = np.linspace(1 / num_percentiles, 1 - (1 / num_percentiles), num_percentiles)
        values = np.interp(percentiles, dfs.index, dfs[variable])
        dfs = pd.DataFrame(values, columns=[variable], index=percentiles)
    return dfs


def nonstationary_epdf(
    data: pd.DataFrame, variable: str, wlen: float = 14 / 365.25, no_values: int = 14
):

    """Computes the empirical PDF using a moving window.

    Args:
        data (pd.DataFrame): Time series.
        variable (str): Name of the variable.
        wlen (float): Length of window in years (default 14 days).
        no_values (int): Number of values for the PDF.

    Returns:
        pd.DataFrame: Values of the non-stationary PDF.
    """

    nlen = len(data)
    ndates = np.arange(0, 1, 1 / 365.25)
    values_ = np.linspace(data[variable].min(), data[variable].max(), no_values)
    pdf_ = pd.DataFrame(-1, index=ndates, columns=(values_[:-1] + values_[1:]) / 2)

    columns = pdf_.columns
    for ind_, col_ in enumerate(columns[:-1]):
        for i in pdf_.index:
            if i > (1 - wlen):
                final_offset = i + wlen - 1
                mask = (
                    (data["n"] > i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                ) | (data["n"] <= final_offset)
            elif i < wlen:
                initial_offset = i - wlen
                mask = (
                    (data["n"] >= i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                ) | (data["n"] >= 1 + initial_offset)
            else:
                mask = (
                    (data["n"] >= i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                )
            pdf_.loc[i, col_] = np.sum(mask) / nlen

    return pdf_


def epdf(df: pd.DataFrame, variable: str, num_bins: int = 14) -> pd.DataFrame:
    """Compute the empirical probability distribution function (PDF).

    Creates a histogram-based empirical PDF by binning the variable values
    and calculating probability densities for each bin.

    Args:
        df (pd.DataFrame): Raw time series data.
        variable (str): Name of the variable column to analyze.
        num_bins (int, optional): Number of bins for the histogram. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with bin centers as index and 'prob' column
            containing probability densities.
    """
    dfs = df[variable].sort_values().to_frame()
    dfs["prob"] = np.arange(1, len(dfs) + 1)
    count_ = pd.DataFrame(-1, index=dfs[variable].unique(), columns=["prob"])

    for _, ind_ in enumerate(count_.index):
        count_.loc[ind_] = np.sum(dfs[variable] == ind_)

    values_ = np.linspace(df[variable].min(), df[variable].max(), num_bins)
    pdf_ = pd.DataFrame(-1, index=(values_[:-1] + values_[1:]) / 2, columns=["prob"])
    for ind_, index_ in enumerate(pdf_.index):
        # range_ = np.interp(values_, pdf_["prob"], pdf_[variable])
        val_ = np.sum(
            count_.loc[
                ((count_.index < values_[ind_ + 1]) & (count_.index > values_[ind_]))
            ].values
        )
        pdf_.loc[index_, "prob"] = val_
    pdf_.loc[:, "prob"] = pdf_["prob"] / (np.sum(pdf_).values * np.diff(values_))
    return pdf_


def acorr(data: np.ndarray | pd.Series, max_lags: int = 24):
    """Compute autocorrelation function of a time series.

    Calculates the normalized autocorrelation for a range of lags using
    matplotlib's autocorrelation function.

    Args:
        data (np.ndarray or pd.Series): Input time series data.
        max_lags (int): Maximum number of lags to compute. Defaults to 24.

    Returns:
        tuple: (lags, autocorrelation) - Arrays of lag values and corresponding
            autocorrelation coefficients.
    """
    lags, c_, _, _ = plt.acorr(data, usevlines=False, maxlags=max_lags, normed=True)
    plt.close()
    # lags, c_ = lags[-maxlags:], c_[-maxlags:]
    return lags, c_


def bidimensional_ecdf(data1: np.ndarray, data2: np.ndarray, num_bins: int):
    """Compute empirical 2D cumulative distribution function (ECDF).

    Calculates the joint empirical CDF for two variables using a 2D histogram
    approach with cumulative summation.

    Args:
        data1 (np.ndarray): Values of the first variable.
        data2 (np.ndarray): Values of the second variable.
        num_bins (int): Number of bins for the 2D histogram in each dimension.

    Returns:
        tuple: (x_mesh, y_mesh, ecdf_values) - 2D meshgrids of bin centers and
            corresponding cumulative probability values.
    """
    f, xedges, yedges = np.histogram2d(data1, data2, bins=num_bins)
    Fe = np.cumsum(np.cumsum(f, axis=0), axis=1) / (np.sum(f) + 1)
    Fe = np.flipud(np.rot90(Fe))

    xmid, ymid = (xedges[0:-1] + xedges[1:]) / 2, (yedges[0:-1] + yedges[1:]) / 2
    xe, ye = np.meshgrid(xmid, ymid)

    return xe, ye, Fe




def bias_adjustment(
    obs,
    hist,
    rcp,
    variable,
    funcs=["gumbel_l", "gumbel_r"],
    quantiles=[0.1, 0.9],
    params=None,
):
    """
    Bias adjustment for climate data using parametric quantile mapping.

    Args:
        obs (pd.DataFrame): Observed data.
        hist (pd.DataFrame): Historical simulation data.
        rcp (pd.DataFrame): Scenario/projection data.
        variable (str): Variable name to adjust.
        funcs (list, optional): List of distribution names. Defaults to ["gumbel_l", "gumbel_r"].
        quantiles (list, optional): Quantiles for tail adjustment. Defaults to [0.1, 0.9].
        params (dict, optional): Precomputed distribution parameters. Defaults to None.

    Returns:
        tuple: (hist, rcp) with bias-adjusted values in column 'unbiased'.
    """
    funcs = funcs.copy()
    for index, fun in enumerate(funcs):
        funcs[index] = getattr(st, fun)

    hist["unbiased"] = 0
    rcp["unbiased"] = 0

    # Low tail
    low_tail_hist = hist.loc[
        hist[variable] <= hist[variable].quantile(quantiles[0]), variable
    ]

    if isinstance(params, dict):
        if "obs_low" in params:
            params_obs_low = params["obs_low"]
    else:
        params_obs_low = funcs[0].fit(
            obs.loc[obs[variable] <= obs[variable].quantile(quantiles[0]), variable]
        )

    if isinstance(params, dict):
        if "hist_low" in params:
            params_hist_low = params["hist_low"]
    else:
        params_hist_low = funcs[0].fit(low_tail_hist)

    hist.loc[hist[variable] <= hist[variable].quantile(quantiles[0]), "unbiased"] = (
        funcs[0].ppf(funcs[0].cdf(low_tail_hist, *params_hist_low), *params_obs_low)
    )
    low_tail_rcp = rcp.loc[
        rcp[variable] <= rcp[variable].quantile(quantiles[0]), variable
    ]
    rcp.loc[rcp[variable] <= rcp[variable].quantile(quantiles[0]), "unbiased"] = funcs[
        0
    ].ppf(funcs[0].cdf(low_tail_rcp, *params_hist_low), *params_obs_low)

    # High tail
    high_tail_hist = hist.loc[
        hist[variable] >= hist[variable].quantile(quantiles[1]), variable
    ]

    if isinstance(params, dict):
        if "obs_high" in params:
            params_obs_high = params["obs_high"]
    else:
        params_obs_high = funcs[1].fit(
            obs.loc[obs[variable] >= obs[variable].quantile(quantiles[1]), variable]
        )

    if isinstance(params, dict):
        if "hist_high" in params:
            params_hist_high = params["hist_high"]
    else:
        params_hist_high = funcs[1].fit(high_tail_hist)

    hist.loc[hist[variable] >= hist[variable].quantile(quantiles[1]), "unbiased"] = (
        funcs[1].ppf(funcs[1].cdf(high_tail_hist, *params_hist_high), *params_obs_high)
    )
    high_tail_rcp = rcp.loc[
        rcp[variable] >= rcp[variable].quantile(quantiles[1]), variable
    ]
    rcp.loc[rcp[variable] >= rcp[variable].quantile(quantiles[1]), "unbiased"] = funcs[
        1
    ].ppf(funcs[1].cdf(high_tail_rcp, *params_hist_high), *params_obs_high)

    # Body
    n_obs = len(obs)
    obs_sort = np.sort(obs[variable])

    # body_obs_quantile = np.linspace(quantiles[0], quantiles[1], 20)

    # cdf_obs = ecdf(obs, variable)
    # body_obs_values = obs[variable].quantile(body_obs_quantile)
    cdf_hist = ecdf(hist, variable)
    # body_hist_values = hist[variable].quantile(body_obs_quantile)

    values_body_hist = hist.loc[
        (hist[variable] > hist[variable].quantile(quantiles[0]))
        & (hist[variable] < hist[variable].quantile(quantiles[1])),
        variable,
    ]
    hist_adj_list = list()
    Fe_Fhist_hist = np.interp(values_body_hist, cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Fhist_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        hist_adj_list.append(obs_sort[pos])
    hist.loc[
        (hist[variable] > hist[variable].quantile(quantiles[0]))
        & (hist[variable] < hist[variable].quantile(quantiles[1])),
        "unbiased",
    ] = np.asarray(hist_adj_list)

    values_body_rcp = rcp.loc[
        (rcp[variable] > rcp[variable].quantile(quantiles[0]))
        & (rcp[variable] < rcp[variable].quantile(quantiles[1])),
        variable,
    ]
    rcp_adj_list = list()
    Fe_Frcp_hist = np.interp(values_body_rcp, cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Frcp_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        rcp_adj_list.append(obs_sort[int(pos)])
    rcp.loc[
        (rcp[variable] > rcp[variable].quantile(quantiles[0]))
        & (rcp[variable] < rcp[variable].quantile(quantiles[1])),
        "unbiased",
    ] = np.asarray(rcp_adj_list)

    fit_params = {
        "hist_low": params_hist_low,
        "hist_high": params_hist_high,
        "obs_low": params_obs_low,
        "obs_high": params_obs_high,
    }

    return hist, rcp, fit_params


def probability_mapping(obs, hist, rcp, variable, func):
    """
    Apply parametric probability mapping for bias correction.

    Args:
        obs (pd.DataFrame): Observed data.
        hist (pd.DataFrame): Historical simulation data.
        rcp (pd.DataFrame): Scenario/projection data.
        variable (str): Variable name to adjust.
        func (str): Name of the distribution to use.

    Returns:
        tuple: (hist, rcp) with bias-adjusted values in column 'unbiased'.
    """

    func = getattr(st, func)

    params_obs = func.fit(obs[variable])
    params_hist = func.fit(hist[variable])

    rcp["unbiased"] = func.ppf(func.cdf(rcp[variable], *params_hist), *params_obs)
    hist["unbiased"] = func.ppf(func.cdf(hist[variable], *params_hist), *params_obs)

    return hist, rcp


def empirical_cdf_mapping(obs, hist, rcp, variable):
    """
    Apply empirical CDF mapping for bias correction.

    Args:
        obs (pd.DataFrame): Observed data.
        hist (pd.DataFrame): Historical simulation data.
        rcp (pd.DataFrame): Scenario/projection data.
        variable (str): Variable name to adjust.

    Returns:
        tuple: (hist, rcp) with bias-adjusted values in column 'unbiased'.
    """

    n_obs = len(obs)
    obs_sort = np.sort(obs[variable])

    cdf_hist = ecdf(hist, variable)

    hist_adj_list = list()
    Fe_Fhist_hist = np.interp(hist[variable], cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Fhist_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        hist_adj_list.append(obs_sort[pos])
    hist["unbiased"] = np.asarray(hist_adj_list)

    rcp_adj_list = list()
    Fe_Frcp_hist = np.interp(rcp[variable], cdf_hist[variable], cdf_hist["prob"])
    for vari_Fe in Fe_Frcp_hist:
        pos = int(n_obs * vari_Fe)
        if pos >= n_obs:
            print(pos)
            pos = n_obs - 1
        rcp_adj_list.append(obs_sort[pos])
    rcp["unbiased"] = np.asarray(rcp_adj_list)

    return hist, rcp


def rotate_geo2nav(ang):
    """
    Convert angles from geographic (0°=East, 90°=North) to navigational (0°=North, 90°=East).

    Args:
        ang (array-like): Array or Series of angles in degrees.

    Returns:
        np.ndarray: Rotated angles in degrees.
    """
    ang = np.fmod(270 - ang + 360, 360)
    return ang


def uv_to_magnitude_angle(u: pd.Series | np.ndarray, v: pd.Series | np.ndarray, labels: list = ["magnitude", "angle"]):
    """Convert u, v vector components to magnitude and direction.

    Transforms Cartesian velocity/wind components (u, v) to polar form
    (magnitude, direction) using standard meteorological convention.

    Args:
        u (pd.Series or np.ndarray): Zonal (east-west) component.
        v (pd.Series or np.ndarray): Meridional (north-south) component.
        labels (list): Output column names for [magnitude, direction].
            Defaults to ["magnitude", "angle"].

    Returns:
        pd.DataFrame: DataFrame with two columns containing magnitude (sqrt(u²+v²))
            and angle in degrees [0, 360).
    """
    ang = np.fmod(np.arctan2(v, u) * 180 / np.pi + 360, 360)
    data = pd.DataFrame(
        np.vstack([np.sqrt(u**2 + v**2), ang]).T, columns=labels, index=ang.index
    )
    return data


def optimize_rbf_epsilon(coords, data, n_train, method="gaussian", smooth=0.5, eps0=1, optimizer="local", metric="rmse"):
    """
    Optimize epsilon and smooth parameters for RBF by minimizing validation error (RMSE or MAE).
    Allows local (SLSQP) or global (differential_evolution) optimization.

    Args:
        coords (np.ndarray): Input coordinates (n_samples, n_features).
        data (np.ndarray): Target values (n_samples,).
        n_train (int): Number of samples for training (rest for validation).
        method (str, optional): RBF function type. Default 'gaussian'.
        smooth (float, optional): Initial smooth value. Default 0.5.
        eps0 (float, optional): Initial epsilon value. Default 1.
        optimizer (str, optional): 'local' (SLSQP) or 'global' (differential_evolution).
        metric (str, optional): 'rmse' or 'mae'.

    Returns:
        tuple: (epsilon_opt, smooth_opt)
    """
    min_epsilon = 1e-3
    max_epsilon = 1e5
    min_smooth = 1e-6
    max_smooth = 10.0
    bounds = [(min_epsilon, max_epsilon), (min_smooth, max_smooth)]

    # Selección aleatoria de muestras para ajuste y validación
    rng = np.random.default_rng()
    indices = np.arange(coords.shape[0])
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]

    def objective(params):
        return rbf_error_metric(params, coords, data, train_idx, valid_idx, method, metric)

    try:
        if optimizer == "global":
            res_ = differential_evolution(objective, bounds, polish=True, tol=1e-5, maxiter=100)
        else:
            res_ = minimize(
                objective,
                [eps0, smooth],
                bounds=bounds,
                method="SLSQP",
                options={"ftol": 1e-7, "eps": 1e-4, "maxiter": 1e4},
            )
        if isinstance(res_["x"], (np.ndarray, list)) and len(res_["x"]) == 2:
            epsilon_opt, smooth_opt = float(res_["x"][0]), float(res_["x"][1])
        else:
            epsilon_opt, smooth_opt = float(res_["x"]), smooth
        print(f"[optimize_rbf_epsilon] {optimizer} | metric={metric} | Success: {res_.get('success', False)}, epsilon_opt={epsilon_opt}, smooth_opt={smooth_opt}, fun={res_.get('fun', None)}")
        return epsilon_opt, smooth_opt
    except Exception as e:
        print(f"Warning: optimize_rbf_epsilon failed ({e}), using epsilon=1.0, smooth={smooth}")
        return 1.0, smooth




def rbf_error_metric(params, coords, data, train_idx, valid_idx, method, metric="rmse"):
    """
    Compute the error of an RBF for given epsilon and smooth values.

    Args:
        params (list): [epsilon, smooth].
        coords (np.ndarray): Input coordinates.
        data (np.ndarray): Target values.
        train_idx (array): Indices for training samples.
        valid_idx (array): Indices for validation samples.
        method (str): RBF function type.
        metric (str): 'rmse' or 'mae'.

    Returns:
        float: Error value (RMSE or MAE).
    """
    epsilon, smooth = params
    
    # Reordenar los conjuntos de entrenamiento y validación
    func = Rbf(
        *coords[train_idx, :].T, data[train_idx], function=method, smooth=smooth, epsilon=epsilon
    )
    validation = func(*coords[valid_idx, :].T)
    if metric == "mae":
        error = np.mean(np.abs(validation - data[valid_idx]))
    else:
        error = np.sqrt(np.mean((validation - data[valid_idx]) ** 2))
        if np.isnan(error).any():
            error = 1e10  # Penalización por NaN
    return error


def outliers_detection(
    data, outliers_fraction, method="Local Outlier Factor", scaler_method="MinMaxScaler"
):
    """
    Detect outliers in data using various sklearn algorithms.

    Args:
        data (array-like): Input data (2D array expected).
        outliers_fraction (float): Fraction of outliers to detect (contamination parameter).
        method (str, optional): Outlier detection method. Options are:
            - "Robust covariance": Uses EllipticEnvelope
            - "One-Class SVM": Uses OneClassSVM
            - "Isolation Forest": Uses IsolationForest
            - "Local Outlier Factor": Uses LocalOutlierFactor (default)
        scaler_method (str, optional): Scaling method to apply before detection. 
            If None, no scaling is applied. Defaults to "MinMaxScaler".

    Returns:
        np.ndarray: Boolean mask indicating outliers (True) and inliers (False).
    """

    algorithms = {
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "One-Class SVM": svm.OneClassSVM(
            nu=outliers_fraction, kernel="rbf", gamma="scale"
        ),
        "Isolation Forest": IsolationForest(
            contamination=outliers_fraction, behaviour="new"
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=25, contamination=outliers_fraction
        ),
    }

    # Apply scaling if requested
    if scaler_method:
        transformed_data, _ = scaler(data, method=scaler_method)
    else:
        transformed_data = data.copy()

    # Select the algorithm and fit/predict
    algorithm = algorithms[method]
    
    if method == "Local Outlier Factor":
        # LocalOutlierFactor returns -1 for outliers, 1 for inliers
        y_pred = algorithm.fit_predict(transformed_data)
    else:
        # Other methods use fit().predict()
        y_pred = algorithm.fit(transformed_data).predict(transformed_data)

    # Create boolean mask: True for inliers, False for outliers
    inliers = y_pred == 1
    
    return ~inliers  # Return True for outliers


def scaler(data, method="MinMaxScaler", transform=True, scale=False):
    """
    Scale or inverse-scale data using sklearn scalers.

    Args:
        data (array-like or pd.DataFrame): Data to scale.
        method (str, optional): Scaling method. Defaults to "MinMaxScaler".
        transform (bool, optional): If True, transform; if False, inverse transform. Defaults to True.
        scale (sklearn scaler, optional): Pre-fitted scaler to use. Defaults to False.

    Returns:
        tuple: (transformed_data, scaler)
    """
    algorithms = {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }

    np_array = False
    if not isinstance(data, pd.DataFrame):
        np_array = True

    if transform & (not scale):
        scale = algorithms[method].fit(data)

    if not transform:
        if np_array:
            transformed_data = scale.inverse_transform(data)
        else:
            transformed_data = pd.DataFrame(
                scale.inverse_transform(data), index=data.index, columns=data.columns
            )
            scale = None
    else:
        if np_array:
            transformed_data = scale.transform(data)
        else:
            transformed_data = pd.DataFrame(
                scale.transform(data), index=data.index, columns=data.columns
            )

    return transformed_data, scale


def string_to_function(param: dict, variable: str = None):
    """Convert string function names to scipy.stats function objects.

    Replaces string representations of statistical distribution names with
    actual scipy.stats function objects in parameter dictionary.

    Args:
        param (dict): Parameter dictionary containing 'fun' key with function
            name strings to convert.
        variable (str, optional): Specific variable name to process. If None,
            processes all entries in param['fun']. Defaults to None.

    Returns:
        dict: Updated parameter dictionary with function objects instead of strings.
    """
    if variable is None:
        for key in param["fun"].keys():
            if isinstance(param["fun"][key], str):
                if param["fun"][key] == "wrap_norm":
                    param["fun"][key] = core.wrap_norm()
                else:
                    param["fun"][key] = getattr(st, param["fun"][key])
    else:
        for key in param[variable]["fun"].keys():
            if isinstance(param[variable]["fun"][key], str):
                if param[variable]["fun"][key] == "wrap_norm":
                    param[variable]["fun"][key] = core.wrap_norm()
                else:
                    param[variable]["fun"][key] = getattr(st, param[variable]["fun"][key])

    return param


def data_over_threshold(data: pd.DataFrame, variable: str, threshold: float, duration: float):
    """Extract extreme events exceeding a threshold for minimum duration.

    Identifies continuous periods where variable values exceed a threshold and
    persist for at least the specified duration. Counts events per year.

    Args:
        data (pd.DataFrame): Time series data with datetime index.
        variable (str): Column name of the variable to analyze.
        threshold (float): Threshold value to identify extreme events.
        duration (float): Minimum duration (in time units matching the index)
            required to classify as an extreme event.

    Returns:
        tuple: (events, events_per_year)
            - events (pd.DataFrame): Time series of all values during threshold
              exceedance events.
            - events_per_year (pd.DataFrame): Count of events per year with
              'eventno' column.
    """
    asc_roots = data.loc[
        ((data.iloc[:-1].shift(1) < threshold) & (data.iloc[1:] >= threshold)), variable
    ]
    dec_roots = data.loc[
        ((data.iloc[:-1] >= threshold) & (data.iloc[1:].shift(-1) < threshold)),
        variable,
    ]

    if asc_roots.index[0] > dec_roots.index[0]:
        dec_roots.drop(index=dec_roots.index[0])

    if asc_roots.index[-1] > dec_roots.index[-1]:
        asc_roots.drop(index=asc_roots.index[-1])

    events = pd.DataFrame(columns=[variable])
    eventsno = pd.DataFrame(0, index=asc_roots.index.year.unique(), columns=["eventno"])
    for indexdate in range(len(asc_roots)):
        aux = pd.DataFrame(
            data.loc[asc_roots.index[indexdate] : dec_roots.index[indexdate]],
            columns=[variable],
        )
        if (aux.index[-1] - aux.index[0]).total_seconds() / 3600 >= duration:
            events = pd.concat([events, aux])
            eventsno.loc[aux.index[0].year] += 1

    return events, eventsno


def extract_isolines(data: dict, iso_values: list = None) -> dict:
    """Extract contour lines at specified values from gridded data.

    Creates contour plot and extracts coordinates of contour lines at specified
    iso-values. Returns longest path for each iso-value.

    Args:
        data (dict): Dictionary containing gridded data with keys:
            - 'x': x-coordinates (2D array)
            - 'y': y-coordinates (2D array)
            - 'z': values at each (x, y) point (2D array)
        iso_values (list, optional): Values at which to extract contours.
            Defaults to [0].

    Returns:
        dict: Dictionary mapping each iso-value to a DataFrame with 'x' and 'y'
            columns containing coordinates along the contour line.
    """
    if iso_values is None:
        iso_values = [0]

    fig, ax = plt.subplots(figsize=(8, 8))
    cs = ax.contour(data["x"], data["y"], data["z"], iso_values)
    # ax.contourf(data['x'], data['y'], data['z'])

    isolevels = dict()
    for i, j in enumerate(iso_values):
        p = cs.collections[i].get_paths()[0]  # usually the longest path
        v = p.vertices
        isolevels[j] = pd.DataFrame(
            v[:, :2], index=np.arange(len(v)), columns=["x", "y"]
        )

    plt.close(fig)
    return isolevels


def pre_ensemble_plot(models: list, param: dict, variable: str, file_name: str = None):
    """Compute ensemble statistics for multiple probability models across percentiles.

    Evaluates multiple probability models at various percentile levels,
    computes their mean and standard deviation across a normalized time grid,
    and prepares data for ensemble visualization.

    Args:
        models (list): List of model names to evaluate.
        param (dict): Dictionary containing parameters for each model, including:
            - 'fun': Probability distribution functions for each variable
            - Model-specific parameters for statistical fitting
        variable (str): Variable name to analyze (e.g., 'Hs', 'Tp', or direction variables).
        file_name (str, optional): Filename for saving results. Defaults to None.

    Returns:
        dict: Dictionary where keys are percentile strings (e.g., '0.05', '0.5', '0.95')
            and values are DataFrames containing:
            - 'n': Normalized time coordinate [0, 1]
            - 'prob': Probability value for this percentile
            - One column per model with computed values
            - 'mean': Mean across all models
            - 'std': Standard deviation across all models
    """

    probs = [0.05, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
    if variable.lower().startswith("d"):
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n = np.linspace(0, 1, 24 * 365.25)

    ppfs = dict()
    for prob in probs:
        df = pd.DataFrame(np.ones(len(n)) * prob, index=n, columns=["prob"])
        df["n"] = n
        ppfs[str(prob)] = df.copy()
        for model in models:
            param[variable][model] = string_to_function(param[variable][model], None)
            res = core.ppf(df.copy(), param[variable][model])
            # Transformed timeserie if required
            if param[variable][model]["transform"]["make"]:
                res[variable] = analysis.inverse_transform(res[[variable]], param[variable][model])
            ppfs[str(prob)].loc[:, model] = res[variable]

        ppfs[str(prob)]["mean"] = ppfs[str(prob)].loc[:, models].mean(axis=1)
        ppfs[str(prob)]["std"] = ppfs[str(prob)].loc[:, models].std(axis=1)

    return ppfs



def smooth_1d(data: np.ndarray, window_length: int = None, poly_order: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay filter for 1D data smoothing.

    Uses the Savitzky-Golay filter to smooth 1D data by fitting successive
    sub-sets of adjacent data points with a low-degree polynomial.

    Args:
        data (np.ndarray): 1D array of data values to smooth.
        window_length (int, optional): Length of the filter window (number of coefficients).
            Must be a positive odd integer. If None, defaults to len(data)/51.
            Defaults to None.
        poly_order (int, optional): Order of the polynomial used to fit the samples.
            Must be less than window_length. Defaults to 3.

    Returns:
        np.ndarray: Smoothed data array with the same length as input data.
    """

    if window_length is None:
        # TODO: Replace 51 with a target value based on analysis
        window_length = int(len(data) / 51)

    if not window_length % 2:
        window_length += 1

    sm_data = savgol_filter(data, window_length, poly_order)
    return sm_data


def find_nearest_point(data, point: tuple):
    """Find the nearest point in a spatial dataset to a given coordinate.

    Uses Euclidean distance to identify the closest point in the dataset
    to the specified target coordinates.

    Args:
        data (xr.Dataset or pd.DataFrame): Dataset with 'x' and 'y' coordinates.
        point (tuple): Target point coordinates as (x, y).

    Returns:
        int: Index of the nearest point in the dataset.
    """
    idx = np.argmin(np.sqrt((data.x - point[0]) ** 2 + (data.y - point[1]) ** 2))
    return idx


def date_to_julian(dates: list, calendar: str = "julian") -> np.ndarray:
    """Convert datetime objects to Julian dates.

    Transforms a list of datetime objects into Julian date numbers based on
    the specified calendar system.

    Args:
        dates (list): List of datetime objects to convert.
        calendar (str, optional): Calendar system to use for conversion.
            Currently only 'julian' is supported. Defaults to 'julian'.

    Returns:
        np.ndarray: Array of Julian date numbers.
    """
    if calendar == "julian":
        julian_dates = np.zeros(len(dates))
        for i, date_ in enumerate(dates):
            julian_dates[i] = (
                date_.toordinal()
                - date(1, 1, 1).toordinal()
                + date_.hour / (24)
                + date_.second / (3600 * 24)
                + 367
            )
        dates = julian_dates

    elif calendar == "gregorian":
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.to_julian_date()
        elif isinstance(dates, pd.DataFrame):
            dates.index = dates.index.to_julian_date()
        else:
            raise ValueError("DatetimeIndex or DataFrame are required.")
    else:
        raise ValueError("Calendar can be Julian or Gregorian.")
    return dates


def mean_dt_param(B, Q):
    nmodels = len(B)
    norders = np.max([np.shape(Bm) for Bm in B], axis=0)

    Bs = np.zeros([norders[0], norders[1], nmodels])
    # Qs = np.zeros([norders[0], norders[0], nmodels])

    for i in range(nmodels):
        Bs[:, : np.shape(B[i])[1], i] = B[i]

    B, Q = np.mean(Bs, axis=2), np.mean(Q, axis=0)
    return B, Q, int((norders[1] - 1) / norders[0])


def rmse(a, b):
    """Calculate Root Mean Square Error between two arrays.

    The RMSE measures the square root of the average of squared differences
    between predicted and observed values. Lower values indicate better fit.

    Args:
        a (array-like): First array (e.g., observed values).
        b (array-like): Second array (e.g., predicted values).

    Returns:
        float: The root mean square error value.
    """
    len_ = len(a)
    value_ = np.sqrt(np.sum((a - b) ** 2) / len_)
    return value_


def maximum_absolute_error(a, b):
    """Calculate Maximum Absolute Error between two arrays.

    The MAE measures the largest absolute difference between predicted
    and observed values. Useful for identifying worst-case errors.

    Args:
        a (array-like): First array (e.g., observed values).
        b (array-like): Second array (e.g., predicted values).

    Returns:
        float: The maximum absolute error value.
    """
    len_ = len(a)
    value_ = np.max(np.abs(a - b))
    return value_

def mean_absolute_error(a, b):
    """Calculate Mean Absolute Error between two arrays.

    The Mean Absolute Error (MAE) measures the average magnitude of errors
    between predicted and observed values without considering their direction.

    Args:
        a (array-like): First array (e.g., observed values).
        b (array-like): Second array (e.g., predicted values).

    Returns:
        float: The mean absolute error value.
    """
    len_ = len(a)
    value_ = np.sum(np.abs(a - b)) / len_
    return value_



def xrnearest(
    ds, lat, lon, lat_name=None, lon_name=None, variable_mask=None, time_mask=0
):
    """Find the nearest grid point to specified coordinates in xarray dataset.

    Locates the dataset grid point closest to the given latitude and longitude
    coordinates, optionally applying a mask to exclude invalid data points.

    Args:
        ds (xarray.Dataset): Input dataset with coordinate information.
        lat (float): Target latitude coordinate.
        lon (float): Target longitude coordinate.
        lat_name (str, optional): Name of latitude variable. Auto-detected if None.
        lon_name (str, optional): Name of longitude variable. Auto-detected if None.
        variable_mask (str, optional): Variable name to use for masking NaN values.
        time_mask (int): Time index to use for masking. Defaults to 0.

    Returns:
        xarray.Dataset: Subset of dataset at the nearest grid point.

    Raises:
        Exception: If coordinate dimensions are not found in dataset.
    """
    if not lat_name or not lon_name:
        lat_name, lon_name = coords_name(ds)

    lats, lons = latslons_values(ds, lat_name, lon_name)

    if variable_mask:
        mask = np.isnan(ds[variable_mask].isel(time=time_mask).values)

        lats = np.ma.masked_array(lats, mask)
        lons = np.ma.masked_array(lons, mask)

    ilat, ilon = find_indexes(lats, lons, lat, lon)

    if "rlat" in ds.dims and "rlon" in ds.dims:
        poi = ds.isel(rlat=ilat, rlon=ilon)
    elif lat_name in ds.dims and lon_name in ds.dims:
        poi = ds.isel(**{lat_name: ilat, lon_name: ilon})
    elif not ilat and not ilon:
        poi = ds
    else:
        raise Exception("Dimensions coordinates not found in dataset")

    return poi


def latslons_values(ds, lat_name, lon_name):
    """Extract latitude and longitude values from dataset.

    Retrieves coordinate values, creating 2D meshgrid if coordinates are 1D arrays.

    Args:
        ds (xarray.Dataset): Input dataset containing coordinate variables.
        lat_name (str): Name of latitude coordinate variable.
        lon_name (str): Name of longitude coordinate variable.

    Returns:
        tuple: (lats, lons) - Arrays of latitude and longitude values.
    """
    if len(ds[lat_name].dims) == 1 and len(ds[lon_name] == 1):
        lats, lons = create_lat_lon_matrix(ds[lat_name].values, ds[lon_name].values)
    else:
        lats, lons = ds[lat_name].values, ds[lon_name].values

    return lats, lons


def find_indexes(latvar, lonvar, lat0, lon0):
    """Find array indices of coordinates nearest to target location.

    Uses great circle distance (tunnel through Earth) as the distance metric
    for finding the closest grid point.

    Args:
        latvar (np.ndarray): Array of latitude values (degrees).
        lonvar (np.ndarray): Array of longitude values (degrees).
        lat0 (float): Target latitude (degrees).
        lon0 (float): Target longitude (degrees).

    Returns:
        tuple: (lat_index, lon_index) - Array indices of nearest point, or (None, None)
            if arrays are empty.

    References:
        https://www.unidata.ucar.edu/blogs/developer/en/entry/accessing_netcdf_data_by_coordinates
    """
    indexes = None, None
    if latvar.size > 1 and lonvar.size > 1:
        rad_factor = np.pi / 180.0

        latvals = latvar * rad_factor
        lonvals = lonvar * rad_factor
        lat_rad = lat0 * rad_factor
        lon_rad = lon0 * rad_factor

        clat, clon = np.cos(latvals), np.cos(lonvals)
        slat, slon = np.sin(latvals), np.sin(lonvals)
        delX = np.cos(lat_rad) * np.cos(lon_rad) - clat * clon
        delY = np.cos(lat_rad) * np.sin(lon_rad) - clat * slon
        delZ = np.sin(lat_rad) - slat
        dist_sq = delX**2 + delY**2 + delZ**2
        minindex_1d = dist_sq.argmin()

        indexes = np.unravel_index(minindex_1d, latvals.shape)

    return indexes


def create_lat_lon_matrix(lat, lon):
    """Create 2D coordinate meshgrid from 1D coordinate vectors.

    Converts separate latitude and longitude vectors into 2D coordinate matrices
    suitable for spatial operations and interpolation.

    Args:
        lat (np.ndarray): 1D array of latitude values.
        lon (np.ndarray): 1D array of longitude values.

    Returns:
        tuple: (lat_matrix, lon_matrix) - 2D arrays of shape (len(lat), len(lon)).
    """
    lat_matrix = np.tile(lat, (lon.shape[0], 1)).T
    lon_matrix = np.tile(lon, (lat.shape[0], 1))

    return lat_matrix, lon_matrix


def coords_name(ds):
    """Detect standard coordinate variable names in dataset.

    Identifies latitude and longitude coordinate names from common conventions.

    Args:
        ds (xarray.Dataset): Input dataset to inspect.

    Returns:
        tuple: (lat_name, lon_name) - Names of latitude and longitude coordinates.

    Raises:
        Exception: If standard coordinate names are not found in dataset.
    """
    if "lat" in ds.coords and "lon" in ds.coords:
        lat_name = "lat"
        lon_name = "lon"
    elif "latitude" in ds.coords and "longitude" in ds.coords:
        lat_name = "latitude"
        lon_name = "longitude"
    else:
        raise Exception("Latitudes and longitudes not found in dataset")

    return lat_name, lon_name
