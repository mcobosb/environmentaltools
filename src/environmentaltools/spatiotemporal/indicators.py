"""
Threshold-Based Spatial Indicators Module
==========================================

This module provides functions for computing threshold-based indicators used in
spatiotemporal analysis, particularly for assessing exceedance patterns and spatial
coverage relative to environmental thresholds.

These indicators are commonly used in:
- Flood risk assessment
- Pollution exposure analysis
- Habitat suitability mapping
- Climate impact studies

References
----------
.. [1] Bogardi, I., & Duckstein, L. (1993). The fuzzy logic paradigm of risk analysis.
       Risk Analysis in Water Resources Engineering, 12(1), 1-12.
"""

import numpy as np
from skimage import measure
from scipy.ndimage import uniform_filter
from environmentaltools.spatiotemporal import utils

def fractional_exceedance_area(data, thresholds=None):
    """
    Compute fractional area exceeding threshold values.

    Calculates the proportion of the spatial domain where values exceed each
    specified threshold. This indicator (RAEH - Ratio of Area Exceeding thresHold)
    is useful for assessing the spatial extent of threshold exceedances.

    Parameters
    ----------
    data : array_like
        1D array of spatial data values to analyze.
    thresholds : array_like, optional
        Array of threshold values to evaluate. If None, generates 100 equally
        spaced thresholds from 0 to the maximum data value.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values used in the analysis.
    exceedance_fractions : np.ndarray
        Fraction of area exceeding each threshold, ranging from 0 to 1.

    Notes
    -----
    The fractional exceedance area is computed as:

    .. math::
        RAEH(t) = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{1}(x_i \\geq t)

    where :math:`N` is the total number of spatial points, :math:`x_i` are the
    data values, :math:`t` is the threshold, and :math:`\\mathbb{1}` is the
    indicator function.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.normal(10, 2, 1000)
    >>> thresholds, fractions = fractional_exceedance_area(data)
    >>> # fractions[0] will be close to 1.0 (most area exceeds low threshold)
    >>> # fractions[-1] will be close to 0.0 (little area exceeds high threshold)
    """
    data = np.asarray(data)
    n_points = len(data)
    
    if thresholds is None:
        thresholds = np.linspace(0, np.max(data), 100)
    else:
        thresholds = np.asarray(thresholds)
    
    exceedance_fractions = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        exceedance_fractions[i] = np.sum(data >= threshold) / n_points

    return thresholds, exceedance_fractions


def mean_exceedance_over_total_area(data, thresholds=None):
    """
    Compute mean value of exceedances normalized by total area.

    Calculates the sum of values exceeding each threshold, normalized by the
    total number of spatial points. This indicator (MEW - Mean Exceedance over
    Whole domain) provides a measure of exceedance intensity averaged over the
    entire spatial domain.

    Parameters
    ----------
    data : array_like
        1D array of spatial data values to analyze.
    thresholds : array_like, optional
        Array of threshold values to evaluate. If None, generates 100 equally
        spaced thresholds from 0 to the maximum data value.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values used in the analysis.
    mean_exceedances : np.ndarray
        Mean exceedance values normalized by total area.

    Notes
    -----
    The mean exceedance over total area is computed as:

    .. math::
        MEW(t) = \\frac{1}{N} \\sum_{i=1}^{N} x_i \\cdot \\mathbb{1}(x_i \\geq t)

    This indicator represents the spatial average of values that exceed the
    threshold, with non-exceedance points contributing zero.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.exponential(5, 1000)
    >>> thresholds, mean_exc = mean_exceedance_over_total_area(data)
    """
    data = np.asarray(data)
    n_points = len(data)
    
    if thresholds is None:
        thresholds = np.linspace(0, np.max(data), 100)
    else:
        thresholds = np.asarray(thresholds)
    
    mean_exceedances = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        mean_exceedances[i] = np.sum(data[data >= threshold]) / n_points

    return thresholds, mean_exceedances


def mean_excess_over_total_area(data, thresholds=None):
    """
    Compute mean excess (difference from threshold) normalized by total area.

    Calculates the average amount by which values exceed each threshold,
    normalized by the total number of spatial points. This indicator (MEDW -
    Mean Excess Difference over Whole domain) measures the average magnitude
    of exceedances over the entire domain.

    Parameters
    ----------
    data : array_like
        1D array of spatial data values to analyze.
    thresholds : array_like, optional
        Array of threshold values to evaluate. If None, generates 100 equally
        spaced thresholds from 0 to the maximum data value.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values used in the analysis.
    mean_excess : np.ndarray
        Mean excess values (difference from threshold) normalized by total area.

    Notes
    -----
    The mean excess over total area is computed as:

    .. math::
        MEDW(t) = \\frac{1}{N} \\sum_{i=1}^{N} (x_i - t) \\cdot \\mathbb{1}(x_i \\geq t)

    This provides a measure of how much, on average across the entire domain,
    values exceed the threshold.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.gamma(2, 2, 1000)
    >>> thresholds, excess = mean_excess_over_total_area(data)
    """
    data = np.asarray(data)
    n_points = len(data)
    
    if thresholds is None:
        thresholds = np.linspace(0, np.max(data), 100)
    else:
        thresholds = np.asarray(thresholds)
    
    mean_excess = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        mean_excess[i] = np.sum(data[data >= threshold] - threshold) / n_points

    return thresholds, mean_excess


def mean_exceedance_over_exceedance_area(data, thresholds=None):
    """
    Compute mean value of exceedances normalized by exceedance area only.

    Calculates the average of values that exceed each threshold, considering
    only the spatial points where exceedance occurs. This indicator (WMEW -
    Weighted Mean Exceedance over exceedance area) provides the conditional
    mean given that the threshold is exceeded.

    Parameters
    ----------
    data : array_like
        1D array of spatial data values to analyze.
    thresholds : array_like, optional
        Array of threshold values to evaluate. If None, generates 100 equally
        spaced thresholds from 0 to the maximum data value.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values used in the analysis.
    conditional_means : np.ndarray
        Mean values conditional on exceeding each threshold.

    Notes
    -----
    The mean exceedance over exceedance area is computed as:

    .. math::
        WMEW(t) = \\frac{\\sum_{i: x_i \\geq t} x_i}{\\sum_{i=1}^{N} \\mathbb{1}(x_i \\geq t)}

    This represents the expected value conditional on exceeding the threshold:
    :math:`E[X | X \\geq t]`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.lognormal(1, 0.5, 1000)
    >>> thresholds, cond_means = mean_exceedance_over_exceedance_area(data)
    """
    data = np.asarray(data)
    
    if thresholds is None:
        thresholds = np.linspace(0, np.max(data), 100)
    else:
        thresholds = np.asarray(thresholds)
    
    conditional_means = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        exceeding_values = data[data >= threshold]
        if len(exceeding_values) > 0:
            conditional_means[i] = np.mean(exceeding_values)
        else:
            conditional_means[i] = np.nan

    return thresholds, conditional_means


def mean_excess_over_exceedance_area(data, thresholds=None):
    """
    Compute mean excess (difference from threshold) over exceedance area only.

    Calculates the average amount by which values exceed each threshold,
    considering only the spatial points where exceedance occurs. This indicator
    (WMDW - Weighted Mean excess Difference over exceedance area) provides the
    conditional mean excess given that the threshold is exceeded.

    Parameters
    ----------
    data : array_like
        1D array of spatial data values to analyze.
    thresholds : array_like, optional
        Array of threshold values to evaluate. If None, generates 100 equally
        spaced thresholds from 0 to the maximum data value.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values used in the analysis.
    conditional_excess : np.ndarray
        Mean excess values conditional on exceeding each threshold.

    Notes
    -----
    The mean excess over exceedance area is computed as:

    .. math::
        WMDW(t) = \\frac{\\sum_{i: x_i \\geq t} (x_i - t)}{\\sum_{i=1}^{N} \\mathbb{1}(x_i \\geq t)}

    This represents the expected excess conditional on exceeding the threshold:
    :math:`E[X - t | X \\geq t]`, also known as the mean excess function in
    extreme value theory.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.pareto(2, 1000)
    >>> thresholds, cond_excess = mean_excess_over_exceedance_area(data)
    """
    data = np.asarray(data)
    
    if thresholds is None:
        thresholds = np.linspace(0, np.max(data), 100)
    else:
        thresholds = np.asarray(thresholds)
    
    conditional_excess = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        exceeding_values = data[data >= threshold]
        if len(exceeding_values) > 0:
            conditional_excess[i] = np.mean(exceeding_values - threshold)
        else:
            conditional_excess[i] = np.nan

    return thresholds, conditional_excess


def exceedance_to_nonexceedance_ratio(data, thresholds=None):
    """
    Compute ratio of exceedance area to non-exceedance area.

    Calculates the ratio between the fraction of area exceeding each threshold
    and the fraction not exceeding it. This indicator (AEAN - Area Exceeding to
    Area Non-exceeding) becomes large when exceedances are prevalent and
    approaches zero when exceedances are rare.

    Parameters
    ----------
    data : array_like
        1D array of spatial data values to analyze.
    thresholds : array_like, optional
        Array of threshold values to evaluate. If None, generates 100 equally
        spaced thresholds from 0 to the maximum data value.

    Returns
    -------
    thresholds : np.ndarray
        Array of threshold values used in the analysis.
    area_ratios : np.ndarray
        Ratio of exceedance area to non-exceedance area for each threshold.

    Notes
    -----
    The exceedance to non-exceedance ratio is computed as:

    .. math::
        AEAN(t) = \\frac{RAEH(t)}{1 - RAEH(t)} = \\frac{N_e}{N - N_e}

    where :math:`N_e` is the number of points exceeding the threshold and
    :math:`N` is the total number of points.

    The ratio approaches infinity as the exceedance fraction approaches 1, and
    equals 0 when no points exceed the threshold.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.beta(2, 5, 1000) * 10
    >>> thresholds, ratios = exceedance_to_nonexceedance_ratio(data)
    
    Warnings
    --------
    Returns infinity for thresholds where all points exceed (100% exceedance).
    """
    data = np.asarray(data)
    n_points = len(data)
    
    if thresholds is None:
        thresholds = np.linspace(0, np.max(data), 100)
    else:
        thresholds = np.asarray(thresholds)
    
    area_ratios = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        exceedance_fraction = np.sum(data >= threshold) / n_points
        if exceedance_fraction < 1.0:
            area_ratios[i] = exceedance_fraction / (1.0 - exceedance_fraction)
        else:
            area_ratios[i] = np.inf

    return thresholds, area_ratios


# def compute_all_indicators_and_plot(moments):
#     """
#     Compute all threshold-based indicators for a single point and create plots.

#     Convenience function that computes all five main threshold-based indicators
#     (RAEH, MEW, MEDW, WMEW, WMDW) from BME moment data and generates comparative
#     visualization plots.

#     Parameters
#     ----------
#     moments : np.ndarray
#         2D array with shape (n_points, n_columns) where the second column
#         (moments[:, 1]) contains the values to analyze. Typically output from
#         BME estimation at a single spatial location across multiple time points
#         or ensemble members.

#     Returns
#     -------
#     None
#         Generates and displays plots using the graphics module.

#     Notes
#     -----
#     This function extracts the mean values (second column) from the moments array
#     and computes:
    
#     - **RAEH**: Fractional area exceeding threshold
#     - **MEW**: Mean exceedance over whole domain
#     - **MEDW**: Mean excess difference over whole domain
#     - **WMEW**: Mean exceedance over exceedance area
#     - **WMDW**: Mean excess difference over exceedance area

#     The results are visualized using the spatiotemporal graphics module.

#     Examples
#     --------
#     >>> import numpy as np
#     >>> from environmentaltools.spatiotemporal import compute_all_indicators_and_plot
#     >>> 
#     >>> # Simulate BME moments for a single point
#     >>> moments = np.random.normal(5, 1.5, (1000, 3))
#     >>> compute_all_indicators_and_plot(moments)
#     """
#     # Define indicator labels
#     labels = ["RAEH", "MEW", "MEDW", "WMEW", "WMDW"]
    
#     # Initialize storage for thresholds and indicator values
#     thresholds = [None] * len(labels)
#     indicator_values = [None] * len(labels)

#     # Extract mean values (second column) from moments
#     data = moments[:, 1]

#     # Compute all indicators
#     thresholds[0], indicator_values[0] = fractional_exceedance_area(data)
#     thresholds[1], indicator_values[1] = mean_exceedance_over_total_area(data)
#     thresholds[2], indicator_values[2] = mean_excess_over_total_area(data)
#     thresholds[3], indicator_values[3] = mean_exceedance_over_exceedance_area(data)
#     thresholds[4], indicator_values[4] = mean_excess_over_exceedance_area(data)

#     # Create visualization
#     figures.indicators(thresholds, indicator_values, labels)
    
#     return


def mean_presence_boundary(data_cube, threshold=None):
    """
    Calculate spatial boundary where temporal mean exceeds a presence threshold.
    
    Computes the contour lines that define areas where the temporal average
    of values exceeds a specified threshold. This is useful for identifying
    persistent zones of influence or presence in spatiotemporal data.

    Parameters
    ----------
    data_cube : np.ndarray, xarray.DataArray, or xarray.Dataset
        3D array with shape (time, lat, lon) containing spatiotemporal data.
        If Dataset, uses the first data variable.
    threshold : float, optional
        Presence threshold. If None, uses the global mean of the temporal
        average map.

    Returns
    -------
    contours : list
        List of contour coordinates defining the presence boundary.
    mean_map : np.ndarray
        2D map of temporal means for each spatial location.

    Notes
    -----
    The function computes temporal means for each spatial cell, then identifies
    contours where these means exceed the specified threshold. The contours
    represent boundaries between areas of high and low temporal presence.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((365, 50, 50))  # Daily data for 1 year
    >>> contours, mean_map = mean_presence_boundary(data_cube, threshold=0.6)
    """
    # Handle xarray Dataset/DataArray
    if hasattr(data_cube, 'values'):
        # xarray DataArray or Dataset
        if hasattr(data_cube, 'data_vars'):
            # It's a Dataset, get first variable
            var_name = list(data_cube.data_vars.keys())[0]
            data_array = data_cube[var_name]
        else:
            # It's already a DataArray
            data_array = data_cube
        
        # Compute mean using xarray's dim parameter
        mean_map = data_array.mean(dim='time').values
    else:
        # It's a numpy array
        mean_map = np.mean(data_cube, axis=0)
    
    # Presence threshold
    if threshold is None:
        threshold = np.mean(mean_map)
    
    # Binary: presence where threshold is exceeded
    presence_mask = mean_map >= threshold
    
    # Extract contours
    contours = measure.find_contours(presence_mask.astype(float), level=0.5)
    
    return contours, mean_map


def maximum_influence_extent(data_cube, percentile=95):
    """
    Calculate maximum spatial extent under extreme conditions.

    Determines the spatial boundary encompassing areas that experience
    extreme values (defined by a percentile threshold) over the time period.
    This is useful for assessing maximum impact zones or worst-case scenarios.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    percentile : float, optional
        Percentile to define extreme condition (e.g., 95 for 95th percentile).
        Default is 95.

    Returns
    -------
    contours : list
        List of contour coordinates defining the maximum influence extent.
    extreme_map : np.ndarray
        2D map of extreme values (specified percentile) for each spatial cell.

    Notes
    -----
    The function computes the specified percentile value for each spatial
    location across time, then identifies contours around areas where these
    extreme values exceed the mean extreme value.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.gamma(2, 1, (100, 30, 30))  # Gamma-distributed data
    >>> contours, extreme_map = maximum_influence_extent(data_cube, percentile=90)
    """
    # Compute extreme value per cell
    extreme_map = np.percentile(data_cube, percentile, axis=0)
    
    # Define presence threshold (can be dynamic or fixed)
    threshold = np.mean(extreme_map)  # or use a physical value
    
    # Create binary mask
    mask = extreme_map >= threshold
    
    # Extract contours
    contours = measure.find_contours(mask.astype(float), level=0.5)
    
    return contours, extreme_map


def threshold_exceedance_frequency(data_cube, threshold):
    """
    Calculate frequency of threshold exceedance for each spatial cell.
    
    Computes how many times each spatial location exceeds a given threshold
    across the time dimension. This provides a measure of temporal persistence
    of exceedance conditions.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    threshold : float or np.ndarray
        Fixed threshold value or 2D array of spatially-varying thresholds
        with shape (lat, lon).

    Returns
    -------
    freq_map : np.ndarray
        2D map showing number of times threshold is exceeded at each location.

    Notes
    -----
    For a fixed threshold, exceedance is computed as data_cube > threshold.
    For spatially-varying thresholds, each spatial cell is compared against
    its corresponding threshold value.

    The frequency map provides insight into which areas are most prone to
    threshold exceedances over time.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.exponential(2, (100, 20, 20))
    >>> freq_map = threshold_exceedance_frequency(data_cube, threshold=3.0)
    >>> print(f"Max exceedances: {freq_map.max()}")
    """
    if isinstance(threshold, (int, float)):
        exceedances = data_cube > threshold
    else:
        # Variable threshold per cell
        exceedances = data_cube > threshold[np.newaxis, :, :]
    
    freq_map = np.sum(exceedances, axis=0)
    return freq_map


def permanently_affected_zone(data_cube, threshold, persistence_ratio=0.8):
    """
    Calculate permanently affected zones based on threshold exceedance persistence.
    
    Identifies spatial areas where a variable exceeds a threshold for a specified
    proportion of the time period. This is useful for identifying zones of
    chronic impact or persistent environmental stress.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    threshold : float or np.ndarray
        Fixed threshold value or 2D array of spatially-varying thresholds
        with shape (lat, lon).
    persistence_ratio : float, optional
        Minimum proportion of time for a cell to be considered permanently
        affected (e.g., 0.8 means 80% of the time). Default is 0.8.

    Returns
    -------
    mask : np.ndarray
        Binary map of permanently affected zones (True = permanently affected).
    freq_map : np.ndarray
        Map of exceedance frequency (0-1) for each spatial location.

    Notes
    -----
    A location is considered permanently affected if its exceedance frequency
    exceeds the persistence_ratio threshold. This helps distinguish between
    areas with occasional versus persistent exceedances.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.lognormal(0, 1, (365, 25, 25))  # Daily data
    >>> mask, freq = permanently_affected_zone(data_cube, threshold=2.0, persistence_ratio=0.75)
    >>> print(f"Permanently affected area: {mask.sum()} cells")
    """
    if isinstance(threshold, (int, float)):
        exceedances = data_cube > threshold
    else:
        exceedances = data_cube > threshold[np.newaxis, :, :]
    
    freq_map = np.sum(exceedances, axis=0) / data_cube.shape[0]
    mask = freq_map >= persistence_ratio
    return mask, freq_map


def mean_representative_value(data_cube, time_window=None):
    """
    Calculate representative mean value for each spatial cell.
    
    Computes the temporal mean for each spatial location, optionally within
    a specified time window. This provides a representative value that
    summarizes the typical condition at each location.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    time_window : tuple of int, optional
        Time indices to use (start, end). If None, uses the entire time period.

    Returns
    -------
    mean_map : np.ndarray
        2D map of representative mean values for each spatial cell.

    Notes
    -----
    When time_window is specified, only the data within that temporal subset
    is used for computing means. This is useful for analyzing seasonal patterns
    or specific time periods of interest.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.normal(10, 2, (365, 30, 30))  # Daily data
    >>> # Full year average
    >>> mean_all = mean_representative_value(data_cube)
    >>> # Summer months (June-August, assuming daily data starting Jan 1)
    >>> mean_summer = mean_representative_value(data_cube, time_window=(150, 243))
    """
    if time_window:
        start, end = time_window
        data_slice = data_cube[start:end]
    else:
        data_slice = data_cube
    
    mean_map = np.mean(data_slice, axis=0)
    return mean_map


from scipy.stats import genextreme

def return_period_extreme_value(data_series, return_period):
    """
    Fit Generalized Extreme Value distribution and calculate return period value.
    
    Fits a GEV (Generalized Extreme Value) distribution to extreme value data
    and computes the value associated with a specified return period. This is
    commonly used in environmental risk assessment and engineering design.

    Parameters
    ----------
    data_series : np.ndarray
        1D array of extreme values (e.g., annual maxima).
    return_period : int
        Return period in years (e.g., 100 for 100-year return period).

    Returns
    -------
    x_T : float
        Extreme value associated with the specified return period.
    params : tuple
        Fitted GEV parameters (shape, location, scale).

    Notes
    -----
    The return period value is calculated as:
    
    .. math::
        x_T = F^{-1}\\left(1 - \\frac{1}{T}\\right)
    
    where :math:`F^{-1}` is the inverse CDF of the fitted GEV distribution
    and :math:`T` is the return period.

    The GEV distribution is commonly used for modeling extreme values in
    environmental applications such as flood analysis, wind speed extremes,
    and temperature extremes.

    Examples
    --------
    >>> import numpy as np
    >>> annual_maxima = np.random.gumbel(10, 2, 50)  # 50 years of data
    >>> x_100, params = return_period_extreme_value(annual_maxima, 100)
    >>> print(f"100-year return value: {x_100:.2f}")
    """
    shape, loc, scale = genextreme.fit(data_series)
    prob = 1 - 1 / return_period
    x_T = genextreme.ppf(prob, shape, loc=loc, scale=scale)
    return x_T, (shape, loc, scale)


def spatial_change_rate(data_cube, dx=1, dy=1):
    """
    Calculate mean spatial change rate for each cell over time.

    Computes the temporal average of spatial gradient magnitudes for each
    location. This measures how rapidly values change spatially and how
    this spatial variability evolves over time.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    dx, dy : float, optional
        Spatial resolution in X and Y directions. Default is 1.

    Returns
    -------
    rate_map : np.ndarray
        2D map of mean spatial change rates.

    Notes
    -----
    For each time step, the function computes spatial gradients using finite
    differences, then calculates the gradient magnitude. The final result is
    the temporal average of these magnitudes.

    High values indicate areas with consistently high spatial variability,
    while low values suggest spatially smooth regions.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((50, 20, 20))
    >>> rate_map = spatial_change_rate(data_cube, dx=0.1, dy=0.1)
    """
    gradients = []
    for t in range(data_cube.shape[0]):
        grad_x, grad_y = utils.spatial_gradient(data_cube[t], dx=dx, dy=dy)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradients.append(magnitude)

    rate_map = np.mean(gradients, axis=0)
    return rate_map


def functional_area_loss(data_cube, threshold, t_start, t_end):
    """
    Calculate functional area loss between two time points.
    
    Compares functional areas (defined by threshold exceedance) between two
    time points to quantify spatial losses. This is useful for assessing
    habitat loss, service area degradation, or similar applications.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    threshold : float
        Threshold defining functional status.
    t_start, t_end : int
        Time indices to compare.

    Returns
    -------
    loss_map : np.ndarray
        Binary map of loss areas (1 = loss occurred, 0 = no loss).
    area_start, area_end : int
        Number of functional cells at start and end times.

    Notes
    -----
    A cell experiences functional loss if it was above threshold at t_start
    but below threshold at t_end. The analysis identifies which specific
    areas have lost functionality between the two time points.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((100, 25, 25))
    >>> loss_map, area_start, area_end = functional_area_loss(data_cube, 0.5, 10, 90)
    >>> print(f"Area lost: {loss_map.sum()} cells")
    """
    mask_start = data_cube[t_start] >= threshold
    mask_end = data_cube[t_end] >= threshold
    
    loss_map = np.logical_and(mask_start, ~mask_end)
    area_start = np.sum(mask_start)
    area_end = np.sum(mask_end)
    
    return loss_map.astype(int), area_start, area_end


from skimage import measure

def critical_boundary_retreat(data_cube, threshold, t_start, t_end):
    """
    Calculate critical boundary retreat between two time points.
    
    Analyzes how the boundary of critical areas (defined by threshold exceedance)
    changes between two time points. This is useful for studying phenomena like
    shoreline retreat, vegetation boundary shifts, or pollution extent changes.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    threshold : float
        Critical threshold defining the boundary.
    t_start, t_end : int
        Time indices to compare.

    Returns
    -------
    contours_start, contours_end : list
        Contour coordinates at start and end times.
    retreat_mask : np.ndarray
        Binary map showing retreat areas (1 = retreat occurred).

    Notes
    -----
    The function identifies contours of the critical threshold at both time
    points and computes areas that were above threshold initially but fell
    below threshold later, indicating boundary retreat.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.exponential(1, (50, 30, 30))
    >>> contours_start, contours_end, retreat = critical_boundary_retreat(
    ...     data_cube, threshold=1.5, t_start=5, t_end=45)
    """
    mask_start = data_cube[t_start] >= threshold
    mask_end = data_cube[t_end] >= threshold
    
    contours_start = measure.find_contours(mask_start.astype(float), level=0.5)
    contours_end = measure.find_contours(mask_end.astype(float), level=0.5)
    
    retreat_mask = np.logical_and(mask_start, ~mask_end)
    return contours_start, contours_end, retreat_mask.astype(int)



def neighbourhood_mean(data_cube, size=3):
    """
    Calculate neighborhood mean value for each cell.
    
    Applies a spatial moving average filter to compute the mean value within
    a specified neighborhood around each cell. This is useful for spatial
    smoothing and analyzing local context.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    size : int, optional
        Neighborhood size (e.g., 3 for 3x3 neighborhood). Default is 3.

    Returns
    -------
    cube_filtered : np.ndarray
        3D array with neighborhood means, same shape as input.

    Notes
    -----
    The function applies a uniform filter that computes the mean value within
    a square neighborhood of the specified size around each cell. Edge effects
    are handled using reflection mode.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((10, 20, 20))
    >>> smoothed = neighbourhood_mean(data_cube, size=5)
    """
    cube_filtered = uniform_filter(data_cube, size=(1, size, size), mode='reflect')
    return cube_filtered


def neighbourhood_gradient_influence(data_cube, size=3):
    """
    Calculate gradient magnitude between each cell and its neighborhood.
    
    Computes the absolute difference between each cell's value and its
    neighborhood mean. This measures how much each cell deviates from its
    local spatial context.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    size : int, optional
        Neighborhood size. Default is 3.

    Returns
    -------
    influence_map : np.ndarray
        2D map of average gradient influence.

    Notes
    -----
    High values indicate cells that consistently differ from their neighbors,
    suggesting local anomalies or gradient boundaries. Low values indicate
    cells that are well-integrated with their spatial context.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((20, 15, 15))
    >>> influence_map = neighbourhood_gradient_influence(data_cube, size=5)
    """
    cube_filtered = neighbourhood_mean(data_cube, size=size)
    diff = data_cube - cube_filtered
    influence_map = np.mean(np.abs(diff), axis=0)
    return influence_map


def environmental_convergence(data_cube, size=3):
    """
    Calculate environmental convergence as reduction of neighborhood differences over time.
    
    Measures how the difference between each cell and its neighborhood changes
    over time. Negative values indicate convergence (becoming more similar to
    neighbors), while positive values indicate divergence.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    size : int, optional
        Neighborhood size. Default is 3.

    Returns
    -------
    convergence_map : np.ndarray
        2D map of convergence trends (negative = convergence, positive = divergence).

    Notes
    -----
    The function computes the temporal trend of absolute differences between
    each cell and its neighborhood mean. Areas with negative trends are
    becoming more spatially homogeneous over time.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((50, 20, 20))
    >>> convergence_map = environmental_convergence(data_cube, size=3)
    """
    cube_filtered = neighbourhood_mean(data_cube, size=size)
    diff = np.abs(data_cube - cube_filtered)
    trend = np.polyfit(np.arange(data_cube.shape[0]), diff.reshape(data_cube.shape[0], -1), deg=1)[0]
    convergence_map = trend.reshape(data_cube.shape[1], data_cube.shape[2])
    return convergence_map


def neighbourhood_polarization(data_cube, size=3):
    """
    Calculate polarization as local standard deviation.
    
    Computes the temporal standard deviation of differences between each cell
    and its neighborhood mean. This measures local variability and identifies
    areas of high temporal polarization.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    size : int, optional
        Neighborhood size. Default is 3.

    Returns
    -------
    polarization_map : np.ndarray
        2D map of average polarization.

    Notes
    -----
    High polarization values indicate areas where the relationship between
    a cell and its neighbors varies significantly over time. Low values
    suggest stable neighborhood relationships.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((30, 25, 25))
    >>> polarization_map = neighbourhood_polarization(data_cube, size=3)
    """
    cube_filtered = neighbourhood_mean(data_cube, size=size)
    polarization = np.std(data_cube - cube_filtered, axis=0)
    return polarization



def local_persistence(data_cube, size=3):
    """
    Calculate local persistence as proportion of time cell exceeds its neighborhood.
    
    Computes the fraction of time that each cell has values higher than its
    neighborhood mean. This measures local dominance and persistence patterns.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    size : int, optional
        Neighborhood size. Default is 3.

    Returns
    -------
    persistence_map : np.ndarray
        2D map of persistence values (0-1 scale).

    Notes
    -----
    Values near 1 indicate cells that consistently exceed their neighborhood
    (local maxima or persistent hotspots). Values near 0 indicate cells that
    consistently fall below their neighborhood (local minima or cold spots).

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.exponential(1, (40, 20, 20))
    >>> persistence_map = local_persistence(data_cube, size=5)
    """
    cube_filtered = neighbourhood_mean(data_cube, size=size)
    dominance = data_cube > cube_filtered
    persistence_map = np.mean(dominance, axis=0)
    return persistence_map


def environmental_risk(data_cube, threshold, size=3):
    """
    Calculate environmental risk as frequency of extreme values combined with polarization.
    
    Combines threshold exceedance frequency with neighborhood polarization to
    assess environmental risk. Areas with both high exceedance frequency and
    high polarization are considered highest risk.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    threshold : float
        Threshold defining extreme/hazardous conditions.
    size : int, optional
        Neighborhood size for polarization calculation. Default is 3.

    Returns
    -------
    risk_map : np.ndarray
        2D map of environmental risk.

    Notes
    -----
    The risk is computed as the product of exceedance frequency and
    neighborhood polarization. This combines information about both the
    likelihood of extreme events and the spatial variability of conditions.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.gamma(2, 1, (100, 30, 30))
    >>> risk_map = environmental_risk(data_cube, threshold=3.0, size=3)
    """
    exceedance = data_cube > threshold
    freq = np.mean(exceedance, axis=0)
    polarization = neighbourhood_polarization(data_cube, size=size)
    risk_map = freq * polarization
    return risk_map



def directional_influence(data_cube, dx=1, dy=1):
    """
    Calculate mean direction of spatial gradient for each cell.
    
    Computes the temporal average direction and magnitude of spatial gradients.
    This identifies the predominant direction of spatial influence and the
    strength of directional patterns.

    Parameters
    ----------
    data_cube : np.ndarray
        3D array with shape (time, lat, lon) containing spatiotemporal data.
    dx, dy : float, optional
        Spatial resolution in X and Y directions. Default is 1.

    Returns
    -------
    angle_map : np.ndarray
        2D map of mean gradient directions in radians.
    magnitude_map : np.ndarray
        2D map of mean gradient magnitudes.

    Notes
    -----
    The function computes spatial gradients for each time step, then averages
    both the direction (angle) and magnitude across time. This reveals
    persistent spatial patterns and directional influences.

    Examples
    --------
    >>> import numpy as np
    >>> data_cube = np.random.random((50, 25, 25))
    >>> angles, magnitudes = directional_influence(data_cube, dx=0.1, dy=0.1)
    """
    grad_x_list = []
    grad_y_list = []
    
    for t in range(data_cube.shape[0]):
        grad_x, grad_y = utils.spatial_gradient(data_cube[t], dx=dx, dy=dy)
        grad_x_list.append(grad_x)
        grad_y_list.append(grad_y)
    
    mean_grad_x = np.mean(grad_x_list, axis=0)
    mean_grad_y = np.mean(grad_y_list, axis=0)
    
    angle_map = np.arctan2(mean_grad_y, mean_grad_x)
    magnitude_map = np.sqrt(mean_grad_x**2 + mean_grad_y**2)
    
    return angle_map, magnitude_map


def multivariate_neighbourhood_synergy(cube_list, size=3):
    """
    Calculate neighborhood synergy between multiple variables.
    
    Computes the synergy between multiple environmental variables within
    spatial neighborhoods. High synergy indicates variables that co-vary
    coherently in space, while low synergy suggests independent patterns.

    Parameters
    ----------
    cube_list : list of np.ndarray
        List of 3D arrays (time, lat, lon), one per variable.
    size : int, optional
        Neighborhood size. Default is 3.

    Returns
    -------
    synergy_map : np.ndarray
        2D map of neighborhood synergy (0-1 scale, higher = more synergistic).

    Notes
    -----
    Synergy is computed as the inverse of the coefficient of variation across
    variables within each neighborhood. High synergy means variables have
    similar relative patterns within neighborhoods.

    Examples
    --------
    >>> import numpy as np
    >>> cube1 = np.random.random((20, 15, 15))
    >>> cube2 = np.random.random((20, 15, 15))
    >>> synergy = multivariate_neighbourhood_synergy([cube1, cube2], size=3)
    """
    
    neigh_cubes = [neighbourhood_mean(cube, size=size) for cube in cube_list]
    mean_maps = [np.mean(cube, axis=0) for cube in neigh_cubes]
    
    stacked = np.stack(mean_maps, axis=-1)  # (lat, lon, variables)
    synergy_map = np.zeros(stacked.shape[:2])
    
    for i in range(stacked.shape[0]):
        for j in range(stacked.shape[1]):
            vec = stacked[i, j, :]
            synergy_map[i, j] = np.std(vec) / (np.mean(vec) + 1e-6)  # inverse coefficient of variation
    
    return 1 - synergy_map  # higher synergy = lower dispersion


def spatiotemporal_coupling(cube_x, cube_y):
    """
    Calculate spatiotemporal coupling between two variables.
    
    Computes the temporal correlation between two variables for each spatial
    location. This identifies areas where variables are strongly coupled
    versus areas where they vary independently.

    Parameters
    ----------
    cube_x, cube_y : np.ndarray
        3D arrays with shape (time, lat, lon) for the two variables.

    Returns
    -------
    coupling_map : np.ndarray
        2D map of temporal correlation coefficients (-1 to 1).

    Notes
    -----
    Correlation is computed independently for each spatial location across
    the time dimension. Values near 1 indicate strong positive coupling,
    values near -1 indicate strong negative coupling, and values near 0
    indicate weak or no coupling.

    Examples
    --------
    >>> import numpy as np
    >>> cube_x = np.random.random((100, 20, 20))
    >>> cube_y = np.random.random((100, 20, 20))
    >>> coupling = spatiotemporal_coupling(cube_x, cube_y)
    """
    coupling_map = np.zeros(cube_x.shape[1:])
    for i in range(cube_x.shape[1]):
        for j in range(cube_x.shape[2]):
            x = cube_x[:, i, j]
            y = cube_y[:, i, j]
            if np.std(x) > 0 and np.std(y) > 0:
                coupling_map[i, j] = np.corrcoef(x, y)[0, 1]
            else:
                coupling_map[i, j] = 0
    return coupling_map


def multivariate_threshold_exceedance(cube_list, thresholds):
    """
    Calculate frequency of simultaneous threshold exceedance for multiple variables.
    
    Computes how often all variables simultaneously exceed their respective
    thresholds. This is useful for identifying areas prone to compound
    environmental extremes.

    Parameters
    ----------
    cube_list : list of np.ndarray
        List of 3D arrays (time, lat, lon), one per variable.
    thresholds : list of float
        List of threshold values, one per variable.

    Returns
    -------
    exceedance_map : np.ndarray
        2D map of simultaneous exceedance frequency (0-1 scale).

    Notes
    -----
    The function identifies time steps when ALL variables exceed their
    respective thresholds simultaneously. This is more restrictive than
    individual threshold exceedances and identifies true compound events.

    Examples
    --------
    >>> import numpy as np
    >>> cube1 = np.random.exponential(2, (50, 15, 15))
    >>> cube2 = np.random.gamma(2, 1, (50, 15, 15))
    >>> freq_map = multivariate_threshold_exceedance([cube1, cube2], [3.0, 2.5])
    """
    exceedance = np.ones_like(cube_list[0], dtype=bool)
    for cube, th in zip(cube_list, thresholds):
        exceedance &= cube > th
    freq_map = np.mean(exceedance, axis=0)
    return freq_map


def directional_coevolution(cube_x, cube_y):
    """
    Calculate directional coevolution between two variables.
    
    Measures how often two variables change in the same direction over time.
    This identifies areas where variables have coordinated temporal dynamics
    versus areas where they evolve independently or oppositely.

    Parameters
    ----------
    cube_x, cube_y : np.ndarray
        3D arrays with shape (time, lat, lon) for the two variables.

    Returns
    -------
    coevolution_map : np.ndarray
        2D map of directional agreement (0-1 scale).

    Notes
    -----
    The function computes temporal differences (changes) for both variables,
    then calculates the proportion of time when both variables change in
    the same direction (both increase or both decrease).

    Values near 1 indicate strong coevolution, values near 0 indicate
    independent evolution, and values near 0.5 suggest random relationships.

    Examples
    --------
    >>> import numpy as np
    >>> cube_x = np.random.random((100, 20, 20))
    >>> cube_y = np.random.random((100, 20, 20))
    >>> coevolution = directional_coevolution(cube_x, cube_y)
    """
    dx = np.diff(cube_x, axis=0)
    dy = np.diff(cube_y, axis=0)
    agreement = np.sign(dx) == np.sign(dy)
    coevolution_map = np.mean(agreement, axis=0)
    return coevolution_map


def multivariate_persistence(cube_list, thresholds):
    """
    Calculate multivariate persistence as proportion of time with simultaneous conditions.
    
    Computes the temporal persistence of compound conditions where multiple
    variables simultaneously exceed their thresholds. This measures the
    duration and frequency of multivariate extreme states.

    Parameters
    ----------
    cube_list : list of np.ndarray
        List of 3D arrays (time, lat, lon), one per variable.
    thresholds : list of float
        List of threshold values, one per variable.

    Returns
    -------
    persistence_map : np.ndarray
        2D map of compound persistence (0-1 scale).

    Notes
    -----
    This is similar to multivariate_threshold_exceedance but emphasizes the
    temporal persistence aspect. High values indicate areas where compound
    extreme conditions persist for extended periods.

    Examples
    --------
    >>> import numpy as np
    >>> cube1 = np.random.lognormal(0, 1, (365, 20, 20))  # Daily data
    >>> cube2 = np.random.gamma(2, 1, (365, 20, 20))
    >>> persistence = multivariate_persistence([cube1, cube2], [2.0, 3.0])
    """
    condition = np.ones_like(cube_list[0], dtype=bool)
    for cube, th in zip(cube_list, thresholds):
        condition &= cube > th
    persistence_map = np.mean(condition, axis=0)
    return persistence_map


