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
from environmentaltools.graphics import spatiotemporal as figures


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


def compute_all_indicators_and_plot(moments):
    """
    Compute all threshold-based indicators for a single point and create plots.

    Convenience function that computes all five main threshold-based indicators
    (RAEH, MEW, MEDW, WMEW, WMDW) from BME moment data and generates comparative
    visualization plots.

    Parameters
    ----------
    moments : np.ndarray
        2D array with shape (n_points, n_columns) where the second column
        (moments[:, 1]) contains the values to analyze. Typically output from
        BME estimation at a single spatial location across multiple time points
        or ensemble members.

    Returns
    -------
    None
        Generates and displays plots using the graphics module.

    Notes
    -----
    This function extracts the mean values (second column) from the moments array
    and computes:
    
    - **RAEH**: Fractional area exceeding threshold
    - **MEW**: Mean exceedance over whole domain
    - **MEDW**: Mean excess difference over whole domain
    - **WMEW**: Mean exceedance over exceedance area
    - **WMDW**: Mean excess difference over exceedance area

    The results are visualized using the spatiotemporal graphics module.

    Examples
    --------
    >>> import numpy as np
    >>> from environmentaltools.spatiotemporal import compute_all_indicators_and_plot
    >>> 
    >>> # Simulate BME moments for a single point
    >>> moments = np.random.normal(5, 1.5, (1000, 3))
    >>> compute_all_indicators_and_plot(moments)
    """
    # Define indicator labels
    labels = ["RAEH", "MEW", "MEDW", "WMEW", "WMDW"]
    
    # Initialize storage for thresholds and indicator values
    thresholds = [None] * len(labels)
    indicator_values = [None] * len(labels)

    # Extract mean values (second column) from moments
    data = moments[:, 1]

    # Compute all indicators
    thresholds[0], indicator_values[0] = fractional_exceedance_area(data)
    thresholds[1], indicator_values[1] = mean_exceedance_over_total_area(data)
    thresholds[2], indicator_values[2] = mean_excess_over_total_area(data)
    thresholds[3], indicator_values[3] = mean_exceedance_over_exceedance_area(data)
    thresholds[4], indicator_values[4] = mean_excess_over_exceedance_area(data)

    # Create visualization
    figures.indicators(thresholds, indicator_values, labels)
    
    return
