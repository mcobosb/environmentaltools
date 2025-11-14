"""
Spatiotemporal Analysis Module
===============================

This module provides tools for analyzing spatiotemporal environmental data,
with a focus on Bayesian Maximum Entropy (BME) methods for optimal estimation
and uncertainty quantification.

Main Components
---------------

**BME Estimation**
    - Optimal spatiotemporal prediction combining prior knowledge with data
    - Integration of hard (exact) and soft (probabilistic) observations
    - Local neighborhood-based estimation with adaptive data selection
    - Cross-validation for model performance assessment

**Spatiotemporal Covariance**
    - Empirical covariance calculation from spatiotemporal data
    - Theoretical covariance models (exponential, non-separable)
    - Directional (anisotropic) covariance analysis
    - Automated covariance model parameter fitting

**Threshold-Based Indicators**
    - Fractional area exceeding threshold (raeh)
    - Mean exceedance values (mew, wmew)
    - Mean difference exceedance (medw, wmdw)
    - Area-weighted indicators (aean)
    - Single-point moment extraction (one_point)

**Multi-Criteria Decision Analysis**
    - TOPSIS method for spatial prioritization
    - Multiple weighting schemes
    - Visualization and mapping tools

**Raster Analysis**
    - Spatiotemporal binary matrix generation
    - Input validation and preprocessing
    - NetCDF output generation

Examples
--------
BME estimation workflow:
    >>> import numpy as np
    >>> from environmentaltools.spatiotemporal import (
    ...     compute_spatiotemporal_covariance,
    ...     fit_covariance_model,
    ...     compute_bme_moments
    ... )
    >>> 
    >>> # Compute empirical covariance
    >>> empcov, pairs, dists, distt = compute_spatiotemporal_covariance(
    ...     dfh, dfs, slag=np.linspace(0, 100, 20), tlag=np.linspace(0, 30, 10)
    ... )
    >>> 
    >>> # Fit theoretical model
    >>> import scipy.optimize as opt
    >>> result = opt.minimize(
    ...     fit_covariance_model,
    ...     x0=[1.0, 20.0, 5.0, 0.1],
    ...     args=(empcov, [dists, distt], 'exponentialST')
    ... )
    >>> 
    >>> # Compute BME moments
    >>> moments = compute_bme_moments(
    ...     dfk, dfh, dfs, 'exponentialST', result.x,
    ...     nmax=[50, 100], dmax=[100, 30, 3], order=[1, 1],
    ...     options=[100, 3, 0.95], path='./cache', name='bme_run'
    ... )
"""

# BME estimation functions
from environmentaltools.spatiotemporal.bme import (
    compute_bme_moments,
    estimate_local_mean_bme,
    calculate_moments,
    integrate_moment_vector,
    apply_data_smoothing,
    perform_cross_validation,
    select_neighbours,
    estimate_bme_regression,
    create_design_matrix,
    smooth_data,
    create_spatiotemporal_matrix,
    coordinates_to_distance,
    coordinates_to_distance_angle,
    find_pairs_by_distance,
)

# Covariance functions
from environmentaltools.spatiotemporal.covariance import (
    compute_spatiotemporal_covariance,
    compute_directional_covariance,
    calculate_theoretical_covariance,
    fit_covariance_model,
)

# Threshold-based indicators
from environmentaltools.spatiotemporal.indicators import (
    fractional_exceedance_area,
    mean_exceedance_over_total_area,
    mean_excess_over_total_area,
    mean_exceedance_over_exceedance_area,
    mean_excess_over_exceedance_area,
    exceedance_to_nonexceedance_ratio,
    # compute_all_indicators_and_plot,
    # Advanced spatiotemporal indicators
    mean_presence_boundary,
    maximum_influence_extent,
    threshold_exceedance_frequency,
    permanently_affected_zone,
    mean_representative_value,
    return_period_extreme_value,
    spatial_change_rate,
    functional_area_loss,
    critical_boundary_retreat,
    neighbourhood_mean,
    neighbourhood_gradient_influence,
    environmental_convergence,
    neighbourhood_polarization,
    local_persistence,
    environmental_risk,
    directional_influence,
    multivariate_neighbourhood_synergy,
    spatiotemporal_coupling,
    multivariate_threshold_exceedance,
    directional_coevolution,
    multivariate_persistence,
)

# Multi-criteria decision analysis
from environmentaltools.spatiotemporal.multicriteria import (
    run_topsis_mcda,
    create_weights_visualization,
    create_topsis_maps,
)

# Raster analysis

from environmentaltools.spatiotemporal.raster import (
    check_inputs,
    post_treatment,
    binary_matrix,
    analysis,
    calculate_temporal_differences,
    save_results,
    load_results,
)

__all__ = [
    # BME estimation
    'compute_bme_moments',
    'estimate_local_mean_bme',
    'calculate_moments',
    'integrate_moment_vector',
    'apply_data_smoothing',
    'perform_cross_validation',
    'select_neighbours',
    'estimate_bme_regression',
    'create_design_matrix',
    'smooth_data',
    'create_spatiotemporal_matrix',
    'coordinates_to_distance',
    'coordinates_to_distance_angle',
    'find_pairs_by_distance',
    
    # Covariance
    'compute_spatiotemporal_covariance',
    'compute_directional_covariance',
    'calculate_theoretical_covariance',
    'fit_covariance_model',
    
    # Indicators
    'fractional_exceedance_area',
    'mean_exceedance_over_total_area',
    'mean_excess_over_total_area',
    'mean_exceedance_over_exceedance_area',
    'mean_excess_over_exceedance_area',
    'exceedance_to_nonexceedance_ratio',
    # 'compute_all_indicators_and_plot',
    # Advanced spatiotemporal indicators
    'mean_presence_boundary',
    'maximum_influence_extent',
    'threshold_exceedance_frequency',
    'permanently_affected_zone',
    'mean_representative_value',
    'return_period_extreme_value',
    'spatial_change_rate',
    'functional_area_loss',
    'critical_boundary_retreat',
    'neighbourhood_mean',
    'neighbourhood_gradient_influence',
    'environmental_convergence',
    'neighbourhood_polarization',
    'local_persistence',
    'environmental_risk',
    'directional_influence',
    'multivariate_neighbourhood_synergy',
    'spatiotemporal_coupling',
    'multivariate_threshold_exceedance',
    'directional_coevolution',
    'multivariate_persistence',
    
    # Multi-criteria
    'run_topsis_mcda',
    'create_weights_visualization',
    'create_topsis_maps',
    
    # Raster
    'check_inputs',
    'post_treatment',
    'binary_matrix',
    'analysis',
    'calculate_temporal_differences',
    'save_results',
    'load_results',
]


