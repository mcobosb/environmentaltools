"""Temporal module for time series analysis and simulation.

This module provides comprehensive tools for statistical characterization
and simulation of environmental time series, including:
- Marginal and joint distribution fitting
- Non-stationary analysis
- Extreme value analysis (POT and annual maxima)
- Time series simulation
- Classification and regime analysis
"""

# Analysis functions
from .analysis import (
    fit_marginal_distribution,
    check_marginal_params,
    add_noise_to_array,
    look_models,
    storm_series,
    storm_properties,
    dependencies,
    check_dependencies_params,
    fit_var_model,
    varfit_OLS,
    ensemble_dt,
    iso_indicators,
    confidence_bands,
    generate_outputfilename,
)

# Statistical fitting core functions
from .core import (
    stationary_analysis,
    fit_distribution,
    nonstationary_analysis,
    nonst_fit,
    fourier_initialization,
    fourier_expansion,
    initial_params,
    matching_lower_bound,
    fit,
    negative_log_likelihood,
    get_params,
    ppf,
    cdf,
    transform,
    inverse_transform,
    numerical_cdf_pdf_at_n,
    ensemble_cdf,
    ensemble_ppf,
    params_t_expansion,
)

# Simulation functions
from .simulation import (
    simulation,
    check_parameters,
    var_simulation,
    class_seasons,
)

# Regime analysis functions
from .regimes import (
    confidence_intervals,
    bootstrapping,
    probability_model_fit,
    l_mom,
    lmom_genpareto,
    pot_method,
    au2,
    fit_gpd_bootstrap,
    annual_maxima_method,
)

# Classification functions
from .classification import (
    class_storm_seasons,
    classification,
    maximum_dissimilarity_algorithm,
    reconstruction,
    regression,
    normalize,
)

# Copula analysis
from .copula import Copula

# Utility functions
from .utils import (
    extreme_events,
    events_duration,
    values_over_threshold,
    interpolation_series,
    interpolation_boundaries_index,
    interpolation_nearest,
    extreme_indexes,
    near_events,
)

__all__ = [
    # Analysis
    "fit_marginal_distribution",
    "check_marginal_params",
    "add_noise_to_array",
    "look_models",
    "storm_series",
    "storm_properties",
    "dependencies",
    "check_dependencies_params",
    "fit_var_model",
    "varfit_OLS",
    "ensemble_dt",
    "iso_indicators",
    "confidence_bands",
    "generate_outputfilename",
    # Statistical fitting
    "stationary_analysis",
    "fit_distribution",
    "nonstationary_analysis",
    "nonst_fit",
    "fourier_initialization",
    "fourier_expansion",
    "initial_params",
    "matching_lower_bound",
    "fit",
    "negative_log_likelihood",
    "get_params",
    "ppf",
    "cdf",
    "transform",
    "inverse_transform",
    "numerical_cdf_pdf_at_n",
    "ensemble_cdf",
    "ensemble_ppf",
    "params_t_expansion",
    # Simulation
    "simulation",
    "check_parameters",
    "var_simulation",
    "class_seasons",
    # Regimes
    "confidence_intervals",
    "bootstrapping",
    "probability_model_fit",
    "l_mom",
    "lmom_genpareto",
    "pot_method",
    "au2",
    "fit_gpd_bootstrap",
    "annual_maxima_method",
    "automatico_lmom_boot",
    # Classification
    "class_storm_seasons",
    "classification",
    "maximum_dissimilarity_algorithm",
    "reconstruction",
    "regression",
    "normalize",
    # Copula
    "Copula",
    # Utilities
    "extreme_events",
    "events_duration",
    "values_over_threshold",
    "interpolation_series",
    "interpolation_boundaries_index",
    "interpolation_nearest",
    "extreme_indexes",
    "near_events",
]
