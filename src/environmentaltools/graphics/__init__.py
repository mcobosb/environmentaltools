"""Graphics module for environmental data visualization.

This module provides comprehensive plotting and visualization tools for environmental data,
including time series, spatial data, statistical distributions, and regime analysis.

Submodules:
    temporal: Core plotting functions for time series and statistical analysis
    spatial: Spatial data visualization and mapping
    processes: Process-specific visualizations
    joint: Joint and conditional distribution plots
    spatiotemporal: Spatiotemporal covariance plots
    utils: Utility functions for plot handling
"""

# Core plotting functions
from .temporal import (
    mda,
    timeseries,
    storm_timeseries,
    test_normality,
    cadency,
    spectra,
    cdf,
    boxplot,
    qq,
    probplot,
    nonstationary_percentiles,
    nonstationary_qq_plot,
    scatter_error_dependencies,
    scatter,
    look_models,
    crosscorr,
    corr,
    joint_plot,
    bivariate_ensemble_pdf,
    bivariate_pdf,
    nonstationary_cdf,
    nonstat_cdf_ensemble,
    soujourn,
    nonstationary_cdf_ensemble,
    pdf_n_i,
    wrose,
    seasonalbox,
    ensemble_acorr,
    heatmap,
    qqplot,
    line_ci,
    annotate_heatmap,
    peaks_over_threshold,
    threshold_fits,
    dimensionless_fits,
    pot_lmom,
    annual_maxima_analysis,
    serie_peaks,
    serie_peaks_umbral,
)

# Spatial plotting functions
from .spatial import (
    plot_interps,
    plot_mesh,
    plot_profiles,
    onclick,
    plot_preview,
    plot_db,
    plot_ascifiles,
    plot_2d_plan_view,
    folium_map,
    flood_map,
    plot_quiver,
    coastline_ci,
    plot_voronoi_diagram,
    osm_image,
    calc_extent,
    image_spoof,
    include_Andalusian_coast,
    include_coastal_Andalusian_cities,
    include_seas,
)


# Process plots
from .processes import (
    pr2flow,
    transport_mode,
    cme_calibration,
)

# Joint distribution plots
from .joint import (
    dscatter,
    smooth1D,
    plot_conditional_regime,
    plot_copula
    
)

# Spatiotemporal plots

from .spatiotemporal import (
    covariance_comparison,
    anisotropic_spatiotemporal_covariance,
    era5_time_series_plot,
    plot_presence_boundary,
)

# Utility functions
from .utils import (
    enable_latex_rendering,
    show,
    handle_axis,
    labels,
)

__all__ = [
    # Core plotting
    "mda",
    "timeseries",
    "storm_timeseries",
    "test_normality",
    "cadency",
    "spectra",
    "cdf",
    "boxplot",
    "qq",
    "probplot",
    "nonstationary_percentiles",
    "nonstationary_qq_plot",
    "scatter_error_dependencies",
    "scatter",
    "look_models",
    "crosscorr",
    "corr",
    "joint_plot",
    "bivariate_ensemble_pdf",
    "bivariate_pdf",
    "nonstationary_cdf",
    "nonstat_cdf_ensemble",
    "soujourn",
    "nonstationary_cdf_ensemble",
    "pdf_n_i",
    "wrose",
    "seasonalbox",
    "ensemble_acorr",
    "heatmap",
    "qqplot",
    "line_ci",
    "annotate_heatmap",
    "peaks_over_threshold",
    "threshold_fits",
    "dimensionless_fits",
    # Spatial
    "plot_interps",
    "plot_mesh",
    "plot_profiles",
    "onclick",
    "plot_preview",
    "plot_db",
    "plot_ascifiles",
    "plot_2d_plan_view",
    "folium_map",
    "flood_map",
    "plot_quiver",
    "coastline_ci",
    "plot_voronoi_diagram",
    "osm_image",
    "calc_extent",
    "image_spoof",
    "include_Andalusian_coast",
    "include_coastal_Andalusian_cities",
    "include_seas",
    # Regimes
    "pot_lmom",
    "annual_maxima_analysis",
    "serie_peaks",
    "serie_peaks_umbral",
    # Processes
    "pr2flow",
    "transport_mode",
    "cme_calibration",
    # Joint distributions
    "dscatter",
    "smooth1D",
    "plot_conditional_regime",
    "plot_copula",
    # Spatiotemporal
    "covariance_comparison",
    "anisotropic_spatiotemporal_covariance",
    "era5_time_series_plot",
    "plot_presence_boundary",
    # Utils
    "enable_latex_rendering",
    "show",
    "handle_axis",
    "labels",
]
