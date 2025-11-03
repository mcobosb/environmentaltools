"""Common module for environmental data utilities.

This module provides essential utilities for data input/output operations,
data manipulation, model configuration, and general-purpose functions used
across the environmentaltools package.

Submodules:
    read: Functions for reading various file formats (CSV, Excel, NetCDF, etc.)
    save: Functions for saving data to various formats
    load: Functions for loading model outputs and configurations
    write: Functions for writing model input files (SWAN, CSHORE, COPLA)
    utils: General utility functions for data analysis and transformations
"""

# Read functions
from .read import (
    keys_as_int,
    keys_as_nparray,
    read_json,
    read_pde,
    csv,
    npy,
    xlsx,
    netcdf,
    ascii_tiff,
    kmz,
    shp,
    mat,
    pdf,
)

# Save functions
from .save import (
    npy2json,
    to_json,
    to_csv,
    to_npy,
    to_xlsx,
    cwriter,
    formats,
    to_esriascii,
    as_float_bool,
    to_geotiff,
    to_txt,
    to_shp,
    to_netcdf,
)

# Load functions
from .load import (
    create_mesh_dictionary,
    cshore_config,
    read_cshore,
    read_copla,
    read_swan,
    delft_raw_files_point,
    delft_raw_files,
)

# Write functions
from .write import (
    write_cshore_input,
    write_swan_input,
    write_copla_input,
    create_project_directory,
)


# Utility functions
from .utils import (
    max_moving_window,
    gaps,
    nonstationary_ecdf,
    best_params,
    ecdf,
    nonstationary_epdf,
    epdf,
    acorr,
    bidimensional_ecdf,
    bias_adjustment,
    probability_mapping,
    empirical_cdf_mapping,
    rotate_geo2nav,
    uv_to_magnitude_angle,
    optimize_rbf_epsilon,
    rbf_error_metric,
    outliers_detection,
    scaler,
    string_to_function,
    data_over_threshold,
    extract_isolines,
    pre_ensemble_plot,
    smooth_1d,
    find_nearest_point,
    date_to_julian,
    mean_dt_param,
    rmse,
    maximum_absolute_error,
    mean_absolute_error,
    xrnearest,
    latslons_values,
    find_indexes,
    create_lat_lon_matrix,
    coords_name,
)


__all__ = [
    # Read functions
    "keys_as_int",
    "keys_as_nparray",
    "read_json",
    "read_pde",
    "csv",
    "npy",
    "xlsx",
    "netcdf",
    "ascii_tiff",
    "kmz",
    "shp",
    "mat",
    "pdf",
    # Save functions
    "npy2json",
    "to_json",
    "to_csv",
    "to_npy",
    "to_xlsx",
    "cwriter",
    "formats",
    "to_esriascii",
    "as_float_bool",
    "to_geotiff",
    "to_txt",
    "to_shp",
    "to_netcdf",
    # Load functions
    "create_mesh_dictionary",
    "cshore_config",
    "read_cshore",
    "read_copla",
    "read_swan",
    "delft_raw_files_point",
    "delft_raw_files",
    # Write functions
    "write_cshore_input",
    "write_swan_input",
    "write_copla_input",
    "create_project_directory",
    # Utility functions
    "max_moving_window",
    "gaps",
    "nonstationary_ecdf",
    "best_params",
    "ecdf",
    "nonstationary_epdf",
    "epdf",
    "acorr",
    "bidimensional_ecdf",
    "bias_adjustment",
    "probability_mapping",
    "empirical_cdf_mapping",
    "rotate_geo2nav",
    "uv_to_magnitude_angle",
    "optimize_rbf_epsilon",
    "rbf_error_metric",
    "outliers_detection",
    "scaler",
    "string_to_function",
    "data_over_threshold",
    "extract_isolines",
    "pre_ensemble_plot",
    "smooth_1d",
    "find_nearest_point",
    "date_to_julian",
    # Additional utilities
    "mean_dt_param",
    "rmse",
    "maximum_absolute_error",
    "mean_absolute_error",
    "xrnearest",
    "latslons_values",
    "find_indexes",
    "create_lat_lon_matrix",
    "coords_name",
]
