Common module
=============

The **common** module provides essential utilities for data input/output operations, 
data manipulation, and general-purpose functions used across the environmentaltools package.

.. automodule:: environmentaltools.common
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


.. currentmodule:: environmentaltools.common

Data Reading
------------

read module
~~~~~~~~~~~

Functions for reading various file formats including Excel, CSV, NetCDF, and other data sources.

.. autosummary::
   :toctree: _autosummary

   keys_as_int
   keys_as_nparray
   read_json
   read_pde
   csv
   npy
   xlsx
   netcdf
   ascii_tiff
   dap_ascii
   kmz
   shp
   mat
   pdf

Data Saving
-----------

save module
~~~~~~~~~~~

Functions for saving processed data to various formats.

.. autosummary::
   :toctree: _autosummary

   npy2json
   to_json
   to_csv
   to_npy
   to_xlsx
   cwriter
   formats
   to_esriascii
   as_float_bool
   to_geotiff
   to_txt
   to_shp
   to_netcdf

Model I/O
---------

The readers and writers of model files (SWAN, COPLA, CSHORE, Delft3D) moved to
:doc:`models`. The old names (``environmentaltools.common.read_swan``, ...) still
work but are deprecated and will be removed in version 4.0.

Statistical Utilities
---------------------

utils module
~~~~~~~~~~~~

General utility functions for statistical analysis, data transformations, and bias correction.

.. autosummary::
   :toctree: _autosummary

   max_moving_window
   gaps
   ecdf
   nonstationary_ecdf
   epdf
   nonstationary_epdf
   best_params
   acorr
   bidimensional_ecdf
   bias_adjustment
   probability_mapping
   empirical_cdf_mapping
   rotate_geo2nav
   uv_to_magnitude_angle
   optimize_rbf_epsilon
   rbf_error_metric
   outliers_detection
   scaler
   string_to_function
   data_over_threshold
   extract_isolines
   pre_ensemble_plot
   smooth_1d
   find_nearest_point
   date_to_julian
   mean_dt_param
   rmse
   maximum_absolute_error
   mean_absolute_error
   xrnearest
   latslons_values
   find_indexes
   create_lat_lon_matrix
   coords_name
