Spectral Module
===============

.. automodule:: environmentaltools.spectral
   :no-index:

The spectral module provides tools for spectral and harmonic analysis of time series data,
including frequency domain analysis and tidal predictions.

Spectral Analysis Functions
----------------------------

.. currentmodule:: environmentaltools.spectral.analysis

.. autosummary::
   :toctree: _autosummary

   lombscargle_periodogram
   fast_fourier_transform

Harmonic (Tidal) Analysis
--------------------------

.. autosummary::
   :toctree: _autosummary

   harmonic_analysis
   reconstruct_tidal_level

Tidal Model Configuration
--------------------------

.. autosummary::
   :toctree: _autosummary

   configure_tidal_model

Time Series Generation
-----------------------

.. autosummary::
   :toctree: _autosummary

   generate_datetime_series_from_range
   generate_datetime_series
.. autofunction:: environmentaltools.spectral.analysis.generate_time_series

Tidal Predictions
-----------------

.. autofunction:: environmentaltools.spectral.analysis.interpolate_tidal_constants
.. autofunction:: environmentaltools.spectral.analysis.predict_tidal_elevations

Output
------

.. autofunction:: environmentaltools.spectral.analysis.save_results_to_csv

Tidal Reconstruction Workflow
------------------------------

.. autofunction:: environmentaltools.spectral.analysis.tidal_reconstruction_from_models
