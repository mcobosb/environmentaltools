Spatiotemporal module
=====================

.. automodule:: environmentaltools.spatiotemporal
   :no-index:

This module provides comprehensive tools for spatiotemporal analysis of environmental data,
with specialized functions for:

* **Bayesian Maximum Entropy (BME)** estimation and uncertainty quantification
* **Spatiotemporal covariance** modeling and fitting
* **Threshold-based indicators** for risk and impact assessment
* **Multi-criteria decision analysis** for spatial prioritization
* **Raster analysis** for binary matrix generation and preprocessing

Bayesian Maximum Entropy
-------------------------

The BME framework provides optimal spatiotemporal estimation by combining prior knowledge
with observational data (both exact and probabilistic). It's particularly useful for
environmental applications where data uncertainty must be quantified.

BME Estimation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Core functions for performing BME spatiotemporal estimation.

.. currentmodule:: environmentaltools.spatiotemporal.bme

.. autosummary::
   :toctree: _autosummary

   compute_bme_moments
   estimate_local_mean_bme
   perform_cross_validation

Support Functions
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   calculate_moments
   integrate_moment_vector
   apply_data_smoothing

Spatiotemporal Covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for computing empirical covariances and fitting theoretical spatiotemporal
covariance models (exponential, non-separable, directional).

.. currentmodule:: environmentaltools.spatiotemporal.covariance

.. autosummary::
   :toctree: _autosummary

   compute_spatiotemporal_covariance
   fit_covariance_model

Advanced Covariance Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   compute_directional_covariance
   calculate_theoretical_covariance

Utility Functions
~~~~~~~~~~~~~~~~~

Helper functions for data preparation, neighborhood selection, and coordinate transformations.

.. currentmodule:: environmentaltools.spatiotemporal.bme

.. autosummary::
   :toctree: _autosummary

   select_neighbours
   estimate_bme_regression
   create_design_matrix
   smooth_data
   create_spatiotemporal_matrix
   coordinates_to_distance
   coordinates_to_distance_angle
   find_pairs_by_distance

Threshold-Based Indicators
---------------------------

Functions for computing spatial indicators based on threshold exceedances. These indicators
are useful for flood risk assessment, pollution exposure analysis, and environmental impact
studies.

**Key Indicators:**

* **RAEH** - Ratio of Area Exceeding thresHold: Fraction of spatial domain exceeding threshold
* **MEW** - Mean Exceedance over Whole domain: Mean exceedance normalized by total area
* **MEDW** - Mean Excess Difference over Whole domain: Mean excess above threshold over total area
* **WMEW** - Weighted Mean Exceedance over exceedance area: Conditional mean given exceedance
* **WMDW** - Weighted Mean excess Difference: Conditional mean excess given exceedance
* **AEAN** - Area Exceeding to Area Non-exceeding: Ratio of exceedance to non-exceedance areas

.. currentmodule:: environmentaltools.spatiotemporal.indicators

.. autosummary::
   :toctree: _autosummary

   fractional_exceedance_area
   mean_exceedance_over_total_area
   compute_all_indicators_and_plot

Advanced Indicators
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   mean_excess_over_total_area
   mean_exceedance_over_exceedance_area
   mean_excess_over_exceedance_area
   exceedance_to_nonexceedance_ratio

Multi-Criteria Decision Analysis
---------------------------------

TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method for
spatial prioritization. Useful for site selection, restoration planning, and resource
allocation based on multiple environmental criteria.

**Features:**

* Multiple weighting schemes (equal, entropy, analytical hierarchy process)
* Comprehensive visualization (ranking maps, isolines, bar charts)
* Sensitivity analysis across weighting methods
* Statistical summaries and publication-ready outputs

.. currentmodule:: environmentaltools.spatiotemporal.multicriteria

.. autosummary::
   :toctree: _autosummary

   run_topsis_mcda
   create_weights_visualization
   create_topsis_maps

Raster Analysis
---------------

Functions for processing spatiotemporal raster data, including binary matrix generation
for threshold exceedances, input validation, and NetCDF output creation.

**Capabilities:**

* Input validation and preprocessing
* Binary matrix generation for threshold analysis
* Temporal aggregation (annual, seasonal)
* NetCDF format output with metadata

.. currentmodule:: environmentaltools.spatiotemporal.raster

.. autosummary::
   :toctree: _autosummary

   check_inputs
   pretratement
   binary_matrix
   main

Examples
--------

**BME Estimation Workflow**

.. code-block:: python

   import numpy as np
   from environmentaltools.spatiotemporal import (
       compute_spatiotemporal_covariance,
       fit_covariance_model,
       compute_bme_moments
   )
   import scipy.optimize as opt

   # Compute empirical covariance
   empcov, pairs, dists, distt = compute_spatiotemporal_covariance(
       dfh, dfs, 
       slag=np.linspace(0, 100, 20), 
       tlag=np.linspace(0, 30, 10)
   )

   # Fit theoretical model
   result = opt.minimize(
       fit_covariance_model,
       x0=[1.0, 20.0, 5.0, 0.1],
       args=(empcov, [dists, distt], 'exponentialST')
   )

   # Compute BME moments
   moments = compute_bme_moments(
       dfk, dfh, dfs, 
       'exponentialST', result.x,
       nmax=[50, 100], 
       dmax=[100, 30, 3], 
       order=[1, 1],
       options=[100, 3, 0.95], 
       path='./cache', 
       name='bme_run'
   )

**Threshold-Based Indicators**

.. code-block:: python

   from environmentaltools.spatiotemporal import (
       fractional_exceedance_area,
       mean_excess_over_exceedance_area
   )

   # Compute exceedance indicators
   thresholds, fractions = fractional_exceedance_area(data)
   thresholds, excess = mean_excess_over_exceedance_area(data)

   # Or compute all at once
   from environmentaltools.spatiotemporal import compute_all_indicators_and_plot
   compute_all_indicators_and_plot(moments)

**Multi-Criteria Analysis**

.. code-block:: python

   from environmentaltools.spatiotemporal import run_topsis_mcda

   # Run TOPSIS analysis with multiple criteria
   results = run_topsis_mcda(
       criteria_layers=['topo.tif', 'landuse.tif', 'distance.tif'],
       output_dir='./results',
       weights_methods=['equal', 'entropy', 'ahp']
   )



