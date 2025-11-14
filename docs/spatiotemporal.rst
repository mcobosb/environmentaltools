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

Basic Indicators
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   fractional_exceedance_area
   mean_exceedance_over_total_area
   mean_excess_over_total_area
   mean_exceedance_over_exceedance_area
   mean_excess_over_exceedance_area
   exceedance_to_nonexceedance_ratio

Spatiotemporal Extent Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   mean_presence_boundary
   maximum_influence_extent
   threshold_exceedance_frequency
   permanently_affected_zone
   mean_representative_value

Extreme Value and Risk Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   return_period_extreme_value
   environmental_risk
   functional_area_loss
   critical_boundary_retreat

Spatial Dynamics Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   spatial_change_rate
   directional_influence
   environmental_convergence

Neighborhood Analysis
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   neighbourhood_mean
   neighbourhood_gradient_influence
   neighbourhood_polarization
   local_persistence

Multivariate Analysis
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   multivariate_neighbourhood_synergy
   spatiotemporal_coupling
   multivariate_threshold_exceedance
   directional_coevolution
   multivariate_persistence

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

Functions for processing spatiotemporal raster data, including configuration management,
binary matrix generation for threshold exceedances, input validation, and NetCDF output creation.

**Capabilities:**

* Configuration file validation and loading
* Temporal difference analysis for change detection
* Input validation and preprocessing
* Post-treatment data preparation and refinement
* Binary matrix generation for threshold analysis
* Temporal aggregation (annual, seasonal)
* NetCDF format output with metadata

.. currentmodule:: environmentaltools.spatiotemporal.raster

.. autosummary::
   :toctree: _autosummary

   calculate_temporal_differences
   check_inputs
   post_treatment
   binary_matrix
   analysis
   save_results
   load_results