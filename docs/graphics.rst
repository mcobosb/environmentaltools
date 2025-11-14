Graphics module
===============

.. automodule:: environmentaltools.graphics
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


.. currentmodule:: environmentaltools.graphics

The graphics module provides comprehensive visualization tools for environmental data analysis.
It builds upon the scientific Python ecosystem, leveraging NumPy [1]_ for numerical computations,
SciPy [2]_ for statistical analysis, and Matplotlib [3]_ for creating publication-quality figures.

Core Plotting Functions
-----------------------

Time Series and Basic Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   enable_latex_rendering
   timeseries
   storm_timeseries
   cadency
   boxplot
   seasonalbox

Statistical Analysis Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   test_normality
   spectra
   cdf
   qq
   probplot
   qqplot
   corr
   ensemble_acorr
   crosscorr

Non-Stationary Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   nonstationary_percentiles
   nonstationary_qq_plot
   nonstationary_cdf
   nonstat_cdf_ensemble
   nonstationary_cdf_ensemble

Multivariate and Joint Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   scatter
   scatter_error_dependencies
   joint_plot
   bivariate_pdf
   bivariate_ensemble_pdf
   plot_copula
   heatmap
   annotate_heatmap

Model Analysis and Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   look_models
   line_ci
   pdf_n_i
   soujourn

Specialized Plots
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   mda
   wrose

Spatial Plotting Functions
---------------------------

Interpolation and Mesh Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   plot_interps
   plot_mesh
   plot_profiles
   plot_ascifiles

Interactive and Database Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   onclick
   plot_preview
   plot_db

Mapping and Geographic Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   folium_map
   flood_map
   osm_image
   calc_extent
   image_spoof

Regional Features
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   include_Andalusian_coast
   include_coastal_Andalusian_cities
   include_seas

Advanced Spatial Plots
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   plot_quiver
   coastline_ci
   plot_voronoi_diagram
   plot_2d_plan_view

Regime Analysis
---------------

Extreme Value Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   peaks_over_threshold
   threshold_fits
   dimensionless_fits
   pot_lmom
   annual_maxima_analysis
   serie_peaks
   serie_peaks_umbral

Process Visualization
---------------------

.. autosummary::
   :toctree: _autosummary

   pr2flow
   transport_mode
   cme_calibration

Joint Distribution Analysis
---------------------------

.. autosummary::
   :toctree: _autosummary

   dscatter
   smooth1D
   plot_conditional_regime

Spatiotemporal Analysis
-----------------------

.. autosummary::
   :toctree: _autosummary

   covariance_comparison
   anisotropic_spatiotemporal_covariance
   era5_time_series_plot
   plot_presence_boundary

Utility Functions
-----------------

.. autosummary::
   :toctree: _autosummary

   show
   handle_axis
   labels

References
----------

.. [1] Harris, Charles R. and Millman, K. Jarrod and van der Walt, Stéfan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J. and Kern, Robert and Picus, Matti and Hoyer, Stephan and van Kerkwijk, Marten H. and Brett, Matthew and Haldane, Allan and Fernández del Río, Jaime and Wiebe, Mark and Peterson, Pearu and Gérard-Marchant, Pierre and Sheppard, Kevin and Reddy, Tyler and Weckesser, Warren and Abbasi, Hameer and Gohlke, Christoph and Oliphant, Travis E. (2020). Array programming with {NumPy}. *Nature*.

.. [2] Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and Haberland, Matt and Reddy, Tyler and Cournapeau, David and Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and Bright, Jonathan and {van der Walt}, Stéfan J. and Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and Kern, Robert and Larson, Eric and Carey, C J and Polat, Ilhan and Feng, Yu and Moore, Eric W. and {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and Harris, Charles R. and Archibald, Anne M. and Ribeiro, Antonio H. and Pedregosa, Fabian and {van Mulbregt}, Paul and {SciPy 1.0 Contributors} (2020). {{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in Python}. *Nature Methods*.

.. [3] John D. Hunter. Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*.
