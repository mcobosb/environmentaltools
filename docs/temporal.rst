Temporal Module
===============

.. automodule:: environmentaltools.temporal
   :no-index:

The subpackage *temporal* package aimed at providing users with a friendly, general code to statistically characterize a vector random process (RP) to obtain realizations of it. It is implemented in Python - an interpreted, high-level, object-oriented programming language widely used in the scientific community - and it makes the most of the Python packages ecosystem. Among the existing Python packages, it uses Numpy, which is the fundamental package for scientific computing in Python [1]_, SciPy, which offers a wide range of optimization and statistics routines [2]_. Pandas [3]_ to analyse and manipulate data.

The tools implemented in the package named *temporal* allow to capture the statistical properties of a **non stationary (NS) vector RP** by using **compound or piecewise parametric PMs** to properly describe all the range of values and to **simulate uni- or multivariate time series** with the same random behavior. The statistical parameters of the distributions are assumed to depend on time and are expanded into a Generalized Fourier Series (GFS) [4]_ in order to reproduce their NS behavior. The applicability of the present approach has been illustrated in several works with different purposes, among others: (i) the observed wave climate variability in the preceding century and expected changes in projections under a climate change scenario [5]_; (ii) the optimal design and management of an oscillating water column system [6]_ [7]_, (iii) the planning of maintenance strategies of coastal structures [8]_, (iv) the analysis of monthly Wolf sunspot number over a 22 year period [4]_, and (v) the simulation of estuarine water conditions for the management of the estuary [9]_.

Analysis Functions
------------------

Marginal Fitting
~~~~~~~~~~~~~~~~

.. currentmodule:: environmentaltools.temporal

.. autosummary::
   :toctree: _autosummary

   fit_marginal_distribution
   check_marginal_params
   add_noise_to_array
   look_models

Storm Analysis
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   storm_series
   storm_properties

Dependencies
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   dependencies
   check_dependencies_params
   fit_var_model
   varfit_OLS
   ensemble_dt

Indicators and Confidence
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   iso_indicators
   confidence_bands
   generate_outputfilename

Statistical Fitting Functions (Core)
-------------------------------------

Stationary Analysis
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   stationary_analysis
   fit_distribution

Non-Stationary Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   nonstationary_analysis
   nonst_fit
   fourier_initialization
   fourier_expansion
   initial_params
   matching_lower_bound

Distribution Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   fit
   negative_log_likelihood
   get_params
   ppf
   cdf
   transform
   inverse_transform
   numerical_cdf_pdf_at_n
   ensemble_cdf
   ensemble_ppf
   params_t_expansion

Simulation Functions
--------------------

.. autosummary::
   :toctree: _autosummary

   simulation
   check_parameters
   var_simulation
   class_seasons

Regime Analysis Functions
--------------------------

Confidence Intervals and Bootstrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   confidence_intervals
   bootstrapping

Probability Models
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   probability_model_fit
   l_mom
   lmom_genpareto

Extreme Value Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   pot_method
   au2
   fit_gpd_bootstrap
   annual_maxima_method

Classification Functions
-------------------------

Storm Classification
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   class_storm_seasons
   classification

MDA and Reconstruction
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   maximum_dissimilarity_algorithm
   reconstruction
   regression
   normalize

Copula Analysis
---------------

Bivariate Dependence Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: environmentaltools.temporal.Copula
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Event Detection
~~~~~~~~~~~~~~~

.. autofunction:: environmentaltools.temporal.extreme_events
.. autofunction:: environmentaltools.temporal.events_duration
.. autofunction:: environmentaltools.temporal.values_over_threshold

Interpolation
~~~~~~~~~~~~~

.. autofunction:: environmentaltools.temporal.interpolation_series
.. autofunction:: environmentaltools.temporal.interpolation_boundaries_index
.. autofunction:: environmentaltools.temporal.interpolation_nearest

Event Analysis
~~~~~~~~~~~~~~

.. autofunction:: environmentaltools.temporal.extreme_indexes
.. autofunction:: environmentaltools.temporal.near_events

References
----------

.. [1] Harris, Charles R. and Millman, K. Jarrod and van der Walt, Stéfan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J. and Kern, Robert and Picus, Matti and Hoyer, Stephan and van Kerkwijk, Marten H. and Brett, Matthew and Haldane, Allan and Fernández del Río, Jaime and Wiebe, Mark and Peterson, Pearu and Gérard-Marchant, Pierre and Sheppard, Kevin and Reddy, Tyler and Weckesser, Warren and Abbasi, Hameer and Gohlke, Christoph and Oliphant, Travis E. (2020). Array programming with {NumPy}. *Nature*.

.. [2] Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and Haberland, Matt and Reddy, Tyler and Cournapeau, David and Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and Bright, Jonathan and {van der Walt}, Stéfan J. and Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and Kern, Robert and Larson, Eric and Carey, C J and Polat, Ilhan and Feng, Yu and Moore, Eric W. and {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and Harris, Charles R. and Archibald, Anne M. and Ribeiro, Antonio H. and Pedregosa, Fabian and {van Mulbregt}, Paul and {SciPy 1.0 Contributors} (2020). {{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in Python}. *Nature Methods*.

.. [3] McKinney, Wes and others (2010). Data structures for statistical computing in python. *Proceedings of the 9th Python in Science Conference*.

.. [4] Cobos, M. and Otíñar, P. and Magaña, P. and Baquerizo, A. (2021). A method to characterize and simulate climate, Earth or environmental vector random processes. Submitted to *Probabilistic Engineering and Mechanics*.

.. [5] Lira-Loarca, Andrea Lira and Cobos, Manuel and Besio, Giovanni and Baquerizo, Asunción (2021). Projected wave climate temporal variability due to climate change. *Stochastic Environmental Research and Risk Assessment*.

.. [6] Jalón, María L and Baquerizo, Asunción and Losada, Miguel A (2016). Optimization at different time scales for the design and management of an oscillating water column system. *Energy*.

.. [7] López-Ruiz, Alejandro and Bergillos, Rafael J and Lira-Loarca, Andrea and Ortega-Sánchez, Miguel (2018). A methodology for the long-term simulation and uncertainty analysis of the operational lifetime performance of wave energy converter arrays. *Energy*.

.. [8] Lira-Loarca, Andrea and Cobos, Manuel and Losada, Miguel Ángel and Baquerizo, Asunción (2020). Storm characterization and simulation for damage evolution models of maritime structures. *Coastal Engineering*.

.. [9] Cobos, Manuel (2020). A model to study the consequences of human actions in the Guadalquivir River Estuary. Tesis Univ. Granada.