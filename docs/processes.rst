Processes Module
================

.. automodule:: environmentaltools.processes
   :no-index:

The processes module provides comprehensive tools for environmental model execution, data processing, and analysis of coastal, wave, and hydrological processes.

Data Loading Functions
----------------------

Model Output Readers
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: environmentaltools.processes

.. autosummary::
   :toctree: _autosummary

   create_mesh_dictionary
   read_cshore
   read_copla
   read_swan
   delft_raw_files_point
   delft_raw_files

Computation Functions
---------------------

Data Processing
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   create_db
   create_mesh
   create_xarray
   slopes
   save_db
   clean

Sediment Transport
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   sediment_transport_Kobayashi
   sediment_transport_CERC

Model Execution
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   nesting
   run_swan
   run_copla
   run_cshore
   run_coastalme

Coastal Morphology
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   equilibrium_plan_shape
   coastline_evolution

Hydrology
~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   precipitation_to_flow
   wet_soil
   dry_soil
   unit_hydrograph_model
   base_flow
   distribute_precipitation
   cumulative_by_events
   hydraulic_radius
   water_elevation
   settling_velocity
   river_sediment_transport
   storm_surge_from_waves
   flood_fill

Physical Properties
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   EOS_sea_water
   bulk_fluid_density

Wave Analysis Functions
-----------------------

Wave Parameters
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   frequency_limits
   wave_number
   calculate_wave_reflection
   closure_depth
   zero_cross

Spectral Analysis
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   clsquare_s

Sediment Properties
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   fall_velocity
.. autofunction:: environmentaltools.processes.density
.. autofunction:: environmentaltools.processes.kinematic_viscosity

File Writing Functions
----------------------

Model Input Writers
~~~~~~~~~~~~~~~~~~~

.. autofunction:: environmentaltools.processes.write_cshore
.. autofunction:: environmentaltools.processes.write_swan
.. autofunction:: environmentaltools.processes.write_copla
.. autofunction:: environmentaltools.processes.directory
