Processes Module
================

.. automodule:: environmentaltools.processes
   :no-index:

The processes module provides physical formulations of waves, coastal and
hydrological processes and the properties of sea water and sediment.

Model set-up, execution and outputs (SWAN, COPLA, CSHORE, CoastalME, Delft3D)
moved to :doc:`models`. The old names (``environmentaltools.processes.read_swan``,
``run_swan``, ``nesting``, ``write_cshore``, ...) still work but are deprecated
and will be removed in version 4.0.

.. currentmodule:: environmentaltools.processes

Waves
-----

.. autosummary::
   :toctree: _autosummary

   frequency_limits
   wave_number
   clsquare_s
   calculate_wave_reflection
   closure_depth
   zero_cross
   storm_surge_from_waves

Coastal Processes
-----------------

.. autosummary::
   :toctree: _autosummary

   slopes
   bed_slope
   kobayashi_transport
   sediment_transport_CERC
   equilibrium_plan_shape
   coastline_evolution
   flood_fill

``sediment_transport_Kobayashi(j, data, grid, params)`` is deprecated; use
:func:`kobayashi_transport`, which takes the fields explicitly.

Hydrology
---------

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
   river_sediment_transport

Physical Properties
-------------------

.. autosummary::
   :toctree: _autosummary

   density
   kinematic_viscosity
   EOS_sea_water
   bulk_fluid_density
   fall_velocity
   settling_velocity
