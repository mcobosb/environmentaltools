Models Module
=============

.. automodule:: environmentaltools.models
   :no-index:

The models module groups the set-up, execution and reading of numerical models.

Which model to use
------------------

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Model
     - Use it for
     - Limitations
   * - SWAN (:mod:`~environmentaltools.models.swan`)
     - Spectral wave propagation in 2-D over real bathymetry (refraction,
       diffraction, nesting), batches of sea states.
     - Waves only: no currents or sediment transport.
   * - SWAN → COPLA → Kobayashi (:mod:`~environmentaltools.models.swan_copla`)
     - Sediment transport field in plan (2-D): longshore transport,
       circulation cells, effect of structures.
     - Depth-averaged currents: no undertow, so the offshore cross-shore
       transport of storms is underestimated. The bed is not updated (the
       result is the transport field; the bed change would need the sediment
       balance, ∂zb/∂t = −∇·q/(1−p)).
   * - CSHORE (:mod:`~environmentaltools.models.cshore`)
     - Cross-shore profiles: surf and swash zone, undertow, runup,
       overtopping and profile change (erosion, bars) during storms.
     - 1-D: each profile is solved independently assuming an alongshore
       uniform beach (∂/∂y = 0); no refraction/diffraction in plan, structures
       or circulation cells.
   * - BARRIER (:mod:`~environmentaltools.models.barrier`)
     - Tidal propagation and salt intrusion along an estuary or tidal river
       (1-D, cross-section averaged), over weeks to years.
     - No lateral or vertical structure (stratification, secondary
       currents); the channel is described by cross-section tables.

Both approaches are complementary. The sediment transport formula of
Kobayashi et al. (2008) used in SWAN → COPLA
(:func:`environmentaltools.processes.kobayashi_transport`) follows the
implementation of its authors in CSHORE; only the hydrodynamic fields come from
SWAN and COPLA instead of CSHORE.

SWAN
----

A SWAN project is described by one YAML file (start from
``environmentaltools/models/swan/swan_template.yaml``) and run in three steps::

    from environmentaltools.models import swan

    cfg = swan.load_config("swan.yaml")
    swan.prepare(cfg)              # grids, nesting points, one folder per case
    swan.run(cfg)                  # where SWAN is installed (resumable)
    points = swan.read_points(cfg) # one row per case, grid and output point
    swan.summary(cfg)              # convergence and errors

Features: curvilinear (Delft3D/RGFGRID ``.grd``/``.dep``) and regular grids,
nesting as in Delft3D-WAVE (2-D spectra on the whole perimeter of the nested
grid), uniform or along-boundary variable parametric boundaries, batches of sea
states × variants, resumable execution on local disk, readers of tables,
2-D spectra and PRINT diagnostics, and figures.

.. currentmodule:: environmentaltools.models.swan

.. autosummary::
   :toctree: _autosummary

   load_config
   prepare
   run
   read_points
   read_map
   read_table
   read_print
   read_spectra
   spectrum_parameters
   summary
   boundary_distances
   plot_grids
   plot_map

SWAN → COPLA → Kobayashi
------------------------

Coupled workflow on regular grids defined in a parameter dict: SWAN on a
global and a nested local grid, COPLA for the wave-induced currents and the
sediment transport of Kobayashi et al. (2008)
(:func:`environmentaltools.processes.kobayashi_transport`)::

    from environmentaltools.models import swan_copla

    params = swan_copla.create_mesh_dictionary("params.xlsx", "values")
    global_db = swan_copla.create_db(params, bathymetry, "global")
    local_db = swan_copla.create_db(params, bathymetry, "local")
    swan_copla.create_project_directory(params, sea_states, global_db, local_db)
    for k, label in enumerate(sea_states.index):
        grid = swan_copla.nesting(k, label, params, sea_states)

.. currentmodule:: environmentaltools.models.swan_copla

.. autosummary::
   :toctree: _autosummary

   create_mesh_dictionary
   create_mesh
   create_xarray
   create_db
   create_project_directory
   write_swan_input
   write_copla_input
   run_swan
   run_copla
   nesting
   read_swan
   read_copla
   save_db
   clean

BARRIER
-------

BARRIER (https://github.com/mcobosb/Barrier) is the 1-D estuarine model of
M. Cobos. The whole workflow runs from one project file (start from
``environmentaltools/models/barrier/barrier_template.yaml``): a base case (a
folder with BARRIER's ``config.yaml`` and its CSV files), the list of cases (the
base case plus changes), the folders, the executable and the postprocessing.
Section ``x = 0`` is the upstream boundary and the last section is the mouth,
so it works for any channel::

    from environmentaltools.models import barrier

    project = barrier.load_project("barrier.yaml")
    barrier.prepare(project)       # create/update the cases + overview to review them
    barrier.run(project)           # run the cases without result
    barrier.postprocess(project)   # series at points, figures, comparison with a reference

or from the terminal: ``python -m environmentaltools.models.barrier all barrier.yaml``.
Each step only does what is pending: a case changed in the list (or a changed
base case) is created, run and postprocessed again; the rest are kept. Example:
``examples/models - E02 - barrier_workflow.py``.

Building the geometry (``cross-sections.csv``, ``along_channel_data.csv``)
from a channel axis and a DEM is not covered yet.

Input files are written with every digit: rounding the geometry or the
along-channel data to 10 significant digits changes the water level by
decimetres.

.. currentmodule:: environmentaltools.models.barrier

.. autosummary::
   :toctree: _autosummary

   load_project
   prepare
   run
   postprocess
   cases_overview
   create_case
   run_case
   postprocess_output
   load_points
   read_config
   input_files
   read_csv
   write_csv
   write_series
   write_tributaries
   section_profile
   set_section_parameter
   check_cross_sections
   fix_cross_sections
   read_output
   read_project
   extract_points
   tidal_mean
   compare
   plot_point_series
   plot_along_channel
   plot_comparison
   plot_section_parameters

Delft3D
-------

.. currentmodule:: environmentaltools.models.delft3d

.. autosummary::
   :toctree: _autosummary

   extract_wave_dat
   raw_files
   raw_files_point
   plot_grids
   plot_wave_field

CSHORE
------

.. currentmodule:: environmentaltools.models.cshore

.. autosummary::
   :toctree: _autosummary

   default_config
   write_input
   run
   read_output

CoastalME
---------

.. currentmodule:: environmentaltools.models.coastalme

.. autosummary::
   :toctree: _autosummary

   run

Deprecated names
----------------

These names still work, emit a ``DeprecationWarning`` and will be removed in
version 4.0:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Old name
     - New name
   * - ``common.read_swan``, ``processes.read_swan``
     - ``models.swan_copla.read_swan``
   * - ``common.read_copla``, ``processes.read_copla``
     - ``models.swan_copla.read_copla``
   * - ``common.write_swan_input``, ``processes.write_swan``
     - ``models.swan_copla.write_swan_input``
   * - ``common.write_copla_input``, ``processes.write_copla``
     - ``models.swan_copla.write_copla_input``
   * - ``common.create_project_directory``, ``processes.directory``
     - ``models.swan_copla.create_project_directory``
   * - ``common.create_mesh_dictionary``, ``processes.create_mesh_dictionary``
     - ``models.swan_copla.create_mesh_dictionary``
   * - ``processes.create_db``, ``create_mesh``, ``create_xarray``
     - ``models.swan_copla.<same name>``
   * - ``processes.save_db``, ``clean``, ``nesting``
     - ``models.swan_copla.<same name>``
   * - ``processes.run_swan``, ``run_copla``
     - ``models.swan_copla.<same name>``
   * - ``common.cshore_config``
     - ``models.cshore.default_config``
   * - ``common.write_cshore_input``, ``processes.write_cshore``
     - ``models.cshore.write_input``
   * - ``common.read_cshore``, ``processes.read_cshore``
     - ``models.cshore.read_output``
   * - ``processes.run_cshore``
     - ``models.cshore.run``
   * - ``processes.run_coastalme``
     - ``models.coastalme.run``
   * - ``common.delft_raw_files``, ``processes.delft_raw_files``
     - ``models.delft3d.raw_files``
   * - ``common.delft_raw_files_point``, ``processes.delft_raw_files_point``
     - ``models.delft3d.raw_files_point``
   * - ``processes.extract_delft_wave_dat``
     - ``models.delft3d.extract_wave_dat``
   * - ``graphics.plot_delft_grids``
     - ``models.delft3d.plot_grids``
   * - ``graphics.plot_delft_wave_field``
     - ``models.delft3d.plot_wave_field``
   * - ``processes.sediment_transport_Kobayashi``
     - ``processes.kobayashi_transport``

Common utilities
----------------

.. currentmodule:: environmentaltools.models.common

.. autosummary::
   :toctree: _autosummary

   Grid
   read_delft_grd
   read_delft_dep
   find_executable
   library_environment
   run_process
   run_executable
