Models Module
=============

.. automodule:: environmentaltools.models
   :no-index:

The models module groups the set-up, execution and reading of numerical models.

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
