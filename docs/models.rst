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
