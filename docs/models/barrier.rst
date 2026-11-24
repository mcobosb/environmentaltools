BARRIER: user guide
===================

.. currentmodule:: environmentaltools.models.barrier

This page explains how to set up, run and analyse BARRIER with
``environmentaltools.models.barrier``. The reference of every function is in
:doc:`../models` (section *BARRIER*).

.. contents:: On this page
   :local:
   :depth: 1


What it does and limitations
----------------------------

BARRIER (https://github.com/mcobosb/Barrier, M. Cobos, GPL-3.0) is a 1-D
estuarine model written in C++. It solves the cross-section averaged
Saint-Venant equations (TVD MacCormack scheme, wetting and drying) coupled with
the transport of salinity and, optionally, temperature and sediment, along a
channel described by cross-section tables. The density gradient (baroclinic
term) can be included in the momentum equation.

Use it for tidal propagation and salt intrusion along an estuary or a tidal
river over weeks to years: tidal amplitude and phase along the channel,
discharge, salinity at intakes, effect of river discharge or of changes in the
channel (dredging, roughness).

Limitations:

- **1-D, cross-section averaged.** No lateral or vertical structure:
  stratification, secondary currents, lateral salinity gradients or flow over
  flood plains in plan are not resolved. Lateral storage is represented only by
  the storage factor ``Sf``.
- **One channel.** No networks or branches; tributaries enter as point
  discharges (lateral inflows).
- **Geometry from tables.** The channel is what ``cross-sections.csv`` and
  ``along_channel_data.csv`` say. Building them from a channel axis and a DEM is
  not covered by this repository yet: for now they come from an existing case.
- **The section average hides processes** that need to be parameterised: salt
  intrusion depends on the longitudinal dispersion ``Kh``, which is a
  calibration parameter, not a result.

Conventions used everywhere in the module:

- ``x`` (m) grows from the **upstream boundary** (``x = 0``, first section) to
  the **mouth** (last section, downstream boundary).
- ``km`` is the distance to the mouth, ``(x_max − x) / 1000``: ``km 0`` is the
  mouth. Points, stations and figures use ``km``.


Executable and installation
---------------------------

**The Python side.** The module is part of ``environmentaltools`` and needs
nothing beyond its base dependencies (``numpy``, ``pandas``, ``xarray``,
``netCDF4``, ``matplotlib``, ``scipy``, ``openpyxl``, ``pyyaml``).

**BARRIER.** It is compiled from its repository with CMake. It needs a C++20
compiler, ``netcdf`` and ``yaml-cpp`` (found with ``pkg-config``) and,
optionally, OpenMP::

    # macOS
    brew install cmake netcdf yaml-cpp libomp pkg-config
    # Ubuntu / WSL
    sudo apt-get install build-essential cmake pkg-config libnetcdf-dev libyaml-cpp-dev libomp-dev

    git clone https://github.com/mcobosb/Barrier.git
    cd Barrier
    cmake -B build -DCMAKE_BUILD_TYPE=ReleaseFast
    cmake --build build -j 4
    # the executable is build/barrier

Use one build folder per computer: a binary built on WSL/Linux does not run on
macOS and vice versa.

**Where the module looks for it** (:func:`barrier_executable`), in this order:

1. the environment variable ``BARRIER_EXE``;
2. ``executable`` in the project file: a path, or a list of paths (one per
   computer; the first that exists is used), e.g.
   ``executable: [/Users/me/Barrier/build-mac/barrier, /home/me/Barrier/build/barrier]``;
3. ``barrier`` on the ``PATH``.

Several builds (e.g. two numerical schemes) can coexist in one project:
``executables: {maccormack: [...], conservative: [...]}`` and a column
``executable`` in the cases table with the name to use in each case. Those cases
ignore ``BARRIER_EXE``.

BARRIER is called as ``barrier /path/to/case/config.yaml`` and writes its output
in the folder it is run from; :func:`run_case` takes care of that.


Inputs
------

A **case** is a folder with BARRIER's ``config.yaml`` and the files it names
(paths relative to that folder). All tables are CSV, comma-separated, with a
header line that starts with ``;`` (lines starting with ``;`` or ``#`` are
comments). Read and write them with :func:`read_csv` and :func:`write_csv`.

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - File (key in ``config.yaml``)
     - Content
     - How to get it with the repository
   * - ``along_channel_data.csv`` (``geometry.along_channel_file``)
     - One row per section: ``x`` (m), ``z`` bed (m), ``nmann`` (s/m\ :sup:`1/3`),
       ``xutm``, ``yutm`` (m), ``AngMd``, ``AngMi`` (bank angles, radians),
       ``beta``, ``Sf`` (≥ 1, optional), ``kh`` (m²/s, optional).
     - From an existing case. Change the parameters with
       :func:`set_section_parameter` / :func:`section_profile`, or from the
       cases table (``along_channel.<column>.<option>``).
   * - ``cross-sections.csv`` (``geometry.cross_sections_file``)
     - Several rows per section (same ``x``), one per water depth ``eta``
       (m above the bed): ``x, z, eta, B, A, P, Rh, sigma, xl, xr, beta``
       (``barrier.CROSS_SECTION_COLUMNS``).
     - From an existing case. Check and correct it with
       :func:`check_cross_sections` and :func:`fix_cross_sections`.
   * - Level at the mouth, e.g. ``tides.csv``
       (``hydrodynamics.boundary_conditions.downstream.file``)
     - ``seconds since run.start_date, level (m)``, same vertical datum as
       ``z``.
     - Measured series, or harmonic prediction:
       :func:`environmentaltools.spectral.harmonic_analysis <environmentaltools.spectral.analysis.harmonic_analysis>` +
       :func:`environmentaltools.spectral.reconstruct_tidal_level <environmentaltools.spectral.analysis.reconstruct_tidal_level>`, or a global
       tide model with
       :func:`environmentaltools.spectral.tidal_reconstruction_from_models <environmentaltools.spectral.analysis.tidal_reconstruction_from_models>`
       (check its units and datum). Write it with :func:`write_series`.
   * - Discharge at the upstream boundary, e.g. ``dam_discharge.csv``
       (``hydrodynamics.boundary_conditions.upstream.file``)
     - ``seconds, Q (m³/s)``.
     - Gauging or dam release data; :func:`write_series`.
   * - Tributaries (``hydrodynamics.forcing.tributaries_file``)
     - Number of hydrographs, then for each one its UTM position and
       ``seconds, Q`` rows. It enters at the closest section.
     - :func:`write_tributaries`.
   * - Salinity at the boundaries, initial salinity
       (``transport.salinity...``)
     - A constant (``value``) or a series file; ``initial_file`` for a
       non-uniform initial state.
     - Constant ocean salinity, or a series (e.g. a buoy, or Copernicus Marine
       via :doc:`environmentaltools.download <../download>`); :func:`write_series`.
   * - Temperature, heat flux, sediment (``transport.temperature``,
       ``transport.sediment``)
     - Series and properties files (only if those modules are enabled).
     - :func:`write_series` for the series.

:func:`input_files` lists the files named by a ``config.yaml``.

**Dated series → BARRIER.** BARRIER wants seconds since ``run.start_date``.
:func:`write_series` converts a series indexed by date, checks that times grow
and warns if it starts after the start of the run::

    start = barrier.start_date(barrier.read_config("base_case"))
    barrier.write_series(tide, "base_case/tides.csv", start=start, names=("time_seconds", "level_m"))

**The** ``config.yaml``. Main sections (see the minimal example below and the
BARRIER repository for every option):

- ``run``: ``name``, ``start_date``, ``end_date`` (or ``duration`` in s),
  ``timestep`` (output interval, s), ``output_variables`` (a list or ``full``),
  ``log_level``, ``continue_simulation`` / ``continue_netcdf_path`` (hot start
  from a previous output).
- ``geometry``: the two geometry files.
- ``hydrodynamics``: ``initial_conditions.type`` (``calm``: level 0 and Q = 0),
  ``boundary_conditions`` (upstream ``discharge`` / ``reflective`` / ``free`` /
  ``elevation``; downstream ``elevation`` / ``reflective`` / ``free``),
  ``forcing``, ``courant_number``, ``tvd_limiter``, ``dry_bed``,
  ``manning_eta``, ``lateral_storage``…
- ``transport``: ``salinity`` (``enabled``, ``dispersion_kh``, ``beta``,
  ``baroclinic``, boundary conditions), ``temperature``, ``sediment``,
  ``density``.
- ``smoothing``: optional smoothing of the bathymetry and of the solution.


Preparing cases and running
---------------------------

The workflow runs from one **project file** (start from
``environmentaltools/models/barrier/barrier_template.yaml``, available as
``barrier.TEMPLATE``):

.. code-block:: yaml

   base_case: base_case            # BARRIER's config.yaml + its CSV files
   cases_file: cases.xlsx          # one row per case, one column per change
   cases_dir: cases                # what BARRIER reads, one folder per case
   results_dir: results            # NetCDF, log, console output, done.json
   postprocess_dir: postprocess    # series and figures
   executable: null                # or a path / list of paths (one per computer)
   work_dir: null                  # true when results_dir is synced (Synology, OneDrive) or on /mnt/c
   postprocess:
     points: {Mouth: 0, Gauge A: 20}
     reference: {case: reference}

**The cases.** Each case is the base case plus changes. They go in the cases
table (:func:`read_cases_table`; the first ``prepare`` creates an example with a
legend sheet) or under ``cases:`` in the project file. An empty cell means "as
in the base case". Columns:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Column
     - Meaning
   * - ``case``
     - Name of the case and of its folder.
   * - ``description``, ``_anything``
     - Free text, ignored.
   * - ``hydrodynamics.courant_number``, ``run.end_date``…
     - Any key of ``config.yaml`` written with dots.
   * - ``along_channel.<column>.<option>``
     - Section parameters (``nmann``, ``kh``, ``Sf``, ``beta``). Options:
       ``value``, ``scale``, ``add``, ``points`` (``[[x, value], …]``),
       ``steps`` (``[[x_from, value], …]``), ``sigma`` (Gaussian smoothing, in
       sections), ``file`` (csv ``x, value``), ``coordinate`` (``x`` or ``km``).
   * - ``files.<name in the case>``
     - File that replaces that input, e.g. ``files.tides.csv = tides_2010.csv``.
   * - ``executable``
     - Name of a build of ``executables`` (or a path).

For example, ``along_channel.kh.scale = [[0, 0.7], [20, 0.7], [22, 1], [105, 1]]``
with ``along_channel.kh.coordinate = km`` multiplies Kh by 0.7 in the first
20 km from the mouth and leaves it unchanged beyond km 22.

**The three steps**, from Python or from the terminal::

    from environmentaltools.models import barrier

    project = barrier.load_project("barrier.yaml")
    barrier.prepare(project)       # create/update the case folders + cases_overview.xlsx
    barrier.run(project)           # run the cases without result
    barrier.postprocess(project)   # series, figures, comparison with the reference

.. code-block:: bash

   python -m environmentaltools.models.barrier prepare barrier.yaml
   python -m environmentaltools.models.barrier run barrier.yaml [--case NAME] [--again]
   python -m environmentaltools.models.barrier postprocess barrier.yaml
   python -m environmentaltools.models.barrier all barrier.yaml

- :func:`prepare` copies the base case into ``cases_dir/<case>`` and applies the
  changes. It writes ``variant.json`` and ``base_signature.txt`` to know when a
  case is out of date (its row, the base case or a replacement file changed),
  ``section_parameters.png`` (n, Kh, Sf and β against the base case) and
  ``cases_overview.xlsx`` (every ``config.yaml`` value and input file side by
  side, differences in orange). **Review the overview before running.** It
  also warns about settings known to give a wrong result without an error
  (:func:`check_config`).
- :func:`run` runs the cases without ``done.json``, one after the other, and
  keeps ``results_dir/run_log.csv``. A run is accepted only if it writes a new
  NetCDF with time steps, readable in full and without NaN.
- :func:`postprocess` processes the new results only.

Every step only does what is pending: change a row and the same three calls
create, run and postprocess that case again, leaving the rest untouched. Folders
of cases removed from the list are reported but not deleted.

The pieces can be used alone: :func:`create_case` (one case folder from a base
case) and :func:`run_case` (one run).


Key parameters and calibration
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Parameter
     - Where
     - Effect
   * - Manning ``nmann``
     - ``along_channel_data``
     - Friction: damping and delay of the tide along the channel. First
       parameter of the level calibration. ``manning_eta: true`` makes it vary
       with the level.
   * - Storage factor ``Sf`` (≥ 1)
     - ``along_channel_data``
     - Lateral storage (marshes, side areas that store water without carrying
       momentum): slows and damps the tide. Ignored (``Sf = 1``) unless
       ``hydrodynamics.lateral_storage: true``.
   * - Dispersion ``kh`` (m²/s)
     - ``along_channel_data`` (or ``transport.salinity.dispersion_kh``, constant)
     - Salt intrusion length. First parameter of the salinity calibration.
       ``kh_discharge_exponent``, ``kh_reference_discharge``,
       ``kh_discharge_window_hours`` and ``kh_discharge_max`` make it depend on
       the river discharge: ``Kh(x, t) = Kh(x)·(Qm(t)/Qref)^m``.
   * - ``beta``
     - ``along_channel_data``, ``cross-sections``
     - Momentum (Boussinesq) coefficient; usually 1.
   * - ``courant_number``
     - ``hydrodynamics``
     - Stability and run time. Lower it if the solution blows up (0.15 in the
       Guadalquivir calibration).
   * - ``tvd_limiter``
     - ``hydrodynamics``
     - Flux limiter (``minmod``, ``vanleer``…). Keep it enabled (see
       *Common problems*).
   * - ``inflow_mixing_alpha``
     - ``transport.salinity.boundary_conditions.downstream``
     - Salinity that enters on the flood: a mix of the ocean value and the
       interior one.

**Calibration with the workflow.** Each trial is a row of the cases table, e.g.
``along_channel.nmann.scale = 1.1`` or a ``kh`` profile by stretches with
``coordinate = km``. With ``postprocess.reference`` every case is compared with
a reference run (:func:`compare`: RMSE and maximum difference per variable and
per section). Comparing with **measurements** is done outside the workflow:
:func:`extract_points` gives the model series at the gauges (by distance to the
mouth), ready to compare with the observed ones, e.g. with
:func:`tidal_mean` for the subtidal signal.

A usual order: first levels (``nmann``, ``Sf``) without salinity
(``transport.salinity.enabled: false`` is faster), then salinity (``kh``,
``inflow_mixing_alpha``) with the calibrated hydrodynamics.


Outputs
-------

In ``results_dir/<case>/``:

- ``<name>_barrier_sim_<date>_CS<sections>_T<days>d_dt<s>_BC<types>_<options>.nc``:
  the NetCDF output (``SAL``, ``TVD``, ``DRY``, ``CFL<courant×100>`` in the name
  tell the options used). The full ``config.yaml`` is stored inside
  (``yaml_configuration``), and the start date in the global attribute
  ``time_offset``.
- ``barrier_console.txt``: the screen output (first place to look when a case
  fails); the log file, if ``log_level`` > 0.
- ``done.json``: marks the case as finished (build used, minutes).

Variables (dimensions ``(time, x)`` unless noted; which ones are written is set
by ``run.output_variables``):

.. list-table::
   :header-rows: 1
   :widths: 16 54 30

   * - Variable
     - Meaning
     - Units
   * - ``time``, ``x``
     - Seconds since ``time_offset``; distance from the upstream boundary
     - s, m
   * - ``eta``
     - Free surface elevation (same datum as ``z`` and the tide)
     - m
   * - ``level``
     - Water depth
     - m
   * - ``A``, ``Rh``
     - Wetted area, hydraulic radius
     - m², m
   * - ``Q``, ``U``
     - Discharge (positive towards the mouth), mean velocity
     - m³/s, m/s
   * - ``q``
     - Lateral inflows
     - m³/s
   * - ``S``, ``T``, ``rho``
     - Salinity, temperature, density
     - psu, °C, kg/m³
   * - ``Qb``, ``Qs``, ``Qt``
     - Bed-load, suspended and total sediment transport
     - m³/s
   * - ``xl``, ``xr``; ``xl_utm_*``, ``xr_utm_*``
     - Left and right banks (distance to the axis; UTM coordinates)
     - m
   * - ``n``, ``Sf``, ``Kh`` (dimension ``x``)
     - Manning n, storage factor and dispersion used
     - s/m\ :sup:`1/3`, –, m²/s

:func:`read_output` opens the newest output of a folder with ``xarray`` and adds
the coordinates ``date`` (on ``time``) and ``km`` (on ``x``) and, when the bank
coordinates are written, ``W``: the surface width between banks (m), consistent
with ``A`` (see :func:`bank_width`).


Postprocessing, data extraction and figures
-------------------------------------------

:func:`postprocess` writes, for each finished case, in ``postprocess_dir/<case>/``:

- ``timeseries.xlsx``: one sheet per variable of ``postprocess.variables`` (one
  column per point, named ``"<name> (km <km of the section used>)"``), the sheet
  ``points`` (section used, UTM of its centre) and ``forcings`` (discharge at the
  upstream boundary and level at the mouth, read from the output), plus a legend;
- ``timeseries_<variable>.png``: raw series and tidal mean (25 h) at each point;
- ``along_channel_<variable>.png``: the variable in every section against time,
  with contour lines (default: salinity, 1, 5, 10 and 20 psu);
- with a ``reference``: ``vs_<reference>.xlsx`` (RMSE and maximum difference
  per variable and per section, tolerances in ``postprocess.tolerances``) and
  ``vs_<reference>.png``.

The same from code, for any output::

    ds = barrier.read_output("results/reference")                 # folder or .nc
    points, series = barrier.extract_points(ds, {"Mouth": 0, "Gauge": 20}, ["eta", "S"])
    series["S"].to_csv("salinity_points.csv")
    sub = barrier.tidal_mean(series["S"], 25, barrier.time_step_hours(ds))

    outputs = barrier.read_project(project)                       # {case: Dataset}
    summary, per_section = barrier.compare(outputs["reference"], outputs["rough"])

    barrier.plot_point_series(series["eta"], "level (m)", file_name="eta.png")
    barrier.plot_along_channel(ds, "S", file_name="salinity.png")
    barrier.plot_comparison(outputs["reference"], outputs["rough"], stations_km=[0, 5, 10])
    along_base = barrier.read_csv("base_case/along_channel_data.csv")
    along_case = barrier.read_csv("cases/rough/along_channel_data.csv")
    barrier.plot_section_parameters({"base": along_base, "rough": along_case})

Any other analysis works on the ``xarray.Dataset`` directly, e.g. the salt
intrusion length as the largest ``km`` where the tidal-mean salinity exceeds
1 psu.


Coupling
--------

BARRIER is not coupled online with other models of the repository. The links
are through files:

- **Upstream of BARRIER (forcings).** Tide at the mouth from
  :doc:`environmentaltools.spectral <../spectral>` (harmonic analysis of a gauge or a global
  tide model), ocean salinity or sea level from Copernicus Marine
  (:doc:`environmentaltools.download <../download>`), discharges and tributaries from
  gauging data. All of them go through :func:`write_series` /
  :func:`write_tributaries`, which handle the time reference.
- **Downstream of BARRIER (results for other analyses).** Series at points
  (``timeseries.xlsx`` or :func:`extract_points`) for intakes, ecology or
  statistics (:doc:`environmentaltools.temporal <../temporal>`); the bank positions in UTM
  (``xl_utm_*``, ``xr_utm_*``) to draw flooded widths on a map.
- **In time.** ``run.continue_simulation`` and ``run.continue_netcdf_path``
  start a run from the last state of a previous output (hot start), e.g. to
  chain a spin-up period and the period of analysis.
- **With 2-D models.** There is no automatic link with SWAN, Delft3D or COPLA.
  A 2-D model of the mouth can provide the level series at the downstream
  boundary, written with :func:`write_series`.


Validation and minimal example
------------------------------

**Validation of the module.** The tests (``tests/test_models_barrier.py``) do
not need BARRIER: they check the CSV round trip, the profiles of section
parameters, the creation and update of cases, the cases table, the readers,
the comparison and the figures (the run is replaced by a fake executable). The
module was also validated against the Guadalquivir calibration of M. Cobos
(2026-02-26, 806 sections): :func:`fix_cross_sections` reproduces his corrected
geometry (maximum difference 1e-12), :func:`extract_points` and :func:`compare`
reproduce the series and RMSE of the project scripts, and a case written by
:func:`create_case` gives bit-identical BARRIER v0.13 results.

**Minimal example.** A 20 km rectangular channel, 300 m wide and 5 m deep,
with an M2 tide of 1 m at the mouth, 50 m³/s from upstream and ocean salinity
36 psu. Two cases: the base case and Manning n × 1.2. It writes everything
(base case, project file) and runs the whole workflow in a few seconds:

.. literalinclude:: barrier_files/minimal_example.py
   :language: python

Expected result (BARRIER, MacCormack build of November 2026): the tide reaches
the head with a slight amplification (±1.1 m at km 10), the salinity stays
between 0 and 36 psu, and the rougher case differs from the reference by up to
8 cm in level (RMSE 4 cm) and 0.66 psu in salinity (RMSE)::

        case status  rmse_eta     rmse_Q   rmse_S     rmse_A
   reference   done       NaN        NaN      NaN        NaN
       rough   done  0.040025  29.053917  0.65889  12.007401

Run it to see the figures: ``postprocess/rough/vs_reference.png`` (level over
the first days and tidal-mean salinity at four stations, both cases) and
``postprocess/<case>/along_channel_S.png`` (tidal-mean salinity along the
channel; the 25-h window leaves the first and last 12 h empty).


Common problems
---------------

**"BARRIER wrote no valid output (return code 0)".** BARRIER ends with "Run
successfully end" and code 0 even when it could not read its configuration.
Open ``results/<case>/barrier_console.txt``. Typical causes:

- ``YAML parsing error: invalid node; first invalid key: "forcing"``: the
  section ``hydrodynamics.forcing`` must exist, even without tributaries
  (``forcing: {tributaries_file: null}``). The same happens without
  ``hydrodynamics.initial_conditions`` (``initial_conditions: {type: calm}``).
- A header line without ``;`` in an input series: BARRIER stops instead of
  reading it as (0, 0). Write the files with :func:`write_csv` /
  :func:`write_series`.

**The tide does not enter the channel** (the level changes only at the
boundary sections, "Total computation time: 0.00 s"): the TVD limiter is
disabled, which is BARRIER's default. Set
``hydrodynamics.tvd_limiter.enabled: true``. ``prepare`` warns about it
(``Warning: hydrodynamics.tvd_limiter.enabled is not true … (cases: …)``).

**"BARRIER could not be started (Exec format error)"**: the executable was
built for another system (e.g. a Linux build from WSL on macOS). The case is
marked as failed and ``run`` goes on with the next one. Give one path per
computer in ``executable`` (list) or set ``BARRIER_EXE``.

**"No BARRIER executable"**: neither ``BARRIER_EXE``, ``executable`` nor
``barrier`` on the ``PATH`` exist.

**"has NaN from t = … h on (the solution blew up)"**: lower
``courant_number``, enable ``dry_bed`` if sections dry out, check the geometry
(:func:`check_cross_sections`) and abrupt changes of ``nmann``/``kh`` (smooth
them with ``sigma``).

**"output is damaged (some time steps cannot be read)"**: the NetCDF was
written in a synced folder (Synology, OneDrive) or on ``/mnt/c`` from WSL. Set
``work_dir: true`` in the project file.

**Results change by decimetres after editing an input "without changing
anything"**: a table was rounded on the way (Excel, ``float_format``).
:func:`read_csv` / :func:`write_csv` keep every digit; do not round the
geometry.

**A new column of** ``along_channel_data`` **has no effect, or changes the
wrong parameter**: BARRIER reads that file **by position**
(``x, z, nmann, xutm, yutm, AngMd, AngMi, beta, Sf, kh``), not by name. Adding
``kh`` to a file without ``Sf`` puts Kh in the place of ``Sf``. Keep every
column in that order.

**The interpolation of the geometry jumps between depths**: the cross-section
tables are not monotonic. :func:`check_cross_sections` lists the violations,
:func:`fix_cross_sections` corrects them.

**A case is run again although "nothing changed"**: its row, the base case or a
replacement file (``files.*``) changed; ``prepare`` prints why. Cases removed
from the list keep their folders until you delete them.
