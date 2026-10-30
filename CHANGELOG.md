# Changelog

All relevant project updates are documented here:

## [Unreleased]
### Added
- new `models` module: `models.common` (Delft3D/regular grids, process execution) and `models.swan` (YAML-driven SWAN projects: grids, nesting, batch cases and variants, resumable runner, readers of tables/spectra/PRINT, plots) (potinar)
- `models.delft3d` (raw outputs, NEFIS `.dat` extraction, grid and wave field plots), `models.cshore`, `models.coastalme` and `models.swan_copla` (SWAN → COPLA → Kobayashi workflow) (potinar)
- `processes.kobayashi_transport`: Kobayashi et al. (2008) transport as a pure function of the 2-D fields, following the implementation of CSHORE (`SEDTRA`, oblique waves) in a frame aligned with the local waves (potinar)
- `models.swan` follows the workflow of `models.barrier`: `prepare` builds every case in memory and only writes what changed (a changed sea state or variant updates its cases, changed grids or points update all of them; their results become pending), writes `cases_overview.xlsx` (cases with their status, variants side by side) and keeps the results when `overwrite` rewrites identical cases; variants in an Excel/csv table (`variants_file`: one row per variant, dotted columns such as `physics.wcap`, `off` removes a command); `states: sheet`; `postprocess` writes `points.xlsx`, `convergence.xlsx` and map figures of chosen cases, only when there are new results; command line `python -m environmentaltools.models.swan {prepare,run,postprocess,all} swan.yaml`; the executable can be a list of candidates and `SWAN_EXE` takes precedence; example `models - E01 - swan_workflow` (potinar)
- `models.common.tables` (tables of changes, Excel output, fingerprints) and `models.common.resolve_executable`, shared by the model workflows (potinar)
- `models.barrier`: cases in an Excel/csv table (`cases_file`: one row per case, one column per change of the base case written with dots, e.g. `along_channel.nmann.scale`, `hydrodynamics.courant_number`, `files.tides.csv`; empty cell = base case); `prepare` creates the table with examples and a legend when it is missing; `read_cases_table`, `write_cases_table`, `cases_to_table` (potinar)
- `models.barrier` workflow driven by one project file for any channel: `prepare` detects cases changed in the list or a changed base case, creates them again and marks their result as pending, draws the section parameters of each case and writes `cases_overview.xlsx` (base case and cases side by side); `postprocess` processes only new results (series at points, figures, comparison with a reference case or NetCDF); command line `python -m environmentaltools.models.barrier {prepare,run,postprocess,all} barrier.yaml`; the executable can be a list of candidates (one per computer) and `work_dir: true` uses the system temporary folder; example `models - E02 - barrier_workflow` (potinar)
- "which model to use" guide in the docs and docstrings of `models` (potinar)
- `processes.bed_slope`: bed slope components on regular, rotated or curvilinear grids (potinar)
- `models.barrier`: BARRIER 1-D estuarine model for any channel: cases from a base case with config and along-channel parameter changes (constant, interpolated, stepwise, Gaussian-smoothed), boundary and tributary files, cross-section monotonicity check and fix, optional YAML project with variants, resumable runner, NetCDF reader with dates and distance to the mouth, series at points, tidal mean, comparison of runs and figures; validated against the Guadalquivir calibration of M. Cobos (potinar)

### Changed
- `models.barrier`: the list of cases of the project file is `cases` (`variants` still accepted); the environment variable `BARRIER_EXE` takes precedence over the configured executable (potinar)
- model readers/writers/runners moved from `common`, `processes` and `graphics` to `models`; the old names are deprecated aliases (removal in 4.0) (potinar)
- `processes` split by topic: `waves`, `coastal`, `hydrology`, `properties` (`processes.compute` kept for compatibility) (potinar)
- Delft3D plots read `.grd` files with `models.common.read_delft_grd` (potinar)

### Fixed
- `nesting` called non-existent `write.copla`/`copla` and `run_swan` without the executable (potinar)
- `read_copla` discarded the `grid` passed to it (potinar)
- `kobayashi_transport` works in a local cross-shore frame (normal to the isobaths, in the sense of the waves), as CSHORE; it takes the bed slope components `Sbx`, `Sby`; validated against CSHORE runs (example in 12_TESTs/cshore_perfil_simple) (potinar)
- `cshore.write_input` followed neither the number of title lines nor the conditional lines of CSHORE (`IWTRAN`, `IPOND`, `INFILT`, `SLPOT`, `RWH`) and glued values to labels; `cshore.read_output` read only the first output block and mislabelled `OYMOME`; it now returns every time step (`time` argument) with plain column names (potinar)
- Kobayashi transport compared with CSHORE: current normalized by the orbital velocity σT (was not), σT from the local Hrms (was the boundary Hs), breaker height (extra division by the wavelength), breaking dissipation coefficient ≥ 1, `Ps <= Pb`, bottom friction dissipation `(fb/2)·GF·σT³` (was `0.5·g·fb·U³`), breaking instead of breaking minus roller dissipation, bedload factor `(0.5 + Qb)`, bounded slope function, transport across the waves; results change (potinar)
- `hydraulic_radius` failed (undefined `df`); trapezoidal, triangular, circular and parabolic sections used the rectangular formulas. `water_elevation` uses the same geometries (potinar)
- `river_sediment_transport`: Yang failed (undefined `h`, wrong mask, second term dropped, natural instead of common logarithms, no result); Brownlie gave NaN (sign of the exponent of the critical Shields parameter, `np.std` of the diameters instead of `sigma_g`, no zero below the critical velocity) and accepts `"brownlie"`; Wilcock-Crowe ignored a given `Fs`, divided instead of multiplied to get the reference shear stress and skipped the last size class; `.loc` instead of chained assignment (silently ignored with pandas copy-on-write) (potinar)
- `settling_velocity` with `Wu_Wang` failed (undefined `d`, `d_ast`) (potinar)
- `coastline_evolution` called the non-existent `utils.nearest` (potinar)

## [3.2.7] - 2026-04-29
### Added
- fix range for upper tail threshold (potinar)

## [3.2.6] - 2026-04-27
### Added
- added environment.yml file (potinar)
- added func for download marine copernicus data (potinar)
- update nonstationary functions to fix bug with circular variables (potinar)

## [3.2.5] - 2025-12-09
### Added
- bug in temporal simulations transforming into degrees circular variables (mcobosb)
- bug creating the output directory for simulations (mcobosb)

## [3.2.0] - 2025-11-27
### Added
- added func for raster indicators (mcobosb)
- added example for presence_boundary (mcobosb)
- added example for influence_extent (mcobosb)
- added example for directional_influence (mcobosb)
- added draft for contributors.md (mcobosb)

## [3.1.2] - 2025-11-14
### Added
- version structure to versem (mcobosb)

## [v2026.3.1.1] - 2025-11-14
### Added
- clarifying version structure (mcobosb)
- rotation 90 heatmap (mcobosb)
- spatiotemporal funct (mcobosb)

## [v2026.3.0] - 2025-11-13
### Added
- added func for raster indicators (mcobosb)

## [v2026.2.5] - 2025-11-12
### Changed
- configuration of package installation (mcobosb)
- handling cartopy import known errors (mcobosb)
- update TODO list (mcobosb)
- added indicator plot (mcobosb)
- remove internal config load (mcobosb)

## [v2026.2.4] - 2025-11-11
### Changed
- update version 2026.2.4 (mcobosb)
- update docs in test-pypi (mcobosb)
- add packages to remove RTD error (mcobosb)
- suppress toc warning (mcobosb)
- update version 2026.2.0 (mcobosb)
- update TODO list (mcobosb)
- udpate examples (mcobosb)
- update functionality (mcobosb)
- update docs and requirements (mcobosb)
- update functions for downloading and drone missions (mcobosb)
- update remove author version at ini file (mcobosb)
- update functions, download and drone examples (mcobosb)
- update downloaded datasets (mcobosb)

## [v2026.2.0] - 2025-11-10
### Changed
- update TODO (mcobosb)
- update drone docs and nb (mcobosb)
- update data for drone examples (mcobosb)
- update alphabetic for dependencies (mcobosb)
- update dron examples (mcobosb)
- update dron functionality (mcobosb)

## [v2026.1.0] - 2025-11-05
### Changed
- update documentation (mcobosb)
- update links to doc in toml file (mcobosb)
- update publish-to-test (mcobosb)
- update raster (mcobosb)
- update todos (mcobosb)
- update to publish (mcobosb)
- update publish-to-test.pypi (mcobosb)
- update for documentation (mcobosb)
- update for documentation (mcobosb)
- update publish test-pypi and conf file ofr RTD (mcobosb)
- update for documentation in RTD (mcobosb)
- update for documentation in RTD (mcobosb)
- update docs file RTD (mcobosb)
- update toml and readme (mcobosb)
- update test-pypi.yml (mcobosb)
- update spatiotemporal functions (mcobosb)
- update docs spatiotemporal (mcobosb)
- update loguru (mcobosb)
- update gitignore (mcobosb)
- update logging to loguru (mcobosb)

## [v2026.0.0] - 2025-11-04
### Changed
- updated bug importing function (mcobosb)
- remove examples in docs and some functions (mcobosb)
- update check for circular variables (mcobosb)

## [v2025.2.0] - 2025-11-03
### Added
- update pyproject.toml (mcobosb)
- updated spatiotemporal functions (mcobosb)
- updated documentation of spatiotemporal (mcobosb)
- Create first battery of examples (mcobosb)
- upload first battery of tests (mcobosb)
- upload temporal package (mcobosb)
- upload spectral package (mcobosb)
- upload spatiotemporal package (mcobosb)
- upload spatial package (mcobosb)
- upload processes package (mcobosb)
- upload graphics package (mcobosb)
- upload download package (mcobosb)
- upload common package (mcobosb)
- include the first battery of test (mcobosb)

## [v2025.1.0] - 2025-10-31
### Changed
- update TODO (mcobosb)
- udpate dependencies (mcobosb)
- added readthedocs file (mcobosb)
- added sphinx documentation (mcobosb)
- update dependencies (mcobosb)
- ignore models folder (mcobosb)
- update modules and latex support for graphics (mcobosb)
- Add ignore to build_ doc files (mcobosb)
- update readme file (mcobosb)
- update readme file (mcobosb)
- update readme file (mcobosb)
- update readme file (mcobosb)
- update configuration and TODO files (mcobosb)
- update configuration and readme files (mcobosb)

## [v2025.0.0] - 2025-10-27
### Added
- Initial commit (Manuel Cobos)