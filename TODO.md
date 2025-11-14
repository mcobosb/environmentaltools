🧩 General
- [X] Define pyproject.toml with modular extras
- [X] Add installation instructions for partial environments on README.md
- [X] Include module table and usage examples in README.md
- [X] Prepare conda-forge recipe for staged-recipes submission
- [X] Link project to readthedocs
- [ ] Project to pypi

📦 pyproject.toml
- [X] Update common dependencies

🧪 Examples
- [ ] Run examples and check that works fine
- [ ] Revise examples, input data, results and output graphics
- [ ] Reduce size of input data for examples
- [ ]   · temporal - E05 - multivariate_simulations.py
- [ ]   · temporal - E06 - multivariate_ensemble_simulation.py
- [ ]   · temporal - E07 - maximum_dissimilitude_analysis.py
- [ ]   · temporal - E08 - copula_fit_bivariate_distributions.py
- [ ]   · temporal - E09 - bootstrapping_with_l_moments.py
- [ ]   · temporal - E10 - pot_with_l_moments.py
- [ ]   · temporal - E11 - annual_maxima_regime.py
- [ ]   · spatiotemporal - E05 - bayesian_maximum_entropy.py
- [ ]   · spatiotemporal - E06 - anisotropic_spatiotemporal_covariance.py
- [ ]   · processes - E01 - wave_height_with_swan_model.py
- [ ]   · processes - E02 - currents_with_copla_model.py
- [ ]   · processes - E03 - currents_with_cshore_model.py
- [ ]   · processes - E04 - sediment_transport_Kobayashi.py
- [ ]   · processes - E05 - coastal_equilibrium_plan_shape.py
- [ ]   · processes - E06 - waves_zero_upcrossing_method.py
- [ ]   · processes - E07 - calculate_wave_reflection.py
- [ ]   · processes - E08 - precipitation_to_runoff_SCS.py
- [ ]   · processes - E09 - compute_hydraulic_radius.py
- [ ]   · processes - E10 - water_elevation_from_manning.py
- [ ]   · processes - E11 - river_sediment_transport.py
- [ ]   · processes - E12 - storm_surge_from_waves.py
- [ ]   · processes - E13 - flood_fill_algorithm.py
- [ ]   · spectral - E01 - lombscargle_periodogram.py
- [ ]   · spectral - E02 - fast_fourier_transform.py
- [ ]   · spectral - E03 - harmonic_analysis_utide.py
- [ ]   · spectral - E04 - tidal_reconstruction_eot20.py
- [ ]   · spatial - E01 - voronoi_diagram.py
- [ ]   · spatial - E02 - mesh_triangulation.py
- [ ]   · spatial - E03 - coastline_from_imagery.py
- [ ]   · risk - E01 - compute_economic_index.py


## 📂 Module-specific TODOs

### risk
- [ ] Add damage curves
- [ ] Add functionality por computing risk on maps

### download.google-earth-engine
- [ ] Generalize the function for others satellite products

### spatiotemporal.raster.analysis
- [X] Check that level series files exist
- [X] Validate that max_level has data for all months and years
- [X] Implement additional pre-treatment steps as required
- [ ] Add 


### temporal
- [X] Update statistical_fit and its dependencies
- [X] Update initialization message in analysis.py - Line 30
- [X] Update docstring for simulation function - Line 62 in simulation.py
- [X] Update docstring for _summary_ function - Line 441 in simulation.py
- [ ] Implement non-normal multivariate analysis (currently only normal distribution) - Line 523 in simulation.py
- [ ] Review value of 1e-6 subtraction in CDF to avoid 1.0 values - Line 355 in regimes.py
- [ ] Review peaks selection function for POT analysis - Line 398 in regimes.py
- [ ] Check groupby count with monthly average weights - Line 221 in analysis.py
- [ ] Check logic that doesn't make much sense - Line 616 in analysis.py
- [ ] Verify that True option works correctly - Line 621 in analysis.py
- [ ] Implement handling for mixed functions - Line 959 in analysis.py
- [ ] Modify function for storm separation (should not fill gaps) - Line 1025 in analysis.py
- [ ] Remove temporary fix for calm period indices (waiting for Pedro's fix) - Line 1124 in analysis.py
- [ ] Modify for more refined and understandable version - Line 1527 in statistical_fit.py
- [ ] Update reconstruction methods of classification "rbf-multiquadric", "rbf-inverse", "rbf-gaussian", 
        "rbf-linear", "rbf-cubic", "rbf-quintic", "rbf-thin_plate",
        "gp-rbf", "gp-exponential", "gp-quadratic", "gp-white".

### graphics
- [ ] label <= or >= in windrose plot
- [X] heatmap plot rotation to 90 degrees


### common
- [ ] Implement nearest neighbor function separately - Line 361 in read.py
- [ ] Change implementation for more than one variable in dataframe conversion - Line 362 in read.py
- [ ] Enable multi-page reading - Line 707 in read.py
- [ ] Include morphology options - Line 111 in load.py
- [ ] Load paths from a file instead of hardcoding - Line 25 in cme.py
- [ ] Improve parameter extraction for order 1 Fourier series - Line 396 and 1430 in auxiliar.py
- [ ] Fix discontinuity limitation - Line 481 and 1515 in auxiliar.py
- [ ] Separate by Fourier order - Line 504 and 1538 in auxiliar.py
- [ ] Generalize for more than two functions - Line 606 and 1640 in auxiliar.py
- [ ] Verify calculation - Line 864 in auxiliar.py
- [ ] Change hardcoded value 51 to a target value - Line 1352 in auxiliar.py


### spectral
- [ ] Reduce harmonic analysis and reconstruction using pyTMD

## 📂 Test
- [ ] Functionality tests to be passed before any update
- [ ] Integration tests