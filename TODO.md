🧩 General
- [ ] Define pyproject.toml with modular extras
- [ ] Add installation instructions for partial environments on README.md
- [ ] Include module table and usage examples in README.md
- [ ] Prepare conda-forge recipe for staged-recipes submission

📦 pyproject.toml
- [ ] Update common dependencies (os, time, pathlib, dotenv, typing, logging, datetime) - Line 34

🧪 Examples
- [ ] Run examples and check that works fine
- [ ] Revise examples, input data, results and output graphics


## 📂 Module-specific TODOs

### spatiotemporal.raster.analysis
- [ ] Check that level series files exist - Line 67
- [ ] Validate that max_level has data for all months and years - Line 67
- [ ] Implement additional pre-treatment steps as required - Line 227

### temporal
- [ ] Update initialization message in analysis.py - Line 30
- [ ] Update docstring for simulation function - Line 62 in simulation.py
- [ ] Update docstring for _summary_ function - Line 441 in simulation.py
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

### utils
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


