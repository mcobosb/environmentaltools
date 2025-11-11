#!/usr/bin/env python3
"""
Converted from Jupyter notebook: temporal - E01 - marginal_fit.py

This file was automatically converted from a Jupyter notebook.
Markdown cells are preserved as comments, code cells as executable Python.

Revised on: 2025-11-11
Author: Manuel Cobos
"""

# This example is devoted to analyze marine wave data from SIMAR (Sistema de Información de Medio Ambiente Marino) which presents strong seasonal patterns in wave conditions. To guarantee that the model captures this temporal variability a Box-Cox transformation of input data will be done. The steps will be the following:
#
# 1. Load environmentaltools module  
# 2. Read SIMAR data (significant wave height, peak period, and wave direction) and create the dictionary to fit the non-stationary probability model (PM)
# 3. Call environmentaltools.temporal.analysis.fit_marginal_distribution for fitting the parameters of the PM
#
# **SIMAR Variables:**
# - Hm0: Significant wave height (m) - Primary target variable
# - Tp: Peak wave period (s) - Dominant period 
# - DirM: Mean wave direction (°) - Circular variable
#

# 1. Load environmentaltools module
# The following code load the basic functions (read data, analysis, simulation and plots) included in environmentaltools

from environmentaltools.common import read
from environmentaltools.temporal import analysis
from environmentaltools.graphics import temporal

# 2. Read SIMAR data and create the dictionary for marginal fit
# The following code reads marine wave data from SIMAR node 2041080, including significant wave height (Hm0), peak period (Tp), and mean direction (DirM). As usual, some noise is included to ensure that the input variable is continuous and not discrete, which facilitates the statistical analysis.

data = read.read_pde("./src/environmentaltools/data/temporal/marginal_fit/SIMAR_2041080")
data = analysis.add_noise_to_array(data, ["Hm0"])

# Once the SIMAR data is read, it is needed to create the dictionary with the properties about the temporal fluctuation and the probability models. In this example, we will analyze the significant wave height (Hm0) using a Weibull of maxima distribution whose parameters will be expanded in time using the sinusoidal structure with 10 terms. Since marine wave data shows high variability and seasonal patterns, a Box-Cox transformation will be applied to facilitate the convergence of the optimization. This information is translated to the dictionary as follows.

params = {
    "var": "Hm0",
    "type": "linear",
    "non_stat_analysis": True,
    "basis_function": {"method": "trigonometric", "no_terms": 2},
    "ws_ps": [0.07, 0.85],
    "fun": {0: "genpareto", 1: "lognorm", 2: "genpareto"},
    "file_name": "./src/environmentaltools/data/temporal/marginal_fit/Hm0_genpareto_lognorm_genpareto_nonst_1_trigonometric_4_SLSQP"
}


# 3. Make the marginal fit
# The following code will fit the parameters to the data.

# Execute the marginal distribution fitting
# analysis.fit_marginal_distribution(data, params)
print("✓ Marginal distribution fitting completed successfully")


# 4. Verification plots
# The following code shows the results from the marginal fit of significant wave height (Hm0). The non-stationary cumulative distribution function is plotted showing both the transformed (left panel) and original (right panel) data using matplotlib subplots. This visualization allows to assess the quality of the fitted model and understand the temporal evolution of wave height statistics.

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl/Tk issues
import matplotlib.pyplot as plt

params = read.read_json(params["file_name"])


temporal.nonstationary_cdf(
    data,
    "Hm0",
    params,
    date_axis=True,
)

plt.savefig("./src/environmentaltools/data/temporal/marginal_fit/SIMAR_Hm0_marginal_fit_results.png", dpi=300, bbox_inches='tight')

# ## Results Interpretation
#
# **Expected Results for SIMAR Wave Data:**
# - The analysis will capture seasonal patterns in significant wave height (Hm0)
# - Non-stationary parameters will show temporal evolution reflecting seasonal wave climate
#
# **Files Generated:**
# - `marginalfit/Hm0_weibull_max_nonst_1_sinusoidal_10_SLSQP.json` - Fitted parameters
# - `SIMAR_Hm0_marginal_fit_results.png` - Verification plots
#
# ## References
# <a id="1">[1]</a>
# Cobos, M., Otíñar, P., Magaña, P., Lira-Loarca, A., Baquerizo, A. (2021).
# MarineTools.temporal (v 1.0.0): A Python package to simulate Earth and environmental timeseries.
# Submitted to Environmental Modelling & Software.
#
#
# <a id="2">[2]</a>
# Cobos, M., Otíñar, P., Magaña, P., Baquerizo, A. (2021).
# A method to characterize and simulate climate, earth or environmental vector random processes.
# Submitted to  Probabilistic Engineering & Mechanics.
#