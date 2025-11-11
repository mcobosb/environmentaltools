#!/usr/bin/env python3
"""
Converted from Jupyter notebook: spatiotemporal - E01 - raster-indicators.ipynb

This file was automatically converted from a Jupyter notebook.
Markdown cells are preserved as comments, code cells as executable Python.
"""

# ------------------------------------------------------------
# Markdown Cell 1
# ------------------------------------------------------------
# # Spatiotemporal Indicators Examples
#
# This notebook demonstrates the usage of various spatiotemporal indicators available in the `environmentaltools` package. These indicators are designed to analyze environmental data patterns across space and time.
#
# ## Overview
#
# The indicators are organized into several categories:
# 1. **Boundary and Extent Indicators**: Analyze spatial boundaries and influence zones
# 2. **Threshold-based Indicators**: Focus on exceedance patterns and critical values
# 3. **Temporal Analysis**: Study changes over time
# 4. **Neighborhood Analysis**: Examine local spatial relationships
# 5. **Multivariate Analysis**: Analyze relationships between multiple variables
#
# ## Data Setup
#
# We'll start by importing the necessary libraries and creating synthetic data to demonstrate each indicator.
#
# ------------------------------------------------------------
# Code Cell 2
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import environmentaltools.spatiotemporal.indicators as indicators
import numpy as np

# Data simulation: (time, latitude, longitude)
data_cube = np.random.rand(12, 100, 100)
threshold = 0.8
size = 3  # Neighborhood size

# ------------------------------------------------------------
# Markdown Cell 3
# ------------------------------------------------------------
# ## 1. Boundary and Extent Indicators
#
# These indicators analyze spatial boundaries and zones of influence in environmental data.
#
# ------------------------------------------------------------
# Markdown Cell 4
# ------------------------------------------------------------
# ### Mean Presence Boundary
#
# This indicator identifies the average spatial boundary where environmental conditions are present, based on the mean values across time.
#
# ------------------------------------------------------------
# Code Cell 5
# ------------------------------------------------------------

# Indicator 1: Mean presence boundary
contours, mean_map = indicators.mean_presence_boundary(data_cube)

plt.imshow(mean_map, cmap='viridis')
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], color='red')
plt.title("Mean Presence Boundary")
plt.colorbar(label='Mean Value')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 6
# ------------------------------------------------------------
# ### Maximum Influence Extent
#
# This indicator identifies the spatial extent of maximum influence by analyzing extreme values in the temporal dimension.
#
# ------------------------------------------------------------
# Code Cell 7
# ------------------------------------------------------------

# Indicator 2: Maximum Influence Extent
contours, extreme_map = indicators.maximum_influence_extent(data_cube)

import matplotlib.pyplot as plt
plt.imshow(extreme_map, cmap='inferno')
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], color='cyan')
plt.title("Maximum Influence Extent")
plt.colorbar(label='Extreme Value')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 8
# ------------------------------------------------------------
# ## 2. Threshold-based Indicators
#
# These indicators analyze patterns related to threshold exceedance and critical values.
#
# ------------------------------------------------------------
# Markdown Cell 9
# ------------------------------------------------------------
# ### Threshold Exceedance Frequency
#
# This indicator counts how frequently each spatial location exceeds a given threshold value across the temporal dimension.
#
# ------------------------------------------------------------
# Code Cell 10
# ------------------------------------------------------------

# Indicator 3: Threshold Exceedance Frequency
threshold = 0.8  # Fixed threshold example
freq_map = indicators.threshold_exceedance_frequency(data_cube, threshold)

plt.figure()
plt.imshow(freq_map, cmap='magma')
plt.title("Threshold Exceedance Frequency")
plt.colorbar(label='Exceedance Count')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 11
# ------------------------------------------------------------
# ### Permanently Affected Zone
#
# This indicator identifies areas that are persistently affected (above threshold for a specified proportion of time).
#
# ------------------------------------------------------------
# Code Cell 12
# ------------------------------------------------------------

# Indicator 4: Permanently Affected Zone
threshold = 0.7
persistence_ratio = 0.8
mask, freq_map = indicators.permanently_affected_zone(data_cube, threshold, persistence_ratio)

plt.figure()
plt.imshow(freq_map, cmap='plasma')
plt.contour(mask, levels=[0.5], colors='white')
plt.title("Permanently Affected Zone")
plt.colorbar(label='Proportion of Time Above Threshold')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 13
# ------------------------------------------------------------
# ## 3. Temporal Analysis Indicators
#
# These indicators focus on temporal patterns and changes over time.
#
# ------------------------------------------------------------
# Markdown Cell 14
# ------------------------------------------------------------
# ### Mean Representative Value
#
# This indicator calculates the mean value over a specific time window for each spatial location.
#
# ------------------------------------------------------------
# Code Cell 15
# ------------------------------------------------------------

# Indicator 5: Mean Representative Value
# Example: mean between months 3 and 9
mean_map = indicators.mean_representative_value(data_cube, time_window=(3, 9))

plt.figure()
plt.imshow(mean_map, cmap='coolwarm')
plt.title("Mean Representative Value")
plt.colorbar(label='Mean Value')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 16
# ------------------------------------------------------------
# ### Return-period Based Extreme Value
#
# This indicator estimates extreme values based on return period analysis using Generalized Extreme Value (GEV) distribution.
#
# ------------------------------------------------------------
# Code Cell 17
# ------------------------------------------------------------

# Indicator 6: Return-period Based Extreme Value
# Simulate a series of annual maxima
max_series = np.max(data_cube, axis=0).flatten()  # or use annual maxima if you have dates

return_period = 100
x_T, params = indicators.return_period_extreme_value(max_series, return_period)

print(f"Extreme value for {return_period} years: {x_T:.2f}")
print(f"GEV parameters: shape={params[0]:.3f}, loc={params[1]:.2f}, scale={params[2]:.2f}")

# ------------------------------------------------------------
# Markdown Cell 18
# ------------------------------------------------------------
# ### Spatial Change Rate
#
# This indicator measures the rate of spatial change by analyzing gradient magnitudes over time.
#
# ------------------------------------------------------------
# Code Cell 19
# ------------------------------------------------------------

# Indicator 7: Spatial Change Rate
rate_map = indicators.spatial_change_rate(data_cube)

plt.figure()
plt.imshow(rate_map, cmap='cividis')
plt.title("Spatial Change Rate")
plt.colorbar(label='Mean Gradient Magnitude')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 20
# ------------------------------------------------------------
# ### Functional Area Loss
#
# This indicator analyzes the loss of functional area between two time periods based on threshold criteria.
#
# ------------------------------------------------------------
# Code Cell 21
# ------------------------------------------------------------

# Indicator 8: Functional Area Loss
threshold = 0.6
loss_map, area_start, area_end = indicators.functional_area_loss(data_cube, threshold, t_start=0, t_end=-1)

plt.figure()
plt.imshow(loss_map, cmap='Reds')
plt.title("Functional Area Loss")
plt.colorbar(label='Loss (1 = lost)')
plt.show()

print(f"Initial functional area: {area_start}")
print(f"Final functional area: {area_end}")

# ------------------------------------------------------------
# Markdown Cell 22
# ------------------------------------------------------------
# ### Critical Boundary Retreat
#
# This indicator analyzes the retreat of critical boundaries by comparing contours between two time periods.
#
# ------------------------------------------------------------
# Code Cell 23
# ------------------------------------------------------------

# Indicator 9: Critical Boundary Retreat
threshold = 0.6
contours_start, contours_end, retreat_mask = indicators.critical_boundary_retreat(data_cube, threshold, t_start=0, t_end=-1)

plt.figure()
plt.imshow(retreat_mask, cmap='Purples')
for c in contours_start:
    plt.plot(c[:, 1], c[:, 0], color='green', label='Start')
for c in contours_end:
    plt.plot(c[:, 1], c[:, 0], color='orange', label='End')
plt.title("Critical Boundary Retreat")
plt.legend()
plt.show()

# ------------------------------------------------------------
# Markdown Cell 24
# ------------------------------------------------------------
# ## 4. Neighborhood Analysis Indicators
#
# These indicators examine local spatial relationships and neighborhood effects.
#
# ------------------------------------------------------------
# Code Cell 25
# ------------------------------------------------------------

# 10. Neighbourhood Mean
neigh_cube = indicators.neighbourhood_mean(data_cube, size=size)
neigh_mean_map = np.mean(neigh_cube, axis=0)
plt.figure()
plt.imshow(neigh_mean_map, cmap='viridis')
plt.title("Neighbourhood Mean")
plt.colorbar()

# 11. Neighbourhood Gradient Influence
influence_map = indicators.neighbourhood_gradient_influence(data_cube, size=size)
plt.figure()
plt.imshow(influence_map, cmap='plasma')
plt.title("Neighbourhood Gradient Influence")
plt.colorbar()

# 12. Environmental Convergence
convergence_map = indicators.environmental_convergence(data_cube, size=size)
plt.figure()
plt.imshow(convergence_map, cmap='coolwarm')
plt.title("Environmental Convergence (Trend of Difference)")
plt.colorbar()

# 13. Neighbourhood Polarization
polarization_map = indicators.neighbourhood_polarization(data_cube, size=size)
plt.figure()
plt.imshow(polarization_map, cmap='magma')
plt.title("Neighbourhood Polarization")
plt.colorbar()

# 14. Local Persistence
persistence_map = indicators.local_persistence(data_cube, size=size)
plt.figure()
plt.imshow(persistence_map, cmap='cividis')
plt.title("Local Persistence")
plt.colorbar(label='Proportion of Time Dominant')

# 15. Environmental Risk
risk_map = indicators.environmental_risk(data_cube, threshold=threshold, size=size)
plt.figure()
plt.imshow(risk_map, cmap='inferno')
plt.title("Environmental Risk")
plt.colorbar(label='Risk Index')

plt.show()

# ------------------------------------------------------------
# Markdown Cell 26
# ------------------------------------------------------------
# ### Directional Influence
#
# This indicator analyzes the directional patterns of change by computing gradient directions and magnitudes.
#
# ------------------------------------------------------------
# Code Cell 27
# ------------------------------------------------------------

# Indicator 16: Directional Influence
angle_map, magnitude_map = indicators.directional_influence(data_cube)

plt.figure()
plt.imshow(angle_map, cmap='twilight')
plt.title("Directional Influence (Angle in radians)")
plt.colorbar(label='Angle (rad)')

plt.figure()
plt.imshow(magnitude_map, cmap='YlOrBr')
plt.title("Directional Influence (Magnitude)")
plt.colorbar(label='Mean Gradient Magnitude')
plt.show()

# ------------------------------------------------------------
# Markdown Cell 28
# ------------------------------------------------------------
# ## 5. Multivariate Analysis Indicators
#
# These indicators analyze relationships and patterns between multiple environmental variables.
#
# ------------------------------------------------------------
# Markdown Cell 29
# ------------------------------------------------------------
# First, let's set up data for multivariate analysis with two variables:
#
# ------------------------------------------------------------
# Code Cell 30
# ------------------------------------------------------------

# Simulation of two variables: (time, lat, lon)
cube_x = np.random.rand(12, 100, 100)
cube_y = np.random.rand(12, 100, 100)
cube_list = [cube_x, cube_y]
thresholds = [0.8, 0.75]
size = 3

# ------------------------------------------------------------
# Code Cell 31
# ------------------------------------------------------------

# 1. Multivariate Neighbourhood Synergy
synergy_map = indicators.multivariate_neighbourhood_synergy(cube_list, size=size)
plt.figure()
plt.imshow(synergy_map, cmap='viridis')
plt.title("Multivariate Neighbourhood Synergy")
plt.colorbar(label='Synergy Index')

# 2. Spatiotemporal Coupling
coupling_map = indicators.spatiotemporal_coupling(cube_x, cube_y)
plt.figure()
plt.imshow(coupling_map, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Spatiotemporal Coupling")
plt.colorbar(label='Temporal Correlation')

# 3. Multivariate Threshold Exceedance
exceedance_map = indicators.multivariate_threshold_exceedance(cube_list, thresholds)
plt.figure()
plt.imshow(exceedance_map, cmap='magma')
plt.title("Multivariate Threshold Exceedance")
plt.colorbar(label='Joint Exceedance Frequency')

# 4. Directional Co-evolution
coevolution_map = indicators.directional_coevolution(cube_x, cube_y)
plt.figure()
plt.imshow(coevolution_map, cmap='cividis')
plt.title("Directional Co-evolution")
plt.colorbar(label='Directional Agreement (0–1)')

# 5. Multivariate Persistence
persistence_map = indicators.multivariate_persistence(cube_list, thresholds)
plt.figure()
plt.imshow(persistence_map, cmap='inferno')
plt.title("Multivariate Persistence")
plt.colorbar(label='Persistence Ratio')

plt.show()

# ------------------------------------------------------------
# Markdown Cell 32
# ------------------------------------------------------------
# ## Conclusions
#
# This notebook has demonstrated the comprehensive suite of spatiotemporal indicators available in the `environmentaltools` package. These indicators provide powerful tools for:
#
# 1. **Boundary Analysis**: Understanding spatial extents and influence zones
# 2. **Threshold Analysis**: Identifying critical values and exceedance patterns
# 3. **Temporal Analysis**: Tracking changes and trends over time
# 4. **Neighborhood Analysis**: Examining local spatial relationships
# 5. **Multivariate Analysis**: Analyzing interactions between multiple variables
#
# Each indicator is designed to help environmental scientists and researchers gain insights into complex spatiotemporal patterns in their data. The indicators can be combined and compared to provide comprehensive assessments of environmental systems.
#
# For more detailed information about each indicator, please refer to the function documentation and the package's API reference.
#