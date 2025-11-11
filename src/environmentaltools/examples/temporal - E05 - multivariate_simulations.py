"""
Temporal Analysis - Multivariate Time Series Simulation

This example demonstrates multivariate time series simulation of wind fields using previously
fitted marginal distributions and temporal dependencies. The simulation generates statistically
equivalent realizations of wind velocity and direction data based on historical patterns from
SIMAR meteorological station in the Gulf of Cádiz.

Data Source: SIMAR Station 1052048 (Puertos del Estado, Spain)
Location: 37°N, 7°W (Gulf of Cádiz)
Height: 10 m above mean sea level
Period: 1958-2011 (≈56 years, 3-hour temporal resolution)
Institution: Puertos del Estado (https://puertos.es/)

Features:
- Multivariate simulation using pre-fitted marginal distributions
- Vector Autoregressive (VAR) temporal dependency preservation
- Statistical validation through wind roses and joint PDFs
- Multiple realization generation for uncertainty analysis
- Compressed output format for efficient storage

Variables:
- Wv: Wind velocity (m/s) - Fitted with GEV/LogNormal/Pareto mixture
- Wd: Wind direction (°) - Fitted with wrapped normal distribution

Author: Manuel Cobos
Created: 2025-10-28
Updated: 2025-11-11
"""

# ============================================================================
# IMPORT REQUIRED MODULES
# ============================================================================

from environmentaltools.common import read
from environmentaltools.temporal import simulation
from environmentaltools.graphics import temporal
import numpy as np
import pandas as pd

print("✅ Required modules loaded successfully")

# ============================================================================
# LOAD MARGINAL DISTRIBUTION AND DEPENDENCY PARAMETERS
# ============================================================================

print("\n📊 Loading pre-fitted model parameters...")

# Initialize parameter dictionary
params = {}

# Load marginal distribution parameters from previous analyses
# Wv: Wind velocity with complex threshold mixture model
params["Wv"] = read.rjson(
    "marginalfit/Wv_genpareto_lognorm_genpareto_0.05_0.96_nonst_1_trigonometric_4_SLSQP"
)

# Wd: Wind direction with wrapped normal distribution
params["Wd"] = read.rjson(
    "marginalfit/Wd_norm_norm_0.5_nonst_1_sinusoidal_12_SLSQP"
)

print("✅ Marginal distribution parameters loaded:")
print(f"   💨 Wv (velocity): GEV/LogNormal/Pareto mixture (thresholds: 0.05, 0.96)")
print(f"   🧭 Wd (direction): Wrapped normal distribution")

# Load temporal dependency parameters from previous VAR analysis
params["TD"] = read.rjson("dependency/Wv_Wd_72_VAR", "td")

print("✅ Temporal dependency parameters loaded:")
print(f"   🔄 Model: Vector Autoregressive (VAR)")
print(f"   📈 Variables: Wv, Wd")
print(f"   ⏱️  Maximum order tested: 72 lags")

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================

print("\n🔧 Configuring multivariate time series simulation...")

# Configure simulation parameters
params["TS"] = {
    "start": "2026/02/01 00:00:00",    # Simulation start date/time
    "end": "2046/01/01 00:00:00",       # Simulation end date/time (20 years)
    "nosim": 5,                         # Number of independent realizations
    "folder": "simulations_SIMAR_Cadiz", # Output folder for simulation files
}

print(f"📋 Simulation configuration:")
print(f"   📅 Period: {params['TS']['start']} to {params['TS']['end']}")
print(f"   ⏰ Duration: ~20 years")
print(f"   🔢 Number of realizations: {params['TS']['nosim']}")
print(f"   💾 Output folder: {params['TS']['folder']}")
print(f"   📦 Expected file format: simulation_XXXX.zip (compressed CSV)")

# ============================================================================
# EXECUTE SIMULATION
# ============================================================================

print("\n🎯 Starting multivariate time series simulation...")
print("⚠️  Note: Large-scale simulation may take several minutes")
print("🔄 Generating statistically equivalent wind field realizations...")

# Execute the simulation with configured parameters
simulation.simulation(params)

print("✅ Multivariate simulation completed successfully!")
print(f"📁 Results saved to: {params['TS']['folder']}/")
print(f"📦 Generated {params['TS']['nosim']} compressed simulation files")
print(f"🗂️  File pattern: simulation_0001.zip to simulation_{params['TS']['nosim']:04d}.zip")

simulation_completed = True

# ============================================================================
# SIMULATION VALIDATION AND ANALYSIS
# ============================================================================

print("\n📊 Loading and validating simulation results...")

# Load one simulation for validation
sim = read.csv(f"{params['TS']['folder']}/simulation_0001.zip")

print("✅ Sample simulation loaded successfully")
print(f"   📏 Shape: {sim.shape}")
print(f"   📅 Period: {sim.index[0]} to {sim.index[-1]}")
print(f"   🌊 Variables: {list(sim.columns)}")

# Note: Original data loading would be needed for comparison
print("\n💡 For complete validation, load original wind data:")
print("   data = read.csv('original_SIMAR_wind_data.csv')")

# ============================================================================
# STATISTICAL VALIDATION PLOTS
# ============================================================================

print("\n🎨 Generating validation plots...")


print("📊 Available validation plot types:")
print("   📈 Bivariate PDF comparison (simulated vs observed)")
print("   🌹 Wind rose comparison")
print("   📊 Marginal distribution validation")

# Example validation plots (requires original data)
# Uncomment when original data is available:
joint.bivariate_pdf(sim, data, ["Wv", "Wd"])
temporal.wrose(data["Wd"], data["Wv"])  # Original data wind rose
temporal.wrose(sim["Wd"], sim["Wv"])    # Simulated data wind rose

print("💡 To generate validation plots:")
print("   plots.bivariate_pdf(sim, data, ['Wv', 'Wd'])")
print("   plots.wrose(data['Wd'], data['Wv'])  # Original")
print("   plots.wrose(sim['Wd'], sim['Wv'])    # Simulated")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("MULTIVARIATE TIME SERIES SIMULATION RESULTS")
print("="*70)

print(f"📁 Location: Gulf of Cádiz SIMAR Station 1052048 (37°N, 7°W)")
print(f"📊 Variables simulated: Wind velocity (Wv) & direction (Wd)")
print(f"⏰ Simulation period: {params['TS']['start']} to {params['TS']['end']}")
print(f"🔢 Number of realizations: {params['TS']['nosim']}")

if simulation_completed:
    print(f"\n✅ Simulation Status: COMPLETED")
    print(f"   💾 Output folder: {params['TS']['folder']}/")
    print(f"   📦 File format: Compressed CSV (.zip)")
    print(f"   📏 Each realization: ~20 years of 3-hourly wind data")
else:
    print(f"\n❌ Simulation Status: NOT COMPLETED")
    print(f"   🔍 Check prerequisite parameter files")

print(f"\n📚 Validation methods:")
print(f"   📈 Joint probability density functions")
print(f"   🌹 Wind rose directional distributions")
print(f"   📊 Marginal distribution preservation")
print(f"   🔄 Temporal correlation structure")

print(f"\n💾 Generated simulation files:")
for i in range(1, params['TS']['nosim'] + 1):
    print(f"   📄 simulation_{i:04d}.zip")

print(f"\n🔬 Applications:")
print(f"   💨 Wind resource assessment and forecasting")
print(f"   🏗️ Offshore engineering and wind farm design")
print(f"   🌊 Coastal and marine structure analysis")
print(f"   📈 Long-term climate impact studies")

print("\n" + "="*70)

# ============================================================================
# USAGE NOTES
# ============================================================================
# 
# Simulation Quality:
# - Preserves marginal distributions of both variables
# - Maintains temporal dependencies through VAR model
# - Reproduces multivariate correlation structure
# - Generates realistic wind field patterns
#
# File Format:
# - Each simulation saved as compressed CSV (.zip)
# - Significant space savings compared to raw CSV
# - Easy to load with read.csv() function
# - Contains datetime index and variable columns
#
# Validation Steps:
# 1. Compare marginal distributions (histograms, Q-Q plots)
# 2. Check joint distributions (scatter plots, 2D PDFs)
# 3. Validate directional patterns (wind roses)
# 4. Verify temporal correlations (autocorrelation functions)
#
# Next Steps:
# - Load and compare multiple realizations
# - Calculate ensemble statistics and uncertainty bounds
# - Apply simulations to engineering design problems
# - Extend to longer simulation periods or different locations
#
# ============================================================================

# ============================================================================
# REFERENCES
# ============================================================================
# [1] Cobos, M., et al. (2021). MarineTools.temporal (v 1.0.0): A Python 
#     package to simulate Earth and environmental timeseries. 
#     Environmental Modelling & Software.
#
# [2] Cobos, M., et al. (2021). A method to characterize and simulate climate, 
#     earth or environmental vector random processes. 
#     Probabilistic Engineering & Mechanics.
#
# [3] Puertos del Estado: https://puertos.es/
#     SIMAR oceanographic and meteorological database
# ============================================================================