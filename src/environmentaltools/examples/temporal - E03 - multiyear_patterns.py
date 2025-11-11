"""
Temporal Analysis - Multiyear Pattern Analysis of Sunspot Data

This example analyzes monthly Wolf sunspot number data to detect multiyear cycles
including the well-known 11-year Schwabe cycle and 22-year cycle, using non-stationary
probability modeling with piecewise distribution fitting.

Data Source: WDC-SILSO, Royal Observatory of Belgium, Brussels (1749-present)
Target: Monthly sunspot numbers with multiyear oscillation patterns

Features:
- Piecewise probability model (lognorm + normal distributions)
- Detection of cycles from 22 years down to seasonal scale  
- Modified basis functions with 44 terms for detailed temporal resolution
- Percentile matching at 85% for optimal model transitions

Author: Manuel Cobos
Created: 2025-10-28
Updated: 2025-11-11
"""

# ============================================================================
# IMPORT REQUIRED MODULES
# ============================================================================

from datetime import datetime
import pandas as pd
import numpy as np
from environmentaltools.common import read
from environmentaltools.temporal import analysis
from environmentaltools.graphics import temporal

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Define date parser for monthly data format
monthly_date_parser = lambda x: datetime.strptime(x, "%Y-%m")

# Load sunspot data with proper datetime parsing
data = pd.read_csv(
    "./src/environmentaltools/data/temporal/marginal_fit/sunspots.txt",
    index_col=0,
    parse_dates=["Month"], 
    date_parser=monthly_date_parser,
)

# Add noise to ensure continuous distribution for statistical fitting
data = analysis.add_noise_to_array(data, ["Sunspots"])

print("✅ Sunspot data loaded successfully")
print(f"📊 Data range: {data.index.min()} to {data.index.max()}")
print(f"📏 Total data points: {len(data)}")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Configure piecewise non-stationary probability model
params = {
    # Target variable
    "var": "Sunspots",
    "type": "linear",
    
    # Piecewise model: lognormal (body) + normal (upper tail)  
    "fun": {0: "lognorm", 1: "norm"},
    "piecewise": True,
    "ws_ps": [0.85],  # Percentile matching point at 85%
    
    # Non-stationary analysis configuration
    "non_stat_analysis": True,
    "basis_function": {
        "method": "modified",     # Modified basis functions for complex patterns
        "no_terms": 44,          # 44 terms for detailed temporal resolution
    },
    "basis_period": [22],        # 22-year base period to capture major cycles
    "file_name": "./src/environmentaltools/data/temporal/marginal_fit/Sunspots_lognorm_norm_0.85_nonst_22_modified_44_SLSQP"
}

print("\n🔧 Model Configuration:")
print(f"   🎯 Target variable: {params['var']}")
print(f"   📊 Distributions: {params['fun'][0]} + {params['fun'][1]} (piecewise)")
print(f"   📈 Percentile split: {params['ws_ps'][0]*100}%")
print(f"   🌊 Basis method: {params['basis_function']['method']}")
print(f"   🔢 Number of terms: {params['basis_function']['no_terms']}")
print(f"   📅 Base period: {params['basis_period'][0]} years")

# ============================================================================
# MARGINAL DISTRIBUTION FITTING
# ============================================================================

print("\n🎯 Starting marginal distribution fitting...")
print("⚠️  Note: This analysis is computationally intensive due to:")
print("   - 88 parameter sets (44 terms × 2 models)")  
print("   - 400+ total parameters being optimized")
print("   - Complex piecewise model structure")
print("\n🔄 This may take several minutes to complete...\n")

# Execute the marginal fitting
analysis.fit_marginal_distribution(data, params)

print("✅ Marginal distribution fitting completed successfully!")

# ============================================================================
# RESULTS VERIFICATION
# ============================================================================

print("\n📊 Generating verification plots...")

# Load fitted parameters
fitted_params = read.read_json(params["file_name"])

# Generate non-stationary CDF plots with custom settings for monthly data
temporal.nonstationary_cdf(
    data,
    "Sunspots", 
    fitted_params,
    date_axis=True,
    daysWindowsLength=30,  # Monthly windows for empirical CDF
    equal_windows=True,
    pemp=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),  # Multiple percentiles
    file_name="sunspot_multiyear_patterns_verification.png"
)

print("✅ Verification plots generated successfully!")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*60)
print("MULTIYEAR PATTERN ANALYSIS RESULTS")
print("="*60)
print(f"📁 Results file: marginalfit/Sunspots_lognorm_norm_0.85_nonst_22_modified_44_SLSQP.json")
print(f"📊 Verification plot: sunspot_multiyear_patterns_verification.png")
print(f"🎯 Analysis target: Monthly sunspot numbers")
print(f"📈 Model: Piecewise (lognormal + normal) with 85% split")
print(f"🔍 Detected cycles: 22-year base period with seasonal variations")
print(f"⚙️  Parameters fitted: ~400 parameters across 44 temporal terms")

print(f"\n📚 Expected patterns detected:")
print(f"   🌞 11-year Schwabe solar cycle")
print(f"   📅 22-year magnetic polarity cycle") 
print(f"   🔄 Seasonal and sub-seasonal variations")
print(f"   📈 Long-term secular trends")

print(f"\n💾 Files generated:")
print(f"   📄 JSON parameters: Complete fitted model for simulation")
print(f"   📊 Verification plots: Model quality assessment")

print("\n" + "="*60)

# ============================================================================
# REFERENCES
# ============================================================================
# [1] Usoskin, IG and Mursula, K (2003). Long-term solar cycle evolution: 
#     review of recent developments. Solar Physics, 218(1), 319-343.
# 
# [2] Cobos, M., et al. (2021). MarineTools.temporal (v 1.0.0): A Python 
#     package to simulate Earth and environmental timeseries. 
#     Environmental Modelling & Software.
#
# [3] Cobos, M., et al. (2021). A method to characterize and simulate climate, 
#     earth or environmental vector random processes. 
#     Probabilistic Engineering & Mechanics.
# ============================================================================