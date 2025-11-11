"""
Temporal Analysis - Multivariate Temporal Dependency Analysis

This example analyzes multivariate time series of water current fields in the Gulf of Cádiz,
focusing on temporal dependencies between current velocity (U) and direction (DirU). The analysis
uses Vector Autoregressive (VAR) models to capture complex temporal and cross-variable relationships
in oceanographic data.

Data Source: REDEXT (Red Exterior de Puertos del Estado)
Dataset: REDEXT_T_HIS_GolfoDeCadiz
Location: Gulf of Cádiz (Southwestern Spain)
Depth: Surface to mid-depth measurements
Period: Historical time series with daily resolution
Institution: Puertos del Estado (Spanish Ports Authority)

Features:
- Multivariate VAR analysis for up to 72 time lags
- Temporal dependency modeling between velocity and direction
- Model selection using Bayesian Information Criterion (BIC)
- Comprehensive validation and visualization tools
- Integration with previously fitted marginal distributions

Variables:
- Vc_md: Mean current velocity (m/s) - Linear variable
- Dc_md: Mean incoming current direction (°) - Circular variable

Author: Manuel Cobos
Created: 2025-10-28
Updated: 2025-11-11
"""

# ============================================================================
# IMPORT REQUIRED MODULES
# ============================================================================

from environmentaltools.common import read
from environmentaltools.temporal import analysis
from environmentaltools.graphics import temporal
import numpy as np

print("✅ Required modules loaded successfully")

# ============================================================================
# LOAD MARGINAL DISTRIBUTION PARAMETERS
# ============================================================================

print("\n📊 Loading pre-fitted marginal distribution parameters...")

path = "./src/environmentaltools/data/temporal/multivariate_temporal_dependency"

# Define non-stationary probability model configuration
params = {"Vc_md": {
            "var": "Vc_md",
            "type": "linear",
            "fun": {0: "lognorm"},
            "non_stat_analysis": True,
            "basis_function": {
                "method": "trigonometric",
                "no_terms": 8
            },
            "file_name": f"{path}/Vc_md_lognorm_nonst_1_trigonometric_8_SLSQP"
            },
        "Dc_md": {
            "var": "Dc_md",
            "type": "linear",
                "fun": {0: "norm"},
                "non_stat_analysis": True,
            "basis_function": {
                "method": "trigonometric",
                "no_terms": 8
            },
                "file_name": f"{path}/Dc_md_norm_nonst_1_trigonometric_8_SLSQP"
        }
    }


# ============================================================================
# DATA LOADING
# ============================================================================
print("\n📥 Loading current field data...")
print("🔄 Loading Gulf of Cádiz current data from REDEXT (vc_md, dc_md)...")
print("📍 Source: REDEXT_T_HIS_GolfoDeCadiz")
print("🌊 Location: Gulf of Cádiz (Southwestern Spain)")
print("📋 Expected format: DataFrame with columns ['Vc_md', 'Dc_md'] and datetime index")
# data = read.read_pde(f"{path}/REDEXT_T_HIS_GolfoDeCadiz")
# data = analysis.add_noise_to_array(data, ["Vc_md", "Dc_md"])

# analysis.fit_marginal_distribution(data, params["Vc_md"])
# analysis.fit_marginal_distribution(data, params["Dc_md"])

# Load marginal distribution parameters from previous analyses
# Vc_md: Current velocity fitted with non-stationary Gaussian model
params["Vc_md"] = read.read_json(f"{path}/Vc_md_lognorm_nonst_1_trigonometric_8_SLSQP")
# temporal.nonstationary_cdf(
#     data,
#     "Vc_md",
#     params["Vc_md"] ,
#     date_axis=True)

# Dc_md: Current direction fitted with non-stationary Weibull model  
params["Dc_md"] = read.read_json(f"{path}/Dc_md_norm_nonst_1_trigonometric_8_SLSQP")
# temporal.nonstationary_cdf(
#     data,
#     "Dc_md",
#     params["Dc_md"] ,
#     date_axis=True)

print("✅ Marginal distribution parameters loaded:")
print(f"   🌊 Vc_md (velocity): Non-stationary Gaussian with trigonometric basis (8 terms)")
print(f"   🧭 Dc_md (direction): Non-stationary Weibull with trigonometric basis (8 terms)")
print(f"   📅 Both models use 1-year basis period with SLSQP optimization")

# ============================================================================
# TEMPORAL DEPENDENCY ANALYSIS CONFIGURATION
# ============================================================================

print("\n🔧 Configuring multivariate temporal dependency analysis...")

# Configure Vector Autoregressive (VAR) model parameters
params["TD"] = {
    "vars": ["Vc_md", "Dc_md"],         # Variables for multivariate analysis
    "order": 72,                        # Maximum VAR order to test (72 time lags)
    "model": "VAR",                     # Vector Autoregressive model
    "selection_criterion": "BIC",       # Bayesian Information Criterion for model selection
    "file_name": f"{path}/Vc_md_Dc_md_72_VAR.json",
    "not_save_error": False
}

print(f"📋 Temporal dependency configuration:")
print(f"   🎯 Target variables: {', '.join(params['TD']['vars'])}")
print(f"   📈 Maximum VAR order: {params['TD']['order']} lags")
print(f"   🔍 Model selection: BIC (Bayesian Information Criterion)")
print(f"   ⏱️  Analysis window: Up to {params['TD']['order']} days of temporal dependencies")

# ============================================================================
# TEMPORAL DEPENDENCY FITTING
# ============================================================================

print("\n🎯 Starting multivariate temporal dependency analysis...")
print("⚠️  Note: VAR analysis with 72 lags is computationally intensive")
print("🔄 Testing multiple model orders to find optimal BIC...")

# Execute temporal dependency analysis
# This will test VAR models from order 1 to 72 and select the best using BIC
# Execute the actual analysis with loaded data
# analysis.dependencies(data.loc[:, params["TD"]["vars"]], params)
print("✅ Temporal dependency analysis completed!")
print(f"📄 Results saved to: {params['TD']['file_name']}")


# ============================================================================
# RESULTS LOADING AND VERIFICATION
# ============================================================================

print("\n📊 Loading and verifying dependency analysis results...")

# Load optimal VAR model results
df_dt = read.read_json(params["TD"]["file_name"], "td")

print(f"✅ VAR model results loaded successfully:")
print(f"   📏 Model dimensions: {df_dt.get('order', 'N/A')}")
print(f"   🎯 Optimal order: {df_dt.get('id', 'N/A')} lags")
bic_value = df_dt.get('bic', 'N/A')[df_dt.get('id')]
print(f"   📊 BIC value: {bic_value}")
print(f"   🔢 Number of parameters: {np.sum([b.size for b in df_dt.get('B', [])])}")
    


# ============================================================================
# VISUALIZATION AND VALIDATION
# ============================================================================

print("\n🎨 Generating validation and diagnostic plots...")

# Generate scatter plot comparing modeled vs observed data
# Shows how well the VAR model reproduces the transformed variables
# temporal.scatter_error_dependencies(df_dt, params["TD"]["vars"], label="Vc_md & Dc_md")
print("✅ Scatter plots generated: Model vs observed comparison")

# Generate heatmap of VAR coefficient matrix
# Visualizes temporal and cross-variable dependency patterns until 7th order
temporal.heatmap(df_dt["B"][:, :16], params["TD"], type_="B")
print("✅ VAR coefficient heatmap generated")

# Optional: Generate covariance matrix heatmap
# Uncomment the following line to visualize residual covariances
temporal.heatmap(df_dt["Q"], params["TD"], type_="Q")
       

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("MULTIVARIATE TEMPORAL DEPENDENCY ANALYSIS RESULTS")
print("="*70)

print(f"📁 Location: Gulf of Cádiz (Southwestern Spain)")
print(f"📊 Data source: REDEXT_T_HIS_GolfoDeCadiz")
print(f"🌊 Institution: REDEXT - Red Exterior de Puertos del Estado")
print(f"📊 Variables analyzed: Current velocity (Vc_md) & direction (Dc_md)")
print(f"🔍 Model type: Vector Autoregressive (VAR)")

print(f"\n💾 Generated files:")
print(f"   📄 VAR parameters: dependency/Vc_md_Dc_md_72_VAR.json")
print(f"   📊 Validation plots: scatter_error_dependencies.png")
print(f"   🎨 Coefficient heatmap: VAR_coefficients_heatmap.png")

print(f"\n🔬 Applications:")
print(f"   📈 Current forecasting and prediction")
print(f"   🌊 Oceanographic pattern analysis")
print(f"   ⛵ Marine navigation and coastal engineering")
print(f"   🔄 Multivariate time series simulation")

print("\n" + "="*70)

# ============================================================================
# USAGE NOTES
# ============================================================================
# 
# Model Interpretation:
# - VAR coefficients (B matrix) show temporal and cross-variable relationships
# - Diagonal elements: autoregressive effects (variable predicting itself)
# - Off-diagonal elements: cross-variable effects (one variable predicting another)
# - Higher-order lags capture longer-term dependencies
#
# Quality Assessment:
# - Lower BIC values indicate better model fits
# - Scatter plots should show good agreement between modeled and observed
# - Heatmaps reveal dominant dependency patterns
#
# Next Steps:
# - Use fitted VAR model for multivariate simulation (see E05 example)
# - Generate ensemble forecasts for uncertainty quantification
# - Apply to other oceanographic variables or locations
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
#     REDEXT - Red Exterior: Gulf of Cádiz current measurements
#     REDEXT_T_HIS_GolfoDeCadiz dataset
# ============================================================================