"""
Temporal Analysis - Marginal Distribution Fitting Example

This example demonstrates how to perform non-stationary probability modeling
for environmental time series data using the environmentaltools library.
The analysis focuses on freshwater river flow in a semi-arid basin with
strong seasonality patterns.

Features:
- Box-Cox transformation for improved model convergence
- Non-stationary Weibull distribution fitting
- Sinusoidal temporal expansion of parameters
- Comprehensive visualization and verification plots
- JSON output for reproducible results

Applications:
- River flow analysis in semi-arid regions
- Seasonal environmental data modeling
- Climate change impact assessment
- Extreme event probability estimation
- Water resource management

Technical Details:
- Uses Weibull distribution of maxima
- Sinusoidal basis functions with 10 terms
- Box-Cox transformation for data normalization
- Non-stationary parameter evolution
- SLSQP optimization algorithm

Author: Manuel Cobos
Created: 2025-10-28
Updated: 2025-11-11
"""

# ============================================================================
# TEMPORAL ANALYSIS - MARGINAL DISTRIBUTION FITTING EXAMPLE
# ============================================================================
# This script demonstrates the complete workflow for fitting non-stationary
# probability models to environmental time series with seasonal patterns.
#
# Analysis Steps:
# 1. Load and preprocess environmental data
# 2. Configure non-stationary probability model parameters
# 3. Apply Box-Cox transformation for improved convergence
# 4. Fit Weibull distribution with temporal parameter evolution
# 5. Generate comprehensive verification plots
# 6. Export results for further analysis
#
# Data Requirements:
# - Time series data with seasonal patterns
# - Continuous variables (discrete noise added if needed)
# - Sufficient data length for temporal analysis
#
# Model Configuration:
# - Distribution: Weibull of maxima
# - Temporal basis: Sinusoidal with 10 harmonics
# - Transformation: Box-Cox normalization
# - Optimization: SLSQP algorithm
#
# TROUBLESHOOTING NOTE:
# If you encounter cartopy import errors during graphics module loading:
# 1. The script will continue running with plotting disabled
# 2. Results will still be saved as JSON files
# 3. To fix cartopy: pip install --force-reinstall cartopy
# 4. Alternative: conda install -c conda-forge cartopy proj
# ============================================================================

# Import required modules
from environmentaltools.common import read
from environmentaltools.temporal import analysis
import matplotlib.pyplot as plt
import os
import numpy as np
from environmentaltools.graphics import temporal

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Configure matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10

# Define input and output paths
data_file = "./src/environmentaltools/data/temporal/marginal_fit_simulation/CSC_REMO2009_MPI-ESM-LR_rcp26.xlsx"
output_dir = "./src/environmentaltools/data/temporal/marginal_fit_simulation/"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("TEMPORAL MARGINAL DISTRIBUTION FITTING ANALYSIS")
print("="*70)
print(f"📁 Input file: {data_file}")
print(f"📁 Output directory: {output_dir}")
print()

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("🔄 STEP 1: Loading and preprocessing data...")
print("="*50)

# Load river discharge data from Excel file
print(f"📖 Reading data from: {data_file}")
data = read.xlsx(data_file)

# Add noise to ensure continuous distribution
# This is important for fitting continuous probability distributions
print("🎲 Adding noise to ensure continuous distribution...")
data = analysis.add_noise_to_array(data, ["Qd"])

# Display basic data statistics
qd_data = data["Qd"]
print(f"✅ Data loaded successfully:")
print(f"   📊 Variable: River discharge (Qd)")
print(f"   📏 Data points: {len(qd_data)}")
print(f"   📈 Mean: {np.mean(qd_data):.3f}")
print(f"   📉 Min: {np.min(qd_data):.3f}")
print(f"   📊 Max: {np.max(qd_data):.3f}")
print(f"   📐 Std: {np.std(qd_data):.3f}")

# ============================================================================
# STEP 2: MODEL CONFIGURATION
# ============================================================================

print("\n🔧 STEP 2: Configuring probability model parameters...")
print("="*50)

# Define non-stationary probability model configuration
params = {
    # Target variable for analysis
    "var": "Qd",
    
    # Variable type (circular for directions, linear for most environmental variables)
    "type": "linear",
    
    # Probability distribution family
    # Weibull of maxima is suitable for extreme value analysis
    "fun": {0: "weibull_max"},
    
    # Enable non-stationary analysis
    "non_stat_analysis": True,
    
    # Temporal basis function configuration
    # Sinusoidal functions capture seasonal patterns effectively
    "basis_function": {
        "method": "sinusoidal",    # Sinusoidal basis functions
        "no_terms": 10             # Number of harmonics (captures annual + sub-annual cycles)
    },
    
    # Data transformation configuration
    # Box-Cox transformation improves model convergence for skewed data
    "transform": {
        "make": True,              # Apply transformation
        "plot": False,             # Don't plot transformation during fitting
        "method": "box-cox"        # Box-Cox normalization
    },
    # Output file configuration
    "file_name": "./src/environmentaltools/data/temporal/marginal_fit/Qd_weibull_max_nonst_1_sinusoidal_10_SLSQP.json"

}

print("📋 Model configuration:")
print(f"   🎯 Target variable: {params['var']}")
print(f"   📊 Variable type: {params['type']}")
print(f"   📈 Distribution: {params['fun'][0]}")
print(f"   🔀 Non-stationary: {params['non_stat_analysis']}")
print(f"   🌊 Basis function: {params['basis_function']['method']}")
print(f"   🔢 Number of terms: {params['basis_function']['no_terms']}")
print(f"   🔄 Transformation: {params['transform']['method']}")
print("✅ Model configuration completed")

# ============================================================================
# STEP 3: MARGINAL DISTRIBUTION FITTING
# ============================================================================

print("\n🎯 STEP 3: Fitting marginal distribution...")
print("="*50)

print("🔄 Starting optimization process...")
print("   This may take several minutes depending on data size")
print("   Progress information will be displayed below:")
print()

# Perform marginal distribution fitting
# This fits a non-stationary Weibull distribution with sinusoidal parameter evolution
fitting_result = analysis.fit_marginal_distribution(data, params)

print()
print("✅ Marginal fitting completed successfully")

# The results are automatically saved as JSON file
result_path = params["file_name"]

print(f"💾 Results saved to: {result_path}")

# Get file size
file_size = os.path.getsize(result_path) / 1024  # Size in KB
print(f"📏 File size: {file_size:.2f} KB")

    
# ============================================================================
# STEP 4: RESULTS LOADING AND VERIFICATION
# ============================================================================

print("\n📊 STEP 4: Loading results and generating verification plots...")
print("="*50)


# Load fitted parameters from JSON file
print("📖 Loading fitted parameters...")
fitted_params = read.read_json(result_path)
print("✅ Parameters loaded successfully")

# Create figure for verification plots
print("🎨 Generating verification plots...")
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
axs = axs.flatten()

# Plot 1: Non-stationary CDF with transformation (normalized data)
print("   📈 Plot 1: Non-stationary CDF (transformed data)")
fitted_params["transform"]["plot"] = True
temporal.nonstationary_cdf(
    data,
    "Qd",
    fitted_params,
    date_axis=True,
    ax=axs[0],
)
axs[0].set_title("Non-stationary CDF (Transformed Data)", fontsize=12, fontweight='bold')
axs[0].grid(True, alpha=0.3)

# Plot 2: Non-stationary CDF without transformation (original data)
print("   📈 Plot 2: Non-stationary CDF (original data)")
fitted_params["transform"]["plot"] = False
temporal.nonstationary_cdf(
    data,
    "Qd",
    fitted_params,
    date_axis=True,
    ax=axs[1],
)
axs[1].set_title("Non-stationary CDF (Original Data)", fontsize=12, fontweight='bold')
axs[1].grid(True, alpha=0.3)

# Improve overall figure layout
plt.tight_layout()
plt.suptitle("Marginal Distribution Fitting Results - River Discharge Analysis", 
                fontsize=14, fontweight='bold', y=1.02)

# Save the verification plot
plot_filename = os.path.join(output_dir, "marginal_fit_verification.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"💾 Verification plot saved: {plot_filename}")

# Display the plot
plt.show()

print("✅ Verification plots generated successfully")


# ============================================================================
# STEP 5: ADDITIONAL ANALYSIS AND DIAGNOSTICS
# ============================================================================

print("\n🔍 STEP 5: Additional analysis and diagnostics...")
print("="*50)

try:
    # Generate parameter evolution plots if possible
    print("📈 Analyzing parameter evolution over time...")
    
    if PLOTS_AVAILABLE:
        # Create additional diagnostic plots
        fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot parameter evolution (this would need custom implementation)
        # For now, we'll create placeholder plots showing the concept
        
        # Time series plot
        if 'date' in data or 'time' in data:
            time_key = 'date' if 'date' in data else 'time'
            axs2[0,0].plot(data[time_key], data["Qd"], 'b-', alpha=0.7, linewidth=0.8)
            axs2[0,0].set_title("Original Time Series", fontweight='bold')
            axs2[0,0].set_ylabel("River Discharge (Qd)")
            axs2[0,0].grid(True, alpha=0.3)
        
        # Histogram
        axs2[0,1].hist(data["Qd"], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axs2[0,1].set_title("Data Distribution", fontweight='bold')
        axs2[0,1].set_xlabel("River Discharge (Qd)")
        axs2[0,1].set_ylabel("Frequency")
        axs2[0,1].grid(True, alpha=0.3)
        
        # Box plot by season (simplified)
        seasonal_data = []
        if 'date' in data:
            # This is a simplified seasonal analysis
            months = [d.month if hasattr(d, 'month') else 1 for d in data['date']]
            for season in range(1, 5):  # 4 seasons
                season_months = [(season-1)*3 + 1, (season-1)*3 + 2, (season-1)*3 + 3]
                season_data = [data["Qd"][i] for i, m in enumerate(months) if m in season_months]
                if season_data:
                    seasonal_data.append(season_data)
        
        if seasonal_data:
            axs2[1,0].boxplot(seasonal_data, labels=['Winter', 'Spring', 'Summer', 'Autumn'])
            axs2[1,0].set_title("Seasonal Distribution", fontweight='bold')
            axs2[1,0].set_ylabel("River Discharge (Qd)")
            axs2[1,0].grid(True, alpha=0.3)
        else:
            axs2[1,0].text(0.5, 0.5, "Seasonal analysis\nnot available", 
                           ha='center', va='center', transform=axs2[1,0].transAxes)
            axs2[1,0].set_title("Seasonal Distribution", fontweight='bold')
        
        # Q-Q plot (conceptual)
        axs2[1,1].text(0.5, 0.5, "Q-Q Plot\n(Available after fitting)", 
                       ha='center', va='center', transform=axs2[1,1].transAxes)
        axs2[1,1].set_title("Q-Q Plot", fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle("Additional Diagnostic Plots", fontsize=14, fontweight='bold', y=0.98)
        
        # Save diagnostic plots
        diagnostic_filename = os.path.join(output_dir, "diagnostic_plots.png")
        plt.savefig(diagnostic_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Diagnostic plots saved: {diagnostic_filename}")
        
        plt.show()
        
        print("✅ Additional analysis completed")
    else:
        print("⚠️ Diagnostic plotting skipped due to graphics module unavailability")
        print("📊 Basic analysis completed - results available in JSON format")
        
except Exception as e:
    print(f"⚠️ Some diagnostic plots could not be generated: {e}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS RESULTS SUMMARY")
print("="*70)

print(f"📁 Output directory: {output_dir}")
print(f"🎯 Analysis target: River discharge (Qd)")
print(f"📊 Distribution fitted: Weibull of maxima (non-stationary)")
print(f"🌊 Basis functions: Sinusoidal with {params['basis_function']['no_terms']} terms")
print(f"🔄 Transformation: {params['transform']['method']}")

# List generated files
output_files = []
if os.path.exists(output_dir):
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

marginal_files = []
if os.path.exists("marginalfit"):
    marginal_files = [f for f in os.listdir("marginalfit") if f.endswith('.json')]

print(f"\n📋 Generated files:")
print(f"   📊 Plots: {len(output_files)} files in {output_dir}/")
for file in output_files:
    print(f"      📈 {file}")

print(f"   📄 Results: {len(marginal_files)} files in marginalfit/")
for file in marginal_files:
    print(f"      💾 {file}")
    
if not output_files and not marginal_files:
    print("   ⚠️ No output files found (files may be in different location)")


print(f"\n📚 References:")
print(f"   [1] Cobos, M., Otíñar, P., Magaña, P., Lira-Loarca, A., Baquerizo, A. (2021)")
print(f"       MarineTools.temporal (v 1.0.0): A Python package to simulate Earth and")
print(f"       environmental timeseries. Environmental Modelling & Software.")
print(f"   [2] Cobos, M., Otíñar, P., Magaña, P., Baquerizo, A. (2021)")
print(f"       A method to characterize and simulate climate, earth or environmental")
print(f"       vector random processes. Probabilistic Engineering & Mechanics.")

print("\n" + "="*70)

# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================
#
# Common Issues and Solutions:
#
# 1. Data File Not Found:
#    - Verify the path to the Excel file
#    - Ensure the file exists and is accessible
#    - Check file permissions
#
# 2. Convergence Issues:
#    - Reduce number of basis function terms (try 5-8 instead of 10)
#    - Try different transformation methods
#    - Check data quality (outliers, missing values)
#
# 3. Memory Issues with Large Datasets:
#    - Reduce data size by subsampling
#    - Process data in chunks
#    - Use less complex basis functions
#
# 4. Plotting Errors:
#    - Ensure matplotlib backend is properly configured
#    - Check data types and formats
#    - Verify all required data fields are present
#
# 5. JSON Export Issues:
#    - Ensure write permissions in marginalfit directory
#    - Check disk space availability
#    - Verify no conflicting processes accessing files
#
# 6. Graphics/Cartopy Import Issues:
#    - Script continues with plotting disabled if graphics import fails
#    - Results still available in JSON format
#    - To fix: pip install --force-reinstall cartopy
#    - Alternative: conda install -c conda-forge cartopy proj
#    - Windows users may need: conda install -c conda-forge proj-data
#
# ============================================================================