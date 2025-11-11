"""
Temporal Analysis - Maximum Dissimilarity Algorithm for Environmental Data Reconstruction

This example demonstrates the Maximum Dissimilarity Algorithm (MDA) for selecting representative
environmental cases from large datasets and reconstructing data at different locations using
interpolation techniques. MDA is particularly valuable for reducing computational costs while
preserving the full range of environmental conditions.

Application Domain: Coastal and marine engineering, climate downscaling, wave propagation
Method: Maximum Dissimilarity Algorithm with Radial Basis Function interpolation
Dataset: SIMAR coastal wave data (Spanish marine database)

Features:
- Representative case selection using maximum dissimilarity criteria
- Multi-variable environmental analysis (wave height, period, direction)
- Data reconstruction using multiple interpolation methods
- Validation and visualization tools for quality assessment
- Transfer function establishment between locations

Variables:
- Hm0: Significant wave height (m)
- Tp: Peak wave period (s)
- DirM: Mean wave direction (degrees)

Use Cases:
- Downscaling climate model data to coastal locations
- Wave propagation from offshore to nearshore environments
- Creating boundary conditions for high-resolution numerical models
- Transfer of environmental data between measurement locations

Author: Manuel Cobos
Created: 2025-10-28
Updated: 2025-11-11
"""

# ============================================================================
# IMPORT REQUIRED MODULES
# ============================================================================

# Standard libraries for data manipulation and visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Environmentaltools modules for MDA and reconstruction
from environmentaltools.graphics import plots as fig
from environmentaltools.temporal.classification import mda, reconstruction
from environmentaltools.temporal import analysis
from environmentaltools.common import read, save

print("✅ Required modules loaded successfully")

# ============================================================================
# CONFIGURATION AND PARAMETERS
# ============================================================================

print("\n🔧 Configuring Maximum Dissimilarity Analysis...")

# Analysis configuration
ncasos = 500  # Number of representative cases to select
vars_ = ["Hm0", "Tp", "DirM"]  # Variables for multivariate analysis
mvar = "Hm0"  # Main variable for case ordering (wave height)
file_name = "SIMAR_2041080"  # Source dataset identifier

print(f"📊 MDA Configuration:")
print(f"   🔢 Representative cases: {ncasos}")
print(f"   🌊 Variables analyzed: {', '.join(vars_)}")
print(f"   🎯 Ordering variable: {mvar}")
print(f"   📄 Source dataset: {file_name}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n📥 Loading and preprocessing source dataset...")

try:
    # Load the SIMAR wave data
    data = read.PdE(f"data/{file_name}")
    
    # Preprocess: select variables and apply noise reduction for continuity
    data = analysis.add_noise_to_array(data, vars_)[vars_]
    
    print(f"✅ Dataset loaded successfully:")
    print(f"   📏 Records: {len(data):,}")
    print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   🌊 Variables: {list(data.columns)}")
    print(f"\n📊 Data summary:")
    print(data.describe())
    
    data_loaded = True
    
except FileNotFoundError as e:
    print(f"❌ Data loading failed: {e}")
    print("📋 Expected data structure:")
    print(f"   📄 data/{file_name}.csv - Source wave dataset")
    data_loaded = False

# ============================================================================
# MAXIMUM DISSIMILARITY ALGORITHM APPLICATION
# ============================================================================

print("\n🎯 Applying Maximum Dissimilarity Algorithm...")

if data_loaded:
    try:
        # Define output file for selected cases
        fname = f"cases/cases_{ncasos}_{file_name}.csv"
        
        print("🔄 MDA algorithm progress:")
        print("   1. Starting with maximum wave height case")
        print("   2. Iteratively selecting most dissimilar cases")
        print("   3. Computing Euclidean distances in normalized space")
        print("   4. Ensuring coverage of entire environmental range")
        
        # Execute MDA case selection
        cases = mda(data, vars_, ncasos, mvar, fname)
        
        print(f"\n✅ MDA completed successfully:")
        print(f"   📊 Selected {len(cases)} representative cases")
        print(f"   💾 Cases saved to: {fname}")
        print(f"   🎯 Coverage: {len(cases)/len(data)*100:.2f}% of original dataset")
        
        print(f"\n📊 Selected cases summary:")
        print(cases.describe())
        
        mda_completed = True
        
    except Exception as e:
        print(f"❌ MDA algorithm failed: {e}")
        mda_completed = False
else:
    print("⏸️  MDA skipped - source data not loaded")
    mda_completed = False

# ============================================================================
# MDA VISUALIZATION
# ============================================================================

print("\n🎨 Generating MDA visualization...")

if mda_completed:
    try:
        # Create MDA visualization showing case distribution
        fname_fig = f"figures/mda_{ncasos}_{file_name}"
        
        print("🎨 Creating 3D scatter plot:")
        print("   🔘 All data points: gray with transparency")
        print("   ⚫ Selected cases: black markers")
        print("   📊 Demonstrates coverage of environmental space")
        
        # Generate 3D plot showing MDA case selection
        fig.plot_mda(data, cases, vars_, fname=fname_fig)
        
        print(f"✅ MDA visualization saved to: {fname_fig}")
        
    except Exception as e:
        print(f"⚠️ Warning: MDA visualization failed: {e}")

# ============================================================================
# TRANSFER CASE LOADING
# ============================================================================

print("\n📊 Loading transfer function cases...")

try:
    # Load MDA cases from deep water location (source)
    cases_deep = read.csv(f"cases/cases_500_{file_name}.csv", index_col=True)
    
    # Load corresponding cases from shallow water location (target)
    # These establish the transfer function between locations
    cases_shallow = read.csv("data/seastates_449100_4063000.csv", index_col=True)
    
    print(f"✅ Transfer cases loaded:")
    print(f"   🌊 Deep water cases: {len(cases_deep)} records")
    print(f"   🏖️ Shallow water cases: {len(cases_shallow)} records")
    
    print(f"\n🌊 Deep water statistics:")
    print(cases_deep.describe())
    
    print(f"\n🏖️ Shallow water statistics:")
    print(cases_shallow.describe())
    
    transfer_cases_loaded = True
    
except FileNotFoundError as e:
    print(f"⚠️ Transfer cases not found: {e}")
    print("📋 Required files:")
    print(f"   📄 cases/cases_500_{file_name}.csv - Source MDA cases")
    print("   📄 data/seastates_449100_4063000.csv - Target location cases")
    transfer_cases_loaded = False

# ============================================================================
# TIME SERIES LOADING FOR RECONSTRUCTION
# ============================================================================

print("\n📥 Loading time series for reconstruction...")

try:
    # Load full time series from deep water location for reconstruction
    file_name_sim = "data_CCLM4-CanESM2"
    data_deep = pd.read_csv(f"data/{file_name_sim}.csv", index_col=0, parse_dates=True)
    
    # Preprocess the time series data
    data_deep = data_deep[["Hm0", "Tp", "DirM"]]
    data_deep = analysis.add_noise_to_array(data_deep, ["Hm0", "Tp", "DirM"])[["Hm0", "Tp", "DirM"]]
    
    print(f"✅ Time series loaded:")
    print(f"   📏 Records: {len(data_deep):,}")
    print(f"   📅 Period: {data_deep.index[0]} to {data_deep.index[-1]}")
    print(f"   📊 Source: {file_name_sim}")
    
    print(f"\n📊 Time series statistics:")
    print(data_deep.describe())
    
    timeseries_loaded = True
    
except FileNotFoundError as e:
    print(f"⚠️ Time series data not found: {e}")
    print("📋 Expected file:")
    print(f"   📄 data/{file_name_sim}.csv - Deep water time series")
    timeseries_loaded = False

# ============================================================================
# DATA RECONSTRUCTION CONFIGURATION
# ============================================================================

print("\n🔧 Configuring data reconstruction parameters...")

# Define reconstruction parameters
base_vars = ["Hm0", "Tp", "DirM"]  # Variables to use as basis for interpolation
recons_vars = ["Hm0", "Tp", "DirM"]  # Variables to reconstruct

# Test multiple interpolation methods
methods = ["linear", "nearest", "cubic"]

print(f"📊 Reconstruction configuration:")
print(f"   🔢 Input variables: {', '.join(base_vars)}")
print(f"   🎯 Output variables: {', '.join(recons_vars)}")
print(f"   🔧 Interpolation methods: {', '.join(methods)}")
print(f"   📊 Number of MDA cases: {ncasos}")

if timeseries_loaded:
    print(f"   ⏰ Time steps to reconstruct: {len(data_deep):,}")

print(f"\n🔬 Method descriptions:")
print(f"   📊 Linear: Good balance of accuracy and speed")
print(f"   ⚡ Nearest: Fastest but less smooth results")
print(f"   🎨 Cubic: Smoothest but may oscillate beyond case ranges")

# ============================================================================
# RECONSTRUCTION EXECUTION AND ANALYSIS
# ============================================================================

if transfer_cases_loaded and timeseries_loaded:
    print("\n🎯 Starting data reconstruction analysis...")
    
    # Get time index for reconstruction
    index = data_deep.index
    
    # Loop through interpolation methods
    for method in methods:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {method.upper()} INTERPOLATION")
        print(f"{'='*60}")
        
        try:
            print("🔄 Executing reconstruction...")
            print("   📊 Using RBF interpolation with MDA transfer function")
            print("   🌊 Transferring from deep to shallow water conditions")
            
            # Perform reconstruction using selected method
            data_reconstructed = reconstruction(
                cases_deep,        # MDA cases from source location
                data_deep,         # Full time series to reconstruct
                cases_shallow,     # Corresponding MDA cases at target location
                index,             # Time index for output
                base_vars,         # Variables to use as input
                recons_vars,       # Variables to reconstruct
                method=method,     # Interpolation method
                eps=1.0,           # RBF shape parameter
                optimize=True,     # Optimize interpolation parameters
                optimizer="global", # Use global optimization
                num=ncasos,        # Number of cases to use
                scale_data=False,  # Don't normalize data
                scaler_method="MinMaxScaler",  # Scaler type (if enabled)
            )
            
            print(f"✅ Reconstruction completed:")
            print(f"   📏 Output records: {len(data_reconstructed):,}")
            print(f"   📅 Coverage: {data_reconstructed.index[0]} to {data_reconstructed.index[-1]}")
            
            print(f"\n📊 Reconstructed data summary:")
            print(data_reconstructed.describe())
            
            # Create comparison visualization
            print("🎨 Generating comparison plots...")
            
            # Create subplot layout for three variables
            _, axs = plt.subplots(1, 3, figsize=(12, 5))
            axs = axs.flatten()
            
            # Plot 1: Significant Wave Height (Hm0)
            axs[0].plot(
                data_deep["Hm0"],
                data_reconstructed["Hm0"],
                ".",
                color="cyan",
                alpha=0.5,
                label="RBF reconstruction"
            )
            axs[0].plot(
                cases_deep["Hm0"],
                cases_shallow["Hm0"],
                "+k",
                markersize=8,
                label="MDA selection"
            )
            axs[0].plot([0, 10], [0, 10], "k--", alpha=0.3, label="1:1 line")
            axs[0].set_xlabel(r"Deep water $H_{m0}$ (m)")
            axs[0].set_ylabel(r"Shallow water $H_{m0}$ (m)")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # Plot 2: Peak Period (Tp)
            axs[1].plot(
                data_deep["Tp"], 
                data_reconstructed["Tp"], 
                ".", 
                color="cyan",
                alpha=0.5
            )
            axs[1].plot(cases_deep["Tp"], cases_shallow["Tp"], "+k", markersize=8)
            axs[1].plot([0, 25], [0, 25], "k--", alpha=0.3)
            axs[1].set_xlabel(r"Deep water $T_p$ (s)")
            axs[1].set_ylabel(r"Shallow water $T_p$ (s)")
            axs[1].grid(True, alpha=0.3)
            
            # Plot 3: Mean Direction (DirM)
            axs[2].plot(
                data_deep["DirM"], 
                data_reconstructed["DirM"], 
                ".", 
                color="cyan",
                alpha=0.5
            )
            axs[2].plot(cases_deep["DirM"], cases_shallow["DirM"], "+k", markersize=8)
            axs[2].plot([0, 360], [0, 360], "k--", alpha=0.3)
            axs[2].set_xlabel(r"Deep water $\theta_m$ (°)")
            axs[2].set_ylabel(r"Shallow water $\theta_m$ (°)")
            axs[2].grid(True, alpha=0.3)
            
            # Overall plot formatting
            plt.suptitle(f"Reconstruction using {method.upper()} interpolation", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            fig_path = f"figures/reconstruction_{method}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"📊 Figure saved: {fig_path}")
            plt.show()
            
            # Save reconstructed data
            output_file = f"data/SIMAR_2041080_reconstructed_449100_4063000_{method}.csv"
            save.to_csv(data_reconstructed, output_file)
            print(f"💾 Data saved: {output_file}")
            
        except Exception as e:
            print(f"❌ Reconstruction failed for {method}: {e}")
            
else:
    print("⏸️  Reconstruction skipped - required data not loaded")
    print("📋 Requirements:")
    print("   📊 Transfer cases (deep and shallow water)")
    print("   ⏰ Time series data for reconstruction")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("MAXIMUM DISSIMILARITY ALGORITHM ANALYSIS RESULTS")
print("="*70)

print(f"📁 Source location: SIMAR Station 2041080")
print(f"🎯 Target location: Shallow water (coordinates: 449100, 4063000)")
print(f"🌊 Variables: Wave height, period, direction")
print(f"🔢 Representative cases selected: {ncasos}")

if data_loaded and mda_completed:
    print(f"\n✅ MDA Status: COMPLETED")
    print(f"   📊 Cases selected from {len(data):,} original records")
    print(f"   🎯 Coverage: {ncasos/len(data)*100:.2f}% of dataset")
    print(f"   💾 Cases file: cases/cases_{ncasos}_{file_name}.csv")
else:
    print(f"\n❌ MDA Status: NOT COMPLETED")

if timeseries_loaded and transfer_cases_loaded:
    print(f"\n✅ Reconstruction Status: AVAILABLE")
    print(f"   🔧 Methods tested: {', '.join(methods)}")
    print(f"   ⏰ Time series length: {len(data_deep):,} records")
else:
    print(f"\n❌ Reconstruction Status: DATA MISSING")

print(f"\n📚 Key analysis components:")
print(f"   🎯 Maximum dissimilarity case selection")
print(f"   🔄 Transfer function establishment")
print(f"   📊 Multi-method interpolation comparison")
print(f"   🎨 Validation visualization")

print(f"\n🔬 Applications:")
print(f"   🌊 Wave propagation modeling")
print(f"   🏖️ Coastal impact assessment")
print(f"   📈 Climate data downscaling")
print(f"   ⚖️ Boundary condition generation")

print(f"\n💾 Generated outputs:")
print(f"   📄 MDA case selection files")
print(f"   📊 Reconstruction comparison plots")
print(f"   🗂️ Reconstructed time series datasets")
print(f"   🎨 Quality assessment visualizations")

print("\n" + "="*70)

# ============================================================================
# USAGE NOTES
# ============================================================================
# 
# Algorithm Performance:
# - MDA selects cases that maximize environmental space coverage
# - Representative cases capture both typical and extreme conditions
# - Computational cost scales linearly with number of selected cases
# - Quality improves with larger case numbers but plateaus at ~500-1000 cases
#
# Reconstruction Quality:
# - Linear interpolation: Best general purpose method
# - Nearest neighbor: Fast but discontinuous results
# - Cubic splines: Smooth but may extrapolate beyond physical limits
# - Scatter around 1:1 line indicates reconstruction accuracy
#
# Interpretation Guidelines:
# - Black crosses (MDA cases) define the transfer relationship
# - Cyan points show reconstruction scatter for all time steps
# - Tight clustering around diagonal indicates good performance
# - Large scatter suggests complex local effects not captured by MDA
#
# Quality Metrics:
# - Visual inspection of 1:1 plots
# - Correlation coefficients between original and reconstructed
# - RMSE and bias calculations (implement separately)
# - Extreme value preservation assessment
#
# Next Steps:
# - Validate against independent observations at target location
# - Test sensitivity to number of MDA cases
# - Apply to different variable combinations
# - Implement automated quality metrics
#
# ============================================================================

# ============================================================================
# REFERENCES
# ============================================================================
# [1] Camus, P., et al. (2011). A hybrid efficient method to downscale wave 
#     climate to coastal areas. Coastal Engineering, 58(9), 851-862.
#
# [2] Cobos, M., et al. (2020). A model to study the consequences of human 
#     actions in the Guadalquivir River Estuary. PhD Thesis, University of 
#     Granada.
#
# [3] Cobos, M., et al. (2021). MarineTools.temporal (v 1.0.0): A Python 
#     package to simulate Earth and environmental timeseries. 
#     Environmental Modelling & Software.
#
# [4] Medina-López, E., et al. (2020). A method based on K-means clustering 
#     to identify sea states from in-situ data for coastal engineering 
#     applications. Applied Ocean Research, 101, 102200.
# ============================================================================