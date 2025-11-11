"""
ERA5 Data Download Example - Copernicus Climate Data Store

This example demonstrates how to download ERA5 reanalysis data from the 
Copernicus Climate Data Store (CDS) using the environmentaltools library.

Features:
- Download any ERA5 variable (waves, wind, temperature, etc.)
- Flexible temporal and spatial configuration
- Automatic data processing and CSV export
- Optional visualization capabilities
- Robust error handling and retry mechanisms

Prerequisites:
- Valid CDS API credentials in ~/.cdsapirc file
- Required packages: cdsapi, pandas, xarray
- Internet connection for data download

Example Variables:
- significant_height_of_combined_wind_waves_and_swell (wave height)
- 10m_u_component_of_wind, 10m_v_component_of_wind (wind components)
- 2m_temperature (air temperature)
- sea_surface_temperature (SST)
- mean_sea_level_pressure (MSLP)

Author: Manuel Cobos
Created: 2025-11-03
Updated: 2025-11-11
"""

# ============================================================================
# ERA5 DATA DOWNLOAD EXAMPLE
# ============================================================================
# This script demonstrates downloading ERA5 reanalysis data from CDS
#
# Requirements:
# - Valid CDS API credentials in ~/.cdsapirc
# - Packages: cdsapi, pandas, environmentaltools
# - Sufficient disk space for downloaded files
#
# CDS API Setup:
# 1. Register at: https://cds.climate.copernicus.eu/
# 2. Get your API key from your profile
# 3. Create ~/.cdsapirc file with:
#    url: https://cds.climate.copernicus.eu/api/v2
#    key: {your-uid}:{your-api-key}
# ============================================================================

# Import required modules
from environmentaltools.download import marine_copernicus


# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# Define your download parameters here. The configuration dictionary supports
# both required and optional parameters for flexible data retrieval.

config = {
    # -------------------------------------------------------------------------
    # REQUIRED PARAMETERS - Must be specified for successful download
    # -------------------------------------------------------------------------
    'start_year': 2018,                        # First year to download
    'end_year': 2020,                          # Last year to download (inclusive)
    'area_bounds': [41.4, -9.0, 41.0, -8.65], # Geographic bounds [N, W, S, E] in decimal degrees
    'output_directory': './src/environmentaltools/data/download/era5/wave_data',    # Local directory for downloaded files
    
    # -------------------------------------------------------------------------
    # OPTIONAL PARAMETERS - Defaults provided if not specified
    # -------------------------------------------------------------------------
    # Dataset and variable configuration
    'dataset_name': 'reanalysis-era5-single-levels',                    # CDS dataset identifier
    'variable': 'significant_height_of_combined_wind_waves_and_swell',   # ERA5 variable name
    'file_prefix': 'waves',                                              # Prefix for output filenames
    
    # Temporal resolution control (uncomment to customize)
    # 'months': ['01', '02', '03'],           # Specific months (default: all 12 months)
    # 'days': ['01', '15'],                   # Specific days (default: all days in month)
    # 'hours': ['00:00', '12:00'],            # Specific hours (default: all 24 hours)
    
    # Download behavior settings
    'retry_attempts': 3,                      # Number of retry attempts for failed downloads
    'retry_delay': 60,                        # Wait time (seconds) between retry attempts
    'min_file_size_mb': 0.1,                 # Minimum expected file size in MB
    
    # Output processing options
    'export_csv': True,                       # Export processed data to CSV format
    'create_plot': False,                     # Generate visualization plots (requires matplotlib)
    
    # Visualization settings (only used if create_plot=True)
    # 'variable_name': 'swh',                 # Column name in processed data
    # 'variable_label': 'Significant Wave Height',  # Label for plot axes
    # 'variable_units': 'm',                  # Units for axis labels
}


# ============================================================================
# EXECUTION SECTION
# ============================================================================

# Display configuration summary
print("="*70)
print("ERA5 DATA DOWNLOAD SCRIPT")
print("="*70)
print("\nConfiguration Summary:")
print(f"  Time Period: {config['start_year']}-{config['end_year']}")
print(f"  Geographic Area: {config['area_bounds']}")
print(f"  Variable: {config['variable']}")
print(f"  Output Directory: {config['output_directory']}")
print(f"  Retry Attempts: {config['retry_attempts']}")
print("\n" + "="*70)

# Execute the download and processing workflow
# This function will:
# 1. Validate configuration parameters
# 2. Download ERA5 data for each specified year
# 3. Process and combine the data
# 4. Export to CSV (if enabled)
# 5. Create plots (if enabled)
results = marine_copernicus.download_era5_data(config)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

# Print comprehensive results summary
print("\n" + "="*70)
print("DOWNLOAD COMPLETED")
print("="*70)

# Download statistics
successful = sum(results['download_results'].values())
total = len(results['download_results'])
print(f"Successfully Downloaded: {successful}/{total} years")

# Output file information
if 'csv_path' in results:
    print(f"CSV Export: {results['csv_path']}")

if 'plot_path' in results:
    print(f"Visualization: {results['plot_path']}")

# Data summary statistics
if results['data'] is not None:
    print(f"Total Records: {len(results['data'])}")
    print(f"Date Range: {results['data'].index.min()} to {results['data'].index.max()}")
    
    # Display basic statistics for the downloaded variable
    if len(results['data'].columns) > 0:
        var_col = results['data'].columns[0]
        var_data = results['data'][var_col]
        print(f"Variable Statistics ({var_col}):")
        print(f"  Mean: {var_data.mean():.3f}")
        print(f"  Min: {var_data.min():.3f}")
        print(f"  Max: {var_data.max():.3f}")

print("="*70)

# ============================================================================
# ADDITIONAL CONFIGURATION EXAMPLES
# ============================================================================
# The following sections demonstrate different use cases and configurations
# for downloading various ERA5 variables. Uncomment and modify as needed.

# ----------------------------------------------------------------------------
# Example 1: Download 10m Wind Components (U and V)
# ----------------------------------------------------------------------------
# Wind data is useful for meteorological analysis and marine applications

wind_config = {
    'start_year': 2019,
    'end_year': 2020,
    'area_bounds': [43.0, -10.0, 40.0, -7.0],  # Larger area for wind analysis
    'output_directory': './src/environmentaltools/data/download/era5/wind_data',
    'variable': '10m_u_component_of_wind',       # Choose u_component or v_component
    'file_prefix': 'wind_u10',
    'export_csv': True,
    'create_plot': True,
    'variable_name': 'u10',
    'variable_label': '10m U Wind Component',
    'variable_units': 'm/s'
}

# To execute: results = marine_copernicus.download_era5_data(wind_config)

# ----------------------------------------------------------------------------
# Example 2: Download Sea Surface Temperature (SST)
# ----------------------------------------------------------------------------
# SST is important for marine ecosystem studies and climate analysis

sst_config = {
    'start_year': 2020,
    'end_year': 2021,
    'area_bounds': [45.0, -12.0, 38.0, -6.0],  # Atlantic coastal region
    'output_directory': './src/environmentaltools/data/download/era5/sst_data',
    'variable': 'sea_surface_temperature',
    'file_prefix': 'sst',
    'export_csv': True,
    'create_plot': True,
    'variable_name': 'sst',
    'variable_label': 'Sea Surface Temperature',
    'variable_units': 'K'  # Note: ERA5 temperatures are in Kelvin
}

# To execute: results = marine_copernicus.download_era5_data(sst_config)

# ----------------------------------------------------------------------------
# Example 3: Download Specific Temporal Subset (Winter Months Only)
# ----------------------------------------------------------------------------
# Demonstrates temporal filtering for seasonal analysis

winter_config = {
    'start_year': 2018,
    'end_year': 2020,
    'area_bounds': [41.4, -9.0, 41.0, -8.65],
    'output_directory': './src/environmentaltools/data/download/era5/winter_waves',
    'variable': 'significant_height_of_combined_wind_waves_and_swell',
    'months': ['12', '01', '02'],                # Winter months: Dec, Jan, Feb
    'hours': ['00:00', '06:00', '12:00', '18:00'], # 6-hourly data only
    'file_prefix': 'waves_winter',
    'export_csv': True,
    'create_plot': True
}

# To execute: results = marine_copernicus.download_era5_data(winter_config)

# ----------------------------------------------------------------------------
# Example 4: Download Multiple Variables (requires separate calls)
# ----------------------------------------------------------------------------
# For downloading multiple variables, make separate calls with different configs

# Air temperature configuration
temp_config = {
    'start_year': 2019,
    'end_year': 2019,  # Single year for demonstration
    'area_bounds': [42.0, -9.5, 40.5, -8.0],
    'output_directory': './src/environmentaltools/data/download/era5/temperature_data',
    'variable': '2m_temperature',
    'file_prefix': 'temp_2m',
    'export_csv': True
}

# Mean sea level pressure configuration
pressure_config = {
    'start_year': 2019,
    'end_year': 2019,
    'area_bounds': [42.0, -9.5, 40.5, -8.0],
    'output_directory': './src/environmentaltools/data/download/era5/pressure_data',
    'variable': 'mean_sea_level_pressure',
    'file_prefix': 'mslp',
    'export_csv': True
}

# To execute multiple downloads:
# temp_results = marine_copernicus.download_era5_data(temp_config)
# pressure_results = marine_copernicus.download_era5_data(pressure_config)

# ============================================================================
# TROUBLESHOOTING NOTES
# ============================================================================
#
# Common Issues and Solutions:
#
# 1. Authentication Error:
#    - Ensure ~/.cdsapirc file exists with correct credentials
#    - Check CDS website for API key
#
# 2. Download Timeout:
#    - Large requests may take hours to process on CDS servers
#    - Consider reducing temporal or spatial extent
#    - Check CDS status page for service availability
#
# 3. Variable Name Errors:
#    - Use exact variable names from CDS documentation
#    - Check available variables for your selected dataset
#
# 4. File Size Issues:
#    - ERA5 files can be large (GB range)
#    - Ensure sufficient disk space
#    - Consider downloading smaller time periods
#
# 5. Memory Issues:
#    - Large datasets may require more RAM for processing
#    - Consider processing data in smaller chunks
#
# ============================================================================
