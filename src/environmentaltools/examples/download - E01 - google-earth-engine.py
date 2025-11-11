"""
Google Earth Engine Sentinel-2 Download Example

This example demonstrates how to download Sentinel-2 satellite imagery 
with vegetation indices using Google Earth Engine API.

Revised on: 2025-11-10
Authors: Manuel Cobos
"""

# ============================================================================
# Sentinel-2 Time Series Download Configuration
# ============================================================================
# This example configures the download of Sentinel-2 satellite imagery time series
# from Google Earth Engine. Key parameters include:
#
# - Study area: Geographic coordinates defining the region of interest
# - Time period: January to February 2020
# - Spatial resolution: 10 meters per pixel
# - Cloud filter: Maximum 10% cloud coverage
# - Output directory: Images saved to './satellite_images'
#
# The configuration enables high-quality temporal and spatial imagery for
# environmental monitoring and land use change analysis.

from environmentaltools.download.google_earth_engine import download_sentinel2_images

# EXAMPLE CONFIGURATION
# To use this code you need:
# 1. Google Cloud Platform project with Earth Engine API enabled
# 2. Authentication configured: run `earthengine authenticate` in terminal
# 3. Replace 'YOUR-GCP-PROJECT-ID' with your actual project_id

config = {
    'project_id': 'YOUR-GCP-PROJECT-ID',  # ⚠️ REEMPLAZAR CON TU PROJECT_ID REAL
    'study_area_coords': [[[-6.1, 36.8], [-6.0, 36.8], [-6.0, 36.9], [-6.1, 36.9]]],
    'output_directory': 'src/environmentaltools/data/download/satellite_images',
    'scale': 10,
    'start_date': '2020-01-01',
    'end_date': '2020-02-28',
    'cloud_percentage': 10.0
}

# USAGE EXAMPLE (commented to avoid errors without proper configuration)
# Once Google Earth Engine is configured, uncomment the following line:
results = download_sentinel2_images(config)

print("✅ Example configuration ready")
print("📍 Study area: Andalusia, Spain")
print("📅 Period: January-February 2020") 
print("🎯 Resolution: 10m")
print("☁️  Cloud filter: max 10%")
print("\n⚠️  To execute:")
print("1. Configure Google Earth Engine: earthengine authenticate")
print("2. Replace 'YOUR-GCP-PROJECT-ID' with your actual project ID")
print("3. Uncomment the line: results = download_sentinel2_images(config)")
