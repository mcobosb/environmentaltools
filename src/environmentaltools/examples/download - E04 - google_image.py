"""
Google Maps Image Download Example

This example demonstrates how to download high-resolution Google Maps images
using the environmentaltools library. The script shows how to download different
types of map layers (satellite, roadmap, terrain) for specified coordinates.

Features:
- Download satellite, roadmap, and terrain imagery
- Configurable zoom levels and image dimensions
- Multiple location examples
- Different map layer types
- High-resolution image stitching from tiles
- Batch downloading capabilities

Prerequisites:
- Internet connection for accessing Google Maps tiles
- PIL (Pillow) library for image processing
- Sufficient disk space for downloaded images

Common Use Cases:
- Environmental monitoring and analysis
- Land use and land cover studies
- Geographic information systems (GIS)
- Research documentation and presentations
- Spatial analysis and mapping

Author: Manuel Cobos
Created: 2025-11-11
"""

# ============================================================================
# GOOGLE MAPS IMAGE DOWNLOAD EXAMPLE
# ============================================================================
# This script demonstrates downloading high-resolution Google Maps images
# by stitching together multiple map tiles.
#
# Map Layers Available:
# - SATELLITE: High-resolution satellite imagery
# - ROADMAP: Standard street map with labels
# - TERRAIN: Topographical map showing elevation
# - HYBRID: Satellite imagery with street labels
# - TERRAIN_ONLY: Terrain without labels
# - ALTERED_ROADMAP: Alternative roadmap style
#
# Zoom Levels:
# - 0-4: Continental/country level
# - 5-9: State/region level  
# - 10-14: City level
# - 15-18: District/neighborhood level
# - 19-23: Street/building level (max detail)
# ============================================================================

# Import required modules
from environmentaltools.download.google_image import (
    GoogleMapDownloader, 
    GoogleMapsLayers, 
    download_google_maps_image
)
import os

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Create output directory for downloaded images
output_dir = "./src/environmentaltools/data/download/google_maps_images"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# EXAMPLE 1: DOWNLOAD SATELLITE IMAGE OF MADRID, SPAIN
# ============================================================================

print("="*70)
print("EXAMPLE 1: MADRID SATELLITE IMAGE")
print("="*70)

# Madrid coordinates (Puerta del Sol)
madrid_lat = 40.4168
madrid_lng = -3.7038

# Download high-resolution satellite image
madrid_success = download_google_maps_image(
    lat=madrid_lat,
    lng=madrid_lng,
    zoom=16,                                    # High detail zoom level
    layer=GoogleMapsLayers.SATELLITE,          # Satellite imagery
    tile_width=4,                              # 4x4 tiles = 1024x1024 pixels
    tile_height=4,
    output_file=os.path.join(output_dir, "madrid_satellite.png")
)

if madrid_success:
    print("✅ Madrid satellite image downloaded successfully")
else:
    print("❌ Failed to download Madrid satellite image")

# ============================================================================
# EXAMPLE 2: DOWNLOAD TERRAIN MAP OF THE PYRENEES
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: PYRENEES TERRAIN MAP")
print("="*70)

# Pyrenees mountains coordinates (Pic du Midi)
pyrenees_lat = 42.9367
pyrenees_lng = 0.1432

# Download terrain map showing elevation
pyrenees_success = download_google_maps_image(
    lat=pyrenees_lat,
    lng=pyrenees_lng,
    zoom=13,                                   # Medium zoom for regional view
    layer=GoogleMapsLayers.TERRAIN,           # Terrain with elevation
    tile_width=6,                             # Larger area: 6x6 tiles
    tile_height=6,
    output_file=os.path.join(output_dir, "pyrenees_terrain.png")
)

if pyrenees_success:
    print("✅ Pyrenees terrain map downloaded successfully")
else:
    print("❌ Failed to download Pyrenees terrain map")

# ============================================================================
# EXAMPLE 3: DOWNLOAD ROADMAP OF BARCELONA
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: BARCELONA ROADMAP")
print("="*70)

# Barcelona coordinates (Sagrada Familia)
barcelona_lat = 41.4036
barcelona_lng = 2.1744

# Download standard roadmap
barcelona_success = download_google_maps_image(
    lat=barcelona_lat,
    lng=barcelona_lng,
    zoom=15,                                  # City-level detail
    layer=GoogleMapsLayers.ROADMAP,          # Standard roadmap
    tile_width=3,                            # Smaller area: 3x3 tiles
    tile_height=3,
    output_file=os.path.join(output_dir, "barcelona_roadmap.png")
)

if barcelona_success:
    print("✅ Barcelona roadmap downloaded successfully")
else:
    print("❌ Failed to download Barcelona roadmap")

# ============================================================================
# EXAMPLE 4: DOWNLOAD COASTAL AREA WITH HYBRID VIEW
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: COSTA BRAVA HYBRID VIEW")
print("="*70)

# Costa Brava coordinates (Tossa de Mar)
costa_brava_lat = 41.7192
costa_brava_lng = 2.9317

# Download hybrid view (satellite + labels)
costa_brava_success = download_google_maps_image(
    lat=costa_brava_lat,
    lng=costa_brava_lng,
    zoom=17,                                  # High detail for coastal features
    layer=GoogleMapsLayers.HYBRID,           # Satellite with labels
    tile_width=5,                            # 5x5 tiles for detailed view
    tile_height=5,
    output_file=os.path.join(output_dir, "costa_brava_hybrid.png")
)

if costa_brava_success:
    print("✅ Costa Brava hybrid view downloaded successfully")
else:
    print("❌ Failed to download Costa Brava hybrid view")

# ============================================================================
# EXAMPLE 5: ADVANCED USAGE WITH CUSTOM DOWNLOADER
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: CUSTOM DOWNLOADER - MULTIPLE VIEWS")
print("="*70)

# Coordinates for environmental monitoring (Doñana National Park)
donana_lat = 37.0042
donana_lng = -6.4686

# Create custom downloader instance
downloader = GoogleMapDownloader(
    lat=donana_lat,
    lng=donana_lng,
    zoom=14,
    layer=GoogleMapsLayers.SATELLITE
)

print(f"Target coordinates: {donana_lat}, {donana_lng}")
print(f"Tile coordinates: {downloader.get_tile_coordinates()}")

try:
    # Generate multiple images with different settings
    print("\nGenerating multiple views...")
    
    # 1. Satellite overview
    sat_image = downloader.generate_image(tile_width=3, tile_height=3)
    sat_output = os.path.join(output_dir, "donana_satellite.png")
    sat_image.save(sat_output)
    print(f"✅ Satellite overview saved: {sat_output}")
    
    # 2. Switch to terrain view
    downloader._layer = GoogleMapsLayers.TERRAIN
    terrain_image = downloader.generate_image(tile_width=3, tile_height=3)
    terrain_output = os.path.join(output_dir, "donana_terrain.png")
    terrain_image.save(terrain_output)
    print(f"✅ Terrain view saved: {terrain_output}")
    
    # 3. High-detail hybrid view
    downloader._layer = GoogleMapsLayers.HYBRID
    downloader._zoom = 16  # Increase zoom for more detail
    hybrid_image = downloader.generate_image(tile_width=2, tile_height=2)
    hybrid_output = os.path.join(output_dir, "donana_hybrid_detail.png")
    hybrid_image.save(hybrid_output)
    print(f"✅ High-detail hybrid saved: {hybrid_output}")
    
except Exception as e:
    print(f"❌ Error in custom downloader: {e}")

# ============================================================================
# EXAMPLE 6: BATCH DOWNLOAD FOR MULTIPLE LOCATIONS
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: BATCH DOWNLOAD - SPANISH CITIES")
print("="*70)

# Define multiple locations for batch download
spanish_cities = [
    {"name": "Sevilla", "lat": 37.3891, "lng": -5.9845},
    {"name": "Valencia", "lat": 39.4699, "lng": -0.3763},
    {"name": "Bilbao", "lat": 43.2630, "lng": -2.9350},
    {"name": "Granada", "lat": 37.1773, "lng": -3.5986}
]

# Download satellite images for each city
successful_downloads = 0
total_cities = len(spanish_cities)

for city in spanish_cities:
    print(f"\nDownloading {city['name']}...")
    
    success = download_google_maps_image(
        lat=city['lat'],
        lng=city['lng'],
        zoom=15,
        layer=GoogleMapsLayers.SATELLITE,
        tile_width=3,
        tile_height=3,
        output_file=os.path.join(output_dir, f"{city['name'].lower()}_satellite.png")
    )
    
    if success:
        successful_downloads += 1
        print(f"✅ {city['name']} downloaded successfully")
    else:
        print(f"❌ Failed to download {city['name']}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DOWNLOAD RESULTS SUMMARY")
print("="*70)

print(f"Output directory: {output_dir}")
print(f"Batch download: {successful_downloads}/{total_cities} cities successful")

# List all downloaded files
try:
    downloaded_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"Total images downloaded: {len(downloaded_files)}")
    print("\nDownloaded files:")
    for file in downloaded_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) / 1024 / 1024  # Size in MB
        print(f"  📁 {file} ({file_size:.2f} MB)")
except Exception as e:
    print(f"Error listing files: {e}")

print("\n" + "="*70)

# ============================================================================
# ADDITIONAL CONFIGURATION EXAMPLES
# ============================================================================

# ----------------------------------------------------------------------------
# Example: Different Zoom Levels for Scale Comparison
# ----------------------------------------------------------------------------

scale_comparison_config = {
    "location": {"lat": 40.4168, "lng": -3.7038},  # Madrid
    "zoom_levels": [12, 14, 16, 18],               # Different scales
    "layer": GoogleMapsLayers.SATELLITE,
    "tile_size": 3,                                # 3x3 tiles each
    "prefix": "madrid_scale"
}

# To execute:
# for i, zoom in enumerate(scale_comparison_config["zoom_levels"]):
#     download_google_maps_image(
#         lat=scale_comparison_config["location"]["lat"],
#         lng=scale_comparison_config["location"]["lng"],
#         zoom=zoom,
#         layer=scale_comparison_config["layer"],
#         tile_width=scale_comparison_config["tile_size"],
#         tile_height=scale_comparison_config["tile_size"],
#         output_file=f"{scale_comparison_config['prefix']}_zoom{zoom}.png"
#     )

# ----------------------------------------------------------------------------
# Example: Environmental Monitoring Locations
# ----------------------------------------------------------------------------

environmental_sites = [
    {"name": "Cabrera_Marine_Park", "lat": 39.1425, "lng": 2.9525},
    {"name": "Tablas_de_Daimiel", "lat": 39.1486, "lng": -3.7069},
    {"name": "Picos_de_Europa", "lat": 43.1567, "lng": -4.8558},
    {"name": "Cabo_de_Gata", "lat": 36.7213, "lng": -2.1544}
]

# To execute environmental monitoring downloads:
# for site in environmental_sites:
#     download_google_maps_image(
#         lat=site["lat"],
#         lng=site["lng"],
#         zoom=15,
#         layer=GoogleMapsLayers.SATELLITE,
#         tile_width=4,
#         tile_height=4,
#         output_file=f"env_monitoring_{site['name'].lower()}.png"
#     )

# ============================================================================
# TROUBLESHOOTING AND BEST PRACTICES
# ============================================================================
#
# Common Issues and Solutions:
#
# 1. Network Connection Errors:
#    - Ensure stable internet connection
#    - Check firewall settings
#    - Try reducing tile_width/tile_height for smaller requests
#
# 2. Coordinate Validation:
#    - Latitude: -90 to +90 degrees
#    - Longitude: -180 to +180 degrees
#    - Use decimal degrees format
#
# 3. Zoom Level Guidelines:
#    - Zoom 10-12: Regional/city overview
#    - Zoom 13-15: District/neighborhood detail
#    - Zoom 16-18: Street level detail
#    - Zoom 19+: Maximum detail (may not be available everywhere)
#
# 4. Image Size Considerations:
#    - Each tile is 256x256 pixels
#    - 5x5 tiles = 1280x1280 pixels (~1.6MP)
#    - Larger images require more memory and processing time
#
# 5. Rate Limiting:
#    - Add delays between downloads for batch processing
#    - Avoid excessive concurrent requests
#    - Respect Google's terms of service
#
# 6. File Management:
#    - Ensure sufficient disk space
#    - Use descriptive filenames
#    - Consider organizing by location or date
#
# ============================================================================