"""
OpenStreetMap Image Download Example

This example demonstrates how to download and visualize OpenStreetMap images
using the environmentaltools library. The script shows how to download different
styles of maps (street maps and satellite imagery) for specified coordinates.

Features:
- Download OpenStreetMap street maps and satellite imagery
- Automatic scale calculation based on area extent
- Multiple location examples with different zoom levels
- Customizable area extent (distance_x, distance_y)
- Interactive matplotlib visualization with gridlines
- Geodesic calculations for accurate geographic extent

Prerequisites:
- Internet connection for accessing OpenStreetMap tiles
- Cartopy library for map projections and tile handling
- Matplotlib for visualization
- PIL (Pillow) for image processing

Technical Details:
- Uses Web Mercator projection (EPSG:3857)
- Automatic zoom level calculation based on extent
- Respects OSM tile usage policies
- Geodesic calculations for accurate positioning

Author: Manuel Cobos
Created: 2025-11-11
"""

# ============================================================================
# OPENSTREETMAP IMAGE DOWNLOAD EXAMPLE
# ============================================================================
# This script demonstrates downloading OpenStreetMap images with different
# styles and scales using cartopy and matplotlib.
#
# Map Styles Available:
# - "map": Standard OpenStreetMap street map with roads, labels, and features
# - "satellite": Satellite imagery (when available from tile providers)
#
# Scale Guidelines (Auto-calculated):
# - Scale 2: Continental/worldwide view
# - Scale 4-6: Country and large state level
# - Scale 6-10: State, region, and city level
# - Scale 10-12: City boundaries and districts
# - Scale 14+: Street level detail (roads, blocks, buildings)
#
# Distance Parameters:
# - distance_x: Horizontal extent from center (meters)
# - distance_y: Vertical extent from center (meters)
# - Total area: (2*distance_x) × (2*distance_y) meters
# ============================================================================

# Import required modules
try:
    # Configure matplotlib backend before importing pyplot
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl/Tk issues
    
    from environmentaltools.download import download_openstreet_map
    from environmentaltools.download.open_street_images import create_osm_image
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    print("✅ All required modules loaded successfully")
    print("🔧 Using matplotlib 'Agg' backend (non-interactive)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📋 Required packages:")
    print("   - cartopy: pip install cartopy")
    print("   - matplotlib: pip install matplotlib") 
    print("   - pillow: pip install pillow")
    print("   - numpy: pip install numpy")
    exit(1)

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Create output directory for downloaded images
output_dir = "./src/environmentaltools/data/download/open_street_maps"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory: {output_dir}")
print(f"📁 Directory exists: {os.path.exists(output_dir)}")
print(f"📁 Directory writable: {os.access(output_dir, os.W_OK)}")
print()

# ============================================================================
# EXAMPLE 1: MADRID CITY CENTER - STREET MAP
# ============================================================================

print("="*70)
print("EXAMPLE 1: MADRID CITY CENTER - STREET MAP")
print("="*70)

# Madrid coordinates (Puerta del Sol)
madrid_lat = 40.4168
madrid_lng = -3.7038

try:
    print(f"Downloading street map for Madrid ({madrid_lat}, {madrid_lng})")
    print("Map style: Street map")
    print("Extent: ±500m from center (1km × 1km area)")
    print("⚠️ Using non-interactive mode - images will be saved but not displayed")
    
    # Download street map of Madrid city center
    download_openstreet_map(
        lon=madrid_lng,
        lat=madrid_lat,
        distance_x=500,     # 500 meters east/west from center
        distance_y=500,     # 500 meters north/south from center  
        site_name="Madrid - Puerta del Sol",
        style="map",        # Street map style
        output_file=os.path.join(output_dir, "madrid_puerta_del_sol_street_map.png"),
        show_plot=False     # Don't show interactively, just save
    )
    
    print("✅ Madrid street map processing completed")
    
except Exception as e:
    print(f"❌ Error downloading Madrid map: {e}")
    print(f"🔧 Error type: {type(e).__name__}")
    import traceback
    print(f"🔍 Full traceback:")
    traceback.print_exc()

# ============================================================================
# EXAMPLE 2: BARCELONA SATELLITE VIEW
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: BARCELONA SATELLITE VIEW")
print("="*70)

# Barcelona coordinates (Sagrada Familia)
barcelona_lat = 41.4036
barcelona_lng = 2.1744

try:
    print(f"Downloading satellite imagery for Barcelona ({barcelona_lat}, {barcelona_lng})")
    print("Map style: Satellite imagery")
    print("Extent: ±750m from center (1.5km × 1.5km area)")
    
    # Download satellite imagery of Barcelona
    download_openstreet_map(
        lon=barcelona_lng,
        lat=barcelona_lat,
        distance_x=750,     # Larger area for satellite view
        distance_y=750,
        site_name="Barcelona - Sagrada Familia",
        style="satellite",  # Satellite imagery style
        output_file=os.path.join(output_dir, "barcelona_sagrada_familia_satellite.png"),
        show_plot=False     # Save only, don't display
    )
    
    print("✅ Barcelona satellite view displayed successfully")
    
except Exception as e:
    print(f"❌ Error downloading Barcelona satellite: {e}")

# ============================================================================
# EXAMPLE 3: SEVILLA DETAILED STREET MAP
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: SEVILLA DETAILED STREET MAP")
print("="*70)

# Sevilla coordinates (Cathedral)
sevilla_lat = 37.3858
sevilla_lng = -5.9933

try:
    print(f"Downloading detailed map for Sevilla ({sevilla_lat}, {sevilla_lng})")
    print("Map style: Street map")
    print("Extent: ±200m from center (400m × 400m area - high detail)")
    
    # Download high-detail street map of Sevilla
    download_openstreet_map(
        lon=sevilla_lng,
        lat=sevilla_lat,
        distance_x=200,     # Smaller area for high detail
        distance_y=200,
        site_name="Sevilla - Cathedral District",
        style="map",
        output_file=os.path.join(output_dir, "sevilla_cathedral_district_detailed.png"),
        show_plot=False
    )
    
    print("✅ Sevilla detailed map displayed successfully")
    
except Exception as e:
    print(f"❌ Error downloading Sevilla map: {e}")

# ============================================================================
# EXAMPLE 4: VALENCIA REGIONAL VIEW
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: VALENCIA REGIONAL VIEW")
print("="*70)

# Valencia coordinates (City of Arts and Sciences)
valencia_lat = 39.4556
valencia_lng = -0.3477

try:
    print(f"Downloading regional view for Valencia ({valencia_lat}, {valencia_lng})")
    print("Map style: Street map")
    print("Extent: ±2000m from center (4km × 4km area - regional view)")
    
    # Download regional view of Valencia
    download_openstreet_map(
        lon=valencia_lng,
        lat=valencia_lat,
        distance_x=2000,    # Large area for regional context
        distance_y=2000,
        site_name="Valencia - Regional View",
        style="map",
        output_file=os.path.join(output_dir, "valencia_regional_view.png"),
        show_plot=False
    )
    
    print("✅ Valencia regional view displayed successfully")
    
except Exception as e:
    print(f"❌ Error downloading Valencia map: {e}")

# ============================================================================
# EXAMPLE 5: COASTAL AREA - SANTANDER
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: COASTAL AREA - SANTANDER")
print("="*70)

# Santander coordinates (Bay area)
santander_lat = 43.4623
santander_lng = -3.8099

try:
    print(f"Downloading coastal view for Santander ({santander_lat}, {santander_lng})")
    print("Map style: Satellite imagery")
    print("Extent: ±1500m from center (3km × 3km area - coastal features)")
    
    # Download coastal satellite view
    download_openstreet_map(
        lon=santander_lng,
        lat=santander_lat,
        distance_x=1500,    # Good for showing coastal features
        distance_y=1500,
        site_name="Santander - Coastal Bay",
        style="satellite",
        output_file=os.path.join(output_dir, "santander_coastal_bay_satellite.png"),
        show_plot=False
    )
    
    print("✅ Santander coastal view displayed successfully")
    
except Exception as e:
    print(f"❌ Error downloading Santander map: {e}")

# ============================================================================
# EXAMPLE 6: MOUNTAIN AREA - PICOS DE EUROPA
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: MOUNTAIN AREA - PICOS DE EUROPA")
print("="*70)

# Picos de Europa coordinates (Naranjo de Bulnes area)
picos_lat = 43.2064
picos_lng = -4.8558

try:
    print(f"Downloading mountain view for Picos de Europa ({picos_lat}, {picos_lng})")
    print("Map style: Street map")
    print("Extent: ±3000m from center (6km × 6km area - mountain terrain)")
    
    # Download mountain terrain map
    download_openstreet_map(
        lon=picos_lng,
        lat=picos_lat,
        distance_x=3000,    # Large area for mountain context
        distance_y=3000,
        site_name="Picos de Europa - Mountain Terrain",
        style="map",
        output_file=os.path.join(output_dir, "picos_de_europa_mountain_terrain.png"),
        show_plot=False
    )
    
    print("✅ Picos de Europa mountain view displayed successfully")
    
except Exception as e:
    print(f"❌ Error downloading Picos de Europa map: {e}")

# ============================================================================
# EXAMPLE 7: CUSTOM EXTENT - RECTANGULAR AREAS
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: CUSTOM EXTENT - RECTANGULAR AREAS")
print("="*70)

# Bilbao coordinates (Guggenheim area)
bilbao_lat = 43.2687
bilbao_lng = -2.9340

try:
    print(f"Downloading custom extent for Bilbao ({bilbao_lat}, {bilbao_lng})")
    print("Map style: Street map")
    print("Extent: ±800m E-W, ±400m N-S (1.6km × 0.8km - rectangular area)")
    
    # Download rectangular area (different x and y distances)
    download_openstreet_map(
        lon=bilbao_lng,
        lat=bilbao_lat,
        distance_x=800,     # Wider east-west extent
        distance_y=400,     # Narrower north-south extent
        site_name="Bilbao - Guggenheim District (Custom Extent)",
        style="map",
        output_file=os.path.join(output_dir, "bilbao_guggenheim_custom_extent.png"),
        show_plot=False
    )
    
    print("✅ Bilbao custom extent displayed successfully")
    
except Exception as e:
    print(f"❌ Error downloading Bilbao map: {e}")

# ============================================================================
# EXAMPLE 8: BATCH PROCESSING - MULTIPLE CITIES
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 8: BATCH PROCESSING - MULTIPLE CITIES")
print("="*70)

# Define multiple Spanish cities for comparison
spanish_cities = [
    {
        "name": "Granada - Alhambra",
        "lat": 37.1760,
        "lng": -3.5881,
        "distance": 600,
        "style": "satellite"
    },
    {
        "name": "Toledo - Historic Center", 
        "lat": 39.8628,
        "lng": -4.0273,
        "distance": 400,
        "style": "map"
    },
    {
        "name": "San Sebastian - La Concha",
        "lat": 43.3183,
        "lng": -1.9812,
        "distance": 800,
        "style": "satellite"
    },
    {
        "name": "Cordoba - Mezquita",
        "lat": 37.8790,
        "lng": -4.7794,
        "distance": 300,
        "style": "map"
    }
]

successful_downloads = 0
total_cities = len(spanish_cities)

print(f"Processing {total_cities} cities in batch mode...")
print("Note: Close each map window to proceed to the next one.\n")

for i, city in enumerate(spanish_cities, 1):
    try:
        print(f"[{i}/{total_cities}] Processing {city['name']}...")
        print(f"   Coordinates: ({city['lat']}, {city['lng']})")
        print(f"   Style: {city['style']}, Extent: ±{city['distance']}m")
        
        # Generate safe filename from city name
        safe_filename = city['name'].lower().replace(' - ', '_').replace(' ', '_').replace('.', '')
        output_filename = f"{safe_filename}_{city['style']}.png"
        
        download_openstreet_map(
            lon=city['lng'],
            lat=city['lat'],
            distance_x=city['distance'],
            distance_y=city['distance'],
            site_name=city['name'],
            style=city['style'],
            output_file=os.path.join(output_dir, output_filename),
            show_plot=False
        )
        
        successful_downloads += 1
        print(f"   ✅ {city['name']} processed successfully")
        
    except Exception as e:
        print(f"   ❌ Error processing {city['name']}: {e}")
    
    print()

# ============================================================================
# ADVANCED USAGE EXAMPLES
# ============================================================================

print("="*70)
print("ADVANCED USAGE EXAMPLES")
print("="*70)

# Example: Using the direct create_osm_image function for more control
print("\nAdvanced Example: Direct function usage with custom parameters")

try:
    # Zaragoza coordinates (Basilica del Pilar)
    zaragoza_lat = 41.6564
    zaragoza_lng = -0.8784
    
    print(f"Creating custom OSM image for Zaragoza ({zaragoza_lat}, {zaragoza_lng})")
    print("Using create_osm_image function directly for maximum control")
    
    # Use the direct function for more control
    create_osm_image(
        lon=zaragoza_lng,
        lat=zaragoza_lat,
        site_name="Zaragoza - Basilica del Pilar (Advanced)",
        style="map",
        distance_x=1000,
        distance_y=600,
        output_file=os.path.join(output_dir, "zaragoza_basilica_pilar_advanced.png"),
        show_plot=False
    )
    
    print("✅ Advanced Zaragoza map created successfully")
    
except Exception as e:
    print(f"❌ Error in advanced example: {e}")

# ============================================================================
# RESULTS SUMMARY AND BEST PRACTICES
# ============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"📁 Output directory: {output_dir}")
print(f"Batch processing results: {successful_downloads}/{total_cities} cities successful")

# List all downloaded files
try:
    if os.path.exists(output_dir):
        downloaded_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"\n📊 Total images saved: {len(downloaded_files)}")
        
        if downloaded_files:
            print("\n📋 Downloaded files:")
            for file in sorted(downloaded_files):
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024 / 1024  # Size in MB
                    print(f"  📁 {file} ({file_size:.2f} MB)")
    else:
        print(f"⚠️ Output directory does not exist: {output_dir}")
        
except Exception as e:
    print(f"❌ Error listing files: {e}")

print("\n📊 Scale and Distance Guidelines:")
print("   • distance_x/y < 300m  → Very high detail (buildings, streets)")
print("   • distance_x/y 300-1000m → High detail (districts, neighborhoods)")  
print("   • distance_x/y 1000-3000m → Medium detail (city areas, regions)")
print("   • distance_x/y > 3000m → Low detail (regional overview)")

print("\n🗺️ Style Recommendations:")
print("   • 'map': Best for urban areas, road networks, administrative boundaries")
print("   • 'satellite': Best for natural features, coastal areas, land use analysis")

print("\n⚠️ Important Considerations:")
print("   • Larger areas require longer download times")
print("   • Very high zoom levels may not be available in all regions")
print("   • Respect OSM tile usage policies for automated downloading")
print("   • Close map windows to proceed in batch processing")

print("\n" + "="*70)

# ============================================================================
# CONFIGURATION EXAMPLES FOR SPECIFIC USE CASES
# ============================================================================

# ----------------------------------------------------------------------------
# Example: Environmental Monitoring Sites
# ----------------------------------------------------------------------------

environmental_monitoring_sites = [
    {
        "name": "Donana National Park",
        "lat": 37.0042,
        "lng": -6.4686,
        "distance_x": 5000,
        "distance_y": 5000,
        "style": "satellite",
        "purpose": "Wetland monitoring"
    },
    {
        "name": "Tablas de Daimiel",
        "lat": 39.1486, 
        "lng": -3.7069,
        "distance_x": 3000,
        "distance_y": 3000,
        "style": "satellite",
        "purpose": "Wetland conservation"
    },
    {
        "name": "Cabo de Gata Natural Park",
        "lat": 36.7213,
        "lng": -2.1544,
        "distance_x": 4000,
        "distance_y": 4000,
        "style": "satellite",
        "purpose": "Coastal ecosystem monitoring"
    }
]

# To execute environmental monitoring downloads:
# for site in environmental_monitoring_sites:
#     download_openstreet_map(
#         lon=site["lng"],
#         lat=site["lat"], 
#         distance_x=site["distance_x"],
#         distance_y=site["distance_y"],
#         site_name=f"{site['name']} - {site['purpose']}",
#         style=site["style"]
#     )

# ----------------------------------------------------------------------------
# Example: Urban Planning Analysis
# ----------------------------------------------------------------------------

urban_planning_config = {
    "high_detail_districts": {
        "distance_range": (200, 500),
        "style": "map",
        "zoom_equivalent": "14-16",
        "best_for": "Street-level planning, pedestrian areas"
    },
    "neighborhood_overview": {
        "distance_range": (500, 1500), 
        "style": "map",
        "zoom_equivalent": "12-14",
        "best_for": "District planning, transportation networks"
    },
    "city_context": {
        "distance_range": (1500, 5000),
        "style": "map", 
        "zoom_equivalent": "8-12",
        "best_for": "Regional planning, metropolitan areas"
    }
}

# ----------------------------------------------------------------------------
# Example: Research Documentation
# ----------------------------------------------------------------------------

research_documentation_config = {
    "field_site_detail": {
        "distance_x": 100,
        "distance_y": 100,
        "style": "satellite",
        "description": "Very high detail for specific research plots"
    },
    "study_area_context": {
        "distance_x": 1000,
        "distance_y": 1000, 
        "style": "map",
        "description": "Medium scale for study area context"
    },
    "regional_setting": {
        "distance_x": 10000,
        "distance_y": 10000,
        "style": "map",
        "description": "Large scale for regional geographic context"
    }
}

# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================
#
# Common Issues and Solutions:
#
# 1. Tcl/Tk Backend Errors (Windows):
#    - Error: "Can't find usable init.tcl" 
#    - Solution: Script automatically uses 'Agg' backend (non-interactive)
#    - Images are saved to files instead of displayed interactively
#    - Alternative: Install tkinter: pip install tk
#
# 2. Cartopy Installation Issues:
#    - Windows: Use conda instead of pip: conda install -c conda-forge cartopy
#    - Linux: Ensure GEOS, PROJ, and GDAL libraries are installed
#    - macOS: Use homebrew for dependencies: brew install proj geos
#
# 3. Slow Download Times:
#    - Reduce distance_x and distance_y parameters
#    - Check internet connection stability
#    - OSM servers may be temporarily slow
#
# 3. Empty or Missing Tiles:
#    - Some remote areas may not have high-resolution tiles
#    - Try different style ('map' vs 'satellite')
#    - Reduce zoom level (increase distance parameters)
#
# 4. Memory Issues with Large Areas:
#    - Large distance parameters require more RAM
#    - Process areas in smaller chunks
#    - Close matplotlib figures between downloads
#
# 5. Coordinate System Issues:
#    - Ensure coordinates are in decimal degrees
#    - Latitude: -90 to +90, Longitude: -180 to +180
#    - Use positive values for North/East, negative for South/West
#
# 6. Display Issues:
#    - Ensure matplotlib backend supports interactive display
#    - Try plt.show(block=True) for non-interactive environments
#    - Check display settings in remote/SSH environments
#
# ============================================================================