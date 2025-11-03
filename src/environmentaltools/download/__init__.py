"""Download module for environmental data acquisition.

This module provides utilities for downloading environmental data from various
sources including CORDEX climate data, Google Earth Engine, Marine Copernicus,
and satellite imagery.

Submodules:
    cordex-data: Download CORDEX climate model data
    google-earth-engine: Download satellite imagery from Google Earth Engine
    google-image: Download Google Maps imagery
    open-street-images: Download OpenStreetMap imagery
    marine-copernicus: Download marine data from Copernicus Marine Service
"""

# CORDEX data functions
from .cordex_data import (
    parse_wget_script_to_queries,
    download_cordex_data,
    batch_download_with_config,
    query_esgf_catalog,
    filter_esgf_queries,
    download_esgf_dataset,
    search_and_download_cordex,
)

# Google Earth Engine functions
from .google_earth_engine import (
    initialize_earth_engine,
    create_study_area_geometry,
    calculate_vegetation_indices,
    create_sentinel2_collection,
    download_image_with_geemap,
    download_single_sentinel2_image,
    download_sentinel2_timeseries,
)

# Google Image functions
from .google_image import (
    GoogleMapsLayers,
    GoogleMapDownloader,
    download_google_maps_image,
)

# OpenStreetMap functions
from .open_street_images import (
    download_openstreet_map,
    create_osm_image,
    calculate_extent,
)

# Marine Copernicus functions
from .marine_copernicus import (
    ERA5DataDownloadConfig,
    ERA5DataDownloader,
    ERA5DataProcessor,
    download_era5_data,
)

__all__ = [
    # CORDEX data
    "parse_wget_script_to_queries",
    "download_cordex_data",
    "batch_download_with_config",
    "query_esgf_catalog",
    "filter_esgf_queries",
    "download_esgf_dataset",
    "search_and_download_cordex",
    # Google Earth Engine
    "initialize_earth_engine",
    "create_study_area_geometry",
    "calculate_vegetation_indices",
    "create_sentinel2_collection",
    "download_image_with_geemap",
    "download_single_sentinel2_image",
    "download_sentinel2_timeseries",
    # Google Image
    "GoogleMapsLayers",
    "GoogleMapDownloader",
    "download_google_maps_image",
    # OpenStreetMap
    "download_openstreet_map",
    "create_osm_image",
    "calculate_extent",
    # Marine Copernicus
    "ERA5DataDownloadConfig",
    "ERA5DataDownloader",
    "ERA5DataProcessor",
    "download_era5_data",
]
