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

from typing import Dict, Union, List, Optional
import pandas as pd

# CORDEX data functions - Hybrid approach
# Import both implementations 
try:
    from .cordex_data_intake import (
        query_esgf_catalog_intake,
        download_esgf_dataset_intake,
        download_with_config_intake,
    )
    HAS_INTAKE_ESGF = True
except ImportError:
    HAS_INTAKE_ESGF = False

try:
    from .cordex_data import (
        query_esgf_catalog_pyesgf,
        download_esgf_dataset_pyesgf,
        download_with_config as download_with_config_pyesgf,
    )
    HAS_PYESGF = True
except ImportError:
    HAS_PYESGF = False


def query_esgf_catalog(
    query: Dict[str, Union[str, List[str]]],
    indices: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """Query ESGF catalog with automatic fallback for CORDEX projects.
    
    Uses intake-esgf for supported projects, falls back to PyESGF for CORDEX.
    
    Args:
        query: Dictionary of query parameters
        indices: List of ESGF indices to search
        **kwargs: Additional arguments
        
    Returns:
        pd.DataFrame: DataFrame containing metadata for matching datasets
        
    Raises:
        ImportError: If neither intake-esgf nor PyESGF is available
    """
    project = query.get('project', '').lower()
    
    # Check if this is a CORDEX project or if intake-esgf is not available
    if project == 'cordex' or not HAS_INTAKE_ESGF:
        if not HAS_PYESGF:
            raise ImportError(
                "PyESGF is required for CORDEX projects. "
                "Install with: pip install esgf-pyclient"
            )
        return query_esgf_catalog_pyesgf(query, indices, **kwargs)
    
    # Use intake-esgf for other projects
    try:
        return query_esgf_catalog_intake(query, indices, **kwargs)
    except Exception as e:
        if 'ProjectNotSupported' in str(e) and HAS_PYESGF:
            # Fallback to PyESGF if project not supported by intake-esgf
            return query_esgf_catalog_pyesgf(query, indices, **kwargs)
        raise


def download_esgf_dataset(
    dataset_metadata: Dict,
    output_folder: str,
    file_filter: Optional[str] = None,
    **kwargs
) -> List[str]:
    """Download ESGF dataset with automatic implementation selection.
    
    Args:
        dataset_metadata: Dataset metadata from query_esgf_catalog
        output_folder: Directory to save downloaded files
        file_filter: Optional filter for specific files
        **kwargs: Additional arguments
        
    Returns:
        List[str]: List of downloaded file paths
    """
    project = dataset_metadata.get('project', '').lower()
    
    # Use PyESGF for CORDEX or if intake-esgf not available
    if project == 'cordex' or not HAS_INTAKE_ESGF:
        if not HAS_PYESGF:
            raise ImportError(
                "PyESGF is required for CORDEX projects. "
                "Install with: pip install esgf-pyclient"
            )
        return download_esgf_dataset_pyesgf(dataset_metadata, output_folder, file_filter, **kwargs)
    
    # Use intake-esgf for other projects
    try:
        return download_esgf_dataset_intake(dataset_metadata, output_folder, file_filter, **kwargs)
    except Exception as e:
        if HAS_PYESGF:
            # Fallback to PyESGF
            return download_esgf_dataset_pyesgf(dataset_metadata, output_folder, file_filter, **kwargs)
        raise


def download_with_config(output_folder: str, config_file: str = "download_config.ini") -> List[str]:
    """Download ESGF data using configuration file with automatic implementation selection.
    
    Args:
        output_folder: Directory to save downloaded files
        config_file: Path to configuration file
        
    Returns:
        List[str]: List of downloaded file paths
    """
    # Read config to determine project
    from configobj import ConfigObj
    config = ConfigObj(config_file)
    project = config.get('project', '').lower()
    
    # Use PyESGF for CORDEX or if intake-esgf not available  
    if project == 'cordex' or not HAS_INTAKE_ESGF:
        if not HAS_PYESGF:
            raise ImportError(
                "PyESGF is required for CORDEX projects. "
                "Install with: pip install esgf-pyclient"
            )
        return download_with_config_pyesgf(output_folder, config_file)
    
    # Use intake-esgf for other projects
    try:
        return download_with_config_intake(output_folder, config_file)
    except Exception as e:
        if HAS_PYESGF:
            # Fallback to PyESGF
            return download_with_config_pyesgf(output_folder, config_file)
        raise


# Google Earth Engine functions
from .google_earth_engine import (
    initialize_earth_engine,
    create_study_area_geometry,
    calculate_vegetation_indices,
    create_sentinel2_collection,
    download_image_with_geemap,
    download_single_sentinel2_image,
    download_sentinel2_images,
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
    # CORDEX data (hybrid: intake-esgf + PyESGF fallback)
    "query_esgf_catalog",
    "download_esgf_dataset", 
    "download_with_config",
    # Direct access to specific implementations
    "query_esgf_catalog_intake",
    "download_esgf_dataset_intake", 
    "download_with_config_intake",
    # Google Earth Engine
    "initialize_earth_engine",
    "create_study_area_geometry",
    "calculate_vegetation_indices",
    "create_sentinel2_collection",
    "download_image_with_geemap",
    "download_single_sentinel2_image",
    "download_sentinel2_images",
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
