"""Google Earth Engine satellite data download utilities.

This module provides functions to download Sentinel-2 satellite imagery and calculate
vegetation indices using Google Earth Engine API. Designed for salt marsh and coastal
vegetation analysis.

Features:
    - Sentinel-2 image collection filtering and download
    - Vegetation indices calculation (SAVI, MSI, LSWI)
    - Robust error handling and retry mechanisms
    - Memory-efficient batch processing
    - Progress tracking and validation
"""

import datetime
import os
from pathlib import Path
from typing import Any
from loguru import logger

# Set HOME environment variable for cross-platform compatibility
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser("~")

# Optional dependencies for Google Earth Engine
import ee
import geemap
HAS_GEE = True



def initialize_earth_engine(project_id: str) -> None:
    """Initialize Google Earth Engine with authentication.

    Attempts to initialize Earth Engine with existing credentials. If that fails,
    triggers the authentication flow and then initializes.

    Args:
        project_id (str): Google Earth Engine project ID.

    Raises:
        RuntimeError: If Earth Engine initialization fails after authentication.

    Example:
        >>> initialize_earth_engine("my-gee-project")
        Earth Engine initialized successfully
    """
    logger.info("Initializing Google Earth Engine...")
    
    try:
        # Try to initialize with existing credentials
        ee.Initialize(project=project_id)
        logger.info(f"Earth Engine initialized with project: {project_id}")
        return
    except Exception as e:
        logger.info("Authentication required...")
        
        # Check for common error types and provide helpful messages
        error_str = str(e)
        if "not registered to use Earth Engine" in error_str:
            logger.error(f"❌ Project '{project_id}' is not registered for Earth Engine")
            logger.error("📋 To fix this:")
            logger.error(f"   1. Visit: https://console.cloud.google.com/earth-engine/configuration?project={project_id}")
            logger.error("   2. Register your project for Earth Engine access")
            logger.error("   3. See: https://developers.google.com/earth-engine/guides/access")
            raise RuntimeError(f"Earth Engine project not registered: {project_id}")
        
        if "PERMISSION_DENIED" in error_str:
            logger.error(f"❌ Permission denied for project '{project_id}'")
            logger.error("📋 To fix this:")
            logger.error("   1. Check if you have access to this Google Cloud project")
            logger.error("   2. Ensure Earth Engine API is enabled")
            logger.error("   3. Run: earthengine authenticate")
            raise RuntimeError(f"Permission denied for Earth Engine project: {project_id}")
    
    # If initialization fails, try authentication first
    try:
        logger.info("Attempting authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        logger.info(f"Earth Engine authenticated and initialized: {project_id}")
    except Exception as e:
        error_str = str(e)
        if "not registered to use Earth Engine" in error_str:
            logger.error(f"❌ Project '{project_id}' is not registered for Earth Engine")
            logger.error("📋 To fix this:")
            logger.error(f"   1. Visit: https://console.cloud.google.com/earth-engine/configuration?project={project_id}")
            logger.error("   2. Register your project for Earth Engine access")
            raise RuntimeError(f"Earth Engine project not registered: {project_id}")
        
        raise RuntimeError(f"Earth Engine initialization failed: {e}")


def create_study_area_geometry(coordinates: list[list[list[float]]]) -> ee.Geometry:
    """Create Earth Engine geometry from coordinate list.

    Args:
        coordinates (list): Nested list of coordinates defining polygon vertices.
            Format: [[[lon1, lat1], [lon2, lat2], ...]]

    Returns:
        ee.Geometry: Earth Engine polygon geometry.

    Example:
        >>> coords = [[[-6.1, 36.8], [-6.0, 36.8], [-6.0, 36.9], [-6.1, 36.9]]]
        >>> geometry = create_study_area_geometry(coords)
    """
    return ee.Geometry.Polygon(coordinates)


def calculate_vegetation_indices(image: ee.Image) -> ee.Image:
    """Calculate vegetation and moisture indices for salt marsh analysis.

    Computes three key indices from Sentinel-2 bands:
    - SAVI: Soil-Adjusted Vegetation Index (reduces soil background effects)
    - MSI: Moisture Stress Index (indicates plant water stress)
    - LSWI: Land Surface Water Index (detects surface water content)

    Args:
        image (ee.Image): Sentinel-2 image with B4 (Red), B8 (NIR), and
            B11 (SWIR1) bands.

    Returns:
        ee.Image: Image with selected bands (B4, B8, B11) plus calculated
            indices (SAVI, MSI, LSWI) and QA60 quality band.

    Note:
        SAVI uses L=0.5 for moderate vegetation coverage typical in salt marshes.

    Example:
        >>> sentinel_image = ee.Image("COPERNICUS/S2_HARMONIZED/...")
        >>> with_indices = calculate_vegetation_indices(sentinel_image)
    """
    # SAVI (Soil-Adjusted Vegetation Index)
    # Formula: ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    # L = 0.5 for moderate vegetation coverage
    L = 0.5
    savi = image.expression(
        "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "L": L
        }
    ).rename("SAVI")
    
    # MSI (Moisture Stress Index)
    # Formula: SWIR1 / NIR
    # Higher values indicate greater moisture stress
    msi = image.expression(
        "SWIR1 / NIR",
        {
            "SWIR1": image.select("B11"),
            "NIR": image.select("B8")
        }
    ).rename("MSI")
    
    # LSWI (Land Surface Water Index)
    # Formula: (NIR - SWIR1) / (NIR + SWIR1)
    # Higher values indicate more surface water
    lswi = image.normalizedDifference(["B8", "B11"]).rename("LSWI")
    
    # Add quality assessment band
    qa = image.select('QA60')
    
    # Return image with essential bands only to reduce file size
    return image.select(['B4', 'B8', 'B11']).addBands([savi, msi, lswi, qa])


def create_sentinel2_collection(
    study_area: ee.Geometry,
    start_date: str = "2015-01-01",
    end_date: str = "2024-01-01",
    cloud_percentage: float = 15.0,
) -> ee.ImageCollection:
    """Create filtered Sentinel-2 image collection with vegetation indices.

    Filters Sentinel-2 imagery based on spatial extent, date range, and cloud cover,
    then calculates vegetation indices for each image.

    Args:
        study_area (ee.Geometry): Geographic area of interest.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            Defaults to "2015-01-01".
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
            Defaults to "2024-01-01".
        cloud_percentage (float, optional): Maximum cloud cover percentage.
            Defaults to 15.0.

    Returns:
        ee.ImageCollection: Filtered collection with vegetation indices.

    Raises:
        ValueError: If no images match the filtering criteria.

    Example:
        >>> area = ee.Geometry.Point([-6.0, 36.8]).buffer(10000)
        >>> collection = create_sentinel2_collection(area, "2020-01-01", "2020-12-31")
    """
    logger.info("Creating Sentinel-2 image collection...")
    
    # Create base collection with filters
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(study_area)
        .filterDate(start_date, end_date)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", cloud_percentage)
        .select(['B4', 'B8', 'B11', 'QA60'])  # Select only needed bands
    )
    
    collection_size = collection.size().getInfo()
    logger.info(f"Found {collection_size} images matching criteria")
    
    if collection_size == 0:
        raise ValueError("No images found matching the criteria")
    
    # Calculate vegetation indices for all images
    logger.info("Calculating vegetation indices...")
    indices_collection = collection.map(calculate_vegetation_indices)
    
    return indices_collection


def download_image_with_geemap(
    image: ee.Image,
    output_file: Path,
    study_area: ee.Geometry,
    scale: int = 10,
) -> bool:
    """Download Earth Engine image to GeoTIFF using geemap.

    Uses geemap's download function to export an Earth Engine image to a local file.
    Validates that the downloaded file exists and has reasonable size.

    Args:
        image (ee.Image): Earth Engine image to download.
        output_file (Path): Output file path for the GeoTIFF.
        study_area (ee.Geometry): Study area for clipping the image.
        scale (int, optional): Spatial resolution in meters. Defaults to 10
            (Sentinel-2 native resolution).

    Returns:
        bool: True if download successful and validated, False otherwise.

    Example:
        >>> img = ee.Image("COPERNICUS/S2_HARMONIZED/20200601T105619_20200601T110145_T30STG")
        >>> area = ee.Geometry.Point([-6.0, 36.8]).buffer(10000)
        >>> success = download_image_with_geemap(img, Path("output.tif"), area)
    """
    try:
        # Download using geemap
        geemap.download_ee_image(
            image,
            str(output_file),
            scale=scale,
            region=study_area,
        )
        
        # Verify file was created and has reasonable size
        if output_file.exists() and output_file.stat().st_size > 1000:  # At least 1KB
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Successfully downloaded: {output_file.name} ({file_size_mb:.2f} MB)")
            return True
        else:
            logger.warning(f"File created but appears incomplete: {output_file}")
            if output_file.exists():
                output_file.unlink()
            return False
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_single_sentinel2_image(
    image_info: dict[str, Any],
    image_index: int,
    total_images: int,
    collection: ee.ImageCollection,
    study_area: ee.Geometry,
    output_directory: Path,
    scale: int = 10,
) -> bool:
    """Download a single Sentinel-2 image with vegetation indices.

    Extracts an image from a collection, clips it to the study area, and downloads
    it using geemap. Skips download if file already exists.

    Args:
        image_info (dict): Image metadata from Earth Engine containing properties
            like 'system:index' and 'system:time_start'.
        image_index (int): Current image index (for progress tracking).
        total_images (int): Total number of images to process.
        collection (ee.ImageCollection): Pre-created collection containing the image.
        study_area (ee.Geometry): Study area geometry for clipping.
        output_directory (Path): Directory where files will be saved.
        scale (int, optional): Spatial resolution in meters. Defaults to 10.

    Returns:
        bool: True if download successful or file exists, False otherwise.

    Example:
        >>> info = {'properties': {'system:index': 'T30STG_20200601', ...}}
        >>> success = download_single_sentinel2_image(
        ...     info, 0, 10, collection, area, Path("./output")
        ... )
    """
    # Extract image metadata
    image_id = image_info['properties']['system:index']
    acquisition_date = image_info['properties'].get('system:time_start', 'unknown')
    
    # Convert timestamp to readable date
    if acquisition_date != 'unknown':
        try:
            date_readable = datetime.datetime.fromtimestamp(
                acquisition_date / 1000
            ).strftime('%Y-%m-%d')
        except Exception:
            date_readable = str(acquisition_date)
    else:
        date_readable = 'unknown'
    
    # Create output filename
    output_file = output_directory / f"sentinel2_indices_{image_id}.tif"
    
    # Skip if file already exists
    if output_file.exists() and output_file.stat().st_size > 1000:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"[{image_index+1}/{total_images}] Skipping existing: {image_id} "
            f"({file_size_mb:.2f} MB)"
        )
        return True
    
    logger.info(f"[{image_index+1}/{total_images}] Downloading: {image_id}")
    logger.info(f"Acquisition date: {date_readable}")
    
    # Reconstruct image from collection
    try:
        # Filter collection for specific image
        target_image = ee.Image(
            collection.filter(ee.Filter.eq('system:index', image_id)).first()
        )
        
        # Clip to study area to reduce size
        target_image = target_image.clip(study_area)
        
        # Verify image exists
        image_properties = target_image.getInfo()
        if not image_properties:
            logger.error(f"Could not retrieve image: {image_id}")
            return False
        
    except Exception as e:
        logger.error(f"Error reconstructing image {image_id}: {e}")
        return False
    
    # Download using geemap
    return download_image_with_geemap(target_image, output_file, study_area, scale)


def download_sentinel2_images(
    config: dict[str, Any]
) -> dict[str, int]:
    """Download time series of Sentinel-2 imagery with vegetation indices.

    Main function that orchestrates the complete download workflow: initialization,
    collection creation, and batch downloading of images.

    Args:
        config (dict): Configuration dictionary containing:
            - project_id (str): Google Earth Engine project ID
            - study_area_coords (list): Coordinates defining the study area polygon
            - output_directory (str or Path): Directory for output files
            - scale (int, optional): Spatial resolution in meters. Default: 10
            - start_date (str, optional): Start date 'YYYY-MM-DD'. Default: "2015-01-01"
            - end_date (str, optional): End date 'YYYY-MM-DD'. Default: "2024-01-01"
            - cloud_percentage (float, optional): Max cloud cover %. Default: 15.0

    Returns:
        dict: Summary statistics with keys:
            - successful (int): Number of successful downloads
            - failed (int): Number of failed downloads
            - total (int): Total number of images processed

    Raises:
        RuntimeError: If Earth Engine initialization fails.
        ValueError: If no images match the filtering criteria.

    Example:
        >>> config = {
        ...     'project_id': 'my-gee-project',
        ...     'study_area_coords': [[[-6.1, 36.8], [-6.0, 36.8], ...]],
        ...     'output_directory': './satellite_images',
        ...     'scale': 10,
        ...     'start_date': '2020-01-01',
        ...     'end_date': '2020-12-31',
        ...     'cloud_percentage': 10.0
        ... }
        >>> results = download_sentinel2_images(config)
        >>> print(f"Downloaded {results['successful']} images")
    """
    # Extract configuration
    project_id = config['project_id']
    study_area_coords = config['study_area_coords']
    output_directory = Path(config['output_directory'])
    scale = config.get('scale', 10)
    start_date = config.get('start_date', '2015-01-01')
    end_date = config.get('end_date', '2024-01-01')
    cloud_percentage = config.get('cloud_percentage', 15.0)
    
    logger.info("=" * 70)
    logger.info("SENTINEL-2 SATELLITE DATA DOWNLOAD")
    logger.info("=" * 70)
    
    # Initialize Earth Engine
    initialize_earth_engine(project_id)
    
    # Create study area geometry
    study_area = create_study_area_geometry(study_area_coords)
    area_km2 = study_area.area().divide(1000000).getInfo()
    logger.info(f"Study area: {area_km2:.2f} km²")
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_directory}")
    
    # Create image collection
    collection = create_sentinel2_collection(
        study_area, start_date, end_date, cloud_percentage
    )
    
    # Get collection metadata
    collection_list = collection.toList(collection.size()).getInfo()
    total_images = len(collection_list)
    
    logger.info("Collection Summary:")
    logger.info(f"  • Total images: {total_images}")
    logger.info(f"  • Date range: {start_date} to {end_date}")
    logger.info(f"  • Max cloud cover: {cloud_percentage}%")
    logger.info(f"  • Spatial resolution: {scale}m")
    logger.info(f"  • Indices: SAVI, MSI, LSWI")
    
    if total_images == 0:
        raise ValueError("No images found. Check date range and study area.")
    
    # Download images
    logger.info("Starting download process...")
    successful_downloads = 0
    
    for i, image_info in enumerate(collection_list):
        if download_single_sentinel2_image(
            image_info, i, total_images, collection, study_area,
            output_directory, scale
        ):
            successful_downloads += 1
            
    
    # Print summary
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETED")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Files saved to: {output_directory}")
    logger.info("=" * 70)
    
    return {
        'successful': successful_downloads,
        'total': total_images,
    }
