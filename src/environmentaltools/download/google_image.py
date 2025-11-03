"""Google Maps static image download utilities.

This module provides classes and functions to download high-resolution Google Maps
images by stitching together map tiles at specified coordinates and zoom levels.
Supports different map layers including satellite, terrain, and roadmap views.
"""

import math
import os
import urllib.request

from PIL import Image


class GoogleMapsLayers:
    """Google Maps layer types for tile requests.
    
    Attributes:
        ROADMAP (str): Standard roadmap view with streets and labels.
        TERRAIN (str): Terrain view showing elevation and natural features.
        ALTERED_ROADMAP (str): Alternative roadmap style.
        SATELLITE (str): Satellite imagery without labels.
        TERRAIN_ONLY (str): Terrain-only view without labels.
        HYBRID (str): Satellite imagery with street labels overlay.
    """
    ROADMAP = "v"
    TERRAIN = "p"
    ALTERED_ROADMAP = "r"
    SATELLITE = "s"
    TERRAIN_ONLY = "t"
    HYBRID = "y"


class GoogleMapDownloader:
    """Download and stitch Google Maps tiles into high-resolution images.
    
    This class generates high-resolution Google Maps images by downloading
    and stitching together multiple map tiles based on geographic coordinates
    and zoom level.
    
    Attributes:
        _lat (float): Latitude of the location.
        _lng (float): Longitude of the location.
        _zoom (int): Zoom level (0-23, where higher values show more detail).
        _layer (str): Map layer type from GoogleMapsLayers.
    
    Example:
        >>> downloader = GoogleMapDownloader(
        ...     lat=40.7128,
        ...     lng=-74.0060,
        ...     zoom=15,
        ...     layer=GoogleMapsLayers.SATELLITE
        ... )
        >>> image = downloader.generate_image(tile_width=3, tile_height=3)
        >>> image.save("map.png")
    """

    def __init__(
        self,
        lat: float,
        lng: float,
        zoom: int = 12,
        layer: str = GoogleMapsLayers.ROADMAP,
    ):
        """Initialize Google Map Downloader.
        
        Args:
            lat (float): Latitude of the center location in decimal degrees.
                Valid range: -90 to 90.
            lng (float): Longitude of the center location in decimal degrees.
                Valid range: -180 to 180.
            zoom (int, optional): Zoom level ranging from 0 (world view) to 23
                (maximum detail). Defaults to 12.
            layer (str, optional): Map layer type. Use GoogleMapsLayers constants.
                Defaults to GoogleMapsLayers.ROADMAP.
        
        Example:
            >>> gmd = GoogleMapDownloader(40.7128, -74.0060, zoom=15)
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._layer = layer

    def get_tile_coordinates(self) -> tuple[int, int]:
        """Calculate tile coordinates from latitude, longitude, and zoom level.
        
        Converts geographic coordinates (lat/lng) to Google Maps tile coordinates
        using Web Mercator projection.
        
        Returns:
            tuple[int, int]: Tile coordinates as (x, y) where x and y are tile
                indices at the current zoom level.
        
        Example:
            >>> gmd = GoogleMapDownloader(40.7128, -74.0060, zoom=12)
            >>> x, y = gmd.get_tile_coordinates()
            >>> print(f"Tile: ({x}, {y})")
        """
        tile_size = 256

        # Calculate number of tiles at this zoom level (2^zoom)
        num_tiles = 1 << self._zoom

        # Calculate x tile coordinate from longitude
        # Maps longitude from -180..180 to 0..num_tiles
        point_x = (
            (tile_size / 2 + self._lng * tile_size / 360.0) * num_tiles // tile_size
        )

        # Convert latitude to radians and calculate sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calculate y tile coordinate using Mercator projection
        # Projects latitude onto y-axis with log scale
        point_y = (
            (
                (tile_size / 2)
                + 0.5
                * math.log((1 + sin_y) / (1 - sin_y))
                * -(tile_size / (2 * math.pi))
            )
            * num_tiles
            // tile_size
        )

        return int(point_x), int(point_y)

    def generate_image(
        self,
        start_x: int | None = None,
        start_y: int | None = None,
        tile_width: int = 5,
        tile_height: int = 5,
    ) -> Image.Image:
        """Generate high-resolution image by stitching map tiles together.
        
        Downloads individual 256x256 pixel tiles from Google Maps and combines
        them into a single high-resolution image.
        
        Args:
            start_x (int, optional): Top-left tile x-coordinate. If None,
                calculated from lat/lng. Defaults to None.
            start_y (int, optional): Top-left tile y-coordinate. If None,
                calculated from lat/lng. Defaults to None.
            tile_width (int, optional): Number of tiles wide (horizontal).
                Defaults to 5.
            tile_height (int, optional): Number of tiles high (vertical).
                Defaults to 5.
        
        Returns:
            Image.Image: PIL Image object containing the stitched map.
        
        Raises:
            IOError: If tile download fails.
            urllib.error.URLError: If network connection fails.
        
        Example:
            >>> gmd = GoogleMapDownloader(40.7128, -74.0060, zoom=15)
            >>> # Generate 1280x1280 pixel image (5x5 tiles)
            >>> img = gmd.generate_image(tile_width=5, tile_height=5)
            >>> img.save("output.png")
        """
        # Get tile coordinates if not provided
        if start_x is None or start_y is None:
            start_x, start_y = self.get_tile_coordinates()

        # Calculate final image dimensions (256 pixels per tile)
        width = 256 * tile_width
        height = 256 * tile_height

        # Create blank canvas for stitched image
        map_img = Image.new("RGB", (width, height))

        # Download and stitch each tile
        for x in range(tile_width):
            for y in range(tile_height):
                # Construct Google Maps tile URL
                url = (
                    f"https://mt0.google.com/vt?lyrs={self._layer}"
                    f"&x={start_x + x}"
                    f"&y={start_y + y}"
                    f"&z={self._zoom}"
                )

                # Use temporary filename for tile
                temp_tile_file = f"tile_{x}_{y}.png"
                
                try:
                    # Download tile
                    urllib.request.urlretrieve(url, temp_tile_file)

                    # Open and paste tile onto canvas
                    tile_image = Image.open(temp_tile_file)
                    map_img.paste(tile_image, (x * 256, y * 256))

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_tile_file):
                        os.remove(temp_tile_file)

        return map_img


def download_google_maps_image(
    lat: float,
    lng: float,
    zoom: int = 13,
    layer: str = GoogleMapsLayers.SATELLITE,
    tile_width: int = 5,
    tile_height: int = 5,
    output_file: str = "google_maps_image.png",
) -> bool:
    """Download and save a Google Maps image for specified coordinates.
    
    Convenience function that creates a GoogleMapDownloader, generates an image,
    and saves it to disk.
    
    Args:
        lat (float): Latitude of center location in decimal degrees.
        lng (float): Longitude of center location in decimal degrees.
        zoom (int, optional): Zoom level (0-23). Defaults to 13.
        layer (str, optional): Map layer type. Defaults to GoogleMapsLayers.SATELLITE.
        tile_width (int, optional): Number of tiles wide. Defaults to 5.
        tile_height (int, optional): Number of tiles high. Defaults to 5.
        output_file (str, optional): Output filename. Defaults to "google_maps_image.png".
    
    Returns:
        bool: True if successful, False otherwise.
    
    Example:
        >>> # Download satellite image of New York City
        >>> success = download_google_maps_image(
        ...     lat=40.7128,
        ...     lng=-74.0060,
        ...     zoom=15,
        ...     layer=GoogleMapsLayers.SATELLITE,
        ...     tile_width=3,
        ...     tile_height=3,
        ...     output_file="nyc_satellite.png"
        ... )
        >>> if success:
        ...     print("Map downloaded successfully")
    """
    # Create downloader instance
    gmd = GoogleMapDownloader(lat, lng, zoom, layer)

    print(f"Tile coordinates: {gmd.get_tile_coordinates()}")

    try:
        # Generate high-resolution image
        img = gmd.generate_image(
            tile_width=tile_width,
            tile_height=tile_height
        )
        
        # Save image to disk
        img.save(output_file)
        print(f"Map successfully saved to: {output_file}")
        return True
        
    except IOError as e:
        print(
            f"Could not generate the image: {e}\n"
            "Try adjusting the zoom level and checking your coordinates"
        )
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
