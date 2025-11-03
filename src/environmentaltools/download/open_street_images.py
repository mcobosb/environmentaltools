"""
OpenStreetMap Image Download and Visualization Module.

This module provides functionality to download and display OpenStreetMap images
(both satellite and map styles) for specified geographic locations using cartopy.
"""

import io
from typing import Literal, Tuple
from urllib.request import Request, urlopen

import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.io.img_tiles as cimgt
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def download_openstreet_map(
    lon: float,
    lat: float,
    distance_x: float,
    distance_y: float,
    site_name: str,
    style: Literal["map", "satellite"] = "satellite",
) -> None:
    """
    Download and display OpenStreetMap image for a specified location.
    
    This is a convenience wrapper function for creating OpenStreetMap visualizations
    with either satellite imagery or street map style.
    
    Args:
        lon: Longitude of the center point in degrees.
        lat: Latitude of the center point in degrees.
        distance_x: Distance from center to edge in the x-direction (meters).
        distance_y: Distance from center to edge in the y-direction (meters).
        site_name: Name of the site/location for the plot title.
        style: Map style, either 'map' for street map or 'satellite' for satellite imagery.
            Default is 'satellite'.
    
    Returns:
        None. Displays the map using matplotlib.
    
    Example:
        >>> download_openstreet_map(
        ...     lon=-3.7038,
        ...     lat=40.4168,
        ...     distance_x=500,
        ...     distance_y=500,
        ...     site_name="Madrid",
        ...     style="satellite"
        ... )
    """
    create_osm_image(
        lon=lon,
        lat=lat,
        site_name=site_name,
        style=style,
        distance_x=distance_x,
        distance_y=distance_y,
    )


def create_osm_image(
    lon: float,
    lat: float,
    site_name: str = "Location",
    style: Literal["map", "satellite"] = "satellite",
    distance_x: float = 500,
    distance_y: float = 500,
) -> None:
    """
    Create and display an OpenStreetMap image with customizable style and extent.
    
    This function downloads OpenStreetMap tiles (either satellite imagery or street map)
    and displays them using matplotlib with cartopy projections. The zoom level is
    automatically calculated based on the requested extent.
    
    Args:
        lon: Longitude of the center point in degrees.
        lat: Latitude of the center point in degrees.
        site_name: Name of the site/location for the plot title. Default is "Location".
        style: Map style, either 'map' for street map or 'satellite' for satellite imagery.
            Default is 'satellite'.
        distance_x: Distance from center to edge in the x-direction (meters). Default is 500.
        distance_y: Distance from center to edge in the y-direction (meters). Default is 500.
    
    Returns:
        None. Displays the map using matplotlib.
    
    Notes:
        - Scale (zoom level) is automatically calculated based on the maximum distance.
        - According to OSM policies, avoid both large scale (>16) and large radius (>1000).
        - Scale guidelines:
            * 2: Coarse image for worldwide or continental scales
            * 4-6: Medium coarseness for countries and larger states
            * 6-10: Medium fineness for smaller states, regions, and cities
            * 10-12: Fine image for city boundaries and zip codes
            * 14+: Extremely fine image for roads, blocks, buildings
    
    References:
        OSM Tile Usage Policy: https://operations.osmfoundation.org/policies/tiles/
    """
    # Configure tile provider based on requested style
    if style == "map":
        # Street map style
        cimgt.OSM.get_image = _image_spoof  # Reformat web request for tile spoofing
        img = cimgt.OSM()
    elif style == "satellite":
        # Satellite imagery style
        cimgt.QuadtreeTiles.get_image = _image_spoof  # Reformat web request
        img = cimgt.QuadtreeTiles()
    else:
        raise ValueError(
            f"Invalid style '{style}'. Must be either 'map' or 'satellite'."
        )

    # Create the figure and projection
    plt.close("all")
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=img.crs)  # Use tile provider's CRS
    
    # Define data coordinate reference system (UTM zone 30)
    data_crs = ccrs.UTM(zone=30)

    ax.set_title(f"{site_name} ({lat}, {lon})", fontsize=15)

    # Auto-calculate zoom scale based on extent
    radius = np.max([distance_x, distance_y])
    scale = int(120 / np.log(radius))
    scale = min(scale, 19)  # Cap at maximum zoom level 19

    # Calculate and set map extent
    extent = calculate_extent(lon, lat, distance_x, distance_y)
    ax.set_extent(extent)
    ax.add_image(img, int(scale))  # Add OSM tiles with calculated zoom level

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, crs=data_crs, color="k", lw=0.5)
    gl.top_labels = False
    gl.right_labels = False

    plt.show()


def calculate_extent(
    lon: float, lat: float, distance_x: float, distance_y: float
) -> list[float]:
    """
    Calculate the geographic extent (bounding box) for a map centered at a location.
    
    This function uses geodesic calculations to determine the map extent based on
    the center point and distances from the center to the edges.
    
    Args:
        lon: Longitude of the center point in degrees.
        lat: Latitude of the center point in degrees.
        distance_x: Distance from center to edge in the x-direction (meters).
        distance_y: Distance from center to edge in the y-direction (meters).
    
    Returns:
        A list of four floats [lon_min, lon_max, lat_min, lat_max] representing
        the bounding box extent in degrees.
    
    Notes:
        Uses cartopy's Geodesic class to accurately calculate positions on the
        Earth's surface, accounting for the Earth's curvature.
    """
    # Calculate the angle from center to corner
    angle = np.rad2deg(np.arctan(distance_y / distance_x))

    # Calculate the distance from center to corner
    dist_corner = np.sqrt(distance_x**2 + distance_y**2)
    
    # Calculate top-left corner coordinates
    top_left = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=-(90 - angle), distances=dist_corner
    )[:, 0:2][0]
    
    # Calculate bottom-right corner coordinates
    bottom_right = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=90 + angle, distances=dist_corner
    )[:, 0:2][0]

    # Return extent as [lon_min, lon_max, lat_min, lat_max]
    extent = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]

    return extent


def _image_spoof(self, tile) -> Tuple[Image.Image, tuple, str]:
    """
    Reformat web requests from OpenStreetMap for cartopy compatibility.
    
    This internal function modifies the default tile request behavior to include
    proper user agent headers, which are required by OpenStreetMap's tile servers.
    
    Args:
        self: The tile provider instance (OSM or QuadtreeTiles).
        tile: The tile coordinates to request.
    
    Returns:
        A tuple containing:
            - PIL Image object with the tile image
            - Tile extent coordinates
            - Origin position ('lower')
    
    Notes:
        This function is used internally to "spoof" the tile request by adding
        appropriate headers. It replaces the default get_image method of tile
        providers.
        
        Heavily based on code by Joshua Hrisko:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """
    # Get the URL for the requested tile
    url = self._image_url(tile)
    
    # Create HTTP request with proper user agent header
    req = Request(url)
    req.add_header("User-agent", "Anaconda 3")
    
    # Fetch the tile image
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())
    fh.close()
    
    # Open and convert image to the desired format
    img = Image.open(im_data)
    img = img.convert(self.desired_tile_form)
    
    # Return image, extent, and origin for cartopy
    return img, self.tileextent(tile), "lower"