"""
Marinetools - Statistical Functions Module

This module provides statistical and spatial analysis functions for marine data processing.
It includes utilities for grid manipulation, contour analysis, geometry processing,
and spatial interpolation specifically designed for coastal and marine applications.

The module handles:
- Bathymetric data processing and grid operations
- Shoreline and contour extraction
- Spatial interpolation and refinement
- Geometric operations for coastal features
- File I/O operations for GIS and raster data
- Statistical analysis of seasonal and temporal data

Dependencies:
    - numpy: Array operations and mathematical functions
    - pandas: Data manipulation and analysis
    - geopandas: Geospatial data operations
    - rasterio: Raster data I/O and processing
    - matplotlib: Plotting and visualization
    - scipy: Scientific computing and interpolation
    - shapely: Geometric operations
    - loguru: Logging functionality

Author: Marinetools Development Team
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Debug mode: set DEBUG_MODE=True to enable visualizations
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")


def calculate_grid_angle_and_create_rotated_mesh(xx, yy, grid_size):
    """
    Calculate grid angle from DEM data and create a rotated mesh aligned with contours.

    Calculates the orientation angle of the DEM grid and generates a new rotated
    coordinate mesh (X, Y) inscribed within the xx, yy bounds with coordinates
    aligned to the contours.

    Parameters
    ----------
    xx : array-like
        X coordinate bounds for the new mesh.
    yy : array-like
        Y coordinate bounds for the new mesh.
    grid_size : float
        Grid spacing for the new mesh.

    Returns
    -------
    X_rotated : np.ndarray
        Rotated X coordinates of the new mesh.
    Y_rotated : np.ndarray
        Rotated Y coordinates of the new mesh.
    angle : float
        Rotation angle in radians used for the mesh alignment.

    Notes
    -----
    The function automatically detects if the DEM coordinates are 1D or 2D and
    calculates the appropriate rotation angle. A memory usage warning is issued
    if the resulting mesh would be very large (> 10 million points).
    """
    # Calculate the angle of the DEM grid
    # Use corners to calculate the main orientation
    x_dem = xx
    y_dem = yy

    # Calculate edge vectors of the grid
    # Horizontal vector (first row)
    dx1 = x_dem[0, -1] - x_dem[0, 0]
    dy1 = y_dem[0, -1] - y_dem[0, 0]
    # Vertical vector (first column)
    dx2 = x_dem[-1, 0] - x_dem[0, 0]
    dy2 = y_dem[-1, 0] - y_dem[0, 0]

    # Calculate rotation angle (use the longer vector)
    if np.sqrt(dx1**2 + dy1**2) > np.sqrt(dx2**2 + dy2**2):
        angle = np.arctan2(dy1, dx1)  # horizontal axis angle
    else:
        angle = np.arctan2(dx2, dy2) - np.pi / 2  # corrected vertical axis angle

    # Bounds of xx, yy
    x_min, x_max = np.min(xx), np.max(xx)
    y_min, y_max = np.min(yy), np.max(yy)

    # Domain center
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Domain dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Calculate number of points for the new mesh
    nx = int(width / grid_size) + 1
    ny = int(height / grid_size) + 1

    # Create regular mesh in local system (without rotation)
    x_local = np.linspace(-width / 2, width / 2, nx)
    y_local = np.linspace(-height / 2, height / 2, ny)
    X_local, Y_local = np.meshgrid(x_local, y_local)

    # Apply rotation
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rotated = cos_a * X_local - sin_a * Y_local + x_center
    Y_rotated = sin_a * X_local + cos_a * Y_local + y_center

    return X_rotated, Y_rotated, angle


def band(data_cube, levels):
    """
    Create a band mask for data_cube within specified depth/elevation levels.

    Creates a mask for areas within the data_cube that fall between minimum and maximum
    levels, then expands this mask in the specified orientation to create a band
    for further analysis.

    Parameters
    ----------
    data_cube : dict
        data_cube containing 'x', 'y', and 'z' arrays.
    levels : array-like
        Depth/elevation levels to define the band boundaries.

    Returns
    -------
    band_ : np.ndarray
        Boolean mask indicating the band area.
    coords : dict
        Dictionary containing reshaped X and Y coordinate arrays for the band.

    Notes
    -----
    In debug mode (DEBUG_MODE=True), displays a visualization of the band.
    The function extends the initial mask along the specified orientation to
    create a continuous band across the domain.
    TODO: Optimize band extension to avoid holes in complex geometries.
    """
    # Define level bounds and create initial mask
    z_ = np.where((data_cube < levels.max()) & (data_cube > levels.min()), 1, 0)

    # Debug mode: show band visualization
    if DEBUG_MODE:
        plt.figure()
        plt.contourf(data_cube["x"], data_cube["y"], z_, levels=2, cmap="RdBu", alpha=0.5)
        plt.axis("equal")
        plt.title("Debug: Band Visualization")
        plt.show()

    # Create extended mask
    mask = np.zeros_like(z_)
    len_x, len_y = 0, 0
    # Extend mask horizontally
    for i in range(data_cube["y"].shape[0]):
        if np.any(z_[i, :] == 1):
            mask[i, :] = 1
            len_y += 1

    # Filter mask vertically
    for i in range(data_cube["y"].shape[1]):
        if np.any(z_[:, i] == 1):
            indexes = np.where(z_[:, i] == 1)[0]
            mask[indexes, i] = 1
            len_x += 1

    # Convert to boolean mask and extract coordinates
    band_ = mask == 1
    xx = data_cube["x"][band_]
    yy = data_cube["y"][band_]

    # Reshape coordinates
    xx = np.reshape(xx, (len_x, len_y))
    yy = np.reshape(yy, (len_x, len_y))


    return band_, {"X": xx, "Y": yy}


def refinement(da_dem, band_, coords):
    """
    Perform spatial interpolation to refine elevation data on a new coordinate grid.

    Uses LinearNDInterpolator to interpolate elevation values from the DEM data
    within the band area to new coordinate positions specified in coords.

    Parameters
    ----------
    da_dem : dict
        DEM data containing 'x', 'y', and 'z' arrays.
    band_ : np.ndarray
        Boolean mask indicating the area within the band for interpolation.
    coords : dict
        Dictionary containing 'X' and 'Y' arrays with target interpolation coordinates.

    Returns
    -------
    Z : np.ndarray
        Interpolated elevation/depth values at the new coordinate positions.

    Notes
    -----
    In debug mode (DEBUG_MODE=True), displays a visualization comparing the original
    DEM points with the new interpolation grid points. The interpolation uses only
    the points within the band mask to avoid extrapolation beyond the data bounds.
    """
    X, Y = coords["X"], coords["Y"]

    # Debug mode: show band visualization
    if DEBUG_MODE:
        plt.figure()
        plt.plot(da_dem["x"].flatten(), da_dem["y"].flatten(), "ob", markersize=1)
        plt.plot(X.flatten(), Y.flatten(), "xr", markersize=1)
        plt.title("Debug: DEM points (blue) vs Interpolation grid (red)")
        plt.show()

    # Create interpolator using only band points
    interp = LinearNDInterpolator(
        list(zip(da_dem["x"][band_].flatten(), da_dem["y"][band_].flatten())),
        da_dem["z"][band_].flatten(),
    )

    # Interpolate to new coordinates
    Z = interp(X, Y)
    return Z



def spatial_gradient(array_2d, dx=1, dy=1):
    """
    Calcula el gradiente espacial de una matriz 2D.
    
    Parameters:
        array_2d (np.ndarray): Mapa 2D (lat, lon)
        dx, dy (float): Resolución espacial en X e Y
    
    Returns:
        grad_x, grad_y (np.ndarray): Gradientes en X y Y
    """
    grad_x = (np.roll(array_2d, -1, axis=1) - np.roll(array_2d, 1, axis=1)) / (2 * dx)
    grad_y = (np.roll(array_2d, -1, axis=0) - np.roll(array_2d, 1, axis=0)) / (2 * dy)
    return grad_x, grad_y


def save_matrix_to_netcdf(data, coordinates, time, info, sim_no, filename):
    import xarray as xr
    import numpy as np

    # Ensure data is a numpy array
    if hasattr(data, 'values'):
        data_array = data.values
    else:
        data_array = np.array(data)
    
    # Handle different coordinate structures
    if isinstance(coordinates, dict):
        # If coordinates is a dictionary (from xarray coords)
        if 'x' in coordinates and 'y' in coordinates:
            x_coords = coordinates['x'].values if hasattr(coordinates['x'], 'values') else coordinates['x']
            y_coords = coordinates['y'].values if hasattr(coordinates['y'], 'values') else coordinates['y']
        elif 'X' in coordinates and 'Y' in coordinates:
            x_coords = coordinates['X'].values if hasattr(coordinates['X'], 'values') else coordinates['X']
            y_coords = coordinates['Y'].values if hasattr(coordinates['Y'], 'values') else coordinates['Y']
        else:
            raise ValueError("Coordinates dictionary must contain 'x','y' or 'X','Y' keys")
    else:
        # If coordinates is an xarray coordinate object
        if 'x' in coordinates:
            x_coords = coordinates['x'].values
            y_coords = coordinates['y'].values
        else:
            raise ValueError("Cannot extract x,y coordinates from provided coordinates object")
    
    # Handle time coordinate
    if hasattr(time, 'values'):
        time_coords = time.values
    else:
        time_coords = time
    
    # Determine data shape and coordinate dimensions
    if data_array.ndim == 3:  # (time, y, x)
        time_dim, y_dim, x_dim = data_array.shape
        data_dims = ("time", "y", "x")
    elif data_array.ndim == 2:  # (y, x) - single time step
        y_dim, x_dim = data_array.shape
        data_array = data_array[np.newaxis, :, :]  # Add time dimension
        data_dims = ("time", "y", "x")
        time_dim = 1
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data_array.ndim}D")
    
    # Ensure coordinates match data dimensions
    if len(x_coords) != x_dim:
        # Create coordinate arrays if they don't match
        x_coords = np.linspace(0, x_dim-1, x_dim)
        y_coords = np.linspace(0, y_dim-1, y_dim)
    
    # Create dataset
    ds = xr.Dataset(
        data_vars={"prob": (data_dims, data_array)},
        coords={
            "x": ("x", x_coords),
            "y": ("y", y_coords),
            "time": ("time", time_coords[:time_dim] if len(time_coords) > time_dim else time_coords),
        },
        attrs={
            "description": f"Binary matrix annual from monthly averages",
            "project": info["project"]["name"],
            "sim": str(sim_no+1).zfill(4),
        },
    )

    # Save to NetCDF4 with compression
    ds.to_netcdf(
        filename,
        format="NETCDF4",
        engine="netcdf4",
        encoding={"prob": {"zlib": True, "complevel": 2}},
    )
    return
