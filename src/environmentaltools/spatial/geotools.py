"""Geospatial tools for topography and bathymetry processing.

This module provides functions for merging land and sea elevation data,
spatial masking, coordinate transformations, and polygon operations.
"""


import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Polygon
from scipy.interpolate import griddata

from environmentaltools.spatial import analysis
from environmentaltools.common import utils


def merge_land_sea(topo, bati, linea0, corners, sea="south"):
    """Merge topography and bathymetry datasets using coastline as boundary.
    
    Combines land elevation data (topography) with sea depth data (bathymetry)
    by defining a sea polygon based on the coastline (zero-elevation line) and
    domain boundaries. Removes land topography within the sea area.

    Parameters
    ----------
    topo : pd.DataFrame
        Land topography data with columns ['x', 'y', 'z']
    bati : pd.DataFrame
        Sea bathymetry data with columns ['x', 'y', 'z']
    linea0 : pd.DataFrame
        Coastline points (zero-elevation contour) with columns ['x', 'y']
    corners : pd.DataFrame
        Domain corner coordinates with columns ['x', 'y']
    sea : str, optional
        Sea location relative to land: 'south', 'east', or 'south-east'.
        Default: 'south'

    Returns
    -------
    pd.DataFrame
        Combined topobathymetry data with columns ['x', 'y', 'z']

    Notes
    -----
    The function creates a polygon defining the sea area using:
    - The coastline (zero-elevation line)
    - Domain boundaries (corners)
    - Side configuration (sea parameter)
    
    A map showing the sea polygon is saved as HTML file.
    """

    # Define sea area polygon based on sea location
    if sea == "south":
        # Southern boundary (bottom edge)
        x, y = np.linspace(corners.x[0], corners.x[1], 2), np.tile(corners.y[1], 2)
        coords = np.array([x, y])

        # Connect to coastline eastern point
        x, y = (corners.x[1], linea0.loc[np.argmax(linea0.x), "y"])
        coords = np.c_[coords, [x, y]]

        # Add coastline points (reversed)
        coords = np.c_[coords, np.flipud(linea0).T]

        # Connect to coastline western point
        x, y = (corners.x[0], linea0.loc[np.argmin(linea0.x), "y"])
        coords = np.c_[coords, [x, y]]

    elif sea == "east":
        # It is defined the sea area using zero line and the boundary (corners)
        x, y = np.linspace(corners.x[0], corners.x[1], 2), np.tile(corners.y[1], 2)
        coords = np.array([x, y])

        x, y = (corners.x[1], linea0.loc[np.argmax(linea0.x), "y"])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(linea0).T]

        # Connect to coastline western point
        x, y = (corners.x[0], linea0.loc[np.argmin(linea0.x), "y"])
        coords = np.c_[coords, [x, y]]

    elif sea == "east":
        # Western boundary (left edge)
        x, y = np.tile(corners.x[0], 2), np.linspace(corners.y[0], corners.y[1], 2)
        coords = np.array([x, y])

        # Connect to coastline northern point
        x, y = (linea0.loc[np.argmax(linea0.y), "x"], corners.y[1])
        coords = np.c_[coords, [x, y]]

        # Add coastline points (reversed)
        coords = np.c_[coords, np.flipud(linea0).T]

        # Connect to coastline southern point
        x, y = (linea0.loc[np.argmin(linea0.y), "x"], corners.y[0])
        coords = np.c_[coords, [x, y]]

    elif sea == "south-east":
        # Corner point (southwest)
        x, y = (corners.x[0], corners.y[1])
        coords = np.array([x, y])

        # Eastern extent (maximum x from coastline)
        xmax = np.max(
            [linea0.loc[np.argmax(linea0.y), "x"], linea0.loc[np.argmax(linea0.x), "x"]]
        )
        x, y = (xmax, corners.y[1])
        coords = np.c_[coords, [x, y]]

        # Add coastline points (reversed)
        coords = np.c_[coords, np.flipud(linea0).T]

        # Southern extent (minimum y from coastline)
        ymin = np.min(
            [linea0.loc[np.argmin(linea0.y), "y"], linea0.loc[np.argmin(linea0.x), "y"]]
        )
        x, y = (corners.x[0], ymin)
        coords = np.c_[coords, [x, y]]

    # Create polygon geometry from coordinates
    polygon_ = Polygon(coords.T)
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon_])

    # Generate map visualization of sea area
    m = folium.Map(location=[np.min(coords[1, :]), np.min(coords[0, :])])
    folium.GeoJson(polygon).add_to(m)

    # Save map to HTML file
    m.save(
        "malla_bat_"
        + str(corners.x[0])
        + "_"
        + str(corners.y[0])
        + "_"
        + str(corners.x[1])
        + "_"
        + str(corners.y[1])
        + ".html"
    )

    # Remove topography data within sea polygon
    topo_mask = spatial_mask(topo, polygon, op="within")

    # Combine masked topography, bathymetry, and coastline
    linea0["z"] = 0
    data = pd.concat([topo_mask, bati, linea0], ignore_index=True)

    return data


def spatial_mask(data, polygon, op="within"):
    """Apply spatial mask to data points based on polygon geometry.
    
    Filters data points based on their spatial relationship with a polygon
    (within or outside the polygon boundary).

    Parameters
    ----------
    data : pd.DataFrame
        Point data with columns ['x', 'y', ...other columns]
    polygon : gpd.GeoDataFrame
        Polygon geometry for masking
    op : str, optional
        Operation type: 'within' (inside polygon) or other (outside polygon).
        Default: 'within'

    Returns
    -------
    pd.DataFrame
        Filtered data containing only points satisfying the spatial condition
    """

    # Convert data points to GeoDataFrame
    geodata = gpd.GeoDataFrame(
        index=data.index, geometry=gpd.points_from_xy(data.x, data.y)
    )

    # Perform spatial join with polygon
    mask = gpd.sjoin(geodata, polygon, op="within", how="left")
    
    # Filter based on operation type
    if op == "within":
        # Keep points inside polygon
        data = data.loc[mask.index_right == polygon.index[0]]
    else:
        # Keep points outside polygon
        data = data.loc[mask.index_right != polygon.index[0]]

    return data


def remove_lowland(data, reference_value: float = 0, replace_value: float = 2):
    """Remove low-lying areas on land side of topobathymetry.
    
    Identifies and replaces low-lying land areas (below sea level) with a
    specified elevation value to prevent unrealistic flooding in models.

    Parameters
    ----------
    data : np.ndarray
        2D array of elevation values (topobathymetry grid)
    reference_value : float, optional
        Elevation threshold for identifying low areas (m). Default: 0 (sea level)
    replace_value : float, optional
        Elevation value to assign to low areas (m). Default: 2

    Returns
    -------
    np.ndarray
        Modified elevation array with low land areas replaced

    Notes
    -----
    The function:
    1. Extracts the coastline (zero-elevation contour)
    2. Creates a land polygon from coastline and domain boundary
    3. Identifies low-lying areas within the land polygon
    4. Replaces elevations below reference_value with replace_value
    
    This prevents unrealistic ponding or flooding in low-lying coastal areas.
    """

    ny, nx = np.shape(data)

    # Prepare bathymetry dictionary for contour extraction
    bathy = dict()
    bathy["z"] = data.copy()
    bathy["x"], bathy["y"] = np.meshgrid(np.arange(nx), np.arange(ny))

    # Extract coastline (zero-elevation contour)
    coastline = utils.extract_isolines(bathy)
    
    # Define upper boundary (domain edge)
    up_side = pd.DataFrame(
        np.asarray([bathy["x"][:, -1], bathy["y"][:, -1]]).T,
        index=np.arange(len(bathy["x"][:, -1])),
        columns=["x", "y"],
    )
    
    # Create land polygon
    land = analysis.create_polygon(up_side, sides=coastline[0])

    # Convert to DataFrame for spatial operations
    bathy = pd.DataFrame(
        np.asarray(
            [np.ravel(bathy["x"]), np.ravel(bathy["y"]), np.ravel(bathy["z"])]
        ).T,
        index=np.arange(bathy["x"].size),
        columns=["x", "y", "z"],
    )
    
    # Apply spatial mask to get land points
    mask = spatial_mask(bathy, land)

    # Identify low areas below reference value
    low_areas = bathy.loc[mask.index, "z"] <= reference_value
    
    # Replace low areas with specified value
    bathy.loc[low_areas.loc[low_areas].index, "z"] = replace_value
    
    # Reshape back to 2D grid
    bathy = np.reshape(bathy.z.values, (ny, nx))

    return bathy



def merge_sea_sea(tb, bd, corners, sea="south"):
    """Merge shallow topobathymetry with deep-water bathymetry.
    
    Combines near-shore topobathymetry data with offshore deep-water bathymetry
    by defining a transition depth contour and blending the datasets.

    Parameters
    ----------
    tb : pd.DataFrame
        Topobathymetry data (shallow/nearshore) with columns ['x', 'y', 'z']
    bd : pd.DataFrame
        Deep-water bathymetry data with columns ['x', 'y', 'z']
    corners : pd.DataFrame
        Domain corner coordinates with columns ['x', 'y']
    sea : str, optional
        Sea location: 'south', 'east', or 'south-east'. Default: 'south'

    Returns
    -------
    pd.DataFrame
        Merged topobathymetry with deep-water bathymetry

    Notes
    -----
    The function:
    1. Extracts -40m depth contour from topobathymetry
    2. Creates polygon from contour and domain boundaries
    3. Replaces topobathymetry data outside polygon with deep bathymetry
    4. Generates HTML map showing the blending boundary
    """

    # Remove NaN values for contour extraction
    mask = tb.z.notna()

    # Extract -40m depth contour as transition boundary
    cs = plt.tricontour(tb.x[mask], tb.y[mask], tb.z[mask], levels=[-40])

    # Find longest contour segment
    datab = [0]
    for collection in cs.collections:
        for path in collection.get_paths():
            if len(datab) < len(path.to_polygons()[0]):
                datab = np.asarray(path.to_polygons()[0])[:-1, :]

    plt.close()
    
    # Build polygon based on sea location
    if sea == "south":
        # Southern boundary
        x, y = np.linspace(corners.x[0], corners.x[1], 2), np.tile(corners.y[1], 2)
        coords = np.array([x, y])

        # Connect to contour
        x, y = (
            np.tile(corners.x[1], 2),
            np.linspace(corners.y[1], datab[np.argmax(datab[:, 0]), 1], 2),
        )
        coords = np.c_[coords, [x, y]]
        coords = np.c_[datab[::-1].T, coords]
        
    elif sea == "east":
        # Western boundary
        x, y = np.tile(corners.x[0], 2), np.linspace(corners.y[0], corners.y[1], 2)
        coords = np.array([x, y])

        # Connect to contour
        x, y = (datab[np.argmax(datab[:, 1]), 0], corners.y[1])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(datab).T]

        x, y = (datab[np.argmin(datab[:, 1]), 0], corners.y[0])
        coords = np.c_[coords, [x, y]]

    elif sea == "south-east":
        # Southwest corner
        x, y = (corners.x[0], corners.y[1])
        coords = np.array([x, y])

        xmax = np.max(
            [datab[np.argmax(datab[:, 0]), 0], datab[np.argmax(datab[:, 1]), 0]]
        )
        x, y = (xmax, corners.y[1])
        coords = np.c_[coords, [x, y]]

        coords = np.c_[coords, np.flipud(datab).T]

        ymin = np.min(
            [datab[np.argmin(datab[:, 0]), 1], datab[np.argmin(datab[:, 1]), 1]]
        )
        x, y = (corners.x[0], ymin)
        coords = np.c_[coords, [x, y]]

    # Create polygon geometry
    polygon = Polygon(coords.T)
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])

    # Generate visualization map
    m = folium.Map(location=[np.min(coords[1, :]), np.min(coords[0, :])])
    folium.GeoJson(polygon).add_to(m)

    # Save map to HTML
    m.save(
        "malla_topobat_"
        + str(corners.x[0])
        + "_"
        + str(corners.y[0])
        + "_"
        + str(corners.x[1])
        + "_"
        + str(corners.y[1])
        + ".html"
    )

    # Convert topobathymetry to GeoDataFrame
    topoxy = gpd.GeoDataFrame(
        tb.loc[:, ["x", "y"]], geometry=gpd.points_from_xy(tb.x, tb.y)
    )

    # Identify points outside polygon (deep water area)
    mask = topoxy.within(polygon.loc[0, "geometry"])
    
    # Replace with deep bathymetry data
    tb.loc[mask[~mask].index, "z"] = bd.loc[mask[~mask].index, "z"]

    return tb


def transform_coordinates(data, proj_from, proj_to, by_columns=False):
    """Transform spatial coordinates between different projection systems.
    
    Converts coordinates from one coordinate reference system (CRS) to another
    using pyproj transformations. Supports common projections used in
    environmental and geographic applications.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or list
        Coordinate data to transform. Format depends on by_columns parameter:
        - If by_columns=False: DataFrame with 'x' and 'y' columns
        - If by_columns=True: Array/list with shape (n, 2) or (2,)
    proj_from : str
        Source projection in EPSG format (e.g., 'epsg:4326', 'epsg:25830').
        Common options:
        - 'epsg:4326': WGS84 geographic (latitude/longitude)
        - 'epsg:23030': European Datum 1950 UTM Zone 30N
        - 'epsg:25830': ETRS89 UTM Zone 30N
        - 'epsg:25829': ETRS89 UTM Zone 29N
        - 'epsg:3857': Web Mercator (Pseudo-Mercator)
    proj_to : str
        Target projection in EPSG format (same format as proj_from)
    by_columns : bool, optional
        If True, data is treated as array with columns [lon/x, lat/y].
        If False, data is treated as DataFrame with 'x' and 'y' attributes.
        Default: False

    Returns
    -------
    pd.DataFrame, np.ndarray, or list
        Transformed coordinates in target projection (same type as input)

    Notes
    -----
    WGS84 (EPSG:4326) uses (longitude, latitude) order, while most projected 
    CRS use (x, y) order. The function handles coordinate order correctly for 
    each projection type.
    
    For UTM zones, consider implementing automatic zone detection based on 
    longitude: zone = int((lon + 180) / 6) + 1

    Examples
    --------
    >>> import pandas as pd
    >>> # Transform from WGS84 to UTM 30N
    >>> data = pd.DataFrame({'x': [-3.5], 'y': [37.0], 'z': [100]})
    >>> transformed = transform_coordinates(data, 'epsg:4326', 'epsg:25830')
    """

    dataout = data.copy()
    
    # Only transform if projections are different
    if proj_from != proj_to:
        p1 = pyproj.Proj(proj_from)
        p2 = pyproj.Proj(proj_to)

        if not by_columns:
            # Transform DataFrame with x, y attributes
            if proj_from != "epsg:4326":
                # For projected CRS: (x, y) order
                dataout.y, dataout.x = pyproj.transform(
                    p1, p2, dataout.x.values, dataout.y.values
                )
            else:
                # For WGS84: swap to (lon, lat) order for transform
                dataout.x, dataout.y = pyproj.transform(
                    p1, p2, dataout.y.values, dataout.x.values
                )

        else:
            # Transform array or list
            if isinstance(dataout, list):
                # List format: [lon/x, lat/y]
                dataout = pyproj.transform(p1, p2, dataout[1], dataout[0])
            elif isinstance(dataout, np.ndarray):
                if dataout.ndim == 1:
                    # 1D array: single point
                    dataout[0], dataout[1] = pyproj.transform(
                        p1, p2, dataout[0], dataout[1]
                    )
                else:
                    # 2D array: multiple points
                    dataout[:, 0], dataout[:, 1] = pyproj.transform(
                        p1, p2, dataout[:, 0], dataout[:, 1]
                    )
            else:
                # Fallback for other array-like types
                dataout[:, 0], dataout[:, 1] = pyproj.transform(
                    p1, p2, dataout[:, 0], dataout[:, 1]
                )

    return dataout


