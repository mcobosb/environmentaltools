"""
Spatial Analysis Utilities
===========================

Utilities for spatial data analysis including coordinate transformations, 
interpolation, profile extraction, Voronoi diagrams, and geometric operations.

"""

import copy
import math
import sys

import geopandas as gpd
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import scipy.spatial as sp
import shapely.speedups
from scipy.interpolate import griddata
from shapely.geometry import LineString, Polygon


def select_data(data, corners):
    """Select spatial data within specified rectangular boundaries.
    
    Filters a DataFrame to include only points within the rectangular region
    defined by corner coordinates. Supports both dictionary and DataFrame
    corner specifications.

    Parameters
    ----------
    data : pd.DataFrame
        Spatial data containing at least 'x' and 'y' coordinate columns
    corners : dict or pd.DataFrame
        Rectangular boundary definition. If dict, format should be:
        {'x': [xmin, xmax], 'y': [ymin, ymax]}.
        If DataFrame, should have 'x' and 'y' columns with [min, max] values.

    Returns
    -------
    pd.DataFrame
        Filtered data containing only points within the specified boundaries.
        Returns a copy of the input data with rows outside the boundary removed.

    Raises
    ------
    ValueError
        If corners is not a dictionary or DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'x': [0, 5, 10], 'y': [0, 5, 10], 'z': [1, 2, 3]})
    >>> corners = {'x': [2, 8], 'y': [2, 8]}
    >>> filtered = select_data(data, corners)
    >>> print(filtered)
       x  y  z
    1  5  5  2
    """
    if isinstance(corners, dict):
        # Filter data using dictionary boundaries
        data = data.loc[(data["x"] > corners["x"][0]) & (data["x"] < corners["x"][1])]
        data = data.loc[(data["y"] > corners["y"][0]) & (data["y"] < corners["y"][1])]
    elif isinstance(corners, pd.DataFrame):
        # Filter data using DataFrame boundaries
        data = data.loc[(data["x"] > corners.x[0]) & (data["x"] < corners.x[1])]
        data = data.loc[(data["y"] > corners.y[0]) & (data["y"] < corners.y[1])]
    else:
        raise ValueError("Corners should be given in dictionary or DataFrame.")

    return data


def interp(base, data, dist=100, method="linear", fill_values=np.nan):
    """Interpolate spatial data onto a regular grid or specified points.
    
    Performs 2D spatial interpolation from scattered base data points to either
    a regular grid or specified target points. Uses scipy.interpolate.griddata
    for the interpolation.

    Parameters
    ----------
    base : pd.DataFrame
        Base spatial data for interpolation. Must contain columns 'x', 'y', and 'z'
        where z is the variable to interpolate.
    data : pd.DataFrame or dict
        Target interpolation domain. If DataFrame with two rows, defines rectangular
        bounds [xmin, xmax], [ymin, ymax] for regular grid creation. If dict with
        'x' and 'y' keys, contains target point coordinates for interpolation.
    dist : float, optional
        Grid spacing in same units as coordinates when creating regular grid.
        Only used if data is DataFrame with bounds. Default is 100.
    method : str, optional
        Interpolation method: 'linear', 'nearest', or 'cubic'. Default is 'linear'.
        See scipy.interpolate.griddata for details.
    fill_values : float, optional
        Value used to fill points outside the convex hull of base data.
        Default is np.nan.

    Returns
    -------
    x : np.ndarray
        X coordinates of interpolation points (2D meshgrid or flattened array)
    y : np.ndarray
        Y coordinates of interpolation points (2D meshgrid or flattened array)
    z : np.ndarray
        Interpolated values at (x, y) locations
    df : pd.DataFrame
        Interpolation results as DataFrame with columns ['x', 'y', 'z']

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create base data
    >>> base = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2], 'z': [10, 20, 30]})
    >>> # Define interpolation bounds
    >>> bounds = pd.DataFrame({'x': [0, 2], 'y': [0, 2]})
    >>> x, y, z, df = interp(base, bounds, dist=0.5)
    """
    if isinstance(data, pd.DataFrame):
        # Create regular grid from bounds
        nx, ny = (
            int((data.x[1] - data.x[0]) / dist),
            int((data.y[1] - data.y[0]) / dist),
        )
        x, y = np.meshgrid(
            np.linspace(data.x[0] + dist / 2, data.x[1] - dist / 2, nx),
            np.linspace(data.y[0] + dist / 2, data.y[1] - dist / 2, ny),
        )
    else:
        # Use provided coordinates
        x, y = data["x"], data["y"]
        if len(np.shape(x)) == 1:
            nx, ny = len(x), 1
        else:
            nx, ny = np.shape(x)
    
    # Perform spatial interpolation
    z = griddata(
        base.loc[:, ["x", "y"]].values,
        base.loc[:, "z"].values,
        (x, y),
        method=method,
        fill_value=fill_values,
    )
    
    # Format results as DataFrame
    df = pd.DataFrame(
        np.asarray([np.ravel(x), np.ravel(y), np.ravel(z)]).T,
        index=np.arange(int(nx * ny)),
        columns=["x", "y", "z"],
    )

    return x, y, z, df


def fillna(data, var_="z", method="nearest"):
    """Fill missing values in spatial data using interpolation from valid data.
    
    Replaces NaN values in specified variable using spatial interpolation based
    on non-missing neighboring values. Useful for gap-filling in spatial datasets.

    Parameters
    ----------
    data : pd.DataFrame
        Spatial data containing columns 'x', 'y', and the variable to fill.
        Missing values should be NaN.
    var_ : str, optional
        Name of the variable column to fill. Default is 'z'.
    method : str, optional
        Interpolation method: 'nearest', 'linear', or 'cubic'. Default is 'nearest'.
        'nearest' is robust and works well for filling gaps.
        See scipy.interpolate.griddata for details.

    Returns
    -------
    pd.DataFrame
        Input data with NaN values filled in the 'z' column using spatial
        interpolation from valid data points.

    Notes
    -----
    The function uses scipy.interpolate.griddata to perform the spatial interpolation.
    Only locations with NaN values are updated; existing valid data is preserved.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
    ...     'x': [0, 1, 2, 0, 1, 2],
    ...     'y': [0, 0, 0, 1, 1, 1],
    ...     'z': [10, np.nan, 30, 40, 50, np.nan]
    ... })
    >>> filled = fillna(data, var_='z', method='nearest')
    """
    # Create mask for valid (non-NaN) data points
    mask = pd.notna(data.loc[:, var_])
    
    # Interpolate values at NaN locations using valid data
    z = griddata(
        data.loc[mask, ["x", "y"]].values,
        data.loc[mask, "z"].values,
        (data.x, data.y),
        method=method,
    )
    
    # Update the z column with filled values
    data.z = z
    return data


def normal_profiles(topobat, info, n_points=1000):
    """Extract normal profiles from spatial topography/bathymetry data.
    
    Interpolates elevation/depth values along multiple profile lines defined
    by start and end coordinates. Each profile is sampled at equally-spaced
    points along the line connecting its endpoints.

    Parameters
    ----------
    topobat : pd.DataFrame
        Topography/bathymetry data containing columns 'x', 'y', and 'z'.
        'z' represents elevation (positive above datum) or depth (negative below).
    info : dict
        Dictionary defining profile locations with keys:
        - 'x': list of [x_start, x_end] coordinate pairs for each profile
        - 'y': list of [y_start, y_end] coordinate pairs for each profile
        Each pair defines a profile line endpoint.
    n_points : int, optional
        Number of equally-spaced points to sample along each profile.
        Default is 1000. Higher values provide finer resolution but increase
        computation time.

    Returns
    -------
    x : list of np.ndarray
        X coordinates along each profile (n_points per profile)
    y : list of np.ndarray  
        Y coordinates along each profile (n_points per profile)
    z : list of np.ndarray
        Interpolated elevation/depth values along each profile.
        NaN values indicate locations outside the convex hull of input data.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample bathymetry data
    >>> topobat = pd.DataFrame({
    ...     'x': [0, 10, 20, 0, 10, 20],
    ...     'y': [0, 0, 0, 10, 10, 10],
    ...     'z': [-5, -10, -15, 0, -2, -5]
    ... })
    >>> # Define two profile lines
    >>> info = {'x': [[0, 20], [5, 15]], 'y': [[0, 10], [0, 10]]}
    >>> x, y, z = normal_profiles(topobat, info, n_points=500)
    >>> print(f"Number of profiles: {len(x)}, points per profile: {len(x[0])}")
    Number of profiles: 2, points per profile: 500
    """
    x, y, z = [], [], []
    
    # Extract profile along each defined line
    for i, j in enumerate(info["x"]):
        # Create equally-spaced points along the profile line
        x.append(np.linspace(info["x"][i][0], info["x"][i][1], n_points))
        y.append(np.linspace(info["y"][i][0], info["y"][i][1], n_points))
        
        # Interpolate elevation/depth values at profile points
        z.append(
            griddata(
                topobat.loc[:, ["x", "y"]].values,
                topobat.loc[:, "z"].values,
                np.vstack([x[i], y[i]]).T,
                method="linear",
                fill_value=np.nan,
            )
        )
    return x, y, z


def rotate_coords(x, y, angle):
    """Rotate coordinates by specified angle around the origin.
    
    Performs 2D coordinate rotation transformation using angle specified in degrees.
    Rotation is counterclockwise for positive angles.

    Parameters
    ----------
    x : float or np.ndarray
        X coordinate(s) to rotate
    y : float or np.ndarray
        Y coordinate(s) to rotate
    angle : float
        Rotation angle in degrees. Positive values rotate counterclockwise.

    Returns
    -------
    x_rot : float or np.ndarray
        Rotated X coordinate(s)
    y_rot : float or np.ndarray
        Rotated Y coordinate(s)

    Examples
    --------
    >>> import numpy as np
    >>> x, y = 1.0, 0.0
    >>> x_rot, y_rot = rotate_coords(x, y, 90)  # Rotate 90 degrees
    >>> print(f"({x_rot:.2f}, {y_rot:.2f})")
    (0.00, 1.00)
    """
    # Calculate distance from origin and angle for each point
    d = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    
    # Decompose into x and y components
    dx, dy = d * np.cos(angles), d * np.sin(angles)
    
    # Apply rotation transformation
    x = dx * np.cos(np.deg2rad(angle)) - dy * np.sin(np.deg2rad(angle))
    y = dx * np.sin(np.deg2rad(angle)) + dy * np.cos(np.deg2rad(angle))
    
    return x, y


def global_to_local_coords(x_glob, y_glob, alpha, lon_0_glob, lat_0_glob):
    """Transform global coordinates to local rotated coordinate system.
    
    Converts coordinates from a global reference system to a local coordinate
    system defined by an origin point and rotation angle. Useful for analyzing
    data in a local shore-normal/shore-parallel reference frame.

    Parameters
    ----------
    x_glob : float or np.ndarray
        Global X coordinates (e.g., UTM easting or longitude)
    y_glob : float or np.ndarray
        Global Y coordinates (e.g., UTM northing or latitude)
    alpha : float
        Rotation angle in degrees defining local coordinate system orientation
    lon_0_glob : float
        Global X coordinate of local system origin
    lat_0_glob : float
        Global Y coordinate of local system origin

    Returns
    -------
    x_loc : float or np.ndarray
        Local X coordinates (along rotated axis)
    y_loc : float or np.ndarray
        Local Y coordinates (perpendicular to rotated axis)

    Notes
    -----
    The transformation applies:
    1. Translation to move origin to (lon_0_glob, lat_0_glob)
    2. Rotation by angle alpha

    Examples
    --------
    >>> x_glob, y_glob = 100.0, 200.0
    >>> x_loc, y_loc = global_to_local_coords(x_glob, y_glob, 45, 0, 0)
    """
    # Apply rotation transformation with origin translation
    x_loc = (x_glob - lon_0_glob) * np.cos(math.radians(alpha)) + (
        y_glob - lat_0_glob
    ) * np.sin(math.radians(alpha))
    y_loc = (y_glob - lat_0_glob) * np.cos(math.radians(alpha)) - (
        x_glob - lon_0_glob
    ) * np.sin(math.radians(alpha))

    return x_loc, y_loc


def local_to_global_coords(x_loc, y_loc, alpha, lon_0_glob, lat_0_glob):
    """Transform local rotated coordinates back to global coordinate system.
    
    Inverse transformation of global_to_local_coords. Converts coordinates from
    local rotated reference system back to the global coordinate system.

    Parameters
    ----------
    x_loc : float or np.ndarray
        Local X coordinates (along rotated axis)
    y_loc : float or np.ndarray
        Local Y coordinates (perpendicular to rotated axis)
    alpha : float
        Rotation angle in degrees that defines the local coordinate system
    lon_0_glob : float
        Global X coordinate of local system origin
    lat_0_glob : float
        Global Y coordinate of local system origin

    Returns
    -------
    x_glob : float or np.ndarray
        Global X coordinates (e.g., UTM easting or longitude)
    y_glob : float or np.ndarray
        Global Y coordinates (e.g., UTM northing or latitude)

    Notes
    -----
    The transformation applies:
    1. Inverse rotation by angle -alpha
    2. Translation to restore original origin

    This function is the mathematical inverse of global_to_local_coords.

    Examples
    --------
    >>> x_loc, y_loc = 70.71, 70.71
    >>> x_glob, y_glob = local_to_global_coords(x_loc, y_loc, 45, 0, 0)
    >>> print(f"Global: ({x_glob:.2f}, {y_glob:.2f})")
    """
    # Apply inverse rotation transformation with origin translation
    x_glob = (
        ((x_loc) / (np.cos(math.radians(alpha))))
        / (1 + (np.tan(math.radians(alpha))) * (np.tan(math.radians(alpha))))
        - (
            ((y_loc * np.sin(math.radians(alpha))))
            / (np.cos(math.radians(alpha)) * np.cos(math.radians(alpha)))
        )
        / (1 + (np.tan(math.radians(alpha))) * (np.tan(math.radians(alpha))))
        + lon_0_glob
    )
    y_glob = (
        (y_loc) / (np.cos(math.radians(alpha)))
        + (
            (np.sin(math.radians(alpha)))
            * (x_loc - y_loc * np.tan(math.radians(alpha)))
        )
        / (
            (np.cos(math.radians(alpha)))
            * (np.cos(math.radians(alpha)))
            * (1 + (np.tan(math.radians(alpha))) * (np.tan(math.radians(alpha))))
        )
        + lat_0_glob
    )

    return x_glob, y_glob


def continuous_line(data: pd.DataFrame, limiting_distance: float = 1e9):
    """Create continuous line from scattered points by connecting nearest neighbors.
    
    Constructs an ordered sequence of points forming a continuous line by
    iteratively connecting each point to its nearest unconnected neighbor.
    Algorithm starts from the leftmost point (minimum x coordinate) and
    proceeds by selecting the closest remaining point at each step.

    Parameters
    ----------
    data : pd.DataFrame
        Scattered point data containing at least 'x' and 'y' coordinate columns.
        Additional columns are preserved in the output.
    limiting_distance : float, optional
        Maximum allowed distance between consecutive points. If the nearest
        remaining point exceeds this distance, the algorithm terminates.
        Default is 1e9 (effectively unlimited). Units match input coordinates.

    Returns
    -------
    pd.DataFrame
        Ordered points forming a continuous line. Points are sorted by
        connection order starting from the leftmost point. Contains same
        columns as input data. Points that couldn't be connected within
        the limiting distance are excluded.

    Notes
    -----
    This greedy nearest-neighbor algorithm works well for creating ordered
    lines from scattered data along approximately linear features (coastlines,
    transects, etc.). For complex or branching geometries, results may not
    be optimal.

    The algorithm modifies the input DataFrame in place (removes points as
    they are processed).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create scattered points along a curve
    >>> data = pd.DataFrame({
    ...     'x': [0, 2, 1, 3, 4],
    ...     'y': [0, 1, 0.5, 2, 3]
    ... })
    >>> line = continuous_line(data.copy(), limiting_distance=2.0)
    >>> print(line[['x', 'y']])
    """
    cols = data.columns
    
    # Initialize output DataFrame with same structure as input
    cont_line = pd.DataFrame(-1, index=np.arange(len(data.index)), columns=cols)
    
    # Start from leftmost point
    index0 = data.index[data.x.argmin()]
    cont_line.loc[0] = data.loc[index0]
    data.drop(index0, inplace=True)
    cont_line["dist"] = 0

    # Iteratively connect nearest neighbors
    k = 0
    while not data.empty:
        # Calculate distances from current point to all remaining points
        distance = np.sqrt(
            (cont_line.loc[k, "x"] - data["x"]) ** 2
            + (cont_line.loc[k, "y"] - data["y"]) ** 2
        )

        # Find nearest remaining point
        idx = data.index[np.argmin(distance)]
        
        # Check if distance exceeds limit
        if np.min(distance) > limiting_distance:
            break
        
        # Add nearest point to line
        cont_line.loc[k + 1, cols] = data.loc[idx]
        cont_line.loc[k + 1, "dist"] = distance[idx]
        data.drop(idx, inplace=True)
        k += 1

    # Clean up: remove distance column and unconnected points
    cont_line.drop(columns="dist", inplace=True)
    cont_line.drop(cont_line[cont_line["x"] == -1].index, inplace=True)

    return cont_line


def create_polygon(data, crs="epsg:25830", sides=[]):
    """Create a closed polygon from coordinates and optional additional sides.
    
    Constructs a polygon geometry from a primary set of coordinates, optionally
    adding additional side segments. Useful for creating polygons from multiple
    line segments or closing open geometries.

    Parameters
    ----------
    data : pd.DataFrame
        Primary polygon coordinates containing columns 'x' and 'y'.
        These points define the main boundary of the polygon.
    crs : str, optional
        Coordinate reference system in EPSG format (e.g., 'epsg:25830' for
        ETRS89 UTM Zone 30N, 'epsg:4326' for WGS84). Default is 'epsg:25830'.
    sides : list of pd.DataFrame or pd.DataFrame, optional
        Additional side segment(s) to append to the polygon boundary.
        Each DataFrame should contain 'x' and 'y' coordinate columns.
        Coordinates are appended in the order provided. Default is empty list.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing single polygon geometry with specified CRS.
        Has one row with the polygon geometry.

    Notes
    -----
    - Coordinates are concatenated in order: data, then each side in sides list
    - The shapely Polygon constructor automatically closes the polygon
    - Ensure coordinate order forms a valid polygon (no self-intersections)
    - All input DataFrames should use the same coordinate system

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Define main boundary
    >>> data = pd.DataFrame({'x': [0, 10, 10, 0], 'y': [0, 0, 10, 10]})
    >>> # Create closed polygon
    >>> poly = create_polygon(data, crs='epsg:4326')
    >>> print(poly.geometry[0])
    POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))
    
    >>> # Add additional side
    >>> side = pd.DataFrame({'x': [10, 5], 'y': [10, 15]})
    >>> poly = create_polygon(data, crs='epsg:4326', sides=[side])
    """
    # Ensure sides is a list
    if not isinstance(sides, list):
        sides = [sides]

    # Extract main boundary coordinates
    coords = data.loc[:, ["x", "y"]].values
    
    # Append additional sides
    for side in sides:
        coords = np.vstack([coords, side.loc[:, ["x", "y"]].values])

    # Create polygon geometry and GeoDataFrame
    polygon = Polygon(coords)
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon])

    return polygon


def confidence_interval_2d(data, loc0=None, angle=None, n_points=1000):
    """Calculate 2D confidence intervals and statistics for profile data.
    
    Computes mean, standard deviation, minimum, and maximum values across
    multiple 2D profiles by rotating, interpolating, and analyzing the data
    in a common coordinate system. Useful for analyzing variability in
    spatial transects or time series of profiles.

    Parameters
    ----------
    data : dict of pd.DataFrame
        Dictionary where keys identify different profiles and values are
        DataFrames containing 'x' and 'y' columns representing profile coordinates.
        All profiles should span similar spatial extent.
    loc0 : list of float, optional
        Origin point [x0, y0] for coordinate rotation. If None, uses the
        minimum x location from the first profile. Default is None.
    angle : float, optional
        Rotation angle in radians to align profiles. If None, automatically
        calculated from data geometry. Default is None.
    n_points : int, optional
        Number of points for the regular interpolation grid. Default is 1000.
        Higher values provide finer resolution but increase computation time.

    Returns
    -------
    pd.DataFrame
        Statistical summary with index representing x-coordinates and columns:
        - 'mean': Mean y-value across all profiles
        - 'std': Standard deviation of y-values
        - 'min': Minimum y-value across all profiles
        - 'max': Maximum y-value across all profiles
        - 'x': X coordinates (in original coordinate system)
        - 'ini': Y-values from first profile
        - 'end': Y-values from last profile

    Notes
    -----
    The function performs the following steps:
    1. Rotates all profiles to a common reference frame
    2. Interpolates each profile onto a regular n_points-grid
    3. Calculates statistics across all interpolated profiles
    4. Rotates results back to original coordinate system

    Requires all profiles to have comparable spatial extent for meaningful results.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample profile data
    >>> data = {
    ...     1: pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]}),
    ...     2: pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1.5, 3]}),
    ...     3: pd.DataFrame({'x': [0, 1, 2], 'y': [0, 0.5, 1]})
    ... }
    >>> stats = confidence_interval_2d(data, n_points=500)
    >>> print(stats[['mean', 'std']].head())
    """
    rot = data.copy()
    
    # Set origin point if not provided
    if loc0 == None:
        ind_min = data[1].x.argmin()
        loc0 = [data[1].loc[ind_min, "x"], data[1].loc[ind_min, "y"]]

    # Calculate rotation angle if not provided
    if angle == None:
        angle = np.arctan2(data.y, data.x)

    # Initialize array for interpolated y-values
    y = np.zeros([len(data), n_points])
    
    # Process each profile
    for ind_, key in enumerate(data.keys()):
        # Sort by x coordinate
        data[key].sort_values(by="x", inplace=True)
        
        # Rotate coordinates to common reference frame
        rot[key].x, rot[key].y = rotate_coords(
            data[key].x - loc0[0], data[key].y - loc0[1], -angle
        )

        # Define interpolation grid from first profile
        if ind_ == 0:
            x = np.linspace(rot[key].x.min(), rot[key].x.max(), n_points)
            init_key = key

        if ind_ == len(data) - 1:
            end_key = key

        # Interpolate profile onto regular grid
        y[ind_, :] = np.interp(x, rot[key].x, data[key].y)

    # Calculate statistics across all profiles
    mean_ = np.mean(y, axis=0)
    std_ = np.std(y, axis=0)
    min_ = np.min(y, axis=0)
    max_ = np.max(y, axis=0)
    
    # Create statistics DataFrame
    stats = pd.DataFrame(
        np.vstack([mean_, std_, min_, max_]).T,
        index=x,
        columns=["mean", "std", "min", "max"],
    )

    # Rotate statistics back to original coordinate system
    for cols_ in ["mean", "min", "max"]:
        stats["x"], stats[cols_] = rotate_coords(x, stats[cols_].values, angle)
        stats["x"], stats[cols_] = stats["x"] + loc0[0], stats[cols_] + loc0[1]

    # Add first and last profile to results
    _, stats["ini"] = rotate_coords(x, y[0, :], angle)
    stats["ini"] = stats["ini"] + loc0[1]
    _, stats["end"] = rotate_coords(x, y[-1, :], angle)
    stats["end"] = stats["end"] + loc0[1]
    
    return stats



def generate_CVD(points, iterations, bounding_box):
    """Generate Centroidal Voronoi Diagram using Lloyd's algorithm.
    
    Iteratively computes a Centroidal Voronoi Diagram (CVD) where each Voronoi
    cell's generator point converges to the centroid of its cell. This creates
    a more uniform and optimized spatial tessellation compared to standard
    Voronoi diagrams.

    Parameters
    ----------
    points : np.ndarray
        Initial generator points as (N, 2) array of [x, y] coordinates
    iterations : int
        Number of Lloyd's algorithm iterations to perform. More iterations
        produce better convergence to true centroids. Typical values: 5-50.
    bounding_box : tuple or list
        Rectangular bounds as [xmin, xmax, ymin, ymax] defining the region
        for the Voronoi diagram

    Returns
    -------
    scipy.spatial.Voronoi
        Bounded Voronoi diagram object with additional attributes:
        - filtered_points: generator points within bounding box
        - filtered_regions: region indices corresponding to filtered points

    Notes
    -----
    Lloyd's algorithm (1982) iteratively:
    1. Constructs bounded Voronoi diagram from current points
    2. Calculates centroid of each Voronoi cell
    3. Moves generator points to cell centroids
    4. Repeats until convergence or max iterations reached

    Reference: https://www.py4u.net/discuss/21901

    Examples
    --------
    >>> import numpy as np
    >>> # Random initial points
    >>> points = np.random.rand(20, 2) * 100
    >>> # Generate CVD within 100x100 box
    >>> vor = generate_CVD(points, iterations=10, bounding_box=[0, 100, 0, 100])
    """
    p = copy.copy(points)

    # Perform Lloyd's algorithm iterations
    for i in range(iterations):
        # Generate bounded Voronoi diagram
        vor = bounded_voronoi(p, bounding_box)
        centroids = []

        # Calculate centroid of each Voronoi region
        for region in vor.filtered_regions:
            # Extract vertices for this region (close polygon by repeating first vertex)
            vertices = vor.vertices[region + [region[0]], :]
            centroid = centroid_region(vertices)
            centroids.append(list(centroid[0, :]))

        # Update points to region centroids
        p = np.array(centroids)

    # Return final bounded Voronoi diagram
    return bounded_voronoi(p, bounding_box)


# Machine epsilon for numerical stability
eps = sys.float_info.epsilon


def in_box(towers, bounding_box):
    """Check if points are within rectangular bounding box.

    Parameters
    ----------
    towers : np.ndarray
        Point coordinates as (N, 2) array of [x, y] values
    bounding_box : tuple or list
        Rectangular bounds as [xmin, xmax, ymin, ymax]

    Returns
    -------
    np.ndarray
        Boolean array of length N indicating which points are inside the box
        (True if inside, False if outside)

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0, 0], [5, 5], [15, 15]])
    >>> bbox = [0, 10, 0, 10]
    >>> mask = in_box(points, bbox)
    >>> print(mask)
    [True, True, False]
    """
    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]
        ),
        np.logical_and(
            bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]
        ),
    )


def bounded_voronoi(towers, bounding_box):
    """Generate bounded Voronoi diagram with finite regions.
    
    Creates a Voronoi diagram constrained to a rectangular bounding box by
    mirroring edge points outside the box. This ensures all Voronoi regions
    within the box are finite and bounded.

    Parameters
    ----------
    towers : np.ndarray
        Generator points as (N, 2) array of [x, y] coordinates
    bounding_box : tuple or list
        Rectangular bounds as [xmin, xmax, ymin, ymax]

    Returns
    -------
    scipy.spatial.Voronoi
        Voronoi diagram object with additional attributes:
        - filtered_points: generator points within bounding box
        - filtered_regions: list of region vertex indices for points in box

    Notes
    -----
    The algorithm mirrors points across each boundary (left, right, top, bottom)
    to create "virtual" points outside the box. This forces edge Voronoi cells
    to close at the boundaries rather than extending to infinity.

    Only the first 1/5 of regions (corresponding to original points) are retained
    in filtered_regions, as the remaining regions belong to mirrored points.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[25, 25], [75, 25], [50, 75]])
    >>> vor = bounded_voronoi(points, [0, 100, 0, 100])
    >>> print(f"Number of filtered regions: {len(vor.filtered_regions)}")
    """
    # Select points inside the bounding box
    i = in_box(towers, bounding_box)
    points_center = towers[i, :]

    # Mirror points across left boundary
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])

    # Mirror points across right boundary
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])

    # Mirror points across bottom boundary
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])

    # Mirror points across top boundary
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

    # Combine all points: original + mirrored
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )

    # Compute Voronoi diagram
    vor = sp.spatial.Voronoi(points)

    # Add custom attributes for filtered results
    vor.filtered_points = points_center  # Original points within bounding box
    vor.filtered_regions = np.array(vor.regions)[
        vor.point_region[: vor.npoints // 5]
    ]  # Regions corresponding to original points (first 1/5 of all regions)

    return vor


def centroid_region(vertices):
    """Calculate centroid of a polygon region.
    
    Computes the geometric centroid (center of mass) of a polygon defined
    by ordered vertices. Uses the standard polygon centroid formula based
    on signed area.

    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices as (N, 2) array of [x, y] coordinates.
        First and last vertex should be identical (closed polygon).

    Returns
    -------
    np.ndarray
        Centroid coordinates as (1, 2) array [[x_centroid, y_centroid]]

    Notes
    -----
    The formula computes the centroid using:
    
    .. math::
        C_x = \\frac{1}{6A} \\sum_{i=0}^{n-1} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
        
        C_y = \\frac{1}{6A} \\sum_{i=0}^{n-1} (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
    
    where A is the signed polygon area.

    Examples
    --------
    >>> import numpy as np
    >>> # Square polygon
    >>> vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    >>> centroid = centroid_region(vertices)
    >>> print(centroid)
    [[0.5, 0.5]]
    """
    # Initialize accumulation variables
    A = 0    # Signed area
    C_x = 0  # Centroid x-coordinate
    C_y = 0  # Centroid y-coordinate
    
    # Accumulate using trapezoidal formula
    for i in range(0, len(vertices) - 1):
        # Cross product term for this edge
        s = vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1]
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    
    # Finalize calculations
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    
    return np.array([[C_x, C_y]])


def triangulation(x, y):
    """Create Delaunay triangulation from coordinate arrays.

    Parameters
    ----------
    x : array-like
        X coordinates of points
    y : array-like
        Y coordinates of points

    Returns
    -------
    matplotlib.tri.Triangulation
        Delaunay triangulation object
    """

    # Create the Triangulation; no triangles specified so Delaunay triangulation created
    triangles = tri.Triangulation(x, y)
    xmid = x.values[triangles.triangles].mean(axis=1)

    # Code for removing specific triangles (currently disabled)
    # triangles2remove = [0, 1, ...]
    # mask = np.ones(len(xmid))
    # for i in range(len(xmid)):
    #     if i not in triangles2remove:
    #         mask[i] = False
    # triangles.set_mask(mask)
    
    return triangles
