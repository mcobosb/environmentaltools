"""Scan waypoint generator for DJI Pilot 2.

Reads a polygon (polygon_to_photograph.gpkg), calculates scan
lines in UTM based on height and field of view, generates waypoints
in zigzag pattern and exports CSV and GeoPackage with WGS84 geometry.

Outputs:
 - waypoints_dji.csv
 - waypoints_dji.gpkg (layer: 'waypoints')
"""

import math
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from loguru import logger


def ground_coverage(height_m, fov_deg):
    """Calculates ground coverage width for a camera.

    Parameters
    ----------
    height_m : float
        Flight height in meters.
    fov_deg : float
        Field of view in degrees.

    Returns
    -------
    float
        Coverage width in meters.
    """
    fov_rad = math.radians(fov_deg)
    return 2 * height_m * math.tan(fov_rad / 2)


def extract_polygons(geometry):
    """Extracts individual polygons from complex geometries.
    
    Parameters
    ----------
    geometry : shapely.geometry
        Shapely geometry object.
        
    Returns
    -------
    list
        List of individual polygons.
    """
    if geometry.geom_type == "Polygon":
        return [geometry]
    elif geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    else:
        return []


def load_study_area(filepath, utm_zone="EPSG:32630"):
    """Loads the study area from GeoPackage file.
    
    Parameters
    ----------
    filepath : str
        Path to GeoPackage file.
    utm_zone : str, optional
        UTM zone for conversion. Default is "EPSG:32630".
        
    Returns
    -------
    tuple
        (gdf_wgs84, gdf_utm) GeoDataFrames in WGS84 and UTM.
    """
    gdf = gpd.read_file(filepath)
    gdf_wgs84 = gdf.to_crs(epsg=4326)  # WGS84 for GPS
    gdf_utm = gdf.to_crs(utm_zone)     # UTM for metric calculations
    return gdf_wgs84, gdf_utm


def filter_polygons(gdf_wgs84, gdf_utm, polygon_indices=None):
    """Filters polygons according to specified indices.
    
    Parameters
    ----------
    gdf_wgs84 : geopandas.GeoDataFrame
        GeoDataFrame in WGS84 coordinate system.
    gdf_utm : geopandas.GeoDataFrame
        GeoDataFrame in UTM coordinate system.
    polygon_indices : list, optional
        Polygon indices to process (1-based). Default is None.
        
    Returns
    -------
    tuple
        (polys_wgs84, polys_utm) Lists of filtered polygons.
    """
    all_polys_wgs84 = []
    all_polys_utm = []
    
    if polygon_indices:
        for idx in polygon_indices:
            if 1 <= idx <= len(gdf_wgs84.geometry):
                all_polys_wgs84.extend(extract_polygons(gdf_wgs84.geometry[idx - 1]))
                all_polys_utm.extend(extract_polygons(gdf_utm.geometry[idx - 1]))
    else:
        for geom_wgs84, geom_utm in zip(gdf_wgs84.geometry, gdf_utm.geometry):
            all_polys_wgs84.extend(extract_polygons(geom_wgs84))
            all_polys_utm.extend(extract_polygons(geom_utm))
    
    return all_polys_wgs84, all_polys_utm


def calculate_scan_parameters(height, fov_deg, lateral_overlap, longitudinal_overlap):
    """Calculates scan parameters based on height and field of view.
    
    Parameters
    ----------
    height : float
        Flight height in meters.
    fov_deg : float
        Field of view in degrees.
    lateral_overlap : float
        Lateral overlap ratio (0-1).
    longitudinal_overlap : float
        Longitudinal overlap ratio (0-1).
        
    Returns
    -------
    tuple
        (spacing, photo_interval) in meters.
    """
    coverage = ground_coverage(height, fov_deg)
    spacing = coverage * (1 - lateral_overlap)
    photo_interval = coverage * (1 - longitudinal_overlap)
    return spacing, photo_interval


def generate_scan_lines(polygon_utm, spacing):
    """Generates scan lines for a polygon.
    
    Parameters
    ----------
    polygon_utm : shapely.geometry.Polygon
        Polygon in UTM coordinates.
    spacing : float
        Spacing between lines in meters.
        
    Returns
    -------
    tuple
        (lines, scan_direction) Scan lines and direction.
    """
    minx, miny, maxx, maxy = polygon_utm.bounds
    width = maxx - minx
    height_poly = maxy - miny
    lines = []
    
    # Determine scan direction based on polygon shape
    if width > height_poly:
        logger.info(f"Generating horizontal lines ({width:.1f}m x {height_poly:.1f}m)")
        y = miny + spacing / 2
        while y <= maxy - spacing / 2:
            line = LineString([(minx, y), (maxx, y)])
            clipped = line.intersection(polygon_utm)
            if not clipped.is_empty:
                lines.append(clipped)
            y += spacing
        scan_direction = "horizontal"
    else:
        logger.info(f"Generating vertical lines ({width:.1f}m x {height_poly:.1f}m)")
        x = minx + spacing / 2
        while x <= maxx - spacing / 2:
            line = LineString([(x, miny), (x, maxy)])
            clipped = line.intersection(polygon_utm)
            if not clipped.is_empty:
                lines.append(clipped)
            x += spacing
        scan_direction = "vertical"
    
    return lines, scan_direction


def create_waypoints_from_lines(lines, height, photo_interval, spacing):
    """Creates waypoints from scan lines.
    
    Parameters
    ----------
    lines : list
        List of scan lines.
    height : float
        Flight height in meters.
    photo_interval : float
        Interval between photos in meters.
    spacing : float
        Spacing between lines in meters.
        
    Returns
    -------
    list
        List of waypoint dictionaries.
    """
    waypoints = []
    
    for i, line in enumerate(lines):
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
            
        for geom in geoms:
            if geom.length == 0:
                continue
                
            num_points = int(geom.length // spacing)
            distances = [d * spacing for d in range(num_points + 1)]
            if distances[-1] < geom.length:
                distances.append(geom.length)
                
            coords = [geom.interpolate(dist).coords[0] for dist in distances]
            
            # Alternate direction for zigzag pattern
            if i % 2 == 1:
                coords = coords[::-1]
                
            for j, point in enumerate(coords):
                pt_utm = gpd.GeoSeries(
                    [gpd.points_from_xy([point[0]], [point[1]])[0]], crs="EPSG:32630"
                )
                pt_gps = pt_utm.to_crs(epsg=4326).geometry[0]
                
                waypoint = {
                    "latitude": round(pt_gps.y, 8),
                    "longitude": round(pt_gps.x, 8),
                    "altitude(m)": height,
                    "heading(deg)": 0,
                    "curvesize(m)": 5,
                    "rotationdir": 0,
                    "gimbalmode": 0,
                    "gimbalpitch(deg)": -90,
                    "actiontype1": "HOVER_AND_SHOOT",
                    "actionparam1": 0,
                    "actiontype2": "NONE",
                    "actionparam2": 0,
                    "actiontype3": "NONE",
                    "actionparam3": 0,
                    "actiontype4": "NONE",
                    "actionparam4": 0,
                    "actiontype5": "NONE",
                    "actionparam5": 0,
                    "speed(m/s)": 8,
                    "poi_latitude": 0,
                    "poi_longitude": 0,
                    "poi_altitude(m)": 0,
                    "poi_altitudemode": 0,
                    "photo_timeinterval": 2,
                    "photo_distinterval": photo_interval,
                }
                waypoints.append(waypoint)
    
    return waypoints


def save_waypoints_csv(waypoints, output_path, polygon_number=None):
    """Saves waypoints in CSV format.
    
    Parameters
    ----------
    waypoints : list
        List of waypoint dictionaries.
    output_path : str
        Output directory path.
    polygon_number : int, optional
        Polygon number for filename. Default is None.
        
    Returns
    -------
    str
        Path to saved file.
    """
    df = pd.DataFrame(waypoints)
    
    if polygon_number is not None:
        filename = f"waypoints_dji_poly_{polygon_number:03d}.csv"
    else:
        filename = "waypoints_dji.csv"
        
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    
    return filepath


def save_waypoints_gpkg(waypoints, output_path):
    """Saves waypoints in GeoPackage format.
    
    Parameters
    ----------
    waypoints : list
        List of waypoint dictionaries.
    output_path : str
        Output directory path.
        
    Returns
    -------
    str or None
        Path to saved file, or None if no waypoints provided.
    """
    if not waypoints:
        return None
        
    df = pd.DataFrame(waypoints)
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    
    filepath = os.path.join(output_path, "waypoints_dji.gpkg")
    gdf_points.to_file(filepath, layer="waypoints", driver="GPKG")
    
    return filepath

def plot_polygon_flight_plan(polygon_gps, polygon_utm, lines, waypoints, output_path, polygon_number):
    """Creates flight plan plots for an individual polygon.
    
    Parameters
    ----------
    polygon_gps : shapely.geometry.Polygon
        Polygon in GPS coordinates.
    polygon_utm : shapely.geometry.Polygon
        Polygon in UTM coordinates.
    lines : list
        List of scan lines.
    waypoints : list
        List of waypoint dictionaries.
    output_path : str
        Output directory path.
    polygon_number : int
        Polygon number for labeling.
        
    Returns
    -------
    str
        Path to generated image file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: UTM
    gpd.GeoSeries([polygon_utm]).plot(
        ax=ax1, alpha=0.5, color="lightblue", edgecolor="blue", linewidth=2
    )
    
    # Draw scan lines
    for line in lines:
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
        for geom in geoms:
            x_coords, y_coords = zip(*geom.coords)
            ax1.plot(x_coords, y_coords, "r-", linewidth=1.5, alpha=0.7)
    
    # Waypoints UTM
    waypoints_utm_x = []
    waypoints_utm_y = []
    for i, line in enumerate(lines):
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
        for geom in geoms:
            coords = list(geom.coords)
            if i % 2 == 1:
                coords = coords[::-1]
            for point in coords:
                waypoints_utm_x.append(point[0])
                waypoints_utm_y.append(point[1])
    
    ax1.scatter(
        waypoints_utm_x,
        waypoints_utm_y,
        c="red",
        s=30,
        alpha=0.8,
        zorder=5,
        label=f"Waypoints ({len(waypoints)})",
    )
    
    if waypoints_utm_x:
        ax1.scatter(
            waypoints_utm_x[0],
            waypoints_utm_y[0],
            c="green",
            s=60,
            marker="^",
            zorder=6,
            label="Start",
        )
        ax1.scatter(
            waypoints_utm_x[-1],
            waypoints_utm_y[-1],
            c="orange",
            s=60,
            marker="v",
            zorder=6,
            label="End",
        )
    
    ax1.set_title(f"Polygon and Scan Trajectories\n(UTM Coordinates) - Polygon {polygon_number:03d}")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    ax1.legend()
    
    # Plot 2: GPS
    gpd.GeoSeries([polygon_gps]).plot(
        ax=ax2, alpha=0.5, color="lightblue", edgecolor="blue", linewidth=2
    )
    
    for line in lines:
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
        for geom in geoms:
            line_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32630")
            line_gps = line_gdf.to_crs(epsg=4326).geometry[0]
            x_coords, y_coords = zip(*line_gps.coords)
            ax2.plot(x_coords, y_coords, "r-", linewidth=1.5, alpha=0.7)
    
    waypoints_gps_lon = [wp["longitude"] for wp in waypoints]
    waypoints_gps_lat = [wp["latitude"] for wp in waypoints]
    
    ax2.scatter(
        waypoints_gps_lon,
        waypoints_gps_lat,
        c="red",
        s=30,
        alpha=0.8,
        zorder=5,
        label=f"Waypoints ({len(waypoints)})",
    )
    
    if waypoints_gps_lon:
        ax2.scatter(
            waypoints_gps_lon[0],
            waypoints_gps_lat[0],
            c="green",
            s=60,
            marker="^",
            zorder=6,
            label="Start",
        )
        ax2.scatter(
            waypoints_gps_lon[-1],
            waypoints_gps_lat[-1],
            c="orange",
            s=60,
            marker="v",
            zorder=6,
            label="End",
        )
    
    for i in range(0, len(waypoints), max(1, len(waypoints) // 10)):
        ax2.annotate(
            f"{i+1}",
            (waypoints_gps_lon[i], waypoints_gps_lat[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )
    
    ax2.set_title(f"Polygon and Scan Trajectories\n(GPS Coordinates) - Polygon {polygon_number:03d}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")
    ax2.legend()
    
    fig_output = os.path.join(output_path, f"flight_plan_poly_{polygon_number:03d}.png")
    plt.savefig(fig_output)
    plt.close(fig)
    
    return fig_output

def plot_complete_flight_plan(gdf_wgs84, gdf_utm, all_lines, all_waypoints, output_path):
    """Creates complete flight plan plot.
    
    Parameters
    ----------
    gdf_wgs84 : geopandas.GeoDataFrame
        GeoDataFrame in WGS84 coordinate system.
    gdf_utm : geopandas.GeoDataFrame
        GeoDataFrame in UTM coordinate system.
    all_lines : list
        All scan lines.
    all_waypoints : list
        All waypoint dictionaries.
    output_path : str
        Output directory path.
        
    Returns
    -------
    str
        Path to generated image file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: In UTM coordinates (metric)
    gdf_utm.plot(ax=ax1, alpha=0.5, color="lightblue", edgecolor="blue", linewidth=2)

    # Draw scan lines
    for line in all_lines:
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
        for geom in geoms:
            x_coords, y_coords = zip(*geom.coords)
            ax1.plot(x_coords, y_coords, "r-", linewidth=1.5, alpha=0.7)

    # Draw waypoints in UTM
    waypoints_utm_x = []
    waypoints_utm_y = []
    for i, line in enumerate(all_lines):
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
        for geom in geoms:
            coords = list(geom.coords)
            if i % 2 == 1:
                coords = coords[::-1]
            for point in coords:
                waypoints_utm_x.append(point[0])
                waypoints_utm_y.append(point[1])

    ax1.scatter(
        waypoints_utm_x,
        waypoints_utm_y,
        c="red",
        s=30,
        alpha=0.8,
        zorder=5,
        label=f"Waypoints ({len(all_waypoints)})",
    )

    # Draw connection lines between waypoints
    for i in range(len(waypoints_utm_x) - 1):
        ax1.plot(
            [waypoints_utm_x[i], waypoints_utm_x[i + 1]],
            [waypoints_utm_y[i], waypoints_utm_y[i + 1]],
            "g--",
            alpha=0.5,
            linewidth=0.8,
        )

    # Mark start and end points
    if waypoints_utm_x:
        ax1.scatter(
            waypoints_utm_x[0],
            waypoints_utm_y[0],
            c="green",
            s=60,
            marker="^",
            zorder=6,
            label="Start",
        )
        ax1.scatter(
            waypoints_utm_x[-1],
            waypoints_utm_y[-1],
            c="orange",
            s=60,
            marker="v",
            zorder=6,
            label="Fin",
        )

    ax1.set_title("Polygon and Scan Trajectories\n(UTM Coordinates)")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    ax1.legend()

    # Plot 2: In GPS coordinates
    gdf_wgs84.plot(ax=ax2, alpha=0.5, color="lightblue", edgecolor="blue", linewidth=2)

    # Draw scan lines in GPS
    for line in all_lines:
        if line.geom_type == "LineString":
            geoms = [line]
        elif line.geom_type == "MultiLineString":
            geoms = list(line.geoms)
        else:
            continue
        for geom in geoms:
            line_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32630")
            line_gps = line_gdf.to_crs(epsg=4326).geometry[0]
            x_coords, y_coords = zip(*line_gps.coords)
            ax2.plot(x_coords, y_coords, "r-", linewidth=1.5, alpha=0.7)

    # Draw waypoints in GPS
    waypoints_gps_lon = [wp["longitude"] for wp in all_waypoints]
    waypoints_gps_lat = [wp["latitude"] for wp in all_waypoints]

    ax2.scatter(
        waypoints_gps_lon,
        waypoints_gps_lat,
        c="red",
        s=30,
        alpha=0.8,
        zorder=5,
        label=f"Waypoints ({len(all_waypoints)})",
    )

    # Draw connection lines between waypoints
    for i in range(len(waypoints_gps_lon) - 1):
        ax2.plot(
            [waypoints_gps_lon[i], waypoints_gps_lon[i + 1]],
            [waypoints_gps_lat[i], waypoints_gps_lat[i + 1]],
            "g--",
            alpha=0.5,
            linewidth=0.8,
        )

    # Mark start and end points
    if waypoints_gps_lon:
        ax2.scatter(
            waypoints_gps_lon[0],
            waypoints_gps_lat[0],
            c="green",
            s=60,
            marker="^",
            zorder=6,
            label="Start",
        )
        ax2.scatter(
            waypoints_gps_lon[-1],
            waypoints_gps_lat[-1],
            c="orange",
            s=60,
            marker="v",
            zorder=6,
            label="End",
        )

    # Number some waypoints
    for i in range(0, len(all_waypoints), max(1, len(all_waypoints) // 10)):
        ax2.annotate(
            f"{i+1}",
            (waypoints_gps_lon[i], waypoints_gps_lat[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    ax2.set_title("Polygon and Scan Trajectories\n(GPS Coordinates)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")
    ax2.legend()

    fig_output = os.path.join(output_path, "flight_plan.png")
    plt.savefig(fig_output)
    plt.close(fig)
    
    return fig_output


def calculate_flight_time(waypoints, hover_time_per_wp=2, speed_mps=8):
    """Calculates estimated flight time.
    
    Parameters
    ----------
    waypoints : list
        List of waypoint dictionaries.
    hover_time_per_wp : float, optional
        Hover time per waypoint in seconds. Default is 2.
    speed_mps : float, optional
        Flight speed in m/s. Default is 8.
        
    Returns
    -------
    tuple
        (travel_time_min, total_time_min) Flight times in minutes.
    """
    if len(waypoints) < 2:
        return 0, 0
        
    total_distance = 0
    for i in range(1, len(waypoints)):
        lat1, lon1 = waypoints[i - 1]["latitude"], waypoints[i - 1]["longitude"]
        lat2, lon2 = waypoints[i]["latitude"], waypoints[i]["longitude"]
        
        # Convert to UTM for precise distance calculation
        pt1 = (
            gpd.GeoSeries([gpd.points_from_xy([lon1], [lat1])[0]], crs="EPSG:4326")
            .to_crs(epsg=32630)
            .geometry[0]
        )
        pt2 = (
            gpd.GeoSeries([gpd.points_from_xy([lon2], [lat2])[0]], crs="EPSG:4326")
            .to_crs(epsg=32630)
            .geometry[0]
        )
        total_distance += pt1.distance(pt2)
    
    # Time calculations
    travel_time = total_distance / speed_mps  # seconds
    hover_time = len(waypoints) * hover_time_per_wp    # seconds
    total_time = travel_time + hover_time  # seconds
    
    return travel_time / 60, total_time / 60  # minutes


def process_polygon(polygon_gps, polygon_utm, polygon_number, spacing, height, 
                   photo_interval, output_path, polygon_indices=None):
    """Processes an individual polygon generating its flight plan.
    
    Parameters
    ----------
    polygon_gps : shapely.geometry.Polygon
        Polygon in GPS coordinates.
    polygon_utm : shapely.geometry.Polygon
        Polygon in UTM coordinates.
    polygon_number : int
        Polygon number for identification.
    spacing : float
        Spacing between scan lines in meters.
    height : float
        Flight height in meters.
    photo_interval : float
        Interval between photos in meters.
    output_path : str
        Output directory path.
    polygon_indices : list, optional
        Original polygon indices. Default is None.
        
    Returns
    -------
    tuple
        (lines, waypoints) Generated lines and waypoints.
    """
    # Get real polygon number
    if polygon_indices:
        real_polygon_no = polygon_indices[polygon_number - 1]
    else:
        real_polygon_no = polygon_number
    
    # Generate scan lines
    lines, scan_direction = generate_scan_lines(polygon_utm, spacing)
    logger.info(f"Polygon {polygon_number}: {len(lines)} scan lines ({scan_direction})")
    
    # Create waypoints
    waypoints = create_waypoints_from_lines(lines, height, photo_interval, spacing)
    
    # Save individual CSV
    csv_path = save_waypoints_csv(waypoints, output_path, real_polygon_no)
    logger.info(f"- {csv_path} ({len(waypoints)} waypoints)")
    
    # Create individual plot
    plot_path = plot_polygon_flight_plan(
        polygon_gps, polygon_utm, lines, waypoints, output_path, real_polygon_no
    )
    logger.info(f"- {plot_path} (flight plan)")
    
    return lines, waypoints


def analysis(config):
    """Executes the complete waypoint generation analysis.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing all parameters for analysis.
        Must include keys: 'path', 'study_area_filename', 'polygons', 
        'height', 'fov_deg', 'lateral_overlap', 'longitudinal_overlap',
        'utm_zone', 'hover_time_per_wp', 'speed_mps', 'output_folder'.
    
    Returns
    -------
    None
        Saves output files and logs progress information.
    """    
    
    logger.info("=== Scan waypoint generator for DJI Pilot 2 ===")
    logger.info(f"Configuration:")
    logger.info(f"- Flight height: {config['height']} m")
    logger.info(f"- Field of view: {config['fov_deg']}°")
    logger.info(f"- Lateral overlap: {config['lateral_overlap']*100:.0f}%")
    logger.info(f"- Longitudinal overlap: {config['longitudinal_overlap']*100:.0f}%")
    logger.info(f"- Output folder: {config['output_folder']}")
    
    # Create output folder if it doesn't exist
    output_path = Path(config["output_folder"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load study area
    filepath = Path(config["path"]) / config["study_area_filename"]
    logger.info(f"Loading study area from: {filepath}")
    gdf_wgs84, gdf_utm = load_study_area(filepath, config["utm_zone"])
    
    # 2. Calculate scan parameters
    spacing, photo_interval = calculate_scan_parameters(
        config["height"], 
        config["fov_deg"], 
        config["lateral_overlap"], 
        config["longitudinal_overlap"]
    )
    logger.info(f"Line spacing: {spacing:.1f} m")
    logger.info(f"Photo interval: {photo_interval:.1f} m")
    
    # 3. Filter polygons
    polys_wgs84, polys_utm = filter_polygons(gdf_wgs84, gdf_utm, config["polygons"])
    logger.info(f"Processing {len(polys_wgs84)} polygon(s)")
    
    # 4. Process each polygon
    all_lines = []
    all_waypoints = []
    
    for idx, (poly_gps, poly_utm) in enumerate(zip(polys_wgs84, polys_utm), 1):
        logger.info(f"--- Processing polygon {idx} ---")
        
        lines, waypoints = process_polygon(
            poly_gps, poly_utm, idx, spacing, config["height"], 
            photo_interval, config["output_folder"], config["polygons"]
        )
        
        all_lines.extend(lines)
        all_waypoints.extend(waypoints)
    
    # 5. Save consolidated files
    logger.info("--- Generating consolidated files ---")
    
    # Consolidated CSV
    csv_path = save_waypoints_csv(all_waypoints, config["output_folder"])
    logger.info(f"- {csv_path} ({len(all_waypoints)} waypoints)")
    
    # Consolidated GeoPackage
    gpkg_path = save_waypoints_gpkg(all_waypoints, config["output_folder"])
    if gpkg_path:
        logger.info(f"- {gpkg_path} (GeoPackage with {len(all_waypoints)} points)")
    else:
        logger.info("- No waypoints to save in GPKG")
    
    # 6. Create consolidated visualization
    plot_path = plot_complete_flight_plan(
        gdf_wgs84, gdf_utm, all_lines, all_waypoints, config["output_folder"]
    )
    logger.info(f"- {plot_path} (consolidated flight plan)")
    
    # 7. Calculate flight time
    travel_time, total_time = calculate_flight_time(
        all_waypoints, config["hover_time_per_wp"], config["speed_mps"]
    )
    
    logger.info("=== Flight plan summary ===")
    logger.info(f"Total waypoints: {len(all_waypoints)}")
    logger.info(f"Estimated flight time (travel only): {travel_time:.1f} minutes")
    logger.info(f"Estimated flight time (including hover): {total_time:.1f} minutes")
    logger.info(f"Files generated in: {config['output_folder']}")
    logger.info("Process completed successfully!")


