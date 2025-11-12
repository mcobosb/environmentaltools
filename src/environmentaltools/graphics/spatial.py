import io
from urllib.request import Request, urlopen

# Cartopy can fail with debugger in Python 3.13 due to Cython issues
try:
    import cartopy.crs as ccrs
    import cartopy.geodesic as cgeo
    import cartopy.io.img_tiles as cimgt
    HAS_CARTOPY = True
except (ImportError, AssertionError, KeyError) as e:
    HAS_CARTOPY = False
    ccrs = None
    cgeo = None
    cimgt = None
    import warnings
    warnings.warn(
        f"Cartopy import failed: {e}. Map-based plotting functions will not be available. "
        "This can happen when using the debugger with Python 3.13.",
        ImportWarning
    )

import cmocean
import matplotlib.pyplot as plt
import numpy as np
from environmentaltools.graphics.utils import handle_axis, labels, show
from environmentaltools.spatial.geotools import transform_coordinates
from environmentaltools.common import read
from PIL import Image

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)


def plot_interps(x, y, z, zoff, niveles=np.arange(-50, 20, 2), fname=None):
    """Plot comparison of two bathymetry interpolations.

    Creates a side-by-side comparison plot of GEBCO and IGN/MITECO bathymetry data
    with contour lines.

    Args:
        x (np.ndarray): X-coordinates mesh grid (m).
        y (np.ndarray): Y-coordinates mesh grid (m).
        z (np.ndarray): IGN/MITECO bathymetry values (m).
        zoff (np.ndarray): GEBCO bathymetry values (m).
        niveles (np.ndarray, optional): Contour levels. Defaults to np.arange(-50, 20, 2).
        fname (str, optional): Filename to save the figure. Defaults to None.
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cb = axs[0].contour(x, y, zoff, levels=niveles)
    cbar = fig.colorbar(cb, ax=axs[0])
    axs[0].set_title("GEBCO Bathymetry")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    cbar.ax.set_ylabel("z (m)")

    cb = axs[1].contour(x, y, z, levels=niveles)
    cbar = fig.colorbar(cb, ax=axs[1])
    axs[1].set_title("IGN/MITECO Bathymetry")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    cbar.ax.set_ylabel("z (m)")
    show(fname)
    return


def plot_mesh(
    data,
    levels=[-10, -1, 0, 1, 2, 5, 10, 20, 50, 100],
    var_="z",
    title=None,
    ax=None,
    fname=None,
    regular=False,
    cmap=cmocean.cm.deep_r,
    bar_label=r"\textbf{z (m)}",
    xlabel=r"\textbf{x (m)}",
    ylabel=r"\textbf{y (m)}",
    centercolormap=False,
    hide_colorbar=False,
    alpha=1,
):
    """Plot mesh data with contours and color mapping.

    Visualizes spatial mesh data (regular or triangular) with filled contours,
    optional contour lines, and customizable color mapping.

    Args:
        data (xarray.Dataset or dict): Dataset containing 'x', 'y' coordinates and variable data.
        levels (list, optional): Contour line levels to display. If None, no contours shown.
            Defaults to [-10, -1, 0, 1, 2, 5, 10, 20, 50, 100].
        var_ (str, optional): Variable name to plot from data. Defaults to "z".
        title (str, optional): Plot title. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.
        regular (bool, optional): If True, uses regular grid contouring. If False, uses
            triangular mesh. Defaults to False.
        cmap (matplotlib.colors.Colormap, optional): Colormap for filled contours.
            Defaults to cmocean.cm.deep_r.
        bar_label (str, optional): Colorbar label. Defaults to r"\textbf{z (m)}".
        xlabel (str, optional): X-axis label. Defaults to r"\textbf{x (m)}".
        ylabel (str, optional): Y-axis label. Defaults to r"\textbf{y (m)}".
        centercolormap (bool, optional): If True, centers colormap at zero. Defaults to False.
        hide_colorbar (bool, optional): If True, hides the colorbar. Defaults to False.
        alpha (float, optional): Transparency of filled contours (0-1). Defaults to 1.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    ax = handle_axis(ax, figsize=(10, 6))
    from matplotlib import colors

    # Center colormap at zero if requested
    if centercolormap:
        divnorm = colors.TwoSlopeNorm(
            vmin=np.min(data[var_]), vcenter=0.0, vmax=np.max(data[var_])
        )
    else:
        divnorm = None

    # Plot filled contours
    if regular:
        cbf = ax.contourf(
            data["x"], data["y"], data[var_], 100, cmap=cmap, norm=divnorm, alpha=alpha
        )
        if levels != None:
            # Add contour lines with labels
            cb = ax.contour(
                data["x"], data["y"], data[var_], levels=levels, colors="white"
            )
            ax.clabel(cb, inline=True, fontsize=8)
    else:
        # Use triangular mesh for irregular grids
        cbf = ax.tricontourf(
            data["x"].values,
            data["y"].values,
            data[var_].values,
            cmap=cmap,
            alpha=alpha,
        )
        if levels != None:
            cb = ax.tricontour(
                data["x"].values,
                data["y"].values,
                data[var_].values,
                levels=levels,
                cmap=cmap,
            )
            ax.clabel(cb, inline=True, fontsize=8)

    ax.grid("+")
    fig = plt.gcf()
    if not hide_colorbar:
        cbar = fig.colorbar(cbf, ax=ax)
        cbar.ax.set_ylabel(bar_label)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.set_aspect("equal", "box")
    show(fname)

    return ax


def plot_profiles(x, y, z, idx, nos=np.arange(0, 1000, 100)):
    """Plot spatial profiles and bathymetric cross-sections.

    Creates a two-panel plot showing: (1) spatial view with profile lines,
    and (2) depth profiles along selected transects.

    Args:
        x (np.ndarray): X-coordinates mesh grid (m).
        y (np.ndarray): Y-coordinates mesh grid (m).
        z (np.ndarray): Bathymetry or elevation values (m).
        idx (np.ndarray): Indices indicating the merge/reference location.
        nos (np.ndarray, optional): Array of profile line indices to plot.
            Defaults to np.arange(0, 1000, 100).
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cb = axs[0].contourf(x, y, z)
    cbar = fig.colorbar(cb, ax=axs[0])
    cbar.ax.set_ylabel("z (m)")
    axs[0].plot(x[0, :], y[idx, 0], "k", lw=2, label="merge location")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    for i in nos:
        axs[0].plot(x[:, i], y[:, i])
    axs[0].legend()

    for i in nos:
        axs[1].plot(y[:, idx[i]], z[:, i])
        axs[1].plot(y[idx[i], i], z[idx[i], i], ".")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")

    plt.show()

    return


def onclick(event):
    """Handle mouse click events on matplotlib figures.

    Prints information about click location (pixel and data coordinates) to console.
    Useful for interactive data exploration.

    Args:
        event (matplotlib.backend_bases.MouseEvent): Mouse click event containing
            click position and button information.
    """
    print(
        "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
        % (
            "double" if event.dblclick else "single",
            event.button,
            event.x,
            event.y,
            event.xdata,
            event.ydata,
        )
    )


def plot_preview(data):
    """Create an interactive preview scatter plot of spatial data.

    Displays a scatter plot with color-coded elevation/depth values and enables
    click event handling for interactive exploration.

    Args:
        data (pd.DataFrame or dict): Dataset containing 'x', 'y' coordinates and
            'z' values (elevation/depth).
    """

    fig = plt.figure(figsize=(12, 5))
    plt.scatter(data["x"], data["y"], c=data["z"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return


# def plot_nprofiles(
#     d,
#     z,
#     ax=None,
#     label: str = None,
#     show_legend: bool = False,
#     fname: str = None,
# ):
#     """[summary]

#     Args:
#         topobat ([type]): [description]
#         x ([type]): [description]
#         y ([type]): [description]
#         z ([type]): [description]
#         info ([type]): [description]
#         fname ([type]): [description]
#     """
#     ax = handle_axis(ax)

#     ax.plot(d, z, label=label)

#     ax.set_xlabel("x (m)")
#     ax.set_ylabel("z (m)")

#     ax.set_ylim([-40, 10])
#     if show_legend:
#         ax.legend()

#     show(fname)
#     return


# def plot_nprofiles(topobat, x, y, z, fname):
#     """[summary]

#     Args:
#         topobat ([type]): [description]
#         x ([type]): [description]
#         y ([type]): [description]
#         z ([type]): [description]
#         fname ([type]): [description]
#     """

#     # l0 = read_kmz(fname, 'utm') update
#     _, ax = plt.subplots(2, 1)
#     ax[0].plot(topobat["x"], topobat["y"], ".", label="Cota cero")
#     ax[0].plot(x, y, label="profile")
#     ax[0].set_xlabel("x (m)")
#     ax[0].set_ylabel("y (m)")
#     ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

#     d = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
#     ax[1].plot(d, z, label="profile")
#     ax[1].set_xlabel("x (m)")
#     ax[1].set_ylabel("z (m)")
#     ax[1].legend()
#     plt.show()
#     return


def plot_db(
    xarr,
    var_,
    coords=["lon", "lat"],
    ind_=0,
    levels=[0, 1, 2, 5, 10, 40],
    title=None,
    ax=None,
    fname=None,
):
    """Plot spatial database field with optional depth contours.

    Visualizes a variable from an xarray dataset with filled contours and optional
    depth contour lines overlaid.

    Args:
        xarr (xarray.Dataset): Dataset containing coordinates and variables.
        var_ (str): Variable name to plot.
        coords (list, optional): List of coordinate names [x_coord, y_coord].
            Defaults to ['lon', 'lat'].
        ind_ (int, optional): Index for third dimension (e.g., time or vertical level).
            Defaults to 0.
        levels (list, optional): Depth contour levels to display.
            Defaults to [0, 1, 2, 5, 10, 40].
        title (str, optional): Plot title. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    ax = handle_axis(ax)
    cbf = ax.contourf(
        xarr[coords[0]].data,
        xarr[coords[1]].data,
        xarr[var_][:, :, ind_].data,
        cmap=cmocean.cm.deep,
    )
    try:
        cb = ax.contour(
            xarr[coords[0]].data,
            xarr[coords[1]].data,
            xarr["depth"][:, :, ind_].data,
            levels=levels,
            colors="white",
        )
        plt.clabel(cb, inline=True, fontsize=8)
    except:
        pass

    ax.grid("+")
    fig = plt.gcf()
    cbar = fig.colorbar(cbf, ax=ax)

    if title is not None:
        ax.set_title(title, loc="right")

    ax.set_xlabel(labels(coords[0]))
    ax.set_ylabel(labels(coords[1]))
    plt.gca().set_aspect("equal", adjustable="box")
    cbar.ax.set_ylabel(labels(var_))
    show(fname)

    return ax


def plot_ascifiles(data, title=None, fname=None, ax=None):
    """Plot data from ASCII grid files.

    Displays raster data with contour lines, typically used for DEM (Digital Elevation Model)
    or bathymetry visualization from ASCII format files.

    Args:
        data (dict): Dictionary containing 'x', 'y', and 'z' arrays representing
            grid coordinates and elevation/depth values.
        title (str, optional): Plot title. Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    ax = handle_axis(ax)
    cbar = ax.imshow(
        data["z"],
        interpolation="none",
        extent=[data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()],
    )
    ax.contour(data["x"], data["y"], data["z"], colors="white")

    fig = plt.gcf()
    cbar = fig.colorbar(cbar, ax=ax)

    ax.set_xlabel(labels(list(data.keys())[0]))
    ax.set_ylabel(labels(list(data.keys())[1]))
    cbar.ax.set_ylabel(labels(list(data.keys())[2]))

    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    show(fname)
    return ax


def plot_2d_plan_view(topobat, isolines, fname):
    """Plot 2D horizontal plan view with bathymetry contours.

    Creates a plan view visualization of topography/bathymetry with labeled
    contour lines, typically used for coastal and marine engineering applications.

    Args:
        topobat (pd.DataFrame): DataFrame containing 'x', 'y' (UTM coordinates)
            and 'z' (elevation/depth) columns.
        isolines (list or np.ndarray): Contour levels to display.
        fname (str): Filename to save the figure.
    """
    plt.figure()
    cs = plt.tricontour(
        topobat.loc[:, "x"],
        topobat.loc[:, "y"],
        topobat.loc[:, "z"],
        levels=isolines,
        colors="k",
    )
    plt.clabel(cs, fontsize=9, inline=1)
    plt.xlabel(r"x$_{UTM}$ (m)")
    plt.ylabel(r"y$_{UTM}$ (m)")
    plt.show()
    return


def folium_map(data, more=[], fname="folium_map"):
    """Create an interactive Folium web map with spatial data.

    Generates an HTML-based interactive map using Folium library, displaying
    spatial features as line geometries. Coordinates are automatically transformed
    from UTM to WGS84.

    Args:
        data (np.ndarray): Array of coordinates in UTM format (x, y).
        more (list, optional): List of additional GeoJSON-compatible geometries
            to overlay on the map. Defaults to [].
        fname (str, optional): Output HTML filename (without extension).
            Defaults to "folium_map".
    """

    import folium
    import geopandas as gpd
    from shapely.geometry import LineString

    # Transform coordinates from UTM to WGS84 (lat/lon)
    data = transform_coordinates(data, "epsg:25830", "epsg:4326", by_columns=True)
    polygon = LineString(data[:, ::-1])
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])

    # Center map at minimum coordinates
    coords = data.min(axis=0)
    map_ = folium.Map(location=[coords[0], coords[1]], zoom_start=12)
    folium.GeoJson(polygon).add_to(map_)

    # Add additional elements to map
    for element in more:
        folium.GeoJson(element).add_to(map_)

    map_.save(fname + ".html")
    return


def flood_map(
    data, coast_line, flood_line, points, flood_polygon, title=None, fname=None, ax=None
):
    """Plot coastal flood inundation map.

    Visualizes coastal flooding extent showing the initial coastline, flood boundary,
    reference points, and calculated inundation area.

    Args:
        data (xarray.Dataset or dict): Background spatial data (typically bathymetry/topography).
        coast_line (pd.DataFrame or gpd.GeoDataFrame): Initial coastline geometry with x, y coordinates.
        flood_line (dict or pd.DataFrame): Flood extent boundary with 'x' and 'y' coordinates.
        points (pd.DataFrame or gpd.GeoDataFrame): Reference points with x, y coordinates.
        flood_polygon (gpd.GeoDataFrame): Polygon geometry of flooded area with 'area' attribute.
        title (str, optional): Plot title. Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    ax = handle_axis(ax)

    # Plot initial coastline
    ax.plot(
        coast_line.x,
        coast_line.y,
        "--",
        color="k",
        label="Initial coastline",
        ms=2,
    )
    # Plot flood extent
    ax.plot(
        flood_line["x"],
        flood_line["y"],
        "cyan",
        lw=2,
        label="Inundation (A ="
        + str(np.round(flood_polygon.area[0], decimals=2))
        + r" m$^2$)",
    )
    # Plot reference points
    ax.plot(points.x, points.y, "+", color="k", label="Reference points")
    ax.set_xlabel("x UTM (m)", fontweight="bold")
    ax.set_ylabel("y UTM (m)", fontweight="bold")

    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    ax.legend()
    show(fname)
    return ax


def plot_quiver(
    db,
    is_db=True,
    vars_=["U", "DirU"],
    cadency=1,
    title=None,
    scale=1,
    label_="U",
    ax=None,
    fname=None,
):
    """Plot vector field using quiver arrows (e.g., currents, wind).

    Visualizes directional data such as ocean currents or wind fields using
    arrow plots with magnitude-based coloring. Handles both xarray datasets
    and regular arrays.

    Args:
        db (xarray.Dataset or dict): Dataset containing velocity magnitude and direction.
        is_db (bool, optional): If True, treats input as xarray Dataset. If False, treats
            as regular numpy arrays. Defaults to True.
        vars_ (list, optional): List of variable names [magnitude, direction].
            Direction should be in degrees (oceanographic convention: direction TO).
            Defaults to ["U", "DirU"].
        cadency (int, optional): Sampling interval for arrows (plots every nth point).
            Higher values = fewer arrows. Defaults to 1.
        title (str, optional): Plot title. Defaults to None.
        scale (float, optional): Arrow scale factor. Smaller values = longer arrows.
            Defaults to 1.
        label_ (str, optional): Label for the colorbar. Defaults to "U".
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    # Convert direction to U, V components (oceanographic to mathematical convention)
    U, V = (
        db[vars_[0]] * np.cos(np.deg2rad(270 - db[vars_[1]])),
        db[vars_[0]] * np.sin(np.deg2rad(270 - db[vars_[1]])),
    )
    M = np.hypot(U, V)  # Magnitude for coloring

    # Plot background bathymetry/depth
    ax = plot_db(
        db, "depth", coords=["x", "y"], ind_=0, levels=[0], fname="to_axes", ax=ax
    )

    if is_db:
        # For xarray dataset
        cbar = ax.quiver(
            db["x"].data[::cadency, ::cadency],
            db["y"].data[::cadency, ::cadency],
            U[:, :, 0].data[::cadency, ::cadency],
            V[:, :, 0].data[::cadency, ::cadency],
            M[:, :, 0].data[::cadency, ::cadency],
            units="x",
            cmap=cmocean.cm.thermal,
            pivot="tip",
            scale=scale,
            width=10,
        )
    else:
        # For regular numpy arrays
        cbar = ax.quiver(
            db["x"][::cadency, ::cadency],
            db["y"][::cadency, ::cadency],
            U[::cadency, ::cadency],
            V[::cadency, ::cadency],
            M[::cadency, ::cadency],
            units="x",
            cmap=cmocean.cm.thermal,
            scale=scale,
            pivot="tip",
            width=10,
        )

    cbar = fig.colorbar(cbar, ax=ax)
    cbar.ax.set_ylabel(labels(label_))

    ax.set_xlabel(labels("x"), fontweight="bold")
    ax.set_ylabel(labels("y"), fontweight="bold")

    ax.grid(True)
    if title is not None:
        ax.set_title(title, loc="right")

    show(fname)
    return ax


def coastline_ci(coast_lines, title=None, fname=None, ax=None):
    """Plot coastline position evolution with confidence intervals.

    Displays temporal evolution of coastline position showing mean, initial,
    final positions, and envelope (min-max range) over time.

    Args:
        coast_lines (pd.DataFrame): DataFrame containing coastline positions with columns:
            - 'x': Along-shore coordinate
            - 'mean': Mean coastline position
            - 'min': Minimum (most seaward) position
            - 'max': Maximum (most landward) position
            - 'ini': Initial coastline position
            - 'end': Final coastline position
        title (str, optional): Plot title. Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """

    ax = handle_axis(ax)
    
    # Define plot styling
    colors = {"mean": "k", "min": "gray", "max": "k", "ini": "red", "end": "purple"}
    linestyles = {"mean": "--", "min": "--", "max": "--", "ini": "-", "end": "-"}
    lws = {"mean": 1, "min": 1, "max": 1, "ini": 2, "end": 2}
    names = {
        "mean": "Mean profile",
        "min": 1,
        "max": 1,
        "ini": "Initial profile",
        "end": "Final profile",
    }

    # Plot mean, initial, and final coastlines
    for name in ["mean", "ini", "end"]:
        ax.plot(
            coast_lines.x.values,
            coast_lines[name].values,
            label=names[name],
            color=colors[name],
            linestyle=linestyles[name],
            lw=lws[name],
        )

    # Plot envelope (min-max range)
    ax.fill_between(
        coast_lines.x.values,
        coast_lines["max"].values,
        coast_lines["min"].values,
        label="Envelope",
        color="darkblue",
        alpha=0.25,
    )

    ax.set_xlabel(labels("x"), fontweight="bold")
    ax.set_ylabel(labels("y"), fontweight="bold")
    ax.set_ylim(coast_lines.loc[:, "min"].min() - 100, coast_lines.loc[:, "max"].max())

    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight="bold", loc="right", color="gray")

    ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    show(fname)
    return ax


def plot_voronoi_diagram(vor, bounding_box, fname=False):
    """Plot Voronoi diagram with vertices and regions.

    Visualizes a Voronoi tessellation showing input points, region vertices,
    and boundaries within a specified bounding box.

    Args:
        vor (scipy.spatial.Voronoi): Voronoi diagram object with attributes:
            - filtered_points: Input points used for tessellation
            - filtered_regions: List of region vertex indices
            - vertices: Coordinates of Voronoi vertices
        bounding_box (tuple or list): Box limits as (xmin, xmax, ymin, ymax).
        fname (str or bool, optional): Filename to save the figure. If False, doesn't save.
            Defaults to False.

    Returns:
        tuple: (fig, ax) - Figure and axes objects.
    """
    # Initialize pyplot figure
    fig = plt.figure(figsize=(12, 11))
    ax = fig.gca()

    # Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], "b.")

    # Plot Voronoi vertices
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], "go")

    # Plot region boundaries (ridges)
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], "k-")

    # Set axis limits with margin
    margin_percent = 0.1
    width = bounding_box[1] - bounding_box[0]
    height = bounding_box[3] - bounding_box[2]

    ax.set_xlim(
        [
            bounding_box[0] - width * margin_percent,
            bounding_box[1] + width * margin_percent,
        ]
    )
    ax.set_ylim(
        [
            bounding_box[2] - height * margin_percent,
            bounding_box[3] + height * margin_percent,
        ]
    )

    show(fname)

    return fig, ax


def osm_image(
    lons, lats, style="satellite", epsg=None, title=None, ax=None, fname=None
):
    """Create OpenStreetMap background image for spatial visualization.

    Generates a map with OpenStreetMap imagery (satellite or street map style) as background.
    Automatically handles coordinate transformations and scale calculations. Based on code by
    Mathew Lipson (m.lipson@unsw.edu.au).

    Args:
        lons (list or np.ndarray): Longitude or x-coordinate bounds [min, max]. 
            Format depends on epsg parameter.
        lats (list or np.ndarray): Latitude or y-coordinate bounds [min, max].
            Format depends on epsg parameter.
        style (str, optional): Map style - either "satellite" or "map". 
            Defaults to "satellite".
        epsg (int, optional): EPSG code of input coordinates. Supported values:
            - 25829, 25830: UTM zones (automatically transforms to WGS84)
            - 4326, 4328: WGS84 geodetic (lon/lat)
            - None: Assumes WGS84 and calculates distance
            Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object with cartopy projection. 
            Defaults to None.
        fname (str, optional): Filename to save the figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object with the map.

    Note:
        Be careful with scale and radius combinations. Large scale (>16) with large radius (>1000m)
        may violate OSM policies (https://operations.osmfoundation.org/policies/tiles/).
        
        Scale guidelines:
        - 2: Worldwide or continental scales
        - 4-6: Countries and larger states
        - 6-10: Smaller states, regions, and cities
        - 10-12: City boundaries and zip codes
        - 14+: Roads, blocks, buildings
    """
    bEpsg = False
    if str(epsg).startswith("258") and (
        str(epsg).endswith("30") | str(epsg).endswith("29")
    ):
        radius = np.max([np.abs(lons[0] - lons[1]), np.abs(lats[1] - lats[0])])
        # The limits should be included in epsg:3857
        data_CRS = ccrs.epsg(epsg)

        x0, x1 = lons
        y0, y1 = lats

        geodetic_CRS = ccrs.Geodetic()
        lons[0], lats[0] = geodetic_CRS.transform_point(x0, y0, data_CRS)
        lons[1], lats[1] = geodetic_CRS.transform_point(x1, y1, data_CRS)

    elif (epsg == 4326) | (epsg == 4328):
        # The EPSG code must correspond to a “projected coordinate system”, so EPSG codes
        # such as 4326 (WGS-84) which define a “geodetic coordinate system” will not work.

        data_CRS = ccrs.Geodetic()
        utm_proj = ccrs.epsg(25830)
        x0, y0 = utm_proj.transform_point(lons[0], lats[0], data_CRS)
        x1, y1 = utm_proj.transform_point(lons[1], lats[1], data_CRS)

        radius = np.max([np.abs(x0 - x1), np.abs(y1 - y0)])

        bEpsg = True
    else:
        R = 6371  # km
        dLat = np.deg2rad(lats[1] - lats[0])
        dLon = np.deg2rad(lons[1] - lons[0])
        a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.deg2rad(lats[0])) * np.cos(
            np.deg2rad(lats[1])
        ) * np.sin(dLon / 2) * np.sin(dLon / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        radius = R * c / 2 * 1000

        geodetic_CRS = ccrs.Geodetic()
        map_proj = ccrs.epsg(25830)
        x0, y0 = map_proj.transform_point(lons[0], lats[0], geodetic_CRS)
        x1, y1 = map_proj.transform_point(lons[1], lats[1], geodetic_CRS)

    lon, lat = (lons[0] + lons[1]) / 2, (lats[0] + lats[1]) / 2

    if style == "map":
        ## MAP STYLE
        cimgt.OSM.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == "satellite":
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = (
            image_spoof  # reformat web request for street map spoofing
        )
        img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
    else:
        raise ValueError('No valid style. Choose "satellite" or "map".')

    _, ax = handle_axis(ax, dim=0, projection=img.crs)

    # Auto-calculate zoom scale based on radius
    scale = int(120 / np.log(radius))
    scale = (scale < 20) and scale or 19

    # Manual scale override option available
    # NOTE: scale specifications should be selected based on radius
    # Be careful not to have both large scale (>16) and large radius (>1000)
    # This is forbidden under OSM policies: https://operations.osmfoundation.org/policies/tiles/
    # Scale guidelines:
    # -- 2     = coarse image, worldwide or continental scales
    # -- 4-6   = medium coarseness, countries and larger states  
    # -- 6-10  = medium fineness, smaller states, regions, and cities
    # -- 10-12 = fine image, city boundaries and zip codes
    # -- 14+   = extremely fine image, roads, blocks, buildings

    extent = calc_extent(lon, lat, radius * 1.1)
    ax.set_extent(extent)  # Set map extents

    def label_grid(x0, x1, y0, y1, utm=False):
        """Configure grid labels for the map axes.
        
        Warning: Should only be used with small area UTM maps.
        """
        ax = plt.gca()
        nx, ny = 7, 5  # Number of tick labels

        decimals = 3 if not utm else 0

        xtickini, xtickend, ytickini, ytickend = (
            ax.get_xticks()[0],
            ax.get_xticks()[-1],
            ax.get_yticks()[0],
            ax.get_yticks()[-1],
        )
        
        # Create x-axis labels
        xlabels_, xticks_ = [], []
        for k_ in range(nx):
            values_ = np.round(x0 + (x1 - x0) * k_ / (nx - 1), decimals=decimals)
            xticks_.append(
                np.round(
                    xtickini + (xtickend - xtickini) * k_ / (nx - 1), decimals=decimals
                )
            )
            values_ = values_ if not utm else int(values_)
            xlabels_.append(str(values_))
        plt.xticks(xticks_, xlabels_)

        # Create y-axis labels
        ylabels_, yticks_ = [], []
        for k_ in range(ny):
            values_ = np.round(y0 + (y1 - y0) * k_ / (ny - 1), decimals=decimals)
            yticks_.append(
                np.round(
                    ytickini + (ytickend - ytickini) * k_ / (ny - 1), decimals=decimals
                )
            )
            values_ = values_ if not utm else int(values_)
            ylabels_.append(str(values_))
        plt.yticks(yticks_, ylabels_)

        # Configure grid appearance
        for xaxis_ in ax.get_xgridlines():
            xaxis_.set_color("black")

        for yaxis_ in ax.get_ygridlines():
            yaxis_.set_color("black")

        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        plt.grid(True)

    ax.add_image(img, int(scale))  # Add OSM layer with zoom level

    ax.set_extent([lons[0], lons[1], lats[0], lats[1]])
    if not bEpsg:
        label_grid(x0, x1, y0, y1, utm=True)
    else:
        label_grid(lons[0], lons[1], lats[0], lats[1])

    if title is not None:
        ax.set_title(title)

    if not bEpsg:
        ax.set_xlabel(r"\textbf{x (m)}")
        ax.set_ylabel(r"\textbf{y (m)}")
    else:
        ax.set_xlabel(r"\textbf{latitude ($\mathbf{^o}$)")
        ax.set_ylabel(r"\textbf{longitude ($\mathbf{^o}$)")

    show(fname)

    return ax


def calc_extent(lon, lat, dist):
    """Calculate map extent from center point and distance.

    Computes the bounding box coordinates for a map centered at given lon/lat
    with specified distance to edges using geodesic calculations.

    Args:
        lon (float): Center longitude in degrees.
        lat (float): Center latitude in degrees.
        dist (float): Distance from center to edge in meters.

    Returns:
        list: Map extent as [lon_min, lon_max, lat_min, lat_max].
    """
    # Calculate corner points using geodesic distance
    dist_cnr = np.sqrt(2 * dist**2)
    top_left = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=-45, distances=dist_cnr
    )[:, 0:2][0]
    bot_right = cgeo.Geodesic().direct(
        points=(lon, lat), azimuths=135, distances=dist_cnr
    )[:, 0:2][0]

    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]

    return extent


def image_spoof(self, tile):
    """Reformat web requests from OSM for cartopy compatibility.

    Spoofs the user agent in HTTP requests to OpenStreetMap tile servers,
    allowing cartopy to download and display map tiles.

    Heavily based on code by Joshua Hrisko:
    https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy

    Args:
        self: The tile image object (OSM or QuadtreeTiles instance).
        tile: Tile coordinates tuple.

    Returns:
        tuple: (image, extent, origin) compatible with cartopy.
    """

    url = self._image_url(tile)  # Get the URL of the map tile API
    req = Request(url)  # Start HTTP request
    req.add_header("User-agent", "Anaconda 3")  # Add user agent header
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # Download image data
    fh.close()  # Close connection
    img = Image.open(im_data)  # Open image with PIL
    img = img.convert(self.desired_tile_form)  # Convert to desired format
    return img, self.tileextent(tile), "lower"  # Return in cartopy format


def include_Andalusian_coast(path, ax):
    """Add Andalusian coastline to map plot.

    Reads shapefile data and plots the Andalusian coast (two segments) on the
    provided axes, transforming coordinates from UTM to WGS84.

    Args:
        path (str): Path to the shapefile containing coastline data.
        ax (matplotlib.axes.Axes): Axes object to plot the coastline on.
    """
    data = read.shp(path)
    costa = transform_coordinates(data[0], "epsg:25830", "epsg:4326")
    ax.plot(costa.x, costa.y, "dimgrey")
    costa = transform_coordinates(data[1], "epsg:25830", "epsg:4326")
    ax.plot(costa.x, costa.y, "dimgrey")
    return


def include_coastal_Andalusian_cities(ax):
    """Add labels and markers for coastal Andalusian cities to map.

    Plots markers and labels for major coastal cities in Andalusia (Spain)
    including Malaga, Cadiz, Huelva, Almeria, Ceuta, Melilla, Motril, and Algeciras.
    Coordinates are automatically transformed from UTM/EPSG to WGS84.

    Args:
        ax (matplotlib.axes.Axes): Axes object with geographic projection to add city labels to.
    """
    cities = {
        "Malaga": {"coords": [4064894.21433146, 373065.536614759], "proj": "utm"},
        "Cadiz": {"coords": [4045830.06891307, 742762.266396567], "proj": "epsg:25829"},
        "Huelva": {
            "coords": [4125852.79525744, 682253.299985719],
            "proj": "epsg:25829",
        },
        "Almeria": {"coords": [4076597.42632942, 547820.672473857], "proj": "utm"},
        "Ceuta": {"coords": [3974169.33152635, 290472.395171953], "proj": "utm"},
        "Melilla": {"coords": [3905458.34473556, 505628.319059769], "proj": "utm"},
        "Motril": {"coords": [4067337.8742406, 453769.08587931], "proj": "utm"},
        "Algeciras": {"coords": [4001510.6438788, 279492.87333613], "proj": "utm"},
    }

    for city in cities.keys():
        # Transform coordinates to WGS84
        location = transform_coordinates(
            np.array(cities[city]["coords"][::-1]),
            cities[city]["proj"],
            "epsg:4326",
            by_columns=True,
        )
        ax.plot(location[1], location[0], "o", ms=2, color="black")
        
        # Configure label positioning for each city
        rotation = 45
        valign = "bottom"
        halign = "left"
        if city == "Huelva":
            rotation = 0
            valign = "center"
            location[1] = location[1] + 0.05
        elif city == "Ceuta":
            valign = "top"
            halign = "right"
        elif city == "Algeciras":
            halign = "center"

        ax.text(
            location[1],
            location[0],
            city,
            rotation=rotation,
            verticalalignment=valign,
            horizontalalignment=halign,
            color="black",
        )

    return


def include_seas(ax):
    """Add labels for Atlantic Ocean and Mediterranean Sea to map.

    Places text labels for major water bodies adjacent to the Andalusian coast.
    Coordinates are transformed from UTM/EPSG to WGS84.

    Args:
        ax (matplotlib.axes.Axes): Axes object with geographic projection to add sea labels to.
    """
    seas = {
        "Atlantic Ocean": {
            "coords": [3974169.33152635, 660253.299985719],
            "proj": "epsg:25829",
        },
        "Mediterranean Sea": {
            "coords": [3974169.33152635, 440769.08587931],
            "proj": "utm",
        },
    }

    for sea in seas.keys():
        # Transform coordinates to WGS84
        location = transform_coordinates(
            np.array(seas[sea]["coords"][::-1]),
            seas[sea]["proj"],
            "epsg:4326",
            by_columns=True,
        )

        label = r"\textbf{" + sea + "}"
        ax.text(
            location[1],
            location[0],
            label,
            rotation=0,
            horizontalalignment="left",
            color="blue",
            fontsize=12,
        )
    return
