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


# ---------------------------------------------------------------------------
# DELFT3D grid helpers (private)
# ---------------------------------------------------------------------------

def _read_delft_grd(grd_path):
    """Read a DELFT3D ASCII .grd file (RGFGRID format).

    Returns:
        tuple: (X, Y, mmax, nmax) where X and Y are 2-D arrays of shape
        (nmax, mmax) in the coordinate system of the file.
    """
    with open(grd_path, encoding="latin-1") as f:
        lines = f.readlines()

    miss = -999.0
    mmax = nmax = None
    rows, current = [], []
    in_data = False

    for line in lines:
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        if "Missing" in s:
            miss = float(s.split("=")[1].strip())
            continue
        if "Coordinate" in s:
            continue
        if mmax is None and "=" not in s:
            parts = s.split()
            if len(parts) == 2:
                try:
                    mmax, nmax = int(parts[0]), int(parts[1])
                    continue
                except ValueError:
                    pass
        if s == "0 0 0":
            continue
        if s.startswith("ETA="):
            if current:
                rows.append(current)
            current = [float(v) for v in s.split()[2:]]
            in_data = True
        elif in_data:
            current.extend(float(v) for v in s.split())
    if current:
        rows.append(current)

    X = np.array(rows[:nmax], dtype=float)
    Y = np.array(rows[nmax:], dtype=float)
    X[np.isclose(X, miss)] = np.nan
    Y[np.isclose(Y, miss)] = np.nan
    return X, Y, mmax, nmax


def _build_grid_segments(X, Y):
    """Build all cell-edge segments for a DELFT3D curvilinear grid.

    Args:
        X (np.ndarray): 2-D array of X coordinates (nmax, mmax).
        Y (np.ndarray): 2-D array of Y coordinates (nmax, mmax).

    Returns:
        np.ndarray: Array of shape (N, 2, 2) with [start, end] pairs for
        every valid grid edge, ready for use with ``LineCollection``.
    """
    x1h, x2h = X[:, :-1], X[:, 1:]
    y1h, y2h = Y[:, :-1], Y[:, 1:]
    valid_h = ~(np.isnan(x1h) | np.isnan(x2h))
    segs_h = np.stack(
        [np.stack([x1h, y1h], axis=-1), np.stack([x2h, y2h], axis=-1)], axis=-2
    )[valid_h]

    x1v, x2v = X[:-1, :], X[1:, :]
    y1v, y2v = Y[:-1, :], Y[1:, :]
    valid_v = ~(np.isnan(x1v) | np.isnan(x2v))
    segs_v = np.stack(
        [np.stack([x1v, y1v], axis=-1), np.stack([x2v, y2v], axis=-1)], axis=-2
    )[valid_v]

    return np.vstack([segs_h, segs_v])


def _load_coast_segments(coast_path, gap_m=5000):
    """Load an XYZ coastline file and split it into continuous segments.

    The file must be a CSV with ``x`` and ``y`` columns in the same CRS as
    the grids. Segments are separated wherever consecutive points are more
    than *gap_m* metres apart.

    Args:
        coast_path (str or Path): Path to the CSV coastline file.
        gap_m (float): Gap threshold in metres. Defaults to 5000.

    Returns:
        list[np.ndarray]: List of (N, 2) arrays, one per segment.
    """
    import pandas as _pd

    df = _pd.read_csv(coast_path)
    xy = df[["x", "y"]].values
    dist = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    breaks = list(np.where(dist > gap_m)[0] + 1)
    segs, prev = [], 0
    for b in breaks + [len(xy)]:
        segs.append(xy[prev:b])
        prev = b
    return segs


def _build_land_polygon(coast_xy, xmin, xmax, ymin, ymax,
                        land_is_north=True, pad=20000):
    """Build a filled land polygon from an open coastline.

    The coastline must run roughly west–east across the domain. The polygon
    is closed through the *north* (high-Y) edge of the padded extent when
    ``land_is_north=True``, or through the *south* edge otherwise.

    Args:
        coast_xy (np.ndarray): Array of shape (N, 2) with the main coastline.
        xmin, xmax, ymin, ymax (float): Map extent in metres.
        land_is_north (bool): If True, land is north of the coast (default).
        pad (float): Extra padding beyond the extent in metres. Defaults to 20000.

    Returns:
        np.ndarray or None: Array of polygon vertices (M, 2), or None if the
        coastline does not intersect the padded extent.
    """
    p = pad
    mask = (
        (coast_xy[:, 0] >= xmin - p) & (coast_xy[:, 0] <= xmax + p) &
        (coast_xy[:, 1] >= ymin - p) & (coast_xy[:, 1] <= ymax + p)
    )
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    pts = coast_xy[idx[0]:idx[-1] + 1]
    # Ensure west-to-east ordering
    if pts[0, 0] > pts[-1, 0]:
        pts = pts[::-1]
    close_y = ymax + p if land_is_north else ymin - p
    poly = np.vstack([
        [[xmin - p, pts[0, 1]]],
        pts,
        [[xmax + p, pts[-1, 1]]],
        [[xmax + p, close_y]],
        [[xmin - p, close_y]],
    ])
    return poly


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

_DELFT_DEFAULT_STYLES = [
    {"color": "#1565C0", "lw": 0.30, "alpha": 0.65},
    {"color": "#E65100", "lw": 0.40, "alpha": 0.85},
    {"color": "#2E7D32", "lw": 0.35, "alpha": 0.75},
    {"color": "#6A1B9A", "lw": 0.35, "alpha": 0.75},
]


def plot_delft_grids(
    grd_paths,
    coast_xyz=None,
    points=None,
    utm_zone=30,
    margin=8000,
    color_land="#EDE8DF",
    color_ocean="#C8DCF0",
    coast_gap_m=5000,
    land_is_north=True,
    grid_styles=None,
    point_label_col=None,
    point_label_offset=500,
    title=None,
    figsize=(12, 10),
    fname=None,
):
    """Plot DELFT3D-WAVE curvilinear grids with an optional basemap.

    Draws one or more DELFT3D curvilinear grids read from ASCII ``.grd``
    files (RGFGRID format) over a basemap built from a high-resolution
    coastline polyline. Extraction or control points can be overlaid
    optionally.

    All spatial data (grids, coastline and points) must be in the same
    UTM coordinate system.

    Args:
        grd_paths (dict[str, str | Path]): Mapping of grid label → path to
            the ``.grd`` file (e.g. ``{"ext": "Alboran_ext.grd", "int":
            "Alboran_int.grd"}``). Grids are drawn in dict order.
        coast_xyz (str | Path | None): Path to a CSV file with ``x`` and
            ``y`` columns (same UTM CRS) representing the coastline as an
            ordered polyline. When provided, land and ocean are filled with
            *color_land* and *color_ocean* respectively. Defaults to None.
        points (pd.DataFrame | None): Optional table of control or
            extraction points with at least ``x`` and ``y`` columns (same
            UTM CRS). An extra column named *point_label_col* is used for
            annotations when provided. Defaults to None.
        utm_zone (int): UTM zone number used for the cartopy projection.
            Defaults to 30.
        margin (float): Map margin around the outermost grid extent, in
            metres. Defaults to 8000.
        color_land (str): Fill colour for land areas. Defaults to
            ``"#EDE8DF"``.
        color_ocean (str): Fill colour for ocean background. Defaults to
            ``"#C8DCF0"``.
        coast_gap_m (float): Threshold in metres for splitting the
            coastline polyline into separate segments. Defaults to 5000.
        land_is_north (bool): If True, the land polygon is closed through
            the northern edge of the map extent. Set to False for domains
            where land is to the south of the coastline. Defaults to True.
        grid_styles (dict[str, dict] | None): Per-grid style overrides. Keys
            must match those in *grd_paths*. Each value is a dict with any
            of ``color``, ``lw``, ``alpha``. Defaults to None (built-in
            palette).
        point_label_col (str | None): Column in *points* to use as point
            labels. If None, points are drawn without text labels. Defaults
            to None.
        point_label_offset (float): Label offset in metres from the point
            position. Defaults to 500.
        title (str | None): Figure title. Defaults to None.
        figsize (tuple): Figure size in inches. Defaults to (12, 10).
        fname (str | None): Output file path. If None, the figure is
            displayed interactively. Defaults to None.

    Returns:
        matplotlib.axes.GeoAxes: The cartopy axes with all elements drawn.

    Examples:
        >>> from pathlib import Path
        >>> from environmentaltools.graphics import plot_delft_grids
        >>> plot_delft_grids(
        ...     grd_paths={"ext": "Alboran_ext.grd", "int": "Alboran_int.grd"},
        ...     coast_xyz="linea_costa_oficial_utm30n.xyz",
        ...     points=df_points,           # DataFrame with x, y, label columns
        ...     point_label_col="label",
        ...     utm_zone=30,
        ...     title="DELFT3D grids — Alboran Sea",
        ...     fname="mallas_puntos.png",
        ... )
    """
    if not HAS_CARTOPY:
        raise ImportError(
            "cartopy is required for plot_delft_grids. "
            "Install it with: conda install -c conda-forge cartopy"
        )

    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches

    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    # ------------------------------------------------------------------
    # Ocean background
    # ------------------------------------------------------------------
    ax.set_facecolor(color_ocean)

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    legend_handles = []
    x_all, y_all = [], []

    for i, (name, path) in enumerate(grd_paths.items()):
        X, Y, mmax, nmax = _read_delft_grd(path)
        segs = _build_grid_segments(X, Y)

        # Resolve style
        default_st = _DELFT_DEFAULT_STYLES[i % len(_DELFT_DEFAULT_STYLES)]
        if grid_styles and name in grid_styles:
            st = {**default_st, **grid_styles[name]}
        else:
            st = default_st

        lc = LineCollection(
            segs,
            colors=st["color"],
            linewidths=st["lw"],
            alpha=st["alpha"],
            transform=proj,
            zorder=4,
        )
        ax.add_collection(lc)
        legend_handles.append(
            mpatches.Patch(
                facecolor=st["color"],
                alpha=0.6,
                edgecolor=st["color"],
                label=f"{name}  ({mmax}\u00d7{nmax})",
            )
        )
        x_all += [np.nanmin(X), np.nanmax(X)]
        y_all += [np.nanmin(Y), np.nanmax(Y)]

    # ------------------------------------------------------------------
    # Map extent
    # ------------------------------------------------------------------
    extent_utm = [
        min(x_all) - margin, max(x_all) + margin,
        min(y_all) - margin, max(y_all) + margin,
    ]
    ax.set_extent(extent_utm, crs=proj)
    xmin, xmax, ymin, ymax = extent_utm

    # ------------------------------------------------------------------
    # Coastline and land fill
    # ------------------------------------------------------------------
    if coast_xyz is not None:
        coast_segs = _load_coast_segments(coast_xyz, gap_m=coast_gap_m)

        # Merge the two longest open segments into the main coastline
        open_idx = [i for i, s in enumerate(coast_segs)
                    if np.hypot(s[-1, 0] - s[0, 0], s[-1, 1] - s[0, 1]) > coast_gap_m]
        closed_idx = [i for i in range(len(coast_segs)) if i not in open_idx]
        open_segs = [coast_segs[i] for i in open_idx]
        closed_segs = [coast_segs[i] for i in closed_idx]

        if len(open_segs) >= 2:
            # Find and merge the pair that share an endpoint
            s0, s1 = open_segs[0], open_segs[1]
            if np.allclose(s0[-1], s1[0], atol=10):
                main_coast = np.vstack([s0, s1[1:]])
            elif np.allclose(s1[-1], s0[0], atol=10):
                main_coast = np.vstack([s1, s0[1:]])
            else:
                # Fallback: concatenate in order and sort by x range
                main_coast = np.vstack(open_segs)
        elif len(open_segs) == 1:
            main_coast = open_segs[0]
        else:
            main_coast = np.vstack(coast_segs)

        # Land polygon from main coastline
        land_poly = _build_land_polygon(
            main_coast, xmin, xmax, ymin, ymax,
            land_is_north=land_is_north,
        )
        if land_poly is not None:
            ax.fill(land_poly[:, 0], land_poly[:, 1],
                    color=color_land, transform=proj, zorder=1)

        # Fill closed segments (islands / small features)
        for seg in closed_segs:
            if len(seg) >= 3:
                ax.fill(seg[:, 0], seg[:, 1],
                        color=color_land, transform=proj, zorder=1)

        # Draw coastline on top of fill
        for seg in coast_segs:
            ax.plot(seg[:, 0], seg[:, 1],
                    color="#333333", linewidth=0.8, transform=proj, zorder=3)

    # ------------------------------------------------------------------
    # Control / extraction points
    # ------------------------------------------------------------------
    if points is not None:
        ax.scatter(
            points["x"].values, points["y"].values,
            color="#D32F2F", s=60, zorder=7,
            edgecolors="white", linewidths=0.7, transform=proj,
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor="#D32F2F",
                label=f"Points (n={len(points)})",
            )
        )
        if point_label_col and point_label_col in points.columns:
            for _, row in points.iterrows():
                ax.text(
                    row["x"] + point_label_offset,
                    row["y"] + point_label_offset,
                    str(row[point_label_col]),
                    transform=proj,
                    fontsize=7.5,
                    color="#B71C1C",
                    fontweight="bold",
                    zorder=7,
                )

    # ------------------------------------------------------------------
    # Lat/lon grid lines and manual labels
    # ------------------------------------------------------------------
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--",
        zorder=5,
    )
    from pyproj import Transformer as _Transformer
    _tr = _Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + utm_zone}",
                                 always_xy=True)
    for lon in np.arange(-10, 10, 1):
        xu, _ = _tr.transform(lon, (ymin + ymax) / 2 * 9e-6 + 36)
        # coarse lat estimate; accurate enough for tick placement
        xu, _ = _tr.transform(lon, 36.0)
        if xmin <= xu <= xmax:
            lbl = f"{abs(lon):.0f}\u00b0{'W' if lon < 0 else 'E'}"
            ax.text(xu, ymin, lbl, transform=proj, fontsize=7.5,
                    ha="center", va="top", color="#444444", zorder=6)
    for lat in np.arange(30, 45, 0.5):
        _, yu = _tr.transform(-3.0, lat)
        if ymin <= yu <= ymax:
            ax.text(xmin, yu, f"{lat:.1f}\u00b0N", transform=proj, fontsize=7.5,
                    ha="right", va="center", color="#444444", zorder=6)

    # ------------------------------------------------------------------
    # Scale bar
    # ------------------------------------------------------------------
    sb_len_m = round((xmax - xmin) / 6, -3)  # ~1/6 of map width, rounded to km
    sb_len_km = int(sb_len_m / 1000)
    bx = xmin + (xmax - xmin) * 0.82
    by = ymin + (ymax - ymin) * 0.05
    tick_h = (ymax - ymin) * 0.008
    ax.plot([bx, bx + sb_len_m], [by, by],
            color="black", lw=2, transform=proj, zorder=10,
            solid_capstyle="butt")
    ax.plot([bx, bx], [by - tick_h, by + tick_h],
            color="black", lw=1.5, transform=proj, zorder=10)
    ax.plot([bx + sb_len_m, bx + sb_len_m], [by - tick_h, by + tick_h],
            color="black", lw=1.5, transform=proj, zorder=10)
    ax.text(bx + sb_len_m / 2, by + tick_h * 1.5, f"{sb_len_km} km",
            ha="center", va="bottom", fontsize=8, transform=proj, zorder=10)

    # ------------------------------------------------------------------
    # Legend and title
    # ------------------------------------------------------------------
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.90,
        edgecolor="#AAAAAA",
        fancybox=False,
    )
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    show(fname)
    return ax


# ---------------------------------------------------------------------------
# Default colormaps and labels per DELFT3D-WAVE variable
# ---------------------------------------------------------------------------

# Blue → cyan → yellow → orange → red  (QuickPlot / DELFT3D style)
from matplotlib.colors import LinearSegmentedColormap as _LSC
_CMAP_WAVE = _LSC.from_list(
    "delft_wave",
    ["#0000CC", "#0055FF", "#00CCFF", "#FFFF00", "#FF8800", "#CC0000"],
    N=256,
)

_DELFT_FIELD_META = {
    # (colormap_key_or_object,  colorbar_label,  vmin_default, vmax_default)
    # None → auto-scale from data
    "hsign":    (_CMAP_WAVE,  "Hs (m)",              0,    None),
    "dir":      ("phase",     "Dir (\u00b0)",         0,    360),
    "pdir":     ("phase",     "Peak dir (\u00b0)",    0,    360),
    "period":   (_CMAP_WAVE,  "Tm (s)",               0,    15),
    "rtp":      (_CMAP_WAVE,  "Rtp (s)",              0,    15),
    "depth":    ("deep",      "Depth (m)",            0,    None),
    "veloc-x":  ("balance",   "Vel X (m/s)",         -1,    1),
    "veloc-y":  ("balance",   "Vel Y (m/s)",         -1,    1),
    "transp-x": ("balance",   "Transport X",         None,  None),
    "transp-y": ("balance",   "Transport Y",         None,  None),
    "dspr":     ("haline",    "Dir spread (\u00b0)",  0,    90),
    "dissip":   (_CMAP_WAVE,  "Dissipation",         None,  None),
    "leak":     (_CMAP_WAVE,  "Leak",                None,  None),
    "qb":       (_CMAP_WAVE,  "Qb",                   0,    1),
    "ubot":     (_CMAP_WAVE,  "Ubot (m/s)",           0,    None),
    "steepw":   (_CMAP_WAVE,  "Steepness",            0,    None),
    "wlength":  (_CMAP_WAVE,  "Wavelength (m)",       0,    None),
    "tps":      (_CMAP_WAVE,  "Tp (s)",               0,    15),
    "tm02":     (_CMAP_WAVE,  "Tm02 (s)",             0,    15),
    "tmm10":    (_CMAP_WAVE,  "Tm\u22121,0 (s)",      0,    15),
    "drtm01":   (_CMAP_WAVE,  "Drtm01 (s)",          None,  None),
    "setup":    ("balance",   "Setup (m)",           -0.5,   0.5),
    "fx":       ("balance",   "Fx",                  None,  None),
    "fy":       ("balance",   "Fy",                  None,  None),
    "windu":    ("balance",   "Wind U (m/s)",        -20,    20),
    "windv":    ("balance",   "Wind V (m/s)",        -20,    20),
}


def plot_delft_wave_field(
    dat_path,
    grd_path,
    variable,
    coast_xyz=None,
    utm_zone=30,
    cmap=None,
    vmin=None,
    vmax=None,
    color_land="#EDE8DF",
    color_ocean="#C8DCF0",
    coast_gap_m=5000,
    land_is_north=True,
    margin=2000,
    quiver_variable=None,
    quiver_stride=2,
    quiver_scale=40,
    quiver_color="white",
    quiver_only=False,
    title=None,
    cbar_label=None,
    figsize=(10, 8),
    fname=None,
    depth_min=1.0,
):
    """Plot a DELFT3D-WAVE variable field on its curvilinear grid.

    Reads grid node coordinates from the ASCII ``.grd`` file and variable
    data from the NEFIS binary ``.dat`` file, then renders the field with
    ``pcolormesh`` on a cartopy UTM projection. An optional high-resolution
    coastline can be overlaid for geographic context.

    Args:
        dat_path (str | Path): Path to the NEFIS binary ``.dat`` output
            file for one model case.
        grd_path (str | Path): Path to the ASCII ``.grd`` grid file for
            the same case (used for node coordinates).
        variable (str): Wave variable to plot. Available names::

                hsign, dir, pdir, period, rtp, depth,
                veloc-x, veloc-y, transp-x, transp-y,
                dspr, dissip, leak, qb,
                ubot, steepw, wlength, tps, tm02, tmm10,
                drtm01, setup, fx, fy, windu, windv

        coast_xyz (str | Path | None): Path to a CSV with ``x`` and ``y``
            columns (same UTM CRS) for a high-resolution coastline.
            Defaults to None.
        utm_zone (int): UTM zone number for the cartopy projection.
            Defaults to 30.
        cmap (str | Colormap | None): Colormap override. If None, a
            sensible default is chosen per variable. Defaults to None.
        vmin (float | None): Colour scale minimum. Defaults to None
            (auto from data).
        vmax (float | None): Colour scale maximum. Defaults to None
            (auto from data).
        color_land (str): Fill colour for land. Defaults to ``"#EDE8DF"``.
        color_ocean (str): Background colour for ocean. Defaults to
            ``"#C8DCF0"``.
        coast_gap_m (float): Gap threshold in metres for splitting the
            coastline into segments. Defaults to 5000.
        land_is_north (bool): Close land polygon through the northern edge
            of the extent. Defaults to True.
        margin (float): Extra margin in metres around the grid extent.
            Defaults to 2000.
        quiver_variable (str | None): Variable to use for direction arrows.
            Typically ``"dir"`` (mean wave direction in degrees, measured
            clockwise from North, direction of provenance: 0° = from North
            → arrow points South). When set, arrows are drawn with length
            and colour proportional to the main *variable* magnitude using
            the same colormap. Defaults to None (no arrows).
        quiver_stride (int): Subsample every *n* grid cells in both
            directions before drawing arrows. Defaults to 12.
        quiver_scale (float): Quiver scale passed to ``ax.quiver``; smaller
            values produce longer arrows. With magnitude-scaled arrows a
            value around 30–50 works well. Defaults to 40.
        quiver_color (str): Ignored when *quiver_variable* is set (arrows
            are coloured by magnitude). Kept for API compatibility.
        title (str | None): Figure title. If None, auto-generated from
            *variable* and the case directory name. Defaults to None.
        cbar_label (str | None): Colorbar label override. Defaults to None
            (uses variable metadata).
        figsize (tuple): Figure size in inches. Defaults to (10, 8).
        fname (str | None): Output file path. If None, displays
            interactively. Defaults to None.
        depth_min (float): Minimum depth (m) for cells to be included in the
            plot. Cells shallower than this threshold are masked to NaN and
            shown as ocean background. Useful for hiding the very shallow
            coastal fringe (depth < 1 m) where DELFT3D outputs Hs ≈ 0 due
            to complete wave breaking, which otherwise creates a dark-blue
            band along the coast. Set to 0 or None to disable. Not applied
            when *variable* is ``"depth"``. Defaults to 1.0.

    Returns:
        matplotlib.axes.GeoAxes: The cartopy axes with the field plot.

    Examples:
        >>> from pathlib import Path
        >>> from environmentaltools.graphics import plot_delft_wave_field
        >>>
        >>> plot_delft_wave_field(
        ...     dat_path=Path("caso_001/wavm-guad-Alboran_int.dat"),
        ...     grd_path=Path("caso_001/Alboran_int.grd"),
        ...     variable="hsign",
        ...     coast_xyz=Path("linea_costa_oficial_utm30n.xyz"),
        ...     vmax=4.0,
        ...     title="Hs \u2014 caso 001",
        ...     fname="hs_caso001.png",
        ... )
    """
    # if not HAS_CARTOPY:
    #     raise ImportError(
    #         "cartopy is required for plot_delft_wave_field. "
    #         "Install it with: conda install -c conda-forge cartopy"
    #     )

    from pathlib import Path as _Path
    from environmentaltools.processes.load import (
        _delft_grid_params,
        _delft_read_var,
    )

    dat_path = _Path(dat_path)
    grd_path = _Path(grd_path)

    # ------------------------------------------------------------------
    # Grid node coordinates from .grd
    # ------------------------------------------------------------------
    X, Y, _mmax, _nmax = _read_delft_grd(grd_path)

    # ------------------------------------------------------------------
    # Variable field from .dat
    # ------------------------------------------------------------------
    from environmentaltools.processes.load import _delft_read_coords as _read_coords

    gp  = _delft_grid_params(dat_path.parent, dat_path.name, grd_path.name)
    val = _delft_read_var(dat_path, variable, gp, fill_threshold=-9000.0)

    # Mask inactive cells using the XP/YP coordinate mask from the .dat file.
    # Active sea cells have valid UTM coordinates; land/outside cells have XP=0.
    _, _, _active = _read_coords(dat_path, gp, fill_threshold=-9000.0)
    val = np.where(~_active, np.nan, val)

    # Also mask cells where the .grd defines no grid node (outside model extent).
    val = np.where(np.isnan(X) | np.isnan(Y), np.nan, val)

    # Mask very shallow coastal cells (depth < depth_min) as NaN so they
    # appear as ocean background rather than dark-blue Hs≈0.  These cells
    # are correctly computed by DELFT3D but show near-zero wave energy due
    # to complete breaking in the intertidal fringe — hiding them avoids a
    # misleading dark band along the coast.  Skip when plotting depth itself.
    if depth_min and variable != "depth":
        _dep = _delft_read_var(dat_path, "depth", gp, fill_threshold=-9000.0)
        val = np.where(_dep < depth_min, np.nan, val)

    if variable in ("dir", "pdir"):
        val = (270.0 - val) % 360.0

    n_valid = int(np.sum(~np.isnan(val)))
    print(f"[plot_delft] '{variable}'  shape={val.shape}  "
          f"valid={n_valid}/{val.size}  "
          f"min={np.nanmin(val) if n_valid else float('nan'):.4f}  "
          f"max={np.nanmax(val) if n_valid else float('nan'):.4f}")

    # ------------------------------------------------------------------
    # Colormap and label
    # ------------------------------------------------------------------
    import copy as _copy
    from matplotlib.colors import Colormap as _Colormap, Normalize as _Norm

    meta_cmap, meta_label, meta_vmin, meta_vmax = _DELFT_FIELD_META.get(
        variable, (_CMAP_WAVE, variable, None, None)
    )
    if cmap is None:
        if isinstance(meta_cmap, _Colormap):
            cmap = meta_cmap          # already a Colormap object (e.g. _CMAP_WAVE)
        else:
            try:
                cmap = getattr(cmocean.cm, meta_cmap)
            except AttributeError:
                cmap = meta_cmap
    # Resolve string colormaps (from caller or fallback) to Colormap objects.
    if isinstance(cmap, str):
        try:
            cmap = getattr(cmocean.cm, cmap)
        except AttributeError:
            import matplotlib as _mpl
            cmap = _mpl.colormaps[cmap]
    # Make masked/NaN cells appear as ocean background instead of the
    # colormap's default "bad" colour (often white).
    cmap = _copy.copy(cmap)
    cmap.set_bad(color=color_ocean)
    if cbar_label is None:
        cbar_label = meta_label
    # Apply per-variable default scale when caller did not specify vmin/vmax
    if vmin is None:
        vmin = meta_vmin
    if vmax is None:
        vmax = meta_vmax


    # plt.pcolormesh(X,Y,np.ma.masked_invalid(val))
    # plt.show()

    # ------------------------------------------------------------------
    # Figure and projection
    # ------------------------------------------------------------------
    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    ax.set_facecolor(color_ocean)

    xmin = float(np.nanmin(X)) - margin
    xmax = float(np.nanmax(X)) + margin
    ymin = float(np.nanmin(Y)) - margin
    ymax = float(np.nanmax(Y)) + margin
    # set_extent with a projected CRS can fail silently in some Cartopy
    # versions.  Use set_xlim/set_ylim directly in the axes (projected)
    # coordinate space instead — these always work.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ------------------------------------------------------------------
    # Variable field (pcolormesh on curvilinear grid)  [zorder=2]
    # Build a shared Normalize so pcolormesh and quiver use identical scaling.
    # ------------------------------------------------------------------
    _vmin = vmin if vmin is not None else float(np.nanmin(val))
    _vmax = vmax if vmax is not None else float(np.nanmax(val))
    norm  = _Norm(vmin=_vmin, vmax=_vmax)

    # quiver_only without a quiver_variable is a no-op — show the field anyway.
    if quiver_only and quiver_variable is None:
        quiver_only = False


    # plt.imshow(val)
    # plt.show()


    if not quiver_only:
        pm = ax.pcolormesh(
            X, Y, np.ma.masked_invalid(val),
            cmap=cmap, norm=norm,
            transform=proj, zorder=2, shading="auto",
        )
        cbar = fig.colorbar(pm, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.set_ylabel(cbar_label, fontsize=9)

    # ------------------------------------------------------------------
    # Coastline geometry — computed first so it can be used as a mask
    # for the quiver arrows before any drawing takes place.
    # ------------------------------------------------------------------
    land_poly   = None   # numpy array (N,2) of the land polygon vertices
    coast_segs  = []
    closed_segs = []

    if coast_xyz is not None:
        coast_segs = _load_coast_segments(coast_xyz, gap_m=coast_gap_m)
        open_idx   = [i for i, s in enumerate(coast_segs)
                      if np.hypot(s[-1, 0] - s[0, 0], s[-1, 1] - s[0, 1]) > coast_gap_m]
        closed_idx = [i for i in range(len(coast_segs)) if i not in open_idx]
        open_segs   = [coast_segs[i] for i in open_idx]
        closed_segs = [coast_segs[i] for i in closed_idx]

        if len(open_segs) >= 2:
            s0, s1 = open_segs[0], open_segs[1]
            if np.allclose(s0[-1], s1[0], atol=10):
                main_coast = np.vstack([s0, s1[1:]])
            elif np.allclose(s1[-1], s0[0], atol=10):
                main_coast = np.vstack([s1, s0[1:]])
            else:
                main_coast = np.vstack(open_segs)
        elif len(open_segs) == 1:
            main_coast = open_segs[0]
        else:
            main_coast = np.vstack(coast_segs)

        land_poly = _build_land_polygon(
            main_coast, xmin, xmax, ymin, ymax, land_is_north=land_is_north,
        )

    # ------------------------------------------------------------------
    # Direction quiver (optional)  [zorder=3]
    # Arrows are:
    #   • coloured by the main variable magnitude (same cmap/norm as field)
    #   • sized proportional to magnitude (longer = higher value)
    #   • geometrically clipped to ocean (land polygon mask)
    # ------------------------------------------------------------------
    if quiver_variable is not None:
        from matplotlib.path import Path as _MplPath

        dir_arr = _delft_read_var(dat_path, quiver_variable, gp, fill_threshold=-9000.0)
        # Mask dir_arr with the exact same masks applied to val.
        dir_arr = np.where(~_active, np.nan, dir_arr)
        dir_arr = np.where(np.isnan(X) | np.isnan(Y), np.nan, dir_arr)
        if depth_min and variable != "depth":
            dir_arr = np.where(_dep < depth_min, np.nan, dir_arr)

        # Subsample — same X, Y, val as pcolormesh.
        st   = quiver_stride
        Xs   = X[::st, ::st]
        Ys   = Y[::st, ::st]
        Ds   = dir_arr[::st, ::st]
        Mags = val[::st, ::st]

        # Aplicar la misma transformación que usa pcolormesh para dir/pdir,
        # y luego fórmula cartesiana estándar U=cos, V=sin.
        # (270 - dir) % 360 convierte náutico-FROM a cartesiano-TO (CCW desde Este):
        #   dir=0   (del N, va al S) → 270° → (cos270, sin270) = (0,-1) Sur  ✓
        #   dir=90  (del E, va al O) → 180° → (cos180, sin180) = (-1,0) Oeste ✓
        #   dir=270 (del O, va al E) →   0° → (cos0,   sin0)   = (1, 0) Este  ✓
        if quiver_variable in ("dir", "pdir"):
            Ds = (270.0 - Ds) % 360.0
        rad = np.deg2rad(Ds)
        Us  = (-1) * np.sin(rad)
        Vs  = (-1) * np.cos(rad)

        mask_q = ~(np.isnan(Xs) | np.isnan(Ys) | np.isnan(Ds) | np.isnan(Mags))
        # Land polygon clipping removed — land fill (zorder=5) draws on top
        # and naturally hides any arrows that fall on land.

        # Scale arrow length proportional to magnitude.
        # Floor at 0.25 so arrows remain visible even in near-calm cells;
        # colour each arrow with the same cmap/norm as pcolormesh (QuickPlot style).
        mag_vals = Mags[mask_q]
        if _vmax > 0:
            scale_f = 0.25 + 0.75 * np.clip(mag_vals / _vmax, 0.0, 1.0)
        else:
            scale_f = np.ones_like(mag_vals)
        _quiver_kw = dict(
            transform=proj,
            scale=quiver_scale,
            scale_units="width",
            width=0.001,
            headwidth=3,
            headlength=4,
            linewidth=0,
        )
        if quiver_only:
            # quiver_only: flechas coloreadas por magnitud + colorbar propio.
            ax.quiver(
                Xs[mask_q], Ys[mask_q],
                Us[mask_q] * scale_f, Vs[mask_q] * scale_f,
                color="white", alpha=1.0, zorder=3, **_quiver_kw,
            )
            qv = ax.quiver(
                Xs[mask_q], Ys[mask_q],
                Us[mask_q] * scale_f, Vs[mask_q] * scale_f,
                mag_vals, cmap=cmap, norm=norm,
                alpha=0.9, zorder=4, **_quiver_kw,
            )
            import matplotlib.cm as _mcm
            sm = _mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
            cbar.ax.set_ylabel(cbar_label, fontsize=9)
        else:
            # Con pcolormesh de fondo: flechas en color fijo (quiver_color).
            ax.quiver(
                Xs[mask_q], Ys[mask_q],
                Us[mask_q] * scale_f, Vs[mask_q] * scale_f,
                color=quiver_color, alpha=0.9, zorder=3, **_quiver_kw,
            )

    # ------------------------------------------------------------------
    # Coastline and land fill  [zorder=5/6] — above quiver (zorder 3/4)
    # ------------------------------------------------------------------
    if coast_xyz is not None:
        if land_poly is not None:
            ax.fill(land_poly[:, 0], land_poly[:, 1],
                    color=color_land, transform=proj, zorder=5)
        for seg in closed_segs:
            if len(seg) >= 3:
                ax.fill(seg[:, 0], seg[:, 1],
                        color=color_land, transform=proj, zorder=5)
        for seg in coast_segs:
            ax.plot(seg[:, 0], seg[:, 1],
                    color="#333333", linewidth=0.8, transform=proj, zorder=6)

    # ------------------------------------------------------------------
    # Lat/lon gridlines with manual labels
    # ------------------------------------------------------------------
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.4, color="gray", alpha=0.5, linestyle="--", zorder=6,
    )
    from pyproj import Transformer as _Tr
    _tr = _Tr.from_crs("EPSG:4326", f"EPSG:{32600 + utm_zone}", always_xy=True)
    for lon in np.arange(-10, 10, 1):
        xu, _ = _tr.transform(lon, 36.0)
        if xmin <= xu <= xmax:
            lbl = f"{abs(lon):.0f}\u00b0{'W' if lon < 0 else 'E'}"
            ax.text(xu, ymin, lbl, transform=proj, fontsize=7,
                    ha="center", va="top", color="#444444", zorder=7)
    for lat in np.arange(30, 45, 0.5):
        _, yu = _tr.transform(-3.0, lat)
        if ymin <= yu <= ymax:
            ax.text(xmin, yu, f"{lat:.1f}\u00b0N", transform=proj, fontsize=7,
                    ha="right", va="center", color="#444444", zorder=7)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    if title is None:
        title = f"{variable.upper()} \u2014 {dat_path.parent.name}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    show(fname)
    return ax
