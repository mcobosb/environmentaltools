import json

import numpy as np
import pandas as pd


from pyproj import CRS
import rasterio
from affine import Affine
import shapefile


def npy2json(params: dict):
    """Convert dictionary with numpy arrays to JSON format and save to file.

    Serializes numpy arrays to lists and performs custom transformations for
    specific parameter structures before saving to JSON file.

    Args:
        params (dict): Dictionary containing parameters to transform. Must include
            'fname' key for output filename. Arrays are converted to lists, 'mode'
            values to integers, and handles nested structures in 'all' and 'fun' keys.

    Returns:
        None
    """
    for key in params.keys():
        if isinstance(params[key], np.ndarray):
            params[key] = list(params[key])

    for loc, mode in enumerate(params["mode"]):
        params["mode"][loc] = int(mode)

    if "all" in params.keys():
        for loc, mode in enumerate(params["all"]):
            params["all"][loc] = [str(mode[0]), float(mode[1]), mode[2].tolist()]

    for loc, fun in enumerate(params["fun"]):
        if not isinstance(fun, str):
            params["fun"][loc] = params["fun"][loc].name

    to_json(params, params["fname"])

    return


def to_json(params: dict, file_name: str, numpy_array_serialization: bool = False):
    """Save dictionary to JSON file with optional numpy array serialization.

    Exports data to JSON format with optional automatic conversion of numpy
    arrays to lists for JSON compatibility.

    Args:
        params (dict): Data dictionary to save.
        file_name (str): Output file path.
        numpy_array_serialization (bool): If True, recursively converts numpy arrays
            to lists in nested dictionaries. Defaults to False.

    Returns:
        None
    """
    with open(f"{str(file_name)}", "w") as f:
        if numpy_array_serialization:
            for key in params.keys():
                if isinstance(params[key], dict):
                    for subkey in params[key].keys():
                        try:
                            params[key][subkey] = params[key][subkey].tolist()
                        except (AttributeError, TypeError):
                            pass
                else:
                    try:
                        params[key] = params[key].tolist()
                    except (AttributeError, TypeError):
                        pass
        json.dump(params, f, ensure_ascii=False, indent=4)

    return


def to_csv(data: pd.DataFrame, file_name: str, compression: str = "infer"):
    """Save DataFrame to CSV file with optional compression.

    Exports data to CSV format with automatic compression detection or
    explicit zip compression.

    Args:
        data (pd.DataFrame): Data to save.
        file_name (str): Output file path.
        compression (str): Compression type ('infer', 'zip', 'gzip', etc.).
            Defaults to 'infer' (auto-detect from extension).

    Returns:
        None
    """
    if ".zip" in file_name:
        data.to_csv(file_name, compression="zip")
    else:
        data.to_csv(file_name, compression=compression)

    return


def to_npy(data: np.ndarray, file_name: str):
    """Save numpy array to binary .npy file.

    Serializes numpy array to binary format for efficient storage and loading.

    Args:
        data (np.ndarray): Array data to save.
        file_name (str): Output file path (without extension).

    Returns:
        None
    """
    np.save(f"{str(file_name)}.npy", data)
    return


def to_xlsx(data: pd.DataFrame, file_name: str):
    """Save DataFrame to formatted Excel file with styled headers and rows.

    Exports data to Excel with alternating row colors and formatted headers
    for improved readability.

    Args:
        data (pd.DataFrame): Data to save.
        file_name (str): Output Excel file path.

    Returns:
        None
    """

    wbook, wsheet = cwriter(str(file_name))

    # Writting the header
    if data.index.name is not None:
        wsheet.write(0, 0, data.index.name, formats(wbook, "header"))
    else:
        wsheet.write(0, 0, "Index", formats(wbook, "header"))

    for col_num, value in enumerate(data.columns.values):
        wsheet.write(0, col_num + 1, value, formats(wbook, "header"))

    # Adding data
    k = 1
    for i in data.index:
        if k % 2 == 0:
            fmt = "even"
        else:
            fmt = "odd"
        wsheet.write_row(k, 0, np.append(i, data.loc[i, :]), formats(wbook, fmt))
        k += 1

    wbook.close()
    return


def cwriter(file_out: str):
    """Create Excel workbook and worksheet for writing.

    Initializes an Excel file with xlsxwriter engine for formatted output.

    Args:
        file_out (str): Output file path.

    Returns:
        tuple: (workbook, worksheet) - Excel writer objects for formatting.
    """
    writer = pd.ExcelWriter(
        file_out,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}},
    )
    df = pd.DataFrame([0])
    df.to_excel(writer, index=False, sheet_name="Sheet1", startrow=1, header=False)
    wsheet = writer.sheets["Sheet1"]
    wbook = writer.book
    return wbook, wsheet


def formats(wbook, style):
    """Apply predefined formatting styles to Excel workbook.

    Provides styling presets for headers and alternating rows.

    Args:
        wbook (xlsxwriter.Workbook): Excel workbook object.
        style (str): Style name ('header', 'even', or 'odd').

    Returns:
        xlsxwriter.Format: Format object with specified styling.
    """
    fmt = {
        "header": {
            "bold": True,
            "text_wrap": True,
            "valign": "center",
            "font_color": "#ffffff",
            "fg_color": "#5983B0",
            "border": 1,
        },
        "even": {
            "bold": False,
            "text_wrap": False,
            "valign": "center",
            "fg_color": "#DEE6EF",
            "border": 1,
        },
        "odd": {
            "bold": False,
            "text_wrap": False,
            "valign": "center",
            "fg_color": "#FFFFFF",
            "border": 1,
        },
    }

    return wbook.add_format(fmt[style])


def to_esriascii(
    data: np.ndarray,
    ncols: int,
    nrows: int,
    cellsize: float,
    file_name: str,
    x0: float = 0,
    y0: float = 0,
    nodata_value: float = -9999
):
    """Save gridded data to ESRI ASCII raster format.

    Exports 2D array data to ESRI ASCII Grid format (.asc) with header
    information including grid dimensions, origin, cell size, and no-data value.

    Args:
        data (np.ndarray): 2D array of grid values to save.
        ncols (int): Number of columns in the grid.
        nrows (int): Number of rows in the grid.
        cellsize (float): Cell size (resolution) in spatial units.
        file_name (str): Output file path.
        x0 (float): X-coordinate of lower-left corner. Defaults to 0.
        y0 (float): Y-coordinate of lower-left corner. Defaults to 0.
        nodata_value (float): Value representing missing/no data. Defaults to -9999.

    Returns:
        None
    """
    fid = open(str(file_name), "w")
    fid.write("ncols    {}\n".format(ncols))
    fid.write("nrows    {}\n".format(nrows))
    fid.write("xllcorner    {}\n".format(x0))
    fid.write("yllcorner    {}\n".format(y0))
    fid.write("cellsize    {}\n".format(cellsize))
    fid.write("NODATA_value    {}\n".format(nodata_value))
    fid.close()

    with open(str(file_name), "ab") as file:
        np.savetxt(file, data, fmt="%8.3f", newline="\n")
    fid.close()
    return


def as_float_bool(obj: dict):
    """Convert string values in dictionary to appropriate types.

    Performs type conversion on dictionary values: converts numeric strings to
    floats/integers and boolean strings ('True', 'False') to bool type.

    Args:
        obj (dict): Dictionary with string values to convert.

    Returns:
        dict: Dictionary with values converted to appropriate types (float, int, or bool).
    """
    for keys in obj.keys():
        try:
            obj[keys] = float(obj[keys])
            # Convert to int if value is a whole number
            if obj[keys] == np.round(obj[keys]):
                obj[keys] = int(obj[keys])
        except (ValueError, TypeError):
            pass

        # Convert string representations of booleans
        if obj[keys] == "True":
            obj[keys] = True
        elif obj[keys] == "False":
            obj[keys] = False

    return obj


def to_geotiff(
    data: np.ndarray,
    file_name: str,
    profile: dict = None,
    transform: Affine = None,
    auxiliary: dict = None
):
    """Save georeferenced raster data to GeoTIFF format.

    Exports 2D array to GeoTIFF with spatial reference information. Profile can
    be provided directly or constructed from auxiliary parameters.

    Args:
        data (np.ndarray): 2D array of raster values.
        file_name (str): Output GeoTIFF file path.
        profile (dict, optional): Rasterio profile dictionary with metadata (driver,
            dtype, nodata, dimensions, CRS, transform). If None, built from auxiliary.
        transform (Affine, optional): Affine transformation matrix. Ignored if profile
            provided. Defaults to None.
        auxiliary (dict, optional): Dictionary with keys: 'corners' (origin [x, y]),
            'dx', 'dy' (cell sizes), 'angle' (rotation), 'driver', 'dtype', 'nodata',
            'nodesx', 'nodesy' (dimensions), 'count' (bands), 'crsno' (EPSG code).
            Required if profile is None.

    Returns:
        None
    """
    if profile is None:
        # Build affine transform from auxiliary parameters
        transform = (
            Affine.translation(auxiliary["corners"][0], auxiliary["corners"][1])
            * Affine.scale(auxiliary["dx"], auxiliary["dy"])
            * Affine.rotation(auxiliary["angle"])
        )

        # Construct profile from auxiliary dictionary
        profile = {
            "driver": auxiliary["driver"],
            "dtype": auxiliary["dtype"],
            "nodata": auxiliary["nodata"],
            "width": auxiliary["nodesy"],
            "height": auxiliary["nodesx"],
            "count": auxiliary["count"],
            "crs": CRS.from_epsg(auxiliary["crsno"]),
            "transform": transform,
            "tiled": False,
            "interleave": "band",
        }

    with rasterio.Env():
        # Update profile with output specifications
        profile.update(dtype=rasterio.float32, count=1, compress="lzw")

        with rasterio.open(str(file_name), "w", **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)
    return


def to_txt(data: pd.DataFrame, file_name: str, fmt: str = "%9.3f"):
    """Save DataFrame to text file with custom formatting.

    Exports data to plain text file using numpy savetxt with specified format.

    Args:
        data (pd.DataFrame): Data to save.
        file_name (str): Output file path.
        fmt (str): Format string for numeric values (e.g., '%9.3f' for 9-character
            width with 3 decimal places). Defaults to "%9.3f".

    Returns:
        None
    """
    np.savetxt(str(file_name), data, delimiter="", fmt=fmt)
    return


def to_shp(
    file_name: str,
    lon: pd.Series,
    lat: pd.Series,
    geometry_type: str = "point",
    values: pd.Series = None,
):
    """Save spatial data to ESRI shapefile format.

    Creates shapefiles with point, multi-point, line, or multi-line geometries
    from coordinate data.

    Args:
        file_name (str): Output shapefile path (without .shp extension).
        lon (pd.Series or list): Longitude or X coordinates.
        lat (pd.Series or list): Latitude or Y coordinates.
        geometry_type (str): Geometry type to create. Options:
            - 'point': Single point
            - 'multi-point': Multiple separate points
            - 'line': Single polyline
            - 'multi-line': Multiple polylines (requires values parameter)
            Defaults to 'point'.
        values (pd.Series, optional): Values to group coordinates for multi-line
            geometries. Each unique value creates a separate line. Defaults to None.

    Returns:
        None

    Raises:
        ImportError: If pyshp package is not installed.
        ValueError: If geometry_type is not recognized.
    """
    iofile = shapefile.Writer(str(file_name))
    iofile.field("id")

    if geometry_type == "point":
        if isinstance(lon, list):
            for i, j in enumerate(lon):
                iofile.point(j, lat[i])
                iofile.record(str(int(i + 1)))
        else:
            iofile.point(lon, lat)
            iofile.record("1")
    elif geometry_type == "multi-point":
        for ind_, lon_key in enumerate(lon):
            iofile.point(lon_key, lat[ind_])
            iofile.record(str(ind_))
    elif geometry_type == "line":
        coords = [[]]
        for ind_, lon_key in enumerate(lon):
            coords[0].append([lon_key, lat[ind_]])
        iofile.line(coords)
        iofile.record("1")
    elif geometry_type == "multi-line":
        unique_values = values.unique()
        coords = [[] for _ in unique_values]
        for k_index, k in enumerate(unique_values):
            mask = values == k
            for ind_, lon_key in enumerate(lon[mask]):
                coords[k_index].append([lon_key, lat[ind_]])
            iofile.line([coords[k_index]])
            iofile.record(str(k))
    else:
        raise ValueError(
            "Geometry type '{}' not implemented. Options are: point, multi-point, line, or multi-line.".format(
                geometry_type
            )
        )

    iofile.close()
    return


def to_netcdf(data: pd.DataFrame, file_path: str):
    """Save DataFrame to NetCDF4 file format.

    Exports time series data to NetCDF format for efficient storage and
    compatibility with climate/oceanographic data standards.

    Args:
        data (pd.DataFrame): Time series or gridded data to save.
        file_path (str): Output file path (without .nc extension).

    Returns:
        None
    """
    data.to_netcdf(str(file_path) + ".nc")
    return
