import json
import sys
from datetime import timedelta
from zipfile import ZipFile

import numpy as np
import pandas as pd
from matplotlib import dates
from loguru import logger

import xarray as xr
import rasterio
from lxml import etree
import geopandas as gpd
from scipy.io import loadmat as ldm
import PyPDF2
from tabula import read_pdf
from environmentaltools.common import utils

def keys_as_int(obj: dict):
    """Convert the keys at reading json file into a dictionary of integers.

    Args:
        obj (dict): Input dictionary.

    Returns:
        dict: Dictionary with integer keys where possible, original keys otherwise.
    """
    try:
        out = {int(k): v for k, v in obj.items()}
    except (ValueError, TypeError):
        out = {k: v for k, v in obj.items()}
    return out


def keys_as_nparray(obj: dict):
    """Convert the values at reading json file into numpy arrays recursively.

    Recursively processes nested dictionaries up to 3 levels deep, converting
    values to numpy arrays where possible.

    Args:
        obj (dict): Input dictionary with nested structure.

    Returns:
        dict: Dictionary with values converted to numpy arrays where applicable.
    """
    if isinstance(obj, dict):
        out = {}
        for item0, level0 in obj.items():
            if isinstance(level0, dict):
                out[item0] = {}
                for item1, level1 in level0.items():
                    if isinstance(level1, dict):
                        out[item0][item1] = {}
                        for item2, level2 in level1.items():
                            try:
                                out[item0][item1][item2] = np.asarray(level2)
                            except (ValueError, TypeError):
                                out[item0][item1][item2] = level2
                    else:
                        try:
                            out[item0][item1] = np.asarray(level1)
                        except (ValueError, TypeError):
                            out[item0][item1] = level1
            else:
                try:
                    out[item0] = np.asarray(level0)
                except (ValueError, TypeError):
                    out[item0] = level0

    return out


def read_json(file_name: str, conversion_type: str = None):
    """Read data from JSON files with optional type conversion.

    Loads JSON files and converts keys to integers or numpy arrays based on
    the specified conversion type.

    Args:
        file_name (str): Path to the JSON file.
        conversion_type (str, optional): Type of data conversion:
            - "td" (temporal dependency): Converts values to numpy arrays
            - None or other: Converts keys to integers
            Defaults to None.

    Returns:
        dict: Loaded and converted dictionary data.
    """
    if conversion_type == "td":
        params = json.load(open(file_name, "r"), object_hook=keys_as_nparray)
    else:
        params = json.load(open(file_name, "r"), object_hook=keys_as_int)
    return params


def read_pde(file_name: str, new_format: bool = False):
    """Read data from Spanish Puertos del Estado (PdE) wave buoy files.

    Parses wave data files from the Spanish port authority, handling both
    new and legacy file formats.

    Args:
        file_name (str): Path to the PdE data file.
        new_format (bool): If True, uses new PdE file format. If False, uses legacy format
            with auto-detection of data start row. Defaults to False.

    Returns:
        pd.DataFrame: Wave parameters with datetime index. Columns include significant
            wave height (Hs), mean period (Tm), peak period (Tp), mean direction (DirM),
            and swell components. Invalid values (-100, -99.9, -9999) are replaced with NaN.
    """
    if new_format:
        data = pd.read_table(
            file_name,
            delimiter=r"\s+",
            parse_dates={"date": [0, 1, 2, 3]},
            index_col="date",
            skiprows=2,
            header=None,
            engine="python",
        )
        data.columns = [
            "Hs",
            "Tm",
            "Tp",
            "DirM",
            "Hswind",
            "DirMwind",
            "Hsswell1",
            "Tmswell1",
            "DirMswell1",
            "Hsswell2",
            "Tmswell2",
            "DirMswell2",
        ]
    else:
        with open(file_name) as file_:
            content = file_.readlines()

        for ind_, line_ in enumerate(content):
            if "LISTADO DE DATOS" in line_:
                skiprows = ind_ + 2

        data = pd.read_table(
            file_name,
            delimiter=r"\s+",
            parse_dates={"date": [0, 1, 2, 3]},
            index_col="date",
            skiprows=skiprows,
            engine="python",
        )

    # Replace invalid/missing data values with NaN
    invalid_values = [-100, -99.9, -99.99, -9999, -9999.9]
    data.replace(invalid_values, np.nan, inplace=True)
    return data


def csv(
    file_name: str,
    ts: bool = False,
    date_format=None,
    sep: str = ",",
    encoding: str = "utf-8",
    index_col: list = [0],
    non_natural_date: bool = False,
    no_data_values: int = -999,
):
    """Read CSV file with flexible datetime and encoding options.

    Flexible CSV reader with support for time series data, custom separators,
    various encodings, and handling of non-natural date formats (e.g., 30-day months).

    Args:
        file_name (str): Path to CSV file (supports .csv, .txt, .dat, .zip).
        ts (bool): If True, treats first column as datetime index. Defaults to False.
        date_format (str, optional): Date format string for parsing. Defaults to None.
        sep (str): Column separator character. Defaults to ",".
        encoding (str): Character encoding. Defaults to "utf-8".
        index_col (list): Columns to use as index. Defaults to [0].
        non_natural_date (bool): If True, handles model dates with 30-day months.
            Defaults to False.
        no_data_values (int): Value to treat as NaN. Defaults to -999.

    Returns:
        pd.DataFrame: Loaded data with appropriate index type.
    """
    # if not any(item in str(file_name) for item in ["dat", "txt", "csv", "zip"]):
    #     raise ValueError("Not extension filename = str(file_name) + ".csv"
    # else:
    #     filename = str(file_name)

    if non_natural_date:
        ts = False

    if not ts:
        if "zip" in file_name:
            data = pd.read_csv(
                file_name,
                sep=sep,
                index_col=index_col,
                compression="zip",
                engine="python",
            )
        else:
            try:
                data = pd.read_csv(
                    file_name,
                    sep=sep,
                    index_col=index_col,
                    encoding=encoding,
                )
            except (pd.errors.ParserError, UnicodeDecodeError):
                data = pd.read_csv(
                    file_name, sep=sep, engine="python", encoding=encoding
                )

        if non_natural_date:
            start = pd.to_datetime(data.index[0])
            # days = timedelta(np.arange(len(data)))
            index_ = [
                start + timedelta(nodays)
                for nodays in np.arange(len(data), dtype=np.float64)
            ]
            data.index = index_
    else:
        if "zip" in file_name:
            try:
                data = pd.read_csv(
                    file_name,
                    sep=sep,
                    parse_dates=[0],
                    index_col=index_col,
                    compression="zip",
                    date_format=date_format,
                )
            except Exception as e:
                data = pd.read_csv(
                    file_name,
                    sep=sep,
                    parse_dates=[0],
                    index_col=index_col,
                    date_format=date_format,
                )
                logger.info(f"{file_name} is not a zip file, reading as regular CSV")
        else:
            try:
                data = pd.read_csv(
                    file_name,
                    sep=sep,
                    parse_dates=["date"],
                    index_col=["date"],
                    date_format=date_format,
                )
            except (KeyError, pd.errors.ParserError):
                if date_format is None:
                    data = pd.read_csv(
                        file_name,
                        sep=sep,
                        parse_dates=[0],
                        index_col=index_col,
                    )
                else:
                    data = pd.read_csv(
                        file_name,
                        sep=sep,
                        parse_dates=[0],
                        index_col=index_col,
                        date_format=date_format,
                    )
    data = data[data != no_data_values]
    return data


def npy(file_name: str):
    """Read data from NumPy binary file (.npy).

    Loads numpy array or pickled data from .npy file format, with fallback
    to pickle loading for complex objects.

    Args:
        file_name (str): Path to .npy file (extension added automatically if missing).

    Returns:
        np.ndarray or dict: Loaded numpy array or dictionary if pickled object.
    """
    try:
        data = np.load(f"{file_name}.npy")
    except ValueError:
        # Fallback for pickled objects
        data = np.load(f"{file_name}.npy", allow_pickle=True)
        if not isinstance(data, pd.DataFrame):
            data = {i: data.item().get(i) for i in data.item()}

    return data


def xlsx(file_name: str, sheet_name: str = 0, names: str = None):
    """Read Excel file (.xls or .xlsx).

    Reads Excel workbook with support for specific sheets and column naming.

    Args:
        file_name (str): Path to Excel file.
        sheet_name (str or int): Sheet name or index to read. Defaults to 0 (first sheet).
        names (list, optional): Custom column names. Defaults to None (use file headers).

    Returns:
        pd.DataFrame: Data from specified Excel sheet with first column as index.
    """
    xlsx = pd.ExcelFile(file_name)
    data = pd.read_excel(xlsx, sheet_name=sheet_name, index_col=0, names=names)
    return data


def netcdf(
    file_name: str,
    variables: str = None,
    latlon: list = None,
    depth: float = None,
    time_series: bool = True,
    glob: bool = False,
):
    """Read NetCDF4 files with spatial/temporal subsetting options.

    Reads NetCDF files using xarray with support for multi-file datasets,
    spatial point extraction, and time series conversion.

    Args:
        file_name (str): Path to NetCDF file or directory pattern for glob.
        variables (str or list, optional): Variable name(s) to extract. Defaults to None (all).
        latlon (list, optional): [latitude, longitude] for point extraction. Defaults to None.
        depth (float, optional): Depth level for extraction. Defaults to None.
        time_series (bool): If True, converts to time series DataFrame. Defaults to True.
        glob (bool): If True, opens multiple files using pattern matching. Defaults to False.

    Returns:
        pd.DataFrame or xarray.Dataset: Extracted data. If latlon specified, returns
            tuple of (DataFrame, (nearest_lat, nearest_lon)).

    Raises:
        ValueError: If glob files are inconsistent.
    """
    if not glob:
        data = xr.open_dataset(file_name)
    else:
        try:
            data = xr.open_mfdataset(file_name)
        except (ValueError, OSError) as e:
            raise ValueError(
                "NetCDF files are not consistent. "
                "Some variables are not adequately saved. "
                "Glob version cannot be used for this dataset."
            ) from e

    if isinstance(latlon, list):
        if not depth:
            data = utils.xrnearest(data, latlon[0], latlon[1])
            data = data.to_dataframe()

            try:
                data = data.loc[0]
                data.index = pd.to_datetime(data.index)
            except (KeyError, TypeError):
                pass

            nearestLatLon = data.latitude.values[0], data.longitude.values[0]
        else:
            data = data.sel(
                depth=0.494025,
                longitude=latlon[0],
                latitude=latlon[1],
                method="nearest",
            ).to_dataframe()
        # print("Nearest lat-lon point: ", nearestLatLon)
        if variables is not None:
            if len(variables) == 1:
                data = data[[variables]]
            else:
                data = data[variables]
            data = (data, nearestLatLon)
        # data.index = data.to_datetimeindex(unsafe=False)
    else:
        if time_series:
            if not data.indexes["time"].dtype.name == "datetime64[ns]":
                times, goodPoints = [], []
                for index, time in enumerate(data.indexes["time"]):
                    try:
                        times.append(time._to_real_datetime())
                        goodPoints.append(index)
                    except (AttributeError, ValueError):
                        continue
                if variables is not None:
                    data = pd.DataFrame(
                        data.to_dataframe()[variables].values[goodPoints],
                        index=times,
                        columns=[variables],
                    )
                else:
                    pd.DataFrame(data.to_dataframe().values[goodPoints], index=times)
            else:
                if isinstance(data, xr.core.dataset.Dataset):
                    values_ = np.squeeze(data[variables].values)
                    data = pd.DataFrame(
                        values_,
                        index=data.indexes["time"],
                        columns=[variables],
                    )
                else:
                    data = pd.DataFrame(
                        data[variables].data,
                        index=data.indexes["time"],
                        columns=[variables],
                    )

    return data


def ascii_tiff(file_name: str, output_format: str = "row"):
    """Read ASCII or GeoTIFF raster files and extract coordinate data.

    Reads georeferenced raster files using rasterio and extracts coordinates
    and values in either tabular (row) or grid format.

    Args:
        file_name (str): Path to the ASCII or TIFF raster file.
        output_format (str): Output format type. Options are:
            - "row": Returns flattened DataFrame with x, y, z columns (default)
            - "grid": Returns dictionary with 2D arrays for x, y, z

    Returns:
        tuple: A tuple containing:
            - data (pd.DataFrame or dict): Coordinate and value data in specified format.
            - profile (dict): Rasterio profile containing metadata (CRS, transform, etc.).
    """

    data = rasterio.open(file_name)
    z = data.read()[0, :, :]
    profile = data.profile

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))

    T0 = data.transform
    x, y = T0 * (cols, rows)

    if output_format == "row":
        dout = pd.DataFrame(
            np.vstack(
                [
                    np.asarray(x).flatten(),
                    np.asarray(y).flatten(),
                    np.asarray(z).flatten(),
                ]
            ).T,
            columns=["x", "y", "z"],
        ).drop_duplicates(subset=["x", "y"])
    else:
        dout = {"x": x, "y": y, "z": z}

    return dout, profile


def kmz(file_name: str, joint: bool = False):
    """Read KMZ or KML files and extract elevation contour data.

    Parses KMZ (zipped) or KML files to extract coordinate and elevation data
    from placemarks, with support for multiple elevation detection methods.

    Args:
        file_name (str): Path to KMZ or KML file.
        joint (bool): If True, combines all contours into single DataFrame.
            If False, returns separate lists for each contour. Defaults to False.

    Returns:
        pd.DataFrame or tuple: If joint=True, returns DataFrame with columns [x, y, z].
            If joint=False, returns tuple of lists (x, y, z) where each element
            corresponds to a separate contour line.

    Raises:
        SystemExit: If KML parsing fails or file structure is invalid.
    """


    if file_name.endswith("kmz"):
        kmz = ZipFile(file_name, "r")
        kml_file = kmz.open("doc.kml", "r")
    else:
        kml_file = open(file_name, "r")

    try:
        # Process the KML file
        tree = etree.parse(kml_file)
        places = tree.xpath(
            ".//kml:Placemark", namespaces={"kml": "http://www.opengis.net/kml/2.2"}
        )

        # Try to find where the elevation data is stored
        place = places[0]

        try:
            level_detection_method = 1
            float(
                place.xpath(
                    ".//kml:SimpleData[@name='ELEVATION']",
                    namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                )[0].text
            )
        except:
            try:
                level_detection_method = 2
                cdata = place.xpath(
                    ".//kml:description",
                    namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                )[0].text
                cdata_root = etree.HTML(cdata)
                float(cdata_root.xpath("/html/body/table/tr/td[2]")[0].text)
            except:
                try:
                    level_detection_method = 3
                    float(
                        place.xpath(
                            ".//kml:SimpleData[@name='Elevation']",
                            namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                        )[0].text
                    )
                except:
                    level_detection_method = -1
                    pass

        x, y, z = [], [], []
        # Process each placemark in the KMZ file
        for id_, place in enumerate(places):
            if level_detection_method == 1:
                c = float(
                    place.xpath(
                        ".//kml:SimpleData[@name='ELEVATION']",
                        namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                    )[0].text
                )
            elif level_detection_method == 2:
                cdata = place.xpath(
                    ".//kml:description",
                    namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                )[0].text
                cdata_root = etree.HTML(cdata)
                c = float(cdata_root.xpath("/html/body/table/tr/td[2]")[0].text)
            elif level_detection_method == 3:
                c = float(
                    place.xpath(
                        ".//kml:SimpleData[@name='Elevation']",
                        namespaces={"kml": "http://www.opengis.net/kml/2.2"},
                    )[0].text
                )
            else:
                c = "0"

            latlong = place.xpath(
                ".//kml:LineString/kml:coordinates",
                namespaces={"kml": "http://www.opengis.net/kml/2.2"},
            )
            coordinates = latlong[0].text.strip().split(" ")
            if joint:
                z.extend(np.ones(len(coordinates)) * float(c))
            else:
                z.append([])
                z[id_] = np.ones(len(coordinates)) * float(c)

            # Create an entry for each coordinate in the polyline
            if joint:
                for coordinate in coordinates:
                    c = coordinate.split(",")
                    x.append(float(c[0]))
                    y.append(float(c[1]))
            else:
                # Create an entry for each coordinate in the polyline
                x.append([])
                y.append([])
                for coordinate in coordinates:
                    c = coordinate.split(",")
                    x[id_].append(float(c[0]))
                    y[id_].append(float(c[1]))

    except (etree.XMLSyntaxError, KeyError, IndexError) as e:
        logger.error(f"Error parsing KML/KMZ file: {e}")
        sys.exit(-1)

    if joint:
        data = pd.DataFrame(
            np.vstack([np.asarray(x), np.asarray(y), np.asarray(z)]).T,
            columns=["x", "y", "z"],
        ).drop_duplicates(subset=["x", "y"])
    else:
        data = x, y, z

    return data


def shp(file_name: str, joint: bool = False, variable: str = None):
    """Read shapefile and extract geometry coordinates.

    Reads shapefiles using geopandas and extracts coordinates from various geometry
    types (Point, Polygon, LineString, MultiPoint, MultiLineString, MultiPolygon).

    Args:
        file_name (str): Path to shapefile.
        joint (bool): If True, concatenates all geometries into single DataFrame.
            Defaults to False (returns list of DataFrames).
        variable (str, optional): Additional attribute column to extract alongside
            coordinates. Defaults to None.

    Returns:
        pd.DataFrame or list: If joint=True or single geometry, returns DataFrame
            with columns [x, y] or [x, y, variable]. Otherwise returns list of
            DataFrames, one per geometry feature.

    Raises:
        ValueError: If geometry type cannot be processed with available methods.
    """

    shape_file = gpd.read_file(str(file_name))

    no_elements = len(shape_file)

    if variable is not None:
        extra_var = []

    xy, data = [], []

    k, element = 0, 0
    while k < no_elements:
        if shape_file.geometry[k] is not None:
            type_ = shape_file.geometry[k].geom_type
            if type_ == "Point":
                xy.append(np.asarray(shape_file["geometry"][k].coords.xy).T)
            elif type_ == "Polygon":
                xy.append(np.asarray(shape_file["geometry"][k].exterior.coords.xy).T)
            elif type_ == "MultiPoint":
                xy.append(np.asarray(shape_file["geometry"][k].centroid.coords.xy).T)
            elif type_ == "LineString":
                xy.append(np.asarray(shape_file["geometry"][k].coords.xy).T)
            elif type_ == "MultiLineString":
                for linestring_ in shape_file["geometry"][k]:
                    xy.append(np.asarray(linestring_.coords.xy).T)

            elif type_ == "MultiPolygon":
                try:
                    for k_, polygon_ in enumerate(shape_file["geometry"][k]):
                        if polygon_ is not None:
                            if polygon_ == "Polygon":
                                xy.append(np.asarray(polygon_.exterior.coords.xy).T)
                            elif polygon_.geom_type == "linearRing":
                                xy.append(
                                    np.asarray(
                                        shape_file.apply(
                                            lambda x: [y for y in polygon_.coords],
                                            axis=1,
                                        )[k_]
                                    )
                                )
                except (AttributeError, TypeError, KeyError):
                    pass
            else:
                raise ValueError(
                    "Shapefile can not be readed with methods coords or exterior.coords or exterior.coords.xy"
                )

            if variable is not None:
                extra_var.append(shape_file[variable][k])

            element += 1
        k += 1

    for k, element in enumerate(xy):
        if variable is not None:
            data.append(
                pd.DataFrame(
                    np.vstack(
                        [
                            element[:, 0],
                            element[:, 1],
                            extra_var[k],
                        ]
                    ).T,
                    columns=["x", "y", variable],
                )
            )
        else:
            data.append(
                pd.DataFrame(
                    np.vstack([element[:, 0], element[:, 1]]).T,
                    columns=["x", "y"],
                )
            )

    if joint:
        data = pd.concat([element for element in data], ignore_index=True)

    if isinstance(data, list) & (len(data) == 1):
        data = data[0]

    return data


def mat(file_name: str, variable: str = "x", julian: bool = False):
    """Read MATLAB .mat files and extract time series data.

    Loads MATLAB files using scipy.io.loadmat and converts time values to
    pandas datetime format. Assumes data structure with timestamps in first
    column and values in second column.

    Args:
        file_name (str): Path to .mat file.
        variable (str): Variable name to extract from .mat file structure.
            Defaults to "x".
        julian (bool): If True, keeps julian date format. If False, converts
            to datetime using matplotlib date conversion. Defaults to False.

    Returns:
        pd.DataFrame: Time series with datetime index and 'Q' column containing values.
    """

    data = ldm(file_name)
    if not julian:
        date = data[variable][:, 0] + dates.date2num(
            np.datetime64("0000-12-31")
        )  # Added in matplotlib 3.3
        date = [dates.num2date(i - 366, tz=None) for i in date]
    else:
        date = data[variable][:, 0]

    df = pd.DataFrame({variable: data[variable][:, 1]}, index=date)
    df.index = df.index.tz_localize(None)
    return df


def pdf(file_name: str, encoding: str = "latin-1", table: bool = False, guess: bool = False, area: list = None):
    """Read PDF files and extract text or tabular data.

    Extracts content from PDF files using either text extraction (PyPDF2) or
    table extraction (tabula-py) methods.

    Args:
        file_name (str): Path to PDF file.
        encoding (str): Character encoding for table extraction. Defaults to "latin-1".
        table (bool): If True, extracts tables using tabula. If False, extracts
            plain text from first page. Defaults to False.
        guess (bool): If True, tabula will guess table locations. Defaults to False.
        area (list, optional): Coordinates [top, left, bottom, right] defining
            table area for extraction. Defaults to None (auto-detect).

    Returns:
        str or pd.DataFrame: Extracted text string (if table=False) or DataFrame
            with table data (if table=True).
    """

    if not table:
        with open(file_name, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            n_pages = len(reader.pages)
            # TODO: enable multi-page reading
            logger.info(
                f"The file contains {n_pages} pages. Obtaining just the first page."
            )
            page = reader.pages[0]
            data = page.extract_text()
    else:
        data = read_pdf(
            file_name,
            guess=guess,
            pages=1,
            stream=True,
            encoding=encoding,
            area=area,
        )

    return data
