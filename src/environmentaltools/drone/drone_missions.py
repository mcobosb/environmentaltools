#!/usr/bin/env python3
"""
Batch create KMZ files from waypoints CSV using a template KMZ folder.

Requirements implemented:
 1) Given a DJI root folder, list all subfolders except 'capability' and 'map_preview'.
 2) Split `_{polygon_no}.csv` into chunks of `chunk_size` (default 50) and generate one KMZ per chunk until all points are used.
    KMZ files are named using the folder names found in (1), cycling if needed.
 3) Inside the DJI `map_preview` folder, create a subfolder per KMZ (name without .kmz) and place a JPG with the same name.
 4) The JPG contains the text showing the file position relative to total (e.g. '3/7').

The script supports `--dry-run` to only print the planned operations.
"""

import math
import shutil
import tempfile
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import csv
import os
import zipfile
from pathlib import Path


# Third-party imports
import pandas as pd
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt



# XML namespace definitions for DJI WPML format
WPML_NS = "http://www.uav.com/wpmz/1.0.2"
KML_NS = "http://www.opengis.net/kml/2.2"
NS = {"wpml": WPML_NS, "kml": KML_NS}


def qname(tag, ns=WPML_NS):
    """Build qualified XML tag names with namespace.

    Parameters
    ----------
    tag : str
        The XML tag name without namespace.
    ns : str, optional
        The XML namespace URI. Default is WPML_NS.

    Returns
    -------
    str
        Qualified tag name in format "{namespace}tag".

    Examples
    --------
    >>> qname("index")
    "{http://www.uav.com/wpmz/1.0.2}index"
    >>> qname("Document", KML_NS)
    "{http://www.opengis.net/kml/2.2}Document"
    """
    return f"{{{ns}}}{tag}"


def generate_wpml_from_csv(template_path: Path, csv_path: Path, out_path: Path, limit: int = None, take: str = 'first', start_zero: bool = False, start_index: int | None = None):
    """Generate a WPML file from a template and CSV waypoints data.

    Parameters
    ----------
    template_path : Path
        Path to the WPML template file (waylines.wpml).
    csv_path : Path
        Path to the CSV file containing waypoint data with longitude and latitude columns.
    out_path : Path
        Output path for the generated WPML file.
    limit : int, optional
        Maximum number of waypoints to process. If None, processes all waypoints.
    take : str, optional
        Strategy for selecting waypoints when limit is applied. Either "first" or "last".
        Default is "first".
    start_zero : bool, optional
        If True, start waypoint indexing from 0. If False, start from 1. Default is False.
    start_index : int, optional
        Force a specific starting index for waypoint numbering. Overrides start_zero if provided.

    Returns
    -------
    None
        Writes the generated WPML file to the specified output path.

    Raises
    ------
    RuntimeError
        If no <Folder> element is found in the template.
        If no <Placemark> elements are found in the template folder.
    FileNotFoundError
        If the template file or CSV file doesn't exist.

    Notes
    -----
    The function expects CSV data with columns that can be detected as longitude/latitude:
    - Longitude: 'lon', 'lng', 'longitude', 'x'
    - Latitude: 'lat', 'latitude', 'y'

    The template WPML file should contain at least one <Placemark> element that will be
    used as a template for generating new placemarks for each waypoint.

    Action group indices within placemarks are automatically incremented to ensure
    unique identification across all generated waypoints.
    """
    # Parse the WPML template file
    tree = ET.parse(str(template_path))
    root = tree.getroot()

    # Register XML namespaces for proper output formatting
    ET.register_namespace('', KML_NS)
    ET.register_namespace('wpml', WPML_NS)

    # Locate the Folder element within the Document structure
    folder = root.find('.//{http://www.opengis.net/kml/2.2}Folder')
    if folder is None:
        raise RuntimeError('No <Folder> element found in template')

    # Calculate starting index for new waypoints
    # Find all existing wpml:index values to determine next available index
    existing_indexes = [int(idx.text) for idx in root.findall('.//{http://www.uav.com/wpmz/1.0.2}index') if idx.text and idx.text.strip().isdigit()]
    max_index = max(existing_indexes) if existing_indexes else -1
    
    # Determine starting index based on parameters:
    # 1. If start_index explicitly provided, use it
    # 2. If start_zero is requested, begin at 0
    # 3. Otherwise, continue after maximum existing index
    if start_index is not None:
        next_index = int(start_index)
    else:
        next_index = 0 if start_zero else (max_index + 1)

    # Load waypoint data from CSV file (expects latitude,longitude columns)
    df = pd.read_csv(str(csv_path))
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise RuntimeError('CSV must contain latitude and longitude columns')

    # Apply optional limit for batch processing (take 'first' or 'last' rows)
    original_len = len(df)
    if limit is not None and limit >= 0:
        if take == 'last':
            df = df.tail(limit)
        else:  # take == 'first'
            df = df.head(limit)
    processed_len = len(df)

    # Use the first Placemark as template for generating new waypoints
    placemarks = folder.findall('{http://www.opengis.net/kml/2.2}Placemark')
    if not placemarks:
        raise RuntimeError('No <Placemark> found in template')
    first_pm = placemarks[0]

    # Remove all existing Placemark elements from the Folder to avoid
    # keeping template coordinates in the final file. We'll clone
    # the first placemark for each CSV waypoint instead.
    for pm in placemarks:
        folder.remove(pm)

    # Generate new Placemark for each waypoint row in the CSV data
    for i, row in df.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        idx = next_index + i

        # Create a deep copy of the template placemark
        pm = copy.deepcopy(first_pm)

        # Update Point coordinates with waypoint data
        point = pm.find('{http://www.opengis.net/kml/2.2}Point')
        if point is None:
            # Skip waypoint if template structure is unexpected
            continue
        coords = point.find('{http://www.opengis.net/kml/2.2}coordinates')
        if coords is None:
            coords = ET.SubElement(point, '{http://www.opengis.net/kml/2.2}coordinates')
        # Format coordinates to match template indentation
        coords.text = f"\n            {lon},{lat}\n          "

        # Update waypoint index in wpml:index element
        idx_elem = pm.find(qname('index'))
        if idx_elem is None:
            idx_elem = ET.SubElement(pm, qname('index'))
        idx_elem.text = str(idx)

        # Update action group indices to ensure unique identification
        # Find and update actionGroupStartIndex/EndIndex within actionGroup elements
        for ag in pm.findall(qname('actionGroup')):
            start = ag.find(qname('actionGroupStartIndex'))
            end = ag.find(qname('actionGroupEndIndex'))
            if start is not None:
                start.text = str(idx)
            if end is not None:
                # Keep action groups to single waypoint for safety
                end.text = str(idx)

        # Update any additional actionGroup indices deeper in the XML tree (safety measure)
        for el in pm.findall('.//'):
            # Check element tag (may include namespace prefix)
            if el.tag == qname('actionGroupStartIndex') or el.tag.endswith('actionGroupStartIndex'):
                el.text = str(idx)
            if el.tag == qname('actionGroupEndIndex') or el.tag.endswith('actionGroupEndIndex'):
                el.text = str(idx)

        # Add the configured placemark to the folder
        folder.append(pm)

    # Write the generated WPML file to disk
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    # logger.info(f'Generated: {out_path} (added {processed_len} placemarks, indices from {next_index} to {next_index + processed_len - 1})')


def build_kmz_from_template(
    kmz_folder: Path,
    csv_path: Path,
    limit: int | None,
    take: str,
    start_zero: bool,
    out_dir: Path | None,
    start_index: int | None = None,
    polygon_no: str | None = None,  # polygon identifier for output naming
    chunk_no: str | None = None,   # chunk/section identifier for output naming
):
    """Build a KMZ file from a template folder and waypoints CSV data.

    Parameters
    ----------
    kmz_folder : Path
        Template folder containing the wpmz subfolder with template.kml and WPML template.
    csv_path : Path
        Path to the CSV file containing waypoint data.
    limit : int, optional
        Maximum number of waypoints to process. If None, processes all waypoints.
    take : str
        Strategy for selecting waypoints when limit is applied ("first" or "last").
    start_zero : bool
        If True, start waypoint indexing from 0 in the generated WPML.
    out_dir : Path, optional
        Output directory for the generated KMZ file. If None, uses template folder's parent.
    start_index : int, optional
        Force a specific starting index for waypoint numbering. Overrides start_zero if provided.
    polygon_no : str, optional
        Polygon identifier for output file naming structure.
    chunk_no : str, optional
        Chunk/section identifier for output file naming structure.

    Returns
    -------
    Path
        Path to the generated KMZ file.

    Raises
    ------
    SystemExit
        If template folder, wpmz subfolder, template.kml, or WPML template is not found.
    RuntimeError
        If the generated archive is not found after creation.

    Notes
    -----
    The function creates a temporary directory, copies the template.kml file,
    generates a new waypoints.wpml from the CSV data using the WPML template,
    and packages everything into a KMZ (ZIP) archive.

    The output naming convention is:
    - With polygon_no: "polygon_no/chunk_no.kmz"
    - Without polygon_no: "chunk_no.kmz"
    """
    # Validate input parameters and template structure
    if not kmz_folder.exists() or not kmz_folder.is_dir():
        raise SystemExit(f"KMZ folder not found: {kmz_folder}")

    wpmz_dir = kmz_folder / "wpmz"
    if not wpmz_dir.exists() or not wpmz_dir.is_dir():
        raise SystemExit(f"No wpmz folder inside: {wpmz_dir}")

    # Locate required template files
    template_kml = wpmz_dir / "template.kml"
    if not template_kml.exists():
        raise SystemExit(f"template.kml not found in {wpmz_dir}")

    # Find WPML template file (use first *.wpml found in wpmz directory)
    wpml_candidates = list(wpmz_dir.glob("*.wpml"))
    if not wpml_candidates:
        raise SystemExit(f"No .wpml template found in {wpmz_dir}")
    wpml_template = wpml_candidates[0]

    # Create temporary workspace for KMZ generation
    tmp = Path(tempfile.mkdtemp(prefix="kmz_build_"))
    try:
        tmp_wpmz = tmp / "wpmz"
        tmp_wpmz.mkdir(parents=True, exist_ok=True)

        # Copy template.kml to temporary workspace
        shutil.copy2(template_kml, tmp_wpmz / "template.kml")

        # Generate waypoints.wpml from CSV data using the WPML template
        out_wpml = tmp_wpmz / "waypoints.wpml"
        # Call the integrated WPML generator with all parameters
        generate_wpml_from_csv(
            wpml_template,
            csv_path,
            out_wpml,
            limit=limit,
            take=take,
            start_zero=start_zero,
            start_index=start_index,
        )

        # Determine output filename based on polygon and chunk identifiers
        if polygon_no:
            out_name = f"{polygon_no}/{chunk_no}"
        else:
            out_name = f"{chunk_no}"
        
        # Set output directory and create target path
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            target_zip = out_dir / f"{out_name}.kmz"
        else:
            target_zip = kmz_folder.parent / f"{out_name}.kmz"

        # Create KMZ archive from temporary directory contents
        # Use temporary location to avoid naming collisions during multi-threaded operations
        archive_base_tmp = str((tmp / out_name))
        shutil.make_archive(archive_base_tmp, "zip", root_dir=tmp)
        zip_tmp = Path(archive_base_tmp + ".zip")

        # Ensure target directory structure exists
        target_zip.parent.mkdir(parents=True, exist_ok=True)

        # Remove any existing KMZ file at target location to allow overwrite
        if target_zip.exists():
            target_zip.unlink()

        # Move generated archive to final destination and rename .zip to .kmz
        if zip_tmp.exists():
            shutil.move(str(zip_tmp), str(target_zip))
        else:
            raise RuntimeError(f"Expected archive not found: {zip_tmp}")

        # logger.info(f"KMZ generated: {target_zip}")
        return target_zip
    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp)


def list_dji_dirs(root: Path):
    """List DJI directories excluding 'capability' and 'map_preview' folders.

    Parameters
    ----------
    root : Path
        Root directory to scan for DJI folders.

    Returns
    -------
    list of str
        Sorted list of directory names suitable for KMZ naming.
    """
    return [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and p.name not in ("capability", "map_preview")
    ]


def create_preview(
    jpg_path: Path,
    text: str,
    size=(1920, 1080),
    waypoints_csv: Path | None = None,
    dpi: int = 96,
):
    """Create a preview image with text overlay and optional waypoints map.

    Parameters
    ----------
    jpg_path : Path
        Output path for the generated JPEG preview image.
    text : str
        Text to display on the image (e.g., mission number).
    size : tuple, optional
        Image size as (width, height) in pixels. Default is (1920, 1080).
    waypoints_csv : Path, optional
        Path to CSV file with waypoints data for map overlay. Default is None.
    dpi : int, optional
        Resolution in dots per inch for the output image. Default is 96.

    Returns
    -------
    None
        Saves the preview image to the specified path.

    Raises
    ------
    RuntimeError
        If Pillow is not installed and image creation is not possible.
    """
    if Image is None:
        raise RuntimeError(
            "Pillow is required to create preview images. Install with: pip install Pillow"
        )
    
    # Create a white background image
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)

    # Define candidate font paths for different operating systems
    # Prefer bold fonts for better visibility
    font_path = None
    font_candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",       # Windows Arial Bold
        r"C:\Windows\Fonts\ARIALBD.TTF",       # Windows Arial Bold (uppercase)
        r"C:\Windows\Fonts\arial.ttf",         # Windows Arial
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux bold
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux regular
    ]
    # Test each font candidate to find a working one
    for fp in font_candidates:
        if Path(fp).exists():
            font_path = fp
            break

    # If no predefined font found, scan Windows font directory
    if font_path is None:
        win_fonts = Path(r"C:\Windows\Fonts")
        if win_fonts.exists():
            # Priority order: bold fonts -> arial fonts -> any TTF
            candidates = (
                list(win_fonts.glob("*bold*.ttf"))
                + list(win_fonts.glob("*bd*.ttf"))
                + list(win_fonts.glob("*arial*.ttf"))
                + list(win_fonts.glob("*.ttf"))
            )
            for c in candidates:
                if c.exists():
                    font_path = str(c)
                    break

    # Auto-scale font size to fit within canvas dimensions
    font = None
    if font_path:
        max_w = size[0] * 0.99  # Use 99% of width for margin
        max_h = size[1] * 0.99  # Use 99% of height for margin
        
        # Start with large font size (almost full height) and shrink until it fits
        fs = max(12, int(size[1] * 0.995))
            
        while fs >= 8:
            f = ImageFont.truetype(font_path, fs)
                
            # Calculate text dimensions with current font size
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), text, font=f)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            else:
                # Fallback for older PIL versions
                w, h = draw.textsize(text, font=f)

            # Accept font if it fits within both width and height constraints
            if h <= max_h and w <= max_w:
                font = f
                break
                
            # Reduce font size gradually for better fitting
            fs = int(fs * 0.92)

    # Fallback to default font if no suitable font found
    if font is None:
        font = ImageFont.load_default()

    # Calculate final text dimensions for positioning
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    else:
        w, h = draw.textsize(text, font=font)

    # Center the text on the canvas
    x = (size[0] - w) / 2
    y = (size[1] - h) / 2
    
    # Calculate stroke width for text outline (improves readability)
    stroke_w = max(2, int(h * 0.08))
    # Draw text with white outline and black fill for maximum contrast
    if hasattr(draw.text, '__code__') and 'stroke_width' in draw.text.__code__.co_varnames:
        # Modern PIL with stroke support
        draw.text(
            (x, y),
            text,
            fill="black",
            font=font,
            stroke_width=stroke_w,
            stroke_fill="white",
        )
    else:
        # Fallback for older PIL versions without stroke support
        # Create outline effect by drawing white text at multiple offsets
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for ox, oy in offsets:
            draw.text((x + ox, y + oy), text, fill="white", font=font)
        draw.text((x, y), text, fill="black", font=font)

    # Enhanced mode: Create waypoints map overlay if CSV provided
    if waypoints_csv is not None:
        # Load waypoints data from CSV
        dfw = pd.read_csv(str(waypoints_csv))
        
        # Auto-detect longitude and latitude column names
        lon_cols = [
            c for c in dfw.columns if c.lower() in ("lon", "lng", "longitude", "x")
        ]
        lat_cols = [c for c in dfw.columns if c.lower() in ("lat", "latitude", "y")]
        
        if lon_cols and lat_cols:
            lon = dfw[lon_cols[0]].astype(float)
            lat = dfw[lat_cols[0]].astype(float)
            
            # Configure matplotlib figure with exact dimensions and DPI
            fig_dpi = dpi
            fig_w = size[0] / fig_dpi
            fig_h = size[1] / fig_dpi
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
            
            # Set light gray background for map area
            ax.set_facecolor("#f0f0f0")
            
            # Plot waypoints as red scatter points
            ax.scatter(lon, lat, s=10, c="red", alpha=0.8)
            
            # Calculate map bounds with safety margins
            minx, maxx = lon.min(), lon.max()
            miny, maxy = lat.min(), lat.max()
            
            # Handle edge case where all coordinates are identical
            if minx == maxx:
                minx -= 0.0005
                maxx += 0.0005
            if miny == maxy:
                miny -= 0.0005
                maxy += 0.0005
                
            # Add 6% margin around the data bounds
            dx = (maxx - minx) * 0.06
            dy = (maxy - miny) * 0.06
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)
            
            # Remove axis ticks for cleaner appearance
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add large mission number text in the center with white outline
            fs = int(size[1] * 0.45)  # Font size relative to image height
            txt = text
            txt_artist = ax.text(
                0.5 * (minx + maxx),  # Center horizontally
                0.5 * (miny + maxy),  # Center vertically
                txt,
                color="black",
                fontsize=fs,
                ha="center",
                va="center",
                weight="bold",
            )
            # Add white stroke effect for better visibility
            txt_artist.set_path_effects(
                [
                    pe.Stroke(
                        linewidth=max(2, int(fs * 0.06)), foreground="white"
                    ),
                    pe.Normal(),
                ]
            )
                
            # Remove margins and save the matplotlib figure
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(str(jpg_path), dpi=fig_dpi)
            plt.close(fig)
            return

    # PIL-only rendering: Save the text-only image
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(jpg_path), format="JPEG", dpi=(dpi, dpi))


def create(
    template_kmz_folder: Path,
    csv_path: Path,
    chunk_size: int = 50,
    take: str = "first",
    out_dir: Path | None = None,
    # missions_csv: Path | None = None,
    polygon_no: str = "001",  # polygon identifier string for output directory structure
):
    """Generate multiple KMZ mission files by splitting waypoints CSV into chunks.

    This function processes a large waypoints CSV file by dividing it into smaller
    chunks and creating individual KMZ mission files for each chunk. It also
    generates preview images showing the mission area and waypoints.

    Parameters
    ----------
    template_kmz_folder : Path
        Path to the template KMZ folder used as a base for new missions.
    csv_path : Path
        Path to the input CSV file containing waypoint data.
    chunk_size : int, optional
        Maximum number of waypoints per generated KMZ file. Default is 50.
    take : str, optional
        Strategy for selecting waypoints ("first", "last", etc.). Default is "first".
    out_dir : Path, optional
        Base output directory for generated files. If None, uses "missions" directory.
    polygon_no : str, optional
        Polygon identifier used in output directory structure. Default is "001".

    Returns
    -------
    None
        Creates KMZ files and preview images in the output directory structure:
        out_dir/polygon_no/NNN/NNN.kmz and map_preview/NNN/NNN.jpg

    Raises
    ------
    SystemExit
        If template folder doesn't exist, CSV file not found, or CSV is empty.
    FileNotFoundError
        If the KMZ builder utility script is not found.

    Notes
    -----
    The function creates a directory structure like:
    - missions/polygon_no/001/001.kmz
    - missions/polygon_no/002/002.kmz
    - missions/polygon_no/map_preview/001/001.jpg
    - missions/polygon_no/map_preview/002/002.jpg

    Each KMZ file contains a subset of waypoints from the original CSV,
    with waypoint indexing restarting from 0 for each chunk.
    """
    workspace = Path(__file__).resolve().parent
    
    # Configure base output directory structure: missions/polygon_no/
    if out_dir:
        base_output = Path(out_dir) / polygon_no
    else:
        base_output = Path("missions") / polygon_no
    base_output.mkdir(parents=True, exist_ok=True)

    # Get list of available directory names for KMZ naming (legacy feature)
    # Currently uses workspace directories, but missions CSV option is preserved
    dirs = list_dji_dirs(workspace)

    # Validate template folder and input CSV file
    if not template_kmz_folder.exists() or not template_kmz_folder.is_dir():
        raise SystemExit(f"Template KMZ folder not found: {template_kmz_folder}")

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Load and validate waypoints data
    df = pd.read_csv(str(csv_path))
    total = len(df)
    if total == 0:
        raise SystemExit("CSV contains no points")

    # Calculate number of output files needed based on chunk size
    files_needed = math.ceil(total / chunk_size)

    # Create map preview directory structure
    map_preview_dir = base_output / "map_preview"
    if not map_preview_dir.exists():
        map_preview_dir.mkdir(parents=True, exist_ok=True)

    # Process each chunk of waypoints to create individual KMZ files
    for i in range(files_needed):
        # Define chunk boundaries
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = df.iloc[start:end]

        # Generate sequential naming: 001, 002, 003, etc.
        name = f"{i+1:03d}"
        kmz_name = f"{name}.kmz"
        preview_sub = map_preview_dir / name
        preview_jpg = preview_sub / f"{name}.jpg"

        logger.info(f"[{i+1}/{files_needed}] {kmz_name}: points {start}..{end-1}")

        # Create temporary CSV file for this chunk with polygon identifier
        tmpdir = Path(tempfile.mkdtemp(prefix="kmz_chunk_"))
        tmp_csv = tmpdir / f"waypoints_dji_{polygon_no}.csv"
        chunk.to_csv(tmp_csv, index=False)

        # Configure waypoint indexing to start from 0 for each KMZ file
        # This ensures consistent waypoint numbering across chunks
        start_index = 0
        pass_start_zero = True

        # Generate KMZ file using the integrated builder function
        generated_kmz = build_kmz_from_template(
            template_kmz_folder,
            tmp_csv,
            limit=None,
            take=take,
            start_zero=pass_start_zero,
            out_dir=out_dir,
            start_index=start_index,
            polygon_no=polygon_no,  # Pass polygon identifier to builder
            chunk_no=name,  # Pass chunk number to builder
        )

        # Move and rename generated KMZ to target location
        # Structure: base_output/NNN/NNN.kmz
        target_dir = base_output / name
        target = target_dir / kmz_name
        gen_path = Path(generated_kmz)
        
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if not gen_path.exists():
            logger.warning(
                f"Generated KMZ not found at {gen_path}; skipping move to {target}"
            )
        else:
            # Handle file moving with safety checks
            if gen_path.resolve() == target.resolve():
                logger.info(f"Source and target are the same ({target}); skipping move")
            else:
                if target.exists():
                    target.unlink()  # Remove existing file
                shutil.move(str(gen_path), str(target))

        # Create preview image with mission number and waypoint map overlay
        preview_sub.mkdir(parents=True, exist_ok=True)
        create_preview(preview_jpg, f"{i+1}/{files_needed}", waypoints_csv=tmp_csv)
        logger.info(f"  -> created {target} and preview {preview_jpg}")
        
        # Clean up temporary directory
        shutil.rmtree(tmpdir)



def rename(path: str, polygon_no: str, missions_csv: Path):
    """Rename mission files and folders based on names from a CSV file.

    This function renames mission directories and their contents according to a CSV file
    that contains new names. It handles both regular files and KMZ archives, updating
    internal file names within KMZ files and maintaining the directory structure.

    Parameters
    ----------
    path : str
        Base path where the missions are located.
    polygon_no : str
        Polygon number as string, must match the one used in mission creation.
    missions_csv : Path
        Path to CSV file containing new names for missions (one per line).

    Returns
    -------
    None
        Renames files and directories in place.

    Raises
    ------
    FileNotFoundError
        If the polygon directory doesn't exist.
    ValueError
        If directory names cannot be parsed as integers.

    Notes
    -----
    The function expects the following directory structure:
    - path/missions/polygon_no/001/001.kmz
    - path/missions/polygon_no/002/002.kmz
    - path/missions/polygon_no/map_preview/001/001.jpg
    - path/missions/polygon_no/map_preview/002/002.jpg

    CSV file should contain one name per line, corresponding to directories 001, 002, etc.
    The function will rename both the directories and all files containing the old names.

    For KMZ files, internal file names are also updated to maintain consistency.
    """
    # Load all mission names from CSV file
    with open(missions_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        names = [row[0].strip() for row in reader if row and row[0].strip()]

    # Locate the polygon directory containing mission segments
    polygon_dir = Path(path) / "missions" / polygon_no
    if not polygon_dir.exists():
        raise FileNotFoundError(f"Polygon directory not found: {polygon_dir}")

    # Process all existing mission segment directories
    for segment_dir in sorted(polygon_dir.iterdir()):
        if not segment_dir.is_dir():
            continue
            
        segment_no = segment_dir.name
        
        # Skip non-numeric directories (like 'map_preview')
        if not segment_no.isdigit():
            logger.warning(f"Skipping non-numeric directory: {segment_no}")
            continue
        else:
            segment_idx = int(segment_no) - 1  # Convert to zero-based index
            
        if segment_idx < 0:
            logger.warning(f"Skipping invalid segment number: {segment_no}")
            continue
            
        if segment_idx >= len(names):
            logger.warning(f"No name in missions.csv for segment {segment_no}, skipping.")
            continue
            
        new_name = names[segment_idx]
        logger.info(f"Renaming segment {segment_no} -> {new_name}")
        
        # Rename all files in the segment directory that contain the segment number
        for file in segment_dir.iterdir():
            if segment_no in file.name:
                new_file = file.with_name(file.name.replace(segment_no, new_name))
                file.rename(new_file)
                
                # Special handling for KMZ files: update internal file names
                if new_file.suffix.lower() == ".kmz":
                    _update_kmz_internal_names(new_file, segment_dir, segment_no, new_name)
        
        # Rename the segment directory itself
        new_segment_dir = segment_dir.parent / new_name
        segment_dir.rename(new_segment_dir)
        logger.info(f"  -> {segment_no} renamed to {new_name}")

        # Update corresponding map_preview directory and files
        _update_preview_directory(new_segment_dir, segment_no, new_name)


def _update_kmz_internal_names(kmz_file: Path, segment_dir: Path, old_name: str, new_name: str):
    """Update internal file names within a KMZ archive.
    
    Parameters
    ----------
    kmz_file : Path
        Path to the KMZ file to update.
    segment_dir : Path
        Directory containing the KMZ file.
    old_name : str
        Original segment name to replace.
    new_name : str
        New segment name to use.
    """
    # Extract KMZ contents to temporary directory
    with zipfile.ZipFile(kmz_file, "r") as zin:
        tmp_dir = segment_dir / "_tmp_kmz"
        zin.extractall(tmp_dir)
    
    # Rename internal files that contain the old segment name
    for root, dirs, files in os.walk(tmp_dir):
        for fname in files:
            if old_name in fname:
                src = Path(root) / fname
                dst = Path(root) / fname.replace(old_name, new_name)
                src.rename(dst)
    
    # Repackage the KMZ file with updated internal names
    with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as zout:
        for root, dirs, files in os.walk(tmp_dir):
            for fname in files:
                fpath = Path(root) / fname
                arcname = fpath.relative_to(tmp_dir)
                zout.write(fpath, arcname)
    
    # Clean up temporary directory
    for root, dirs, files in os.walk(tmp_dir, topdown=False):
        for fname in files:
            (Path(root) / fname).unlink()
        for dname in dirs:
            (Path(root) / dname).rmdir()
    tmp_dir.rmdir()


def _update_preview_directory(segment_dir: Path, old_name: str, new_name: str):
    """Update map preview directory and files for a renamed segment.
    
    Parameters
    ----------
    segment_dir : Path
        The renamed segment directory.
    old_name : str
        Original segment name.
    new_name : str
        New segment name.
    """
    map_preview_dir = segment_dir.parent / "map_preview"
    old_preview_sub = map_preview_dir / old_name
    new_preview_sub = map_preview_dir / new_name
    
    if old_preview_sub.exists():
        # Rename files within the preview subdirectory
        for file in old_preview_sub.iterdir():
            if old_name in file.name:
                new_file = file.with_name(file.name.replace(old_name, new_name))
                file.rename(new_file)
        
        # Rename the preview subdirectory itself
        old_preview_sub.rename(new_preview_sub)
        logger.info(f"  -> map_preview/{old_name} renamed to map_preview/{new_name}")
