"""Reading of SWAN (``.mat`` block output) and COPLA (velocity) results."""

import numpy as np
import pandas as pd
from scipy.io import loadmat as ldm


def read_copla(file_name: str, grid: dict = None):
    """Read COPLA model velocity output files.

    Loads velocity field data from COPLA
    model output and computes magnitude and direction on a 2D grid with ghost cells.

    Args:
        file_name (str): Path to COPLA velocity output file.
        grid (dict, optional): Existing grid dictionary to update. If None,
            creates new dictionary. Defaults to None.

    Returns:
        dict: Dictionary containing velocity components and derived fields:
            - 'u': East-west velocity component (m/s) with ghost cells
            - 'v': North-south velocity component (m/s) with ghost cells
            - 'U': Velocity magnitude (m/s)
            - 'DirU': Velocity direction (degrees, nautical convention)
    """
    data = pd.read_csv(
        file_name,
        skiprows=7,
        sep=r"\s+",
        header=None,
        index_col=0,
        names=["x", "y", "u", "v"],
    )
    # Create meshgrid for spatial dimensions
    _, x = np.meshgrid(data.y.unique(), data.x.unique())

    if grid is None:
        grid = {}

    nx, ny = np.shape(x)
    
    # Create velocity arrays with ghost cells (boundary padding)
    for var_ in ["u", "v"]:
        grid[var_] = np.zeros([nx + 2, ny + 2])
        # Fill interior cells with data
        grid[var_][1:-1, 1:-1] = data[var_].to_numpy().reshape([nx, ny])

    # Compute velocity magnitude
    grid["U"] = np.sqrt(grid["u"] ** 2 + grid["v"] ** 2)
    
    # Compute velocity direction (nautical convention: 0° = North, clockwise)
    grid["DirU"] = np.fmod(np.rad2deg(np.arctan2(grid["v"], grid["u"])) + 90, 360)

    return grid


def read_swan(file_name: str, grid: dict = None, variables: list = None):
    """Read SWAN model output from MATLAB file format.

    Loads wave field data from SWAN (Simulating WAves Nearshore) model output
    stored in MATLAB format and computes wave number from wavelength.

    Args:
        file_name (str): Path to MATLAB (.mat) file containing SWAN output.
        grid (dict, optional): Existing grid dictionary to update. If None,
            creates new dictionary. Defaults to None.
        variables (list, optional): List of output variable names in order:
            [x, y, depth, Qb, L, Setup, Hs, DirM]. If None, uses default names.
            Defaults to None.

    Returns:
        dict: Dictionary containing wave parameters on grid:
            - 'x', 'y': Coordinates
            - 'depth': Water depth (m)
            - 'Qb': Wave breaking dissipation
            - 'L': Wavelength (m)
            - 'Setup': Wave setup (m)
            - 'Hs': Significant wave height (m)
            - 'DirM': Mean wave direction (degrees)
            - 'kp': Wave number (2π/L) computed from wavelength
    """
    if not variables:
        variables = ["x", "y", "depth", "Qb", "L", "Setup", "Hs", "DirM"]

    if grid is None:
        grid = {}

    # Load MATLAB file with SWAN output
    swan_dictionary = ldm(file_name)
    
    # Map SWAN variable names to user-specified names
    for ind_, var_ in enumerate(
        ["Xp", "Yp", "Depth", "Qb", "Wlen", "Setup", "Hsig", "Dir"]
    ):
        grid[variables[ind_]] = swan_dictionary[var_]
        # Replace NaN values with small number to avoid numerical issues
        grid[variables[ind_]][np.isnan(grid[variables[ind_]])] = 1e-6

    # Compute wave number from wavelength
    grid["kp"] = 2 * np.pi / grid["L"]

    return grid
