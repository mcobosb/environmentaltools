"""Reading of CSHORE output files (``O*``)."""

import pandas as pd


def read_output(file_type: str, path: str, skiprows: int = 1):
    """Read CSHORE model output files.

    Loads and parses output files from the CSHORE numerical model, supporting
    various output types (profiles, transport rates, energy, velocities, etc.).

    Args:
        file_type (str): Type of CSHORE output file to read. Options include:
            'bprof', 'bsusl', 'cross', 'energ', 'longs', 'param', 'rolle',
            'setup', 'swase', 'timse', 'xmome', 'xvelo', 'ymome', 'yvelo'.
        path (str): Directory path containing CSHORE output files.
        skiprows (int): Number of rows to skip when reading file. Defaults to 1.

    Returns:
        pd.DataFrame: Parsed CSHORE output data with appropriate column names
            and spatial index (x-distance in meters).
    """
    header = {
        "bprof": ["z"],
        "bsusl": [r"$P_b$", r"$P_s$", r"$V_s$"],
        "cross": [r"$Q_{b,x}$", r"$Q_{s,x}$", r"$Q_{b,x} + Q_{s,x}$"],
        "crvol": [],
        "energ": [r"Eflux (m3/s)", "Db (m2/s)", "Df (m2/s)"],
        "longs": [r"$Q_{b,y}$", r"$Q_{s,y}$", r"$Q_{b,y} + Q_{s,y}$"],
        "lovol": [],
        "param": ["T (s)", r"$Q_b$ (nondim)", "Sigma* (nondim)"],
        "rolle": ["Rq (m2/s)"],
        "setup": [r"$\eta + S_{tide}$ (m)", "d (m)", r"$\sigma_{eta}$ (m)"],
        "swase": ["de (m)", "Uxe (m/s)", "Qxe (m2/s)"],
        "timse": ["t (id)", "t (s)", "q0 (m2/s)", "qbx,lw (m2/s)", "qsx,lw (m2/s)"],
        "xmome": ["Sxx (m2)", "taubx (m)"],
        "xvelo": [r"$U_x$", r"$U_{x,std}$"],
        "ymome": ["Sxx (m2)", "taubx (m)"],
        "yvelo": ["sin theta (unitary)", r"$U_y$", r"$U_{y,std}$"],
    }

    # TODO: include morphology options
    # EWD: Output exceedance probability 0.015
    # q0: wave overtopping rate, qbx,lw: cross-shore bedload transport rate at the landward end of the computation domain
    
    # Construct output filename (CSHORE prepends 'O' to output files)
    filename = path + "/" + "O" + file_type.upper()
    
    if file_type == "bprof":
        # For beach profile, read number of valid points from header
        fid = open(filename, "rb")
        properties = fid.readline()
        id_ = int(properties.split()[1])
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            skiprows=skiprows,
            index_col=0,
            names=header[file_type],
        )
        # Trim to actual number of profile points
        df = df.iloc[:id_, :]
    else:
        # Standard reading for other output types
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            skiprows=skiprows,
            index_col=0,
            names=header[file_type],
        )
    # Index represents x distance in meters

    # Ensure column names are strings
    df.columns = df.columns.astype("str")

    return df
