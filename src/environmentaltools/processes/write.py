import numpy as np
from environmentaltools.common import utils, save
from scipy.io import loadmat as ldm
import os


def write_cshore(properties: dict, folder: str):
    """Write CSHORE model input file (infile).
    
    Creates the main input file for CSHORE (Coastal Storm Modeling System) with
    all necessary parameters for beach profile evolution, sediment transport,
    wave transformation, and coastal structure simulation.

    Parameters
    ----------
    properties : dict
        Dictionary containing CSHORE model parameters:
        
        - header : str
            Project description header
        - iline : int
            Line number option (1: single line, 2: multiple lines)
        - iprofl : int
            Profile option (0: initial, 1: recorded)
        - isedav : int
            Sediment availability (0: unlimited, 1: limited)
        - iperm : int
            Permeable layer option (0: no, 1: yes)
        - iover : int
            Overtopping and overflow option
        - iwtran : int
            Wave transmission option
        - ipond : int
            Ponding option for runup zone
        - infilt : int
            Infiltration option
        - iwcint : int
            Wave-current interaction option
        - iroll : int
            Roller model option
        - iwind : int
            Wind effects option (0: no wind, 1: with wind)
        - itide : int
            Tidal variation option
        - iveg : int
            Vegetation option
        - dx : float
            Node spacing (m)
        - gamma : float
            Breaking parameter
        - d50 : float
            Median sediment grain size (m)
        - wf : float
            Sediment fall velocity (m/s)
        - sg : float
            Sediment specific gravity
        - effb : float
            Suspended load efficiency factor
        - efff : float
            Bed load efficiency factor
        - slp : float
            Suspended load parameter
        - slpot : float
            Suspended load parameter (offshore transport)
        - tanphi : float
            Tangent of internal friction angle
        - blp : float
            Bed load parameter
        - rwh : float
            Roller parameter
        - ilab : int
            Laboratory or field scale
        - nwave : int
            Number of wave conditions
        - nsurg : int
            Number of surge conditions
        - timebc_wave : float
            Time for boundary condition (s)
        - Tp : float
            Peak wave period (s)
        - Hrms : float
            Root-mean-square wave height (m)
        - Wsetup : float
            Wave setup (m)
        - swlbc : float
            Still water level boundary condition (m)
        - angle : float
            Wave angle (degrees)
        - x : np.ndarray
            Cross-shore coordinates (m)
        - zb : np.ndarray
            Beach elevation (m)
        - fw : np.ndarray
            Bottom friction factor
        - VelV : float, optional
            Wind velocity (m/s), required if iwind=1
        - DirV : float, optional
            Wind direction (degrees), required if iwind=1
        - slgradient : float, optional
            Sea level gradient, required if itide=1
            
    folder : str
        Directory path where the input file will be created

    Returns
    -------
    None
        Creates 'infile' in the specified folder

    Notes
    -----
    The function creates a formatted text file following CSHORE input specifications:
    
    - Model control parameters (lines, profile, sediment options)
    - Physical parameters (grid spacing, breaking, sediment properties)
    - Boundary conditions (waves, wind, tide)
    - Bathymetric profile data
    
    Wind and tide data are conditionally written based on iwind and itide flags.

    Examples
    --------
    >>> props = {
    ...     'header': 'Beach Profile Simulation',
    ...     'iline': 1, 'iprofl': 0, 'isedav': 1,
    ...     'dx': 1.0, 'gamma': 0.4, 'd50': 0.0003,
    ...     'Hrms': 2.5, 'Tp': 8.0, 'angle': 0.0,
    ...     'x': np.arange(0, 500, 1.0),
    ...     'zb': np.linspace(-5, 3, 500),
    ...     # ... more parameters
    ... }
    >>> write_cshore(props, './cshore_run')
    """

    fid = open(folder + "/infile", "w")
    fid.write("3 \n")
    fid.write("{} \n".format(str(properties["header"])))
    fid.write(
        "{}                                               ->ILINE\n".format(
            str(properties["iline"])
        )
    )
    fid.write(
        "{}                                               ->IPROFL\n".format(
            str(properties["iprofl"])
        )
    )
    fid.write(
        "{}                                               ->ISEDAV\n".format(
            str(properties["isedav"])
        )
    )
    fid.write(
        "{}                                               ->IPERM\n".format(
            str(properties["iperm"])
        )
    )
    fid.write(
        "{}                                               ->IOVER\n".format(
            str(properties["iover"])
        )
    )
    fid.write(
        "{}                                               ->IWTRAN\n".format(
            str(properties["iwtran"])
        )
    )
    fid.write(
        "{}                                               ->IPOND\n".format(
            str(properties["ipond"])
        )
    )
    fid.write(
        "{}                                               ->INFILT\n".format(
            str(properties["infilt"])
        )
    )
    fid.write(
        "{}                                               ->IWCINT\n".format(
            str(properties["iwcint"])
        )
    )
    fid.write(
        "{}                                               ->IROLL \n".format(
            str(properties["iroll"])
        )
    )
    fid.write(
        "{}                                               ->IWIND \n".format(
            str(properties["iwind"])
        )
    )
    fid.write(
        "{}                                               ->ITIDE \n".format(
            str(properties["itide"])
        )
    )
    fid.write(
        "{}                                               ->IVEG  \n".format(
            str(properties["iveg"])
        )
    )
    # fid.write('{}                                               ->ICLAY  \n'.format(str(properties['iclay'])))
    fid.write(
        "{:11.4f}                                     ->DX\n".format(properties["dx"])
    )
    fid.write(
        "{:11.4f}                                     ->GAMMA \n".format(
            properties["gamma"]
        )
    )
    fid.write(
        "{:11.4f}{:11.4f}{:11.4f}               ->D50 WF SG\n".format(
            properties["d50"], properties["wf"], properties["sg"]
        )
    )
    fid.write(
        "{:11.4f}{:11.4f}{:11.4f}{:11.4f}               ->EFFB EFFF SLP\n".format(
            properties["effb"],
            properties["efff"],
            properties["slp"],
            properties["slpot"],
        )
    )
    fid.write(
        "{:11.4f}{:11.4f}                          ->TANPHI BLP\n".format(
            properties["tanphi"], properties["blp"]
        )
    )
    fid.write(
        "{:11.4f}                                     ->RWH \n".format(
            properties["rwh"]
        )
    )
    fid.write(
        "{}                                               ->ILAB\n".format(
            str(properties["ilab"])
        )
    )
    fid.write(
        "{}                                               ->NWAVE \n".format(
            str(properties["nwave"])
        )
    )
    fid.write(
        "{}                                               ->NSURGE \n".format(
            str(properties["nsurg"])
        )
    )
    fid.write(
        "{:11.2f}{:11.4f}{:11.4f}{:11.4f}{:11.4f}{:11.4f}\n".format(
            properties["timebc_wave"],
            properties["Tp"],
            properties["Hrms"],
            properties["Wsetup"],
            properties["swlbc"],
            properties["angle"],
        )
    )
    fid.write(
        "{}                             ->NBINP \n".format(str(len(properties["x"])))
    )

    # if properties.iperm==1|properties.isedav >= 1:
    #     fid.write('{}                             ->NPINP \n',length(properties.x_p))
    fid.close()

    fid = open(folder + "/infile", "a")
    dum = np.vstack([properties["x"], properties["zb"], properties["fw"]])
    for line in range(dum.shape[1]):
        fid.write("{:11.4f}{:11.4f}{:11.4f}\n".format(*dum[:, line]))

    # if properties.iperm == 1 | properties.isedav >= 1:
    #     dum = [properties.x_p(:) properties.zb_p(:)]
    #     fid.write('#11.4f#11.4f\n', dum)

    if properties["iwind"]:
        fid.write("1 \n")
        fid.write(
            "{:11.1f}{:11.4f}{:11.4f}\n".format(
                0, properties["VelV"], properties["DirV"]
            )
        )
        fid.write(
            "{:11.1f}{:11.4f}{:11.4f}\n".format(
                properties["timebc_wave"], properties["VelV"], properties["DirV"]
            )
        )

    if properties["itide"]:
        fid.write("1 \n")
        fid.write("{:11.1f}{:13.8f}\n".format(0, properties["slgradient"]))
        fid.write(
            "{:11.1f}{:13.8f}\n".format(
                properties["timebc_wave"], properties["slgradient"]
            )
        )
    fid.close()

    return


def write_swan(i, index_, case_id, data, params, mesh="global", local=False, nested=False):
    """Write SWAN model input files (swaninit and input file).
    
    Creates initialization and input files for SWAN (Simulating WAves Nearshore)
    spectral wave model, including grid definition, boundary conditions, physical
    processes, and output specifications.

    Parameters
    ----------
    i : int
        Time step counter (0-indexed)
    index_ : pd.Timestamp or datetime-like
        Timestamp for current simulation time
    case_id : str
        Case identifier (e.g., '0001', '0002')
    data : pd.DataFrame
        Time series with boundary condition data containing columns:
        
        - eta : float
            Water level (m)
        - Hs : float
            Significant wave height (m)
        - Tp : float
            Peak wave period (s)
        - DirM : float
            Mean wave direction (degrees, nautical convention)
        - Vv : float
            Wind velocity (m/s)
        - DirV : float
            Wind direction (degrees)
            
    params : dict
        Model configuration parameters:
        
        - directory : str
            Working directory path
        - project_name : str
            Project identifier
        - {mesh}_coords_x : float
            Grid origin x-coordinate (m)
        - {mesh}_coords_y : float
            Grid origin y-coordinate (m)
        - {mesh}_angle : float
            Grid rotation angle (degrees)
        - {mesh}_length_x : float
            Grid extent in x-direction (m)
        - {mesh}_length_y : float
            Grid extent in y-direction (m)
        - {mesh}_nodes_x : int
            Number of grid nodes in x-direction
        - {mesh}_nodes_y : int
            Number of grid nodes in y-direction
        - {mesh}_inc_x : float
            Grid spacing in x-direction (m)
        - {mesh}_inc_y : float
            Grid spacing in y-direction (m)
            
    mesh : str, optional
        Mesh identifier ('global' or 'local'). Default: 'global'
    local : bool, optional
        Flag for local mesh (deprecated, not used). Default: False
    nested : bool, optional
        If True, creates nested grid output for downscaling. Default: False

    Returns
    -------
    None
        Creates two files in params['directory']/case_id/:
        
        - swaninit: Initialization file
        - input_{mesh}_{case_id}.swn: SWAN command file

    Notes
    -----
    The function generates SWAN input files with:
    
    **Grid Configuration:**
    
    - Regular Cartesian grid with specified origin, extent, and resolution
    - Directional discretization: 36 bins (10° resolution)
    - Frequency range: 0.05-0.50 Hz
    
    **Boundary Conditions:**
    
    - Global mesh: JONSWAP spectrum with constant parameters
    - Local mesh: Nested boundary from parent grid
    - Two-sided boundaries (North/South and East/West)
    
    **Physical Processes:**
    
    - Wave generation (GEN3 with exponential growth)
    - Triad and quadruplet wave-wave interactions
    - Depth-induced breaking and whitecapping
    - Wave setup and diffraction
    
    **Output:**
    
    - Nested mode: Boundary conditions for nested grid
    - Standard mode: Wave parameters on computational grid (Hs, Tp, Dir, etc.)
    
    Direction convention: SWAN uses nautical convention (waves coming from),
    automatically converted from PdE convention (270 - angle).

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'eta': [0.5], 'Hs': [2.0], 'Tp': [8.0],
    ...     'DirM': [45.0], 'Vv': [10.0], 'DirV': [90.0]
    ... })
    >>> params = {
    ...     'directory': './swan_run',
    ...     'project_name': 'Beach_Wave_Modeling',
    ...     'global_coords_x': 0, 'global_coords_y': 0,
    ...     'global_angle': 0, 'global_length_x': 1000,
    ...     'global_length_y': 500, 'global_nodes_x': 101,
    ...     'global_nodes_y': 51, # ... more parameters
    ... }
    >>> write_swan(0, data.index[0], '0001', data, params, mesh='global')
    """

    swfile = open(params["directory"] + "/" + case_id + "/swaninit", "w")
    swfile.write("4                                   version of initialisation file\n")
    swfile.write("GDFA                                name of institute\n")
    swfile.write("3                                   command file ref. number\n")
    swfile.write("input_" + mesh + "_" + case_id + ".swn\n")
    swfile.write("4                                   print file ref. number\n")
    swfile.write("print_" + mesh + "_" + case_id + ".prt\n")
    swfile.write("4                                   test file ref. number\n")
    swfile.write("                                    test file name\n")
    swfile.write("6                                   screen ref. number\n")
    swfile.write("99                                  highest file ref. number\n")
    swfile.write("$                                   comment identifier\n")
    swfile.write(" 	                                TAB character\n")
    swfile.write("/                                   dir sep char in input file\n")
    swfile.write(
        "/                                   dir sep char replacing previous one\n"
    )
    swfile.write("1                                   default time coding option\n")
    swfile.close()

    imswfile = open(
        params["directory"] + "/" + case_id + "/input_" + mesh + "_" + case_id + ".swn", "w"
    )
    imswfile.write(
        "$*******************************HEADING******************************************\n"
    )
    imswfile.write("$\n")
    imswfile.write("PROJ '" + params["project_name"] + "' '" + case_id + "' \n")
    imswfile.write("$Caso " + mesh + "_" + case_id + "\n")
    imswfile.write(
        "$***************************** MODEL INPUT ****************************************\n"
    )
    imswfile.write(
        "SET LEVEL " + str(np.round(data.loc[index_, "eta"], decimals=2)) + "\n"
    )
    imswfile.write("$                xpc		ypc		alpc		xlenc		ylenc		mxc		myc \n")
    imswfile.write("$\n")
    imswfile.write(
        "CGRID REGULAR   "
        + str(params[mesh + "_coords_x"])
        + "   "
        + str(params[mesh + "_coords_y"])
        + "   "
        + str(np.round(np.remainder(params[mesh + "_angle"], 360), decimals=2))
        + "    "
        + str(params[mesh + "_length_x"])
        + "   "
        + str(params[mesh + "_length_y"])
        + "    "
        + str(params[mesh + "_nodes_x"] - 1)
        + "   "
        + str(params[mesh + "_nodes_y"] - 1)
        + "  CIRCLE 36 0.05 0.50\n"
    )
    imswfile.write("$\n")
    imswfile.write("$xpinp		ypinp		alpinp		mxinp		myinp		dxinp		dyinp\n")
    imswfile.write(
        "INPGRID BOTTOM     "
        + str(params[mesh + "_coords_x"])
        + "   "
        + str(params[mesh + "_coords_y"])
        + "   "
        + str(np.round(np.remainder(params[mesh + "_angle"], 360), decimals=2))
        + "    "
        + str(params[mesh + "_nodes_x"] - 1)
        + "   "
        + str(params[mesh + "_nodes_y"] - 1)
        + "   "
        + str(params[mesh + "_inc_x"])
        + "  "
        + str(params[mesh + "_inc_y"])
        + "\n"
    )
    imswfile.write("$\n")
    imswfile.write(
        "$              fac    fname       idla    nhedf     formato  ((mxinp+1)FN.d)\n"
    )
    imswfile.write("$\n")
    imswfile.write(
        "READINP BOTTOM -1. '"
        + case_id
        + "_"
        + mesh
        + ".dat' 3 0 FORMAT '("
        + str(params[mesh + "_nodes_x"])
        + "F9.3)'\n"
    )
    imswfile.write("$\n")
    imswfile.write(
        "WIND  "
        + str(np.round(data.loc[index_, "Vv"], decimals=2))
        + " "
        + str(np.round(np.remainder(270 - data.loc[index_, "DirV"], 360), decimals=2))
        + "\n"
    )
    imswfile.write("$\n")

    if mesh == "global":
        # Here, direction has PdE convention (waves coming from N: 0º, E: 90º)
        if data.loc[index_, "DirM"] < 90:
            side = ["N", "E"]
        elif (data.loc[index_, "DirM"] < 180) & (data.loc[index_, "DirM"] >= 90):
            side = ["S", "E"]
        elif (data.loc[index_, "DirM"] < 270) & (data.loc[index_, "DirM"] >= 180):
            side = ["S", "W"]
        else:
            side = ["N", "W"]
        imswfile.write("BOUN SHAPESPEC JONSWAP PEAK DSPR POWER\n")
        imswfile.write("$                             Hs	Tp	Dir	dd(spreading power)\n")
        imswfile.write(
            "BOUN SIDE "
            + side[0]
            + " CONSTANT PAR  "
            + str(np.round(data.loc[index_, "Hs"], decimals=2))
            + " "
            + str(np.round(data.loc[index_, "Tp"], decimals=2))
            + " "
            + str(
                np.round(np.remainder(270 - data.loc[index_, "DirM"], 360), decimals=2)
            )
            + " 2.00\n"
        )
        imswfile.write(
            "BOUN SIDE "
            + side[1]
            + " VARIABLE PAR  200 "
            + str(np.round(data.loc[index_, "Hs"], decimals=2))
            + " "
            + str(np.round(data.loc[index_, "Tp"], decimals=2))
            + " "
            + str(
                np.round(np.remainder(270 - data.loc[index_, "DirM"], 360), decimals=2)
            )
            + " 2.00\n"
        )
    else:
        imswfile.write("BOUNdnest1 NEST '" + case_id + ".bnd' CLOSED\n")

    imswfile.write("$*************************WIND GROWTH**************************\n")
    imswfile.write("GEN3 AGROW\n")
    imswfile.write(
        "$*************************WAVE-WAVE INTERACTION**************************\n"
    )
    imswfile.write("TRIAD\n")
    imswfile.write("QUAD\n")
    imswfile.write("$*************************DISSIPATION**************************\n")
    imswfile.write("BREAKING\n")
    imswfile.write("WCAP\n")
    imswfile.write("$*************************************************************\n")
    imswfile.write("SETUP\n")
    imswfile.write("DIFFRACTION\n")
    imswfile.write("NUM ACCUR 0.005 0.01 0.005 99.5 STAT 50 0.1\n")
    if nested:
        imswfile.write(
            "$*************************OUPUT LOCATIONS**************************\n"
        )
        imswfile.write(
            "NGRID '"
            + case_id
            + ".dat"
            + "'  "
            + str(params["local_coords_x"])
            + "   "
            + str(params["local_coords_y"])
            + "   "
            + str(np.round(np.remainder(params["local_angle"], 360), decimals=2))
            + "    "
            + str(params["local_length_x"])
            + "   "
            + str(params["local_length_y"])
            + "   "
            + str(params["local_nodes_x"] - 1)
            + "  "
            + str(params["local_nodes_y"] - 1)
            + "   \n"
        )
        imswfile.write(
            "$***************************** MODEL OUTPUT LOCATIONS***************************************\n"
        )
        imswfile.write("NESTOUT '" + case_id + ".dat" + "' '" + case_id + ".bnd' \n")
    else:
        imswfile.write(
            "$***************************** MODEL COMPUTACIONAL GRID OUTPUT ***************************************\n"
        )
        imswfile.write(
            "BLOCK 'COMPGRID' NOHEAD '"
            + case_id
            + ".mat' LAY 3  XP YP HSIGN TPS DIR QB WLEN DEPTH SETUP\n"
        )
    imswfile.write(
        "$***************************** COMPUTATIONS ***************************************\n"
    )
    imswfile.write("COMPUTE\n")
    imswfile.write("STOP\n")
    imswfile.close()
    return


def write_copla(i, index_, case_id, data, params, mesh="local"):
    """Write COPLA model input files for nearshore profile evolution.
    
    Creates input files for COPLA (Coupled Profile and Area) model, which
    simulates beach profile evolution and morphodynamics. Reads SWAN wave
    output and formats it for COPLA computation.

    Parameters
    ----------
    i : int
        Time step counter (0-indexed)
    index_ : pd.Timestamp or datetime-like
        Timestamp for current simulation time
    case_id : str
        Case identifier (e.g., '0001', '0002')
    data : pd.DataFrame
        Time series with boundary condition data containing:
        
        - Hs : float
            Significant wave height (m)
        - Tp : float
            Peak wave period (s)
        - DirM : float
            Mean wave direction (degrees)
        - eta : float
            Water level / tidal elevation (m)
            
    params : dict
        Model configuration parameters:
        
        - directory : str
            Working directory path
        - {mesh}_nodes_x : int
            Number of cross-shore nodes
        - {mesh}_nodes_y : int
            Number of longshore nodes
        - {mesh}_inc_x : float
            Grid spacing in x-direction (m)
        - {mesh}_inc_y : float
            Grid spacing in y-direction (m)
            
    mesh : str, optional
        Mesh identifier, typically 'local'. Default: 'local'

    Returns
    -------
    None
        Creates three files in params['directory']/case_id/:
        
        - {case_id}out.dat: Output configuration with wave field from SWAN
        - CLAVE.DAT: Key file with case identifier
        - {case_id}in.dat: Input parameters file
        - {case_id}dat: Calibration and solver parameters

    Notes
    -----
    The function performs coordinate system transformation:
    
    - SWAN uses standard coordinates (X: east, Y: north)
    - COPLA uses rotated coordinates: Xc = -Ys, Yc = Xs
    - Wave direction adjusted: Dir_copla = Dir_swan - 90°
    
    **File Structure:**
    
    {case_id}out.dat contains:
    
    - Grid dimensions and coordinates
    - Bathymetry from SWAN depth output
    - Wave height field (Hs/2 for amplitude)
    - Wave direction field
    
    {case_id}in.dat specifies:
    
    - Computational grid parameters
    - Boundary condition type
    - Wave period and tidal level
    - Incident wave amplitude and direction
    
    {case_id}dat contains solver settings:
    
    - Time step and roughness (Manning coefficient)
    - Eddy viscosity and Coriolis parameter
    - Nonlinear terms and flooding options
    
    NaN values are replaced with 1e-6 for numerical stability.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'Hs': [2.0], 'Tp': [8.0],
    ...     'DirM': [45.0], 'eta': [0.5]
    ... })
    >>> params = {
    ...     'directory': './copla_run',
    ...     'local_nodes_x': 50, 'local_nodes_y': 100,
    ...     'local_inc_x': 5.0, 'local_inc_y': 10.0
    ... }
    >>> write_copla(0, data.index[0], '0001', data, params)
    """

    fSwan = ldm(params["directory"] + "/" + case_id + "/" + case_id + ".mat")

    Hsig, h_CoPla = fSwan["Hsig"], fSwan["Depth"]
    Dir = (
        fSwan["Dir"] - 90
    )  # lo giro para adapatarlo al eje de Copla, los ejes de copla son Xc = -Ys, Yc = Xs (c: copla, s:swan)
    nr, mr = params[mesh + "_nodes_x"], params[mesh + "_nodes_y"]
    xp, yp = (
        np.arange(0, mr) * params["local_inc_x"],
        np.arange(0, nr) * params["local_inc_y"],
    )  # Confirmar que es correcto
    Hsig[np.isnan(Hsig)], Dir[np.isnan(Dir)], h_CoPla[np.isnan(h_CoPla)] = (
        1e-6,
        1e-6,
        1e-6,
    )

    fp = open(params["directory"] + "/" + case_id + "/" + case_id + "out.dat", "w")
    fp.write(str(nr) + " " + str(mr) + " 1\n")

    for col in range(0, nr - 1):
        fp.write(str(yp[col]) + " ")
    fp.write(str(yp[-1]) + "\n")

    for row in range(0, mr):
        fp.write(str(xp[row]) + "\n")
        for col in range(0, nr - 1):
            fp.write("%8.3f" % h_CoPla[row, col] + " ")
        fp.write("%8.3f" % h_CoPla[row, -1] + "\n")

        for col in range(0, nr - 1):
            fp.write("%8.3f" % (Hsig[row, col] / 2) + " ")
        fp.write("%8.3f" % (Hsig[row, -1] / 2) + "\n")

        if row != 0:
            for col in range(0, nr - 1):
                fp.write("%8.3f" % (Dir[row, col]) + " ")
            fp.write("%8.3f" % (Dir[row, -1]) + "\n")
    fp.close()

    fp = open(params["directory"] + "/" + case_id + "/CLAVE.DAT", "w")
    fp.write(case_id + "\n")
    fp.close()

    # Escritura de fichero clavein
    fp = open(params["directory"] + "/" + case_id + "/" + case_id + "in.dat", "w")
    htol, nd = 50, 1
    fp.write("nx ny\n")
    fp.write(str(nr) + " " + str(mr) + " 1\n")
    fp.write("iu ntype icur ibc\n")
    fp.write("  1   1   0   1\n")
    fp.write("dx dy htol\n")
    fp.write(
        str(params["local_inc_y"])
        + " "
        + str(params["local_inc_x"])
        + " "
        + str(htol)
        + "\n"
    )
    fp.write("nd\n")
    fp.write(str(nd) + "\n")
    fp.write("if1 if2 if3\n")
    fp.write("1   0   0\n")
    fp.write("iinput ioutput\n")
    fp.write("1  1\n")
    fp.write("T marea\n")
    fp.write(str(data.loc[index_, "Tp"]) + " " + str(data.loc[index_, "eta"]) + "\n")
    fp.write("amp dir(grados)\n")
    fp.write(
        str(data.loc[index_, "Hs"] / 2)
        + " "
        + str(180 - data.loc[index_, "DirM"])
        + " \n"
    )
    fp.close()

    fp = open(params["directory"] + "/" + case_id + "/" + case_id + "dat", "w")
    fp.write("*\n")
    fp.write("*\n")
    fp.write("*        FICHERO DE DATOS PARA CALIBRACION\n")
    fp.write("*\n")
    fp.write("F(2F10.3,3I5)\n")
    fp.write("*        IT      = INTERVALO DE TIEMPO\n")
    fp.write("*        ROZA    = RUGOSIDAD DE CHEZY --> 1/Mannig\n")
    fp.write("*        NT      = NUMERO DE ESCRITURAS EN FICHERO\n")
    fp.write("*        REPE    = NUMERO DE ITERACIONES ENTRE LAS ESCRITURAS\n")
    fp.write("*        IESDAO = NUMERO DE REPEs HASTA LA PRIMERA ESCRITURA\n")
    fp.write("*\n")
    fp.write("*        EN TOTAL LAS ITERACIONES SON --> ((NT-1)*REPE + IESDAO*REPE)\n")
    fp.write(
        "*        HAY QUE CUMPLIR LA CONDICION --> (NN >= (NT-1)*REPE + IESDAO*REPE)\n"
    )
    fp.write("*\n")
    fp.write("*      IT      ROZA    NT REPE IESDAO\n")
    fp.write("******.***######.###*****#####*****\n")
    fp.write("     0.500    15.000    1 1000    1\n")
    fp.write("*\n")
    fp.write("*      EDDY = FACTOR EDDY VISCOSITY\n")
    fp.write("*      CORI = FACTOR DE CORIOLIS\n")
    fp.write("*      NINTER= NUMERO ITERACIONES EN TERMINOS NO LINEALES\n")
    fp.write("F(2F10.3,I5)\n")
    fp.write("*     EDDY     CORI   NINTER\n")
    fp.write("    30.000     0.000    3\n")
    fp.write("* \n")
    fp.write("*       IANL  = TERMINOS NO LINEALES   (SI = 1)\n")
    fp.write("*       IAGUA = INUNDACION DE CELDAS   (SI = 1)\n")
    fp.write("*       ISLIP = CONTORNOS SIN FRICCION (SI = 1)\n")
    fp.write("F(3I5)\n")
    fp.write("* IANL IAGUA ISLIP\n")
    fp.write("    1    0    0\n")
    fp.write("*\n")
    fp.write("*\n")
    fp.write(
        "*      COORDENADAS DE PUNTOS DONDE SE DESEE TENER UN FICHERO EN EL TIEMPO     \n"
    )
    fp.write("*      DE SUPERFICIE LIBRE (ETA), VELOCIDAD (U), VELOCIDAD (V). \n")
    fp.write("F(I5)\n")
    fp.write("*NUMERO DE PUNTOS (MAXIMO 30 PUNTOS)\n")
    fp.write("    0\n")

    return


def directory(params, data, global_db, local_db):
    """Create project directory structure with initialized SWAN model files.
    
    Sets up the complete directory tree for a SWAN modeling project, creating
    folders for each simulation case and initializing bathymetry and input
    files for nested grid computations.

    Parameters
    ----------
    params : dict
        Model configuration parameters containing:
        
        - directory : str
            Root directory path for the project
            
    data : pd.DataFrame
        Time series with boundary conditions, indexed by timestamps.
        Each row represents one simulation case.
    global_db : xr.Dataset
        Global (coarse) grid dataset containing:
        
        - depth : xr.DataArray
            Bathymetry on global grid (m)
            
    local_db : xr.Dataset
        Local (fine) grid dataset containing:
        
        - depth : xr.DataArray
            Bathymetry on local nested grid (m)

    Returns
    -------
    None
        Creates directory structure:
        
        - params['directory']/: Root project folder
        - params['directory']/####/: Case folders (0001, 0002, ...)
        
        Each case folder contains:
        
        - ####_global.dat: Global grid bathymetry
        - ####_local.dat: Local grid bathymetry
        - swaninit: SWAN initialization file
        - input_global_####.swn: SWAN input file

    Notes
    -----
    The function performs the following operations:
    
    1. Creates root project directory (if it doesn't exist)
    2. Loops through all time steps in data
    3. For each case:
    
       - Creates numbered subdirectory (####)
       - Writes global and local bathymetry files
       - Generates SWAN input files for nested run
       
    Case numbering is 1-indexed with zero-padding to 4 digits (0001-9999).
    
    The nested flag in write_swan ensures boundary condition files are
    generated for downscaling from global to local grid.

    Examples
    --------
    >>> import pandas as pd
    >>> import xarray as xr
    >>> import numpy as np
    >>> 
    >>> # Create sample datasets
    >>> data = pd.DataFrame({
    ...     'Hs': [2.0, 2.5, 3.0],
    ...     'Tp': [8.0, 9.0, 10.0],
    ...     'DirM': [45, 60, 75]
    ... }, index=pd.date_range('2024-01-01', periods=3, freq='6h'))
    >>> 
    >>> global_db = xr.Dataset({
    ...     'depth': (['y', 'x'], np.random.rand(50, 100) * 10)
    ... })
    >>> local_db = xr.Dataset({
    ...     'depth': (['y', 'x'], np.random.rand(100, 200) * 10)
    ... })
    >>> 
    >>> params = {'directory': './swan_project'}
    >>> directory(params, data, global_db, local_db)
    >>> # Creates: ./swan_project/0001/, ./swan_project/0002/, ./swan_project/0003/
    """
    os.makedirs(params["directory"], exist_ok=True)
    for ind_, time in enumerate(data.index):
        case_id = str(ind_ + 1).zfill(4)
        os.makedirs(params["directory"] + "/" + case_id, exist_ok=True)
        save.to_txt(
            params["directory"] + "/" + case_id + "/" + case_id + "_global.dat",
            global_db["depth"].data[:, :],
            format="%9.3f",
        )
        save.to_txt(
            params["directory"] + "/" + case_id + "/" + case_id + "_local.dat",
            local_db["depth"].data[:, :],
            format="%9.3f",
        )
        write_swan(ind_, time, case_id, data, params, mesh="global", nested=True)

    return

