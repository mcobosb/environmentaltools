import numpy as np
from environmentaltools.common import save
from scipy.io import loadmat as ldm
import os


def write_cshore_input(properties: dict, output_folder: str):
    """Write CSHORE model input file.
    
    Creates the infile for the CSHORE coastal hydrodynamics model with
    the specified configuration parameters.
    
    Parameters
    ----------
    properties : dict
        Dictionary containing CSHORE model parameters including:
        
        - header: Model header text
        - iline, iprofl, isedav, iperm, iover, iwtran, ipond, infilt, 
          iwcint, iroll, iwind, itide, iveg: Integer flags for model options
        - dx: Spatial grid spacing
        - gamma: Wave breaking parameter
        - d50, wf, sg: Sediment parameters (grain size, fall velocity, specific gravity)
        - effb, efff, slp: Efficiency and slope parameters
        - and other model-specific parameters
        
    output_folder : str
        Path to folder where infile will be created.
    """
    fid = open(output_folder + "/infile", "w")
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

    fid = open(output_folder + "/infile", "a")
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


def write_swan_input(case_index: int, time_index, case_id: str, data, params: dict, 
                     mesh: str = "global", local: bool = False, nested: bool = False):
    """Write SWAN wave model input files.

    Creates initialization and input files for the SWAN (Simulating WAves Nearshore)
    model for a specific case.

    Args:
        case_index (int): Sequential case number (0-based index).
        time_index: Timestamp from the data index for this case.
        case_id (str): String identifier for this case (e.g., '0001', '0002').
        data (pd.DataFrame): Time series of boundary condition data.
        params (dict): Dictionary with model parameters including:
            - directory: Root directory for output files
            - Mesh configuration parameters
            - Wave and wind parameters
        mesh (str, optional): Mesh type identifier ('global' or 'local'). 
            Defaults to 'global'.
        local (bool, optional): Whether to use local mesh configuration. 
            Defaults to False.
        nested (bool, optional): Whether to enable nested grid mode. 
            Defaults to False.
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
        "SET LEVEL " + str(np.round(data.loc[time_index, "eta"], decimals=2)) + "\n"
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
        + str(np.round(data.loc[time_index, "Vv"], decimals=2))
        + " "
        + str(np.round(np.remainder(270 - data.loc[time_index, "DirV"], 360), decimals=2))
        + "\n"
    )
    imswfile.write("$\n")

    if mesh == "global":
        # Here, direction has PdE convention (waves coming from N: 0º, E: 90º)
        if data.loc[time_index, "DirM"] < 90:
            side = ["N", "E"]
        elif (data.loc[time_index, "DirM"] < 180) & (data.loc[time_index, "DirM"] >= 90):
            side = ["S", "E"]
        elif (data.loc[time_index, "DirM"] < 270) & (data.loc[time_index, "DirM"] >= 180):
            side = ["S", "W"]
        else:
            side = ["N", "W"]
        imswfile.write("BOUN SHAPESPEC JONSWAP PEAK DSPR POWER\n")
        imswfile.write("$                             Hs	Tp	Dir	dd(spreading power)\n")
        imswfile.write(
            "BOUN SIDE "
            + side[0]
            + " CONSTANT PAR  "
            + str(np.round(data.loc[time_index, "Hs"], decimals=2))
            + " "
            + str(np.round(data.loc[time_index, "Tp"], decimals=2))
            + " "
            + str(
                np.round(np.remainder(270 - data.loc[time_index, "DirM"], 360), decimals=2)
            )
            + " 2.00\n"
        )
        imswfile.write(
            "BOUN SIDE "
            + side[1]
            + " VARIABLE PAR  200 "
            + str(np.round(data.loc[time_index, "Hs"], decimals=2))
            + " "
            + str(np.round(data.loc[time_index, "Tp"], decimals=2))
            + " "
            + str(
                np.round(np.remainder(270 - data.loc[time_index, "DirM"], 360), decimals=2)
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


def write_copla_input(case_index: int, time_index, case_id: str, data, params: dict, mesh: str = "local"):
    """Write COPLA wave propagation model input files.

    Creates input files for the COPLA (Coastal Propagation of LArge waves) model,
    using SWAN output as boundary conditions.

    Args:
        case_index (int): Sequential case number (0-based index).
        time_index: Timestamp from the data index for this case.
        case_id (str): String identifier for this case (e.g., '0001', '0002').
        data (pd.DataFrame): Time series of boundary condition data.
        params (dict): Dictionary with model parameters including:
            - directory: Root directory for input/output files
            - Mesh configuration parameters
        mesh (str, optional): Mesh type identifier. Defaults to 'local'.
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

    # Write input file
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
    fp.write(str(data.loc[time_index, "Tp"]) + " " + str(data.loc[time_index, "eta"]) + "\n")
    fp.write("amp dir(grados)\n")
    fp.write(
        str(data.loc[time_index, "Hs"] / 2)
        + " "
        + str(180 - data.loc[time_index, "DirM"])
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


def create_project_directory(params: dict, data, global_db, local_db):
    """Create project folder structure with initialized files for SWAN and COPLA models.

    Generates a directory structure with subdirectories for each time step,
    containing bathymetry files and SWAN input files.

    Args:
        params (dict): Dictionary with model parameters including:
            - directory: Root directory path for the project
        data (pd.DataFrame): Time series of boundary condition data with datetime index.
        global_db (xr.Dataset): Global mesh bathymetry dataset with 'depth' variable.
        local_db (xr.Dataset): Local mesh bathymetry dataset with 'depth' variable.

    Returns:
        None: Creates directory structure and files as side effect.
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
        write_swan_input(ind_, time, case_id, data, params, mesh="global", nested=True)

    return