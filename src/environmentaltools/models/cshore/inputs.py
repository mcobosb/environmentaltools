"""Writing of the CSHORE input file (``infile``)."""

import numpy as np


def write_input(properties: dict, output_folder: str):
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
