"""Default CSHORE configuration."""


def default_config():
    """Create default CSHORE model configuration parameters.

    Generates a dictionary with default configuration parameters for the CSHORE
    (Cross-shore) numerical model, including morphology, wave, and sediment
    transport settings.

    Returns:
        dict: Dictionary containing CSHORE configuration parameters including:
            - Model flags (iline, iprofl, isedav, etc.)
            - Physical parameters (gamma, sporo, sg, etc.)
            - Boundary conditions (timebc_wave, swlbc, etc.)
            - Sediment properties (tanphi, blp, slp, etc.)
    """
    # Common configuration file
    props = {}
    props["iline"] = 1  # 1 = single line
    props["iprofl"] = (
        1  # 0 = no morph, 1 = run morph, 1.1 = run morph without initial smoothing
    )
    props["isedav"] = 0  # 0 = unlimited sand, 1 = hard bottom
    props["iperm"] = 0  # 0 = no permeability, 1 = permeable
    props["iover"] = 1  # 0 = no overtopping , 1 = include overtopping
    props["infilt"] = 0  # 1 = include infiltration landward of dune crest
    props["iwtran"] = (
        0  # 0 = no standing water landward of crest, 1 = wave transmission due to overtopping
    )
    props["ipond"] = 0  # 0 = no ponding seaward of SWL
    props["iwcint"] = 1  # 0 = no W & C interaction , 1 = include W & C interaction
    props["iroll"] = 1  # 0 = no roller, 1 = roller
    props["iwind"] = 0  # 0 = no wind effect
    props["itide"] = 0  # 0 = no tidal effect on currents
    props["iclay"] = 0  # Clay layer option
    props["iveg"] = 0  # Vegetation effect (0 = no vegetation, 1 = include vegetation)
    props["veg_Cd"] = 1  # Vegetation drag coefficient
    props["veg_n"] = 100  # Vegetation density (stems per m²)
    props["veg_dia"] = 0.01  # Vegetation stem diameter (m)
    props["veg_ht"] = 0.20  # Vegetation height (m)
    props["veg_rod"] = 0.1  # Vegetation erosion limit below sand for failure (m)
    props["veg_extent"] = [
        0.7,
        1,
    ]  # Vegetation coverage as fraction of total domain length
    props["gamma"] = 0.8  # shallow water ratio of wave height to water depth
    props["sporo"] = 0.4  # sediment porosity
    props["sg"] = 2.65  # specific gravity
    props["effb"] = 0.005  # suspension efficiency due to breaking eB
    props["efff"] = 0.01  # Suspension efficiency due to friction (ef)
    props["slp"] = 0.5  # Suspended load parameter
    props["slpot"] = 0.1  # Overtopping suspended load parameter
    props["tanphi"] = 0.630  # Tangent of sediment friction angle (degrees)
    props["blp"] = 0.001  # Bedload transport parameter
    props["rwh"] = 0.02  # Numerical runup wire height (m)
    props["ilab"] = 1  # Controls boundary condition timing (don't change)
    props["fric_fac"] = 0.015  # Bottom friction factor

    # Boundary conditions and timing parameters
    props["timebc_wave"] = 3600  # Wave boundary condition time interval (seconds)

    props["timebc_surg"] = props["timebc_wave"]  # Surge boundary condition time
    props["nwave"] = 1  # Number of wave time steps
    props["nsurg"] = props["nwave"]  # Number of surge time steps

    props["Wsetup"] = 0  # Wave setup at seaward boundary (meters)
    props["swlbc"] = 0.0  # Water level at seaward boundary (meters)

    return props
