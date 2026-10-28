"""Physical processes.

Submodules:
    waves: Wave theory, reflection analysis, zero crossing, closure depth and storm surge.
    coastal: Bottom slopes, sediment transport (Kobayashi, CERC), plan shape and shoreline evolution.
    hydrology: Rainfall-runoff, unit hydrograph, channel hydraulics and river sediment transport.
    properties: Density and viscosity of sea water, sediment fall velocity.

Model set-up, execution and outputs (SWAN, COPLA, CSHORE, CoastalME, Delft3D)
moved to :mod:`environmentaltools.models`; the old names still work as
deprecated aliases.
"""

from environmentaltools._deprecation import moved
from environmentaltools.processes._moved import COMPUTE, LOAD, WRITE

# Waves
from .waves import (
    frequency_limits,
    clsquare_s,
    wave_number,
    calculate_wave_reflection,
    closure_depth,
    zero_cross,
    storm_surge_from_waves,
)

# Coastal processes
from .coastal import (
    slopes,
    bed_slope,
    kobayashi_transport,
    sediment_transport_Kobayashi,
    sediment_transport_CERC,
    equilibrium_plan_shape,
    coastline_evolution,
    flood_fill,
)

# Hydrology
from .hydrology import (
    precipitation_to_flow,
    wet_soil,
    dry_soil,
    unit_hydrograph_model,
    base_flow,
    distribute_precipitation,
    cumulative_by_events,
    hydraulic_radius,
    water_elevation,
    river_sediment_transport,
)

# Physical properties
from .properties import (
    fall_velocity,
    density,
    kinematic_viscosity,
    settling_velocity,
    EOS_sea_water,
    bulk_fluid_density,
)

__all__ = [
    # Waves
    "frequency_limits",
    "clsquare_s",
    "wave_number",
    "calculate_wave_reflection",
    "closure_depth",
    "zero_cross",
    "storm_surge_from_waves",
    # Coastal processes
    "slopes",
    "bed_slope",
    "kobayashi_transport",
    "sediment_transport_Kobayashi",
    "sediment_transport_CERC",
    "equilibrium_plan_shape",
    "coastline_evolution",
    "flood_fill",
    # Hydrology
    "precipitation_to_flow",
    "wet_soil",
    "dry_soil",
    "unit_hydrograph_model",
    "base_flow",
    "distribute_precipitation",
    "cumulative_by_events",
    "hydraulic_radius",
    "water_elevation",
    "river_sediment_transport",
    # Physical properties
    "fall_velocity",
    "density",
    "kinematic_viscosity",
    "settling_velocity",
    "EOS_sea_water",
    "bulk_fluid_density",
]

# Moved to environmentaltools.models (deprecated aliases)
__getattr__ = moved(__name__, {**LOAD, **WRITE, **COMPUTE})
