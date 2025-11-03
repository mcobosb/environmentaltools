"""Processes module for environmental model computation and I/O.

This module provides tools for:
- Model execution (SWAN, read_copla, CSHORE, CoastalME)
- Data processing and computation
- Wave analysis and calculations
- File I/O for various model formats
"""

# Data loading functions
from .load import (
    create_mesh_dictionary,
    read_cshore,
    read_copla,
    read_swan,
    delft_raw_files_point,
    delft_raw_files,
)

# Computation functions
from .compute import (
    create_db,
    create_mesh,
    create_xarray,
    slopes,
    sediment_transport_Kobayashi,
    sediment_transport_CERC,
    nesting,
    run_swan,
    run_copla,
    run_cshore,
    run_coastalme,
    save_db,
    clean,
    equilibrium_plan_shape,
    coastline_evolution,
    precipitation_to_flow,
    wet_soil,
    dry_soil,
    unit_hydrograph_model,
    base_flow,
    distribute_precipitation,
    cumulative_by_events,
    hydraulic_radius,
    water_elevation,
    settling_velocity,
    river_sediment_transport,
    storm_surge_from_waves,
    flood_fill,
    EOS_sea_water,
    bulk_fluid_density,
)

# Wave analysis functions
from .waves import (
    frequency_limits,
    clsquare_s,
    wave_number,
    calculate_wave_reflection,
    closure_depth,
    fall_velocity,
    density,
    kinematic_viscosity,
    zero_cross,
)

# File writing functions
from .write import (
    write_cshore,
    write_swan,
    write_copla,
    directory,
)

__all__ = [
    # Load
    "create_mesh_dictionary",
    "read_cshore",
    "read_copla",
    "read_swan",
    "delft_raw_files_point",
    "delft_raw_files",
    # Compute - Data processing
    "create_db",
    "create_mesh",
    "create_xarray",
    "slopes",
    "save_db",
    "clean",
    # Compute - Sediment transport
    "sediment_transport_Kobayashi",
    "sediment_transport_CERC",
    # Compute - Model execution
    "nesting",
    "run_swan",
    "run_copla",
    "run_cshore",
    "run_coastalme",
    # Compute - Coastal morphology
    "equilibrium_plan_shape",
    "coastline_evolution",
    # Compute - Hydrology
    "precipitation_to_flow",
    "wet_soil",
    "dry_soil",
    "unit_hydrograph_model",
    "base_flow",
    "distribute_precipitation",
    "cumulative_by_events",
    "hydraulic_radius",
    "water_elevation",
    "settling_velocity",
    "river_sediment_transport",
    "storm_surge_from_waves",
    "flood_fill",
    # Compute - Physical properties
    "EOS_sea_water",
    "bulk_fluid_density",
    # Waves
    "frequency_limits",
    "clsquare_s",
    "wave_number",
    "calculate_wave_reflection",
    "closure_depth",
    "fall_velocity",
    "density",
    "kinematic_viscosity",
    "zero_cross",
    # Write
    "write_cshore",
    "write_swan",
    "write_copla",
    "directory",
]
