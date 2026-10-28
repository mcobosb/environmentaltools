"""Former home of the computation functions, now split by topic.

Kept so that ``environmentaltools.processes.compute.<name>`` keeps working:

- :mod:`~environmentaltools.processes.coastal`: slopes, sediment transport, plan shape, shoreline.
- :mod:`~environmentaltools.processes.hydrology`: rainfall-runoff and river hydraulics.
- :mod:`~environmentaltools.processes.properties`: sea water and sediment properties.
- :mod:`~environmentaltools.processes.waves`: storm surge from waves.
- Model set-up and execution moved to :mod:`environmentaltools.models` (deprecated aliases).
"""

from environmentaltools._deprecation import moved
from environmentaltools.processes._moved import COMPUTE
from environmentaltools.processes.coastal import (  # noqa: F401
    coastline_evolution,
    equilibrium_plan_shape,
    flood_fill,
    kobayashi_transport,
    sediment_transport_CERC,
    sediment_transport_Kobayashi,
    slopes,
)
from environmentaltools.processes.hydrology import (  # noqa: F401
    base_flow,
    cumulative_by_events,
    distribute_precipitation,
    dry_soil,
    hydraulic_radius,
    precipitation_to_flow,
    river_sediment_transport,
    unit_hydrograph_model,
    water_elevation,
    wet_soil,
)
from environmentaltools.processes.properties import EOS_sea_water, bulk_fluid_density, settling_velocity  # noqa: F401
from environmentaltools.processes.waves import storm_surge_from_waves  # noqa: F401

__getattr__ = moved(__name__, COMPUTE)
