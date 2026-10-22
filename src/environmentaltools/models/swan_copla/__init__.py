"""Coupled SWAN → COPLA → Kobayashi workflow on regular grids defined in a parameter dict.

For each sea state, SWAN propagates the waves on a global grid and, nested, on a
local grid; COPLA computes the wave-induced currents from the local SWAN
results; the sediment transport follows Kobayashi et al. (2008)
(:func:`environmentaltools.processes.kobayashi_transport`).

Typical use::

    from environmentaltools.models import swan_copla

    params = swan_copla.create_mesh_dictionary("params.xlsx", "values")
    global_db = swan_copla.create_db(params, bathymetry, "global")
    local_db = swan_copla.create_db(params, bathymetry, "local")
    swan_copla.create_project_directory(params, sea_states, global_db, local_db)
    for k, label in enumerate(sea_states.index):
        grid = swan_copla.nesting(k, label, params, sea_states)

For new SWAN projects (curvilinear grids, nesting, batches of cases) use
:mod:`environmentaltools.models.swan`.
"""

from .grids import create_db, create_mesh, create_xarray, save_db
from .inputs import create_mesh_dictionary, create_project_directory, write_copla_input, write_swan_input
from .read import read_copla, read_swan
from .run import clean, nesting, run_copla, run_swan

__all__ = [
    "create_db", "create_mesh", "create_xarray", "save_db",
    "create_mesh_dictionary", "create_project_directory", "write_copla_input", "write_swan_input",
    "read_copla", "read_swan",
    "clean", "nesting", "run_copla", "run_swan",
]
