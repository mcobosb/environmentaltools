"""Readers of model outputs, moved to :mod:`environmentaltools.models` (deprecated aliases)."""

from environmentaltools._deprecation import moved

_SWAN_COPLA = "environmentaltools.models.swan_copla"
_CSHORE = "environmentaltools.models.cshore"
_DELFT3D = "environmentaltools.models.delft3d"

MOVED = {
    "create_mesh_dictionary": (_SWAN_COPLA, "create_mesh_dictionary"),
    "cshore_config": (_CSHORE, "default_config"),
    "read_cshore": (_CSHORE, "read_output"),
    "read_copla": (_SWAN_COPLA, "read_copla"),
    "read_swan": (_SWAN_COPLA, "read_swan"),
    "delft_raw_files_point": (_DELFT3D, "raw_files_point"),
    "delft_raw_files": (_DELFT3D, "raw_files"),
}

__getattr__ = moved(__name__, MOVED)
