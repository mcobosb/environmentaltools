"""Writers of model inputs, moved to :mod:`environmentaltools.models` (deprecated aliases)."""

from environmentaltools._deprecation import moved

_SWAN_COPLA = "environmentaltools.models.swan_copla"

MOVED = {
    "write_cshore_input": ("environmentaltools.models.cshore", "write_input"),
    "write_swan_input": (_SWAN_COPLA, "write_swan_input"),
    "write_copla_input": (_SWAN_COPLA, "write_copla_input"),
    "create_project_directory": (_SWAN_COPLA, "create_project_directory"),
}

__getattr__ = moved(__name__, MOVED)
