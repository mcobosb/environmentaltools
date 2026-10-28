"""Writers of model inputs, moved to :mod:`environmentaltools.models` (deprecated aliases)."""

from environmentaltools._deprecation import moved
from environmentaltools.processes._moved import WRITE

__getattr__ = moved(__name__, WRITE)
