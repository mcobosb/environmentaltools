"""Readers of model outputs, moved to :mod:`environmentaltools.models` (deprecated aliases)."""

from environmentaltools._deprecation import moved
from environmentaltools.processes._moved import LOAD

__getattr__ = moved(__name__, LOAD)
