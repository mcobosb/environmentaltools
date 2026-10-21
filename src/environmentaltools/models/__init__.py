"""Numerical models: set-up, execution and reading of results.

Submodules:
    common  Structured grids (Delft3D .grd/.dep, regular grids) and process execution.
    swan    SWAN projects driven by a YAML file: prepare → run → read.
"""

from . import common, swan

__all__ = ["common", "swan"]
