"""Numerical models: set-up, execution and reading of results.

Submodules:
    common      Structured grids (Delft3D .grd/.dep, regular grids) and process execution.
    swan        SWAN projects driven by a YAML file: prepare → run → read.
    swan_copla  Coupled SWAN → COPLA → Kobayashi workflow on regular grids.
    delft3d     Delft3D outputs (raw exports, DELFT3D-WAVE NEFIS .dat) and figures.
    cshore      CSHORE cross-shore profile model.
    coastalme   CoastalME coastal morphology model.
"""

from . import coastalme, common, cshore, delft3d, swan, swan_copla

__all__ = ["common", "swan", "swan_copla", "delft3d", "cshore", "coastalme"]
