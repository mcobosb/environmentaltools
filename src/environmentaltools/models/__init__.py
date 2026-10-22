"""Numerical models: set-up, execution and reading of results.

Submodules:
    common      Structured grids (Delft3D .grd/.dep, regular grids) and process execution.
    swan        SWAN projects driven by a YAML file: prepare → run → read.
    swan_copla  Coupled SWAN → COPLA → Kobayashi workflow on regular grids.
    delft3d     Delft3D outputs (raw exports, DELFT3D-WAVE NEFIS .dat) and figures.
    cshore      CSHORE cross-shore profile model.
    coastalme   CoastalME coastal morphology model.

Which model to use:
    swan        Wave propagation in 2-D (refraction, diffraction, nesting) over
                real bathymetry; waves only, no currents or sediment transport.
    swan_copla  Sediment transport field in plan (2-D): longshore transport,
                circulation cells, effect of structures. Depth-averaged currents,
                so no undertow (offshore cross-shore transport is underestimated),
                and the bed is not updated.
    cshore      Cross-shore profile (1-D, alongshore uniform beach): surf and swash
                zone, undertow, runup, overtopping and profile change during storms.
"""

from . import coastalme, common, cshore, delft3d, swan, swan_copla

__all__ = ["common", "swan", "swan_copla", "delft3d", "cshore", "coastalme"]
