"""Delft3D: reading of outputs and figures of grids and DELFT3D-WAVE fields.

Typical use::

    from environmentaltools.models import delft3d

    df = delft3d.extract_wave_dat(cases_dir, "wavm-guad-Alboran_int.dat", "Alboran_int.grd",
                                  points, ["hsign", "dir", "period"])
    delft3d.plot_grids({"ext": "Alboran_ext.grd", "int": "Alboran_int.grd"})
    delft3d.plot_wave_field("caso_001/wavm-guad-Alboran_int.dat", "caso_001/Alboran_int.grd", "hsign")

Grid files (``.grd``/``.dep``) are read with :mod:`environmentaltools.models.common`.
"""

from .read import extract_wave_dat, raw_files, raw_files_point
from .plot import plot_grids, plot_wave_field, plot_wave_field_grid

__all__ = [
    "extract_wave_dat", "raw_files", "raw_files_point",
    "plot_grids", "plot_wave_field", "plot_wave_field_grid",
]
