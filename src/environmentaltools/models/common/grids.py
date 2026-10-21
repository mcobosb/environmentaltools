"""Structured grids shared by the numerical models (SWAN, Delft3D, ...).

A :class:`Grid` stores node coordinates ``x``, ``y`` and, optionally, the depth
``depth`` (positive downwards) as 2-D arrays indexed ``[j, i]``: ``i`` runs
fastest (SWAN ``ix``, Delft3D ``M``) and ``j`` is SWAN ``iy`` (Delft3D ``N``).
Inactive nodes have ``NaN`` coordinates.

Grids can be read from Delft3D/RGFGRID files (``.grd`` + ``.dep``) or built as
regular (optionally rotated) grids, with bathymetry interpolated from scattered
points.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MISSING = -999.999


def read_delft_grd(file_name):
    """Read a Delft3D/RGFGRID ``.grd`` file.

    Args:
        file_name (str | Path): Path to the ``.grd`` file.

    Returns:
        tuple: ``(x, y, spherical)`` with ``x`` and ``y`` arrays of shape
        ``(N, M)`` (``NaN`` at inactive nodes) and ``spherical`` True when the
        file declares ``Coordinate System = Spherical``.
    """
    lines = Path(file_name).read_text(errors="ignore").splitlines()
    spherical, missing = False, None
    k0 = None
    for k, line in enumerate(lines):
        low = line.lower()
        if low.startswith("coordinate system"):
            spherical = "spherical" in low
        elif low.startswith("missing value"):
            missing = float(line.split("=")[1])
        elif re.match(r"^\s*\d+\s+\d+\s*$", line) and k0 is None:
            m, n = (int(v) for v in line.split())
            k0 = k
        elif line.strip().startswith("ETA="):
            break
    first_eta = next(k for k, line in enumerate(lines) if line.strip().startswith("ETA="))
    values = []
    for line in lines[first_eta:]:
        values += [float(v) for v in re.sub(r"ETA=\s*\d+", "", line).split()]
    values = np.asarray(values)
    x = values[: m * n].reshape(n, m)
    y = values[m * n: 2 * m * n].reshape(n, m)
    if missing is not None:
        inactive = np.isclose(x, missing) | np.isclose(y, missing)
    else:                                   # old RGFGRID files: inactive nodes stored as (0, 0)
        inactive = (x == 0) & (y == 0)
    x, y = x.copy(), y.copy()
    x[inactive] = np.nan
    y[inactive] = np.nan
    return x, y, spherical


def read_delft_dep(file_name, shape):
    """Read a Delft3D ``.dep`` file and return the depth at the grid nodes.

    Delft3D stores ``(N + 1) × (M + 1)`` values (an extra dummy row and
    column); they are dropped. ``-999`` values become ``NaN``.

    Args:
        file_name (str | Path): Path to the ``.dep`` file.
        shape (tuple): Node shape ``(N, M)`` of the associated grid.

    Returns:
        np.ndarray: Depth (positive downwards) with shape ``(N, M)``.
    """
    n, m = shape
    values = np.array(Path(file_name).read_text().split(), float)
    if values.size == (n + 1) * (m + 1):
        depth = values.reshape(n + 1, m + 1)[:n, :m]
    elif values.size == n * m:
        depth = values.reshape(n, m)
    else:
        raise ValueError(f"{file_name}: {values.size} values do not match grid {n} × {m}")
    depth = depth.copy()
    depth[depth <= -999] = np.nan
    return depth


@dataclass
class Grid:
    """Structured (regular or curvilinear) grid.

    Attributes:
        x, y (np.ndarray): Node coordinates, shape ``(ny, nx)``.
        depth (np.ndarray | None): Depth at nodes (positive downwards).
        kind (str): ``"curvilinear"`` or ``"regular"``.
        spherical (bool): True for longitude/latitude coordinates.
        regular (dict | None): For regular grids, ``xp, yp, alp, dx, dy``.
        name (str): Grid name.
    """

    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray = None
    kind: str = "curvilinear"
    spherical: bool = False
    regular: dict = None
    name: str = "grid"
    source: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_delft(cls, grd, dep=None, name=None):
        """Build a curvilinear grid from Delft3D ``.grd`` (+ ``.dep``) files."""
        x, y, spherical = read_delft_grd(grd)
        depth = read_delft_dep(dep, x.shape) if dep else None
        return cls(x, y, depth, "curvilinear", spherical, None, name or Path(grd).stem,
                   {"grd": str(grd), "dep": str(dep) if dep else None})

    @classmethod
    def regular_grid(cls, xp, yp, alp, dx, dy, nx, ny, name="grid", spherical=False):
        """Build a regular grid.

        Args:
            xp, yp (float): Origin (node ``i = j = 0``).
            alp (float): Rotation of the x axis, degrees counter-clockwise from East.
            dx, dy (float): Node spacing along the grid axes.
            nx, ny (int): Number of nodes along x and y.
        """
        a = np.deg2rad(alp)
        ii, jj = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
        x = xp + ii * np.cos(a) - jj * np.sin(a)
        y = yp + ii * np.sin(a) + jj * np.cos(a)
        return cls(x, y, None, "regular", spherical,
                   {"xp": xp, "yp": yp, "alp": alp, "dx": dx, "dy": dy}, name)

    def set_depth_from_points(self, xs, ys, zs, method="linear", positive_down=True):
        """Interpolate the depth at the nodes from scattered points.

        Args:
            xs, ys, zs (array-like): Point coordinates and values.
            method (str): ``scipy.interpolate.griddata`` method; nodes left
                undefined are filled with the nearest value.
            positive_down (bool): False when ``zs`` are elevations (they are
                then multiplied by −1).
        """
        from scipy.interpolate import griddata

        pts = np.column_stack([np.ravel(xs), np.ravel(ys)])
        z = np.ravel(zs) * (1 if positive_down else -1)
        target = np.column_stack([self.x.ravel(), self.y.ravel()])
        d = griddata(pts, z, target, method=method)
        if np.isnan(d).any():
            d[np.isnan(d)] = griddata(pts, z, target[np.isnan(d)], method="nearest")
        self.depth = d.reshape(self.x.shape)
        return self

    # ------------------------------------------------------------- geometry
    @property
    def shape(self):
        return self.x.shape

    @property
    def mxc(self):
        """Number of meshes along i (SWAN ``mxc`` = nodes − 1)."""
        return self.x.shape[1] - 1

    @property
    def myc(self):
        """Number of meshes along j (SWAN ``myc`` = nodes − 1)."""
        return self.x.shape[0] - 1

    def perimeter(self):
        """Indices ``(j, i)`` of the perimeter nodes in Delft3D-WAVE nesting order.

        Order: row ``j = 0`` (i increasing), column ``i = nx − 1`` (j
        increasing), row ``j = ny − 1`` (i decreasing), column ``i = 0`` (j
        decreasing); every node appears once.
        """
        ny, nx = self.shape
        j = np.r_[np.zeros(nx, int), np.arange(1, ny), np.full(nx - 1, ny - 1), np.arange(ny - 2, 0, -1)]
        i = np.r_[np.arange(nx), np.full(ny - 1, nx - 1), np.arange(nx - 2, -1, -1), np.zeros(ny - 2, int)]
        return j, i

    def nearest_node(self, x, y, wet_only=True, depmin=0.05):
        """Nearest active (and optionally wet) node to a point.

        Returns:
            tuple: ``(j, i, distance)``.
        """
        d = np.hypot(self.x - x, self.y - y)
        d[np.isnan(d)] = np.inf
        if wet_only and self.depth is not None:
            d[~(self.depth > depmin)] = np.inf
        j, i = np.unravel_index(np.argmin(d), d.shape)
        return int(j), int(i), float(d[j, i])

    def distance_along(self, j, i):
        """Cumulative distance along a polyline of nodes (e.g. a boundary segment).

        Args:
            j, i (array-like): Node indices of the polyline, in order.

        Returns:
            np.ndarray: Distance from the first node (grid units).
        """
        xs, ys = self.x[j, i], self.y[j, i]
        return np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]

    def contains(self, x, y):
        """True if the point lies inside the outline of the grid."""
        from matplotlib.path import Path as MplPath

        j, i = self.perimeter()
        poly = np.column_stack([self.x[j, i], self.y[j, i]])
        poly = poly[~np.isnan(poly).any(axis=1)]
        return bool(MplPath(poly).contains_point((x, y)))
