"""Hydraulics, river sediment transport, settling velocity and coastline evolution."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from environmentaltools.processes import (
    coastline_evolution,
    fall_velocity,
    hydraulic_radius,
    river_sediment_transport,
    settling_velocity,
    water_elevation,
)


@pytest.mark.parametrize("section, info, area, perimeter, top", [
    ("rectangular", {"b": 3, "y": 1.5}, 4.5, 6.0, 3.0),
    ("trapezoidal", {"b": 3, "z": 2, "y": 1.5}, 9.0, 3 + 3 * np.sqrt(5), 9.0),
    ("triangular", {"z": 1, "y": 2}, 4.0, 4 * np.sqrt(2), 4.0),
    ("circular", {"D": 1, "y": 0.5}, np.pi / 8, np.pi / 2, 1.0),
    ("parabolic", {"T": 6, "y": 1.5}, 6.0, 6 + 8 * 1.5**2 / 18, 6.0),
])
def test_hydraulic_radius(section, info, area, perimeter, top):
    df = hydraulic_radius(info, section)
    assert df.loc[0, "wet_area"] == pytest.approx(area)
    assert df.loc[0, "wet_perimeter"] == pytest.approx(perimeter)
    assert df.loc[0, "rh"] == pytest.approx(area / perimeter)
    assert df.loc[0, "water_mirror"] == pytest.approx(top)


@pytest.mark.parametrize("section", ["rectangular", "trapezoidal", "triangular", "circular", "parabolic"])
def test_water_elevation_recovers_depth(section):
    params = dict(n=0.015, S=0.001, w=5, z=2, D=3, T=6)
    geometry = hydraulic_radius(dict(params, b=5, y=1.0), section).iloc[0]
    Q = 1 / params["n"] * geometry.wet_area * geometry.rh ** (2 / 3) * params["S"] ** 0.5
    res = minimize_scalar(water_elevation(section), bounds=(0.01, 2.9), args=(Q, params), method="bounded")
    assert res.x == pytest.approx(1.0, abs=1e-3)


RIVER = dict(rho_w=1000, rho_s=2650, sg=2.65, d50=0.002, S=0.001, w=10, nu=1e-6, n=0.03,
             ds_substrate=np.array([0.0005, 0.001]), frac_substrate=np.array([0.5, 0.5]))


@pytest.mark.parametrize("method", ["meyer-peter-muller", "einstein-brown", "wilcock-crowe", "bagnold", "yang",
                                    "brownlie"])
def test_river_sediment_transport(method):
    qb = river_sediment_transport(pd.DataFrame({"h": [0.5, 1.0, 2.0]}), dict(RIVER), method).to_numpy()
    assert np.isfinite(qb).all() and (qb >= 0).all()
    assert (np.diff(qb) > 0).all()  # more transport with more depth


def test_meyer_peter_muller_value():
    qb = river_sediment_transport(pd.DataFrame({"h": [1.0]}), dict(RIVER), "meyer-peter-muller")
    tau_shi = 1000 * 9.81 * 1.0 * 0.001 / (9.81 * 0.002 * 1000 * 1.65)
    expected = 8 * (tau_shi - 0.047) ** 1.5 * 0.002 * (9.81 * 1.65 * 0.002) ** 0.5 * 3600 * 10
    assert qb.iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("d", [0.0002, 0.001, 0.004])
def test_wu_wang_settling_velocity_close_to_soulsby(d):
    ws = settling_velocity(d, {"nu": 1e-6, "sg": 2.65, "Sp": 0.7}, "Wu_Wang")
    assert ws == pytest.approx(fall_velocity(d * 1000, 20, 35), rel=0.25)


def test_coastline_evolution():
    coastlines = {year: pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [y] * 3})
                  for year, y in [(2000, 0.0), (2001, 1.0), (2002, -2.0)]}
    evolution = coastline_evolution(coastlines, [(1, 0), (2, 0.1)])
    assert list(evolution.index) == [2001, 2002]
    assert evolution.loc[2001].tolist() == [-1.0, -1.0]
    assert evolution.loc[2002].tolist() == [2.0, 2.0]
