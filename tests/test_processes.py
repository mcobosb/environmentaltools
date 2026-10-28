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


# --------------------------------------------------------------------------- Kobayashi (CSHORE)
from scipy.special import erf, erfc  # noqa: E402

from environmentaltools.processes import kobayashi_transport  # noqa: E402
from environmentaltools.processes.coastal import _friction_factor_gf  # noqa: E402

G = 9.81
KOB = dict(fb=0.015, wf=0.03, gamma=0.7, eb=0.005, ef=0.01, S=2.65, aK=0.5, d50=0.0003, Shic=0.05,
           bK=0.002, phi=np.arctan(0.63), local_angle=0.0, tburst=1.0)


def cshore_sedtra(H, Hrms, Tp, k, theta, umean, vmean, bslope, qbreak, p):
    """Transcription of CSHORE SEDTRA (IANGLE = 1, IROLL = 0) at one node of a cross-shore profile."""
    sqr2, sqrg1 = np.sqrt(2), np.sqrt(2 / np.pi)
    tanphi, fb2 = np.tan(p["phi"]), 0.5 * p["fb"]
    cth, sth = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    cp = 2 * np.pi / Tp / k
    sigt = cp * min(Hrms / np.sqrt(8) / H, 1.0)
    usta, vsta = min(umean / sigt, 1.0), vmean / sigt
    rb = np.sqrt(G * (p["S"] - 1) * p["d50"] * p["Shic"] / fb2) / sigt
    rs = p["wf"] / sigt / fb2 ** 0.3333
    wsta = usta * cth + vsta * sth
    vcus = vsta * cth - usta * sth
    fs, fbb = rs**2 - vcus**2, rb**2 - vcus**2
    ps = 1.0 if fs < 0 else 0.5 * (erfc((np.sqrt(fs) + wsta) / sqr2) + erfc((np.sqrt(fs) - wsta) / sqr2))
    pb = 1.0 if fbb < 0 else 0.5 * (erfc((np.sqrt(fbb) + wsta) / sqr2) + erfc((np.sqrt(fbb) - wsta) / sqr2))
    ps = min(ps, pb)
    hm = max(0.88 / k * np.tanh(p["gamma"] * k * H / 0.88), Hrms)
    abreak = max(2 * np.pi / k / H * bslope * cth / 3, 1.0)
    dbsta = 0.25 * abreak * qbreak * hm**2 / Tp
    gf = wsta * (wsta**2 + 3) * erf(wsta / sqr2) + (wsta**2 + 2) * sqrg1 * np.exp(-wsta**2 / 2)  # exact, V_cus = 0
    dfsta = fb2 * gf * sigt**3 / G
    vs = ps * (p["ef"] * dfsta + p["eb"] * dbsta) / (p["wf"] * (p["S"] - 1)) * np.sqrt(1 + bslope**2)
    gslope = tanphi / (tanphi + bslope) if bslope < 0 else (tanphi - 2 * bslope) / (tanphi - bslope)
    aslope = p["aK"] + (np.sqrt(bslope / tanphi) if bslope > 0 else 0.0)
    dum = p["bK"] / (G * (p["S"] - 1)) * (0.5 + qbreak) * pb * sigt**3
    qbx = dum * gslope * (1 + usta * vsta**2 + 2 * sth * vcus)
    qby = dum * (vsta * (1 + usta**2 + vsta**2) + 2 * sth * wsta)
    return {"Pb": pb, "Ps": ps, "Vs": vs, "Qbx": qbx, "Qby": qby, "Qsx": aslope * umean * vs, "Qsy": vmean * vs}


@pytest.mark.parametrize("theta, umean, vmean, bslope, ef", [
    (0.0, -0.2, 0.0, 0.02, 0.01),     # undertow, normal incidence
    (0.0, 0.1, 0.0, -0.01, 0.01),     # onshore current, bed descending (landward of a bar)
    (15.0, -0.15, 0.4, 0.03, 0.0),    # oblique waves and longshore current (CSHORE approximates GF: friction off)
    (-10.0, -0.1, -0.3, 0.05, 0.0),   # oblique from the other side
])
def test_kobayashi_matches_cshore(theta, umean, vmean, bslope, ef):
    p = dict(KOB, ef=ef)
    H, Hs, Tp, k, Qb = 2.0, 1.2, 8.0, 0.18, 0.3
    ref = cshore_sedtra(H, Hs / np.sqrt(2), Tp, k, theta, umean, vmean, bslope, Qb, p)
    a = lambda v: np.array([v])  # noqa: E731
    out = kobayashi_transport(a(H), a(0.0), Hs, Tp, a(k), a(theta), a(Qb), a(np.hypot(umean, vmean)),
                              a(np.degrees(np.arctan2(vmean, umean))), a(bslope), a(0.0), p)
    for key in ref:
        assert out[key][0] == pytest.approx(ref[key], rel=1e-9, abs=1e-15), key
    assert out["qx"][0] == pytest.approx(ref["Qbx"] + ref["Qsx"]) and out["qy"][0] == pytest.approx(ref["Qby"] + ref["Qsy"])


def test_kobayashi_is_rotation_invariant():
    args = dict(depth=np.array([2.0]), setup=np.array([0.0]), Hs=1.2, Tp=8.0, kp=np.array([0.18]),
                Qb=np.array([0.3]), U=np.array([0.3]), params=KOB)
    a = kobayashi_transport(wave_dir=np.array([10.0]), current_dir=np.array([40.0]),
                            Sbx=np.array([0.02]), Sby=np.array([0.0]), **args)
    b = kobayashi_transport(wave_dir=np.array([100.0]), current_dir=np.array([130.0]),
                            Sbx=np.array([0.0]), Sby=np.array([0.02]), **args)
    assert b["qx"][0] == pytest.approx(-a["qy"][0]) and b["qy"][0] == pytest.approx(a["qx"][0])


def test_bed_slope_on_rotated_grid():
    from environmentaltools.processes import bed_slope

    i, j = np.meshgrid(np.arange(5.0), np.arange(4.0))
    ang = np.radians(30)
    x, y = 10 * (i * np.cos(ang) - j * np.sin(ang)), 10 * (i * np.sin(ang) + j * np.cos(ang))
    depth = 5 - 0.02 * x + 0.01 * y           # zb = -5 + 0.02 x - 0.01 y
    sbx, sby = bed_slope(x, y, depth)
    assert np.allclose(sbx, 0.02) and np.allclose(sby, -0.01)


def test_friction_factor_matches_cshore_normal_incidence():
    u = np.linspace(-1.5, 1.5, 7)
    exact = u * (u**2 + 3) * erf(u / np.sqrt(2)) + (u**2 + 2) * np.sqrt(2 / np.pi) * np.exp(-u**2 / 2)
    assert np.allclose(_friction_factor_gf(u, 0 * u), exact, rtol=1e-10)
