"""Tests of environmentaltools.models.swan (no SWAN executable needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from environmentaltools.models import swan
from environmentaltools.models.common import Grid


@pytest.fixture
def project(tmp_path):
    xs, ys = np.meshgrid(np.linspace(-1000, 11000, 40), np.linspace(-1000, 9000, 40))
    np.savetxt(tmp_path / "bathy.xyz", np.column_stack([xs.ravel(), ys.ravel(), 50 - ys.ravel() / 200]))
    pd.DataFrame({"name": ["A", "B"], "x": [5000.0, 7000.0], "y": [6000.0, 7000.0]}).to_csv(
        tmp_path / "points.csv", index=False)
    pd.DataFrame({"id": [1, 2], "Hs": [1.5, 3.0], "Tp": [6.0, 9.0], "DirM": [250.0, 80.0],
                  "Vv": [0.0, 12.0], "DirV": [260.0, 70.0]}).to_csv(tmp_path / "states.csv", index=False)
    cfg = {
        "project": "test",
        "grids": {
            "outer": {"regular": {"xp": 0, "yp": 0, "alp": 0, "dx": 500, "dy": 500, "nx": 21, "ny": 17},
                      "bathymetry": {"file": "bathy.xyz"},
                      "boundaries": [{"side": "S"},
                                     {"segment_ij": [20, 0, 20, 16],
                                      "variable": [{"s": 0, "hs": 0.8, "dir": 10}, {"s": 8000, "hs": 1.0}]}]},
            "inner": {"regular": {"xp": 3000, "yp": 4000, "alp": 0, "dx": 100, "dy": 100, "nx": 41, "ny": 31},
                      "bathymetry": {"file": "bathy.xyz"}, "nested_in": "outer"},
        },
        "outputs": {"points": "points.csv", "maps": {"inner": {"quantities": ["XP", "YP", "HSIGN"]}}},
        "states": {"file": "states.csv", "id": "id"},
        "variants": {"S1": {}, "S0": {"physics": {"wcap": "WCAP KOMEN delta=1", "setup": None}}},
    }
    import yaml

    f = tmp_path / "swan.yaml"
    f.write_text(yaml.safe_dump(cfg))
    return swan.load_config(f)


def test_grid_order_and_perimeter(project):
    assert project.grid_names == ["outer", "inner"]
    g = project.grids()["inner"]
    j, i = g.perimeter()
    assert len(j) == 2 * (41 + 31) - 4
    assert (j[0], i[0]) == (0, 0) and (j[41], i[41]) == (1, 40)


def test_prepare_writes_cases(project):
    man = swan.prepare(project, verbose=False)
    assert len(man) == 4 and set(man.variant) == {"S0", "S1"}
    root = project.cases_dir
    common = root / "_common"
    assert (common / "nest_inner.pts").exists() and (common / "points.pts").exists()
    s1 = (root / "1_S1" / "INPUT_1_outer").read_bytes().decode()
    assert "\r\n" in s1
    s1 = s1.replace("\r\n", "\n")
    assert "CGRID REGULAR 0 0 0 10000 8000 20 16" in s1
    assert "BOUN SIDE S CCW CONSTANT PAR 1.50 6.00 250.00 4.00" in s1
    assert "VARIABLE PAR 0.0 1.20 6.00 260.00 4.00 _" in s1
    assert "WIND 0.01 260.00" in s1                       # min_wind
    assert "SPEC 'NEST1' SPEC2D ABS 'nest_inner.sp2'" in s1
    assert "WCAP KOMEN delta=0" in s1 and "SETUP" in s1
    s0 = (root / "1_S0" / "INPUT_1_outer").read_text()
    assert "WCAP KOMEN delta=1" in s0 and "SETUP" not in s0
    inner = (root / "2_S1" / "INPUT_2_inner").read_text()
    assert "BOUN NEST 'nest_inner.sp2' CLOSED" in inner and "TABLE 'COMPGRID' NOHEAD 'map_inner.tab'" in inner
    meta = json.loads((root / "2_S1" / "case.json").read_text())
    assert [s["grid"] for s in meta["stages"]] == ["outer", "inner"]


def test_read_table_and_print(tmp_path):
    (tmp_path / "p.tab").write_text(
        "%\n% Run:  Table:PTS  SWAN version:41.51.3\n%\n%  Xp  Yp  Hsig  X-Windv  Y-Windv\n%  [m] [m] [m] [m/s] [m/s]\n%\n"
        "  1.0  2.0  0.50  1.0  2.0\n  3.0  4.0  -9.0  1.0  2.0\n")
    t = swan.read_table(tmp_path / "p.tab", ["XP", "YP", "HSIGN", "WIND"])
    assert list(t.columns) == ["x", "y", "hs", "wind_x", "wind_y"] and np.isnan(t.hs[1])
    assert swan.read_table(tmp_path / "p.tab").columns[2] == "Hsig"
    (tmp_path / "PRINT").write_text("DELTARES VERSION NUMBER 41.51.3\n iteration    7; sweep 4\n"
                                    " accuracy OK in  97.50 % of wet grid points ( 97.00 % required)\n")
    p = swan.read_print(tmp_path / "PRINT")
    assert p["converged"] and p["iterations"] == 7 and p["version"] == "41.51.3"


def test_delft_grid_roundtrip(tmp_path):
    g = Grid.regular_grid(0, 0, 30, 10, 20, 5, 4)
    lines = ["Coordinate System = Cartesian", "Missing Value = -999", "       5       4", " 0 0 0"]
    for arr in (g.x, g.y):
        for j in range(4):
            lines.append(f" ETA=    {j + 1}   " + "   ".join(f"{v:.6f}" for v in arr[j]))
    (tmp_path / "g.grd").write_text("\n".join(lines))
    (tmp_path / "g.dep").write_text("\n".join(" ".join(["10.0"] * 6) for _ in range(5)))
    h = Grid.from_delft(tmp_path / "g.grd", tmp_path / "g.dep")
    assert np.allclose(h.x, g.x, atol=1e-5) and h.depth.shape == (4, 5) and (h.mxc, h.myc) == (4, 3)


@pytest.mark.skipif(__import__("platform").system() == "Windows", reason="fake executable is a shell script")
def test_run_with_fake_executable(project, tmp_path):
    exe = tmp_path / "fake_swan"
    exe.write_text("#!/bin/sh\n"
                   "for f in $(grep -o \"'[a-z_]*\\.\\(tab\\|sp2\\)'\" INPUT | tr -d \"'\"); do printf '  1.0 2.0 0.5\\n  3.0 4.0 0.7\\n' > $f; done\n"
                   "printf 'DELTARES VERSION NUMBER 41.51.3\\n iteration 3; sweep 4\\n"
                   " accuracy OK in  98.00 %% of wet grid points ( 97.00 %% required)\\n' > PRINT\n")
    exe.chmod(0o755)
    cfg = swan.load_config(project.file, {"executable": str(exe), "outputs": {"point_quantities": ["XP", "YP", "HSIGN"]},
                                          "variants": {}})
    swan.prepare(cfg, verbose=False)
    log = swan.run(cfg, filter="1", verbose=False)
    assert log.ok.all() and (cfg.results_dir / "1" / "done.json").exists()
    assert swan.run(cfg, verbose=False).shape[0] == 1          # only case "2" was pending
    s = swan.summary(cfg)
    assert len(s) == 4 and s.converged.all()
    pts = swan.read_points(cfg)
    assert set(pts.grid) == {"outer", "inner"} and len(pts) == 2 * 2 * 2 and "Hs" in pts   # cases × grids × points
