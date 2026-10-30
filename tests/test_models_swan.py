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
    s1 = (root / "S1" / "1" / "INPUT_1_outer").read_bytes().decode()
    assert "\r\n" in s1
    s1 = s1.replace("\r\n", "\n")
    assert "CGRID REGULAR 0 0 0 10000 8000 20 16" in s1
    assert "BOUN SIDE S CCW CONSTANT PAR 1.50 6.00 250.00 4.00" in s1
    assert "VARIABLE PAR 0.0 1.20 6.00 260.00 4.00 _" in s1
    assert "WIND 0.01 260.00" in s1                       # min_wind
    assert "SPEC 'NEST1' SPEC2D ABS 'nest_inner.sp2'" in s1
    assert "WCAP KOMEN delta=0" in s1 and "SETUP" in s1
    s0 = (root / "S0" / "1" / "INPUT_1_outer").read_text()
    assert "WCAP KOMEN delta=1" in s0 and "SETUP" not in s0
    inner = (root / "S1" / "2" / "INPUT_2_inner").read_text()
    assert "BOUN NEST 'nest_inner.sp2' CLOSED" in inner and "TABLE 'COMPGRID' NOHEAD 'map_inner.tab'" in inner
    meta = json.loads((root / "S1" / "2" / "case.json").read_text())
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


@pytest.mark.skipif(__import__("platform").system() == "Windows", reason="fake executable is a shell script")
def test_archive_maps_and_figures(project, tmp_path):
    exe = tmp_path / "fake_swan"
    # points: 2 rows (x y hs); maps: the whole grid (x y hs) read from the coordinates of the regular grid
    exe.write_text("#!/bin/sh\n"
                   "for f in $(grep -o \"'points_[a-z]*\\.tab'\" INPUT | tr -d \"'\"); do printf '1 2 0.5\\n3 4 0.7\\n' > $f; done\n"
                   "for f in $(grep -o \"'map_[a-z]*\\.tab'\" INPUT | tr -d \"'\"); do "
                   "awk 'BEGIN{for(i=0;i<20;i++)for(j=0;j<20;j++)print 3000+100*i, 4000+100*j, 1+i/20}' > $f; done\n"
                   "printf ' iteration 3; sweep 4\\n accuracy OK in  98.00 %% of wet grid points ( 97.00 %% required)\\n' > PRINT\n")
    exe.chmod(0o755)
    cfg = swan.load_config(project.file, {
        "executable": str(exe), "archive_dir": str(tmp_path / "archive"), "variants": {},
        "outputs": {"point_quantities": ["XP", "YP", "HSIGN"],
                    "maps": {"inner": {"quantities": ["XP", "YP", "HSIGN"], "store": "archive"}},
                    "figures": {"quantity": "hs", "dpi": 50}}})
    swan.prepare(cfg, verbose=False, overwrite=True)
    log = swan.run(cfg, cases=["1"], verbose=False)
    assert log.ok.all() and "figure_error" not in log
    assert (tmp_path / "archive" / "1" / "map_inner.tab").exists()
    assert not (cfg.results_dir / "1" / "map_inner.tab").exists()
    assert (cfg.results_dir / "1" / "maps_1.png").exists()
    assert len(swan.read_map(cfg, "1", "inner")) == 400              # found in archive_dir
    assert swan.run(cfg, cases=["1"], force=True, verbose=False).ok.all()
    with pytest.raises(ValueError):
        swan.load_config(project.file, {"outputs": {"maps": {"inner": {"quantities": ["HSIGN"], "store": "archive"}}}})


def _fake_swan(tmp_path):
    exe = tmp_path / "fake_swan"
    exe.write_text("#!/bin/sh\n"
                   "for f in $(grep -o \"'[a-z_]*\\.\\(tab\\|sp2\\)'\" INPUT | tr -d \"'\"); do printf '  1.0 2.0 0.5\\n  3.0 4.0 0.7\\n' > $f; done\n"
                   "printf ' iteration 3; sweep 4\\n accuracy OK in  98.00 %% of wet grid points ( 97.00 %% required)\\n' > PRINT\n")
    exe.chmod(0o755)
    return exe


def test_prepare_only_redoes_what_changed(project, tmp_path):
    """A second prepare keeps everything; a changed state updates only its cases; overwrite keeps the results."""
    cfg = swan.load_config(project.file, {"variants": {}, "executable": str(_fake_swan(tmp_path)),
                                          "outputs": {"point_quantities": ["XP", "YP", "HSIGN"]}})
    assert set(swan.prepare(cfg, verbose=False)["status"]) == {"created"}
    assert cfg.overview_file.exists()
    swan.run(cfg, verbose=False)
    assert set(swan.prepare(cfg, verbose=False)["status"]) == {"kept"}
    assert set(swan.prepare(cfg, overwrite=True, verbose=False)["status"]) == {"kept"}
    assert (cfg.results_dir / "1" / "done.json").exists()                   # same content: result kept

    states = pd.read_csv(tmp_path / "states.csv")
    states.loc[states.id == 2, "Hs"] = 3.5
    states.to_csv(tmp_path / "states.csv", index=False)
    status = swan.prepare(cfg, verbose=False).set_index("case")["status"]
    assert status["2"] == "updated" and status["1"] == "kept"
    assert not (cfg.results_dir / "2" / "done.json").exists() and (cfg.results_dir / "1" / "done.json").exists()
    assert list(swan.run(cfg, verbose=False)["case"]) == ["2"]

    # a change of the grids/points (_common) updates every case
    pd.DataFrame({"name": ["A"], "x": [5000.0], "y": [6000.0]}).to_csv(tmp_path / "points.csv", index=False)
    assert set(swan.prepare(cfg, verbose=False)["status"]) == {"updated"}


def test_variants_table(project, tmp_path):
    pd.DataFrame({"variant": ["S1", "S0", "D2"], "description": ["base", "Komen delta 1", ""],
                  "physics.wcap": [None, "WCAP KOMEN delta=1", None], "physics.setup": [None, "off", None],
                  "boundary_spectrum.spreading_value": [None, None, 2]}).to_excel(tmp_path / "variants.xlsx", index=False)
    cfg = swan.load_config(project.file, {"variants": {}, "variants_file": "variants.xlsx"})
    assert cfg["variants"] == {"S1": {}, "S0": {"physics": {"wcap": "WCAP KOMEN delta=1", "setup": None}},
                               "D2": {"boundary_spectrum": {"spreading_value": 2}}}
    man = swan.prepare(cfg, verbose=False)
    assert sorted(man["case"]) == ["D2/1", "D2/2", "S0/1", "S0/2", "S1/1", "S1/2"]
    text = (cfg.cases_dir / "S0" / "1" / "INPUT_1_outer").read_text(encoding="latin-1")
    assert "WCAP KOMEN delta=1" in text and "SETUP" not in text
    overview = pd.read_excel(cfg.overview_file, sheet_name=None)
    assert {"cases", "variants", "grids", "legend"} <= set(overview)
    v = overview["variants"].set_index("parameter")
    assert v.loc["physics.wcap", "S0"] == "WCAP KOMEN delta=1" and v.loc["physics.wcap", "base"] == "WCAP KOMEN delta=0"
    with pytest.raises(ValueError):                       # same variant in the YAML and in the table
        swan.load_config(project.file, {"variants": {"S1": {}}, "variants_file": "variants.xlsx"})


def test_postprocess_and_command_line(project, tmp_path):
    from environmentaltools.models.swan.__main__ import main

    import yaml

    f = tmp_path / "swan_cli.yaml"
    f.write_text(yaml.safe_dump({**yaml.safe_load(project.file.read_text()), "variants": {},
                                 "executable": str(_fake_swan(tmp_path)),
                                 "outputs": {**yaml.safe_load(project.file.read_text())["outputs"], "maps": {},
                                             "point_quantities": ["XP", "YP", "HSIGN"]}}))
    main(["all", str(f)])
    cfg = swan.load_config(f)
    pts = pd.read_excel(cfg.postprocess_dir / "points.xlsx")
    assert len(pts) == 2 * 2 * 2 and {"case", "grid", "point", "hs", "Hs"} <= set(pts.columns)
    assert len(pd.read_excel(cfg.postprocess_dir / "convergence.xlsx")) == 4
    assert set(swan.postprocess(cfg, verbose=False)["status"]) == {"up to date"}


def test_executable_resolution(project, tmp_path, monkeypatch):
    exe = _fake_swan(tmp_path)
    monkeypatch.delenv("SWAN_EXE", raising=False)
    cfg = swan.load_config(project.file, {"executable": [str(tmp_path / "missing"), str(exe)]})
    assert swan.swan_executable(cfg) == exe
    monkeypatch.setenv("SWAN_EXE", str(exe))
    assert swan.swan_executable(swan.load_config(project.file, {"executable": str(tmp_path / "missing")})) == exe
    monkeypatch.delenv("SWAN_EXE")
    with pytest.raises(FileNotFoundError):
        swan.swan_executable(swan.load_config(project.file, {"executable": str(tmp_path / "missing")}))


def test_boundary_stretches_and_inspect_grids(project, tmp_path):
    """Stretches of water along the grid sides, from a project file still without boundaries."""
    from environmentaltools.models.common import Grid

    g = Grid.regular_grid(0, 0, 0, 100, 100, 6, 5, "g")
    g.depth = np.full(g.shape, 50.0)
    g.depth[:, 0] = [0, 0, 5, 6, 0]                                   # side i = 0: a short shallow stretch
    t = g.boundary_stretches()
    deep = t[t.side == "i = 5"].iloc[0]
    assert deep.segment_ij == [5, 0, 5, 4] and deep.nodes == 5 and deep.depth_mean == 50
    shallow = t[t.side == "i = 0"].iloc[0]
    assert shallow.segment_ij == [0, 2, 0, 3] and shallow.depth_max == 6

    import yaml

    raw = yaml.safe_load(project.file.read_text())
    for grid in raw["grids"].values():
        grid.pop("boundaries", None)
    f = tmp_path / "no_boundaries.yaml"
    f.write_text(yaml.safe_dump(raw))
    table = swan.inspect_grids(f, points=tmp_path / "points.csv", file_name=tmp_path / "grids.png",
                               table_file=tmp_path / "stretches.xlsx")
    assert set(table.grid) == {"outer", "inner"} and (tmp_path / "grids.png").exists()
    assert (tmp_path / "stretches.xlsx").exists()


def test_active_variants(project, tmp_path):
    """Only the active variants are prepared and run, each one in its own folder; switching a variant on or off
    does not change the cases of the others."""
    cfg = swan.load_config(project.file, {"executable": str(_fake_swan(tmp_path)), "active_variants": ["S1"],
                                          "outputs": {"point_quantities": ["XP", "YP", "HSIGN"]}})
    man = swan.prepare(cfg, verbose=False)
    assert list(man["case"]) == ["S1/1", "S1/2"] and not (cfg.cases_dir / "S0").exists()
    swan.run(cfg, verbose=False)
    assert (cfg.results_dir / "S1" / "1" / "done.json").exists()
    assert sorted(swan.read_points(cfg)["case"].unique()) == ["S1/1", "S1/2"]

    cfg = swan.load_config(project.file, {"executable": str(_fake_swan(tmp_path)),
                                          "outputs": {"point_quantities": ["XP", "YP", "HSIGN"]}})   # all variants
    status = swan.prepare(cfg, verbose=False).set_index("case")["status"]
    assert status["S1/1"] == "kept" and status["S0/1"] == "created"
    assert [c for c in swan.run(cfg, variants=["S0"], verbose=False)["case"]] == ["S0/1", "S0/2"]
    # a prepared variant that is no longer active is not run
    cfg = swan.load_config(project.file, {"executable": str(_fake_swan(tmp_path)), "active_variants": ["S1"],
                                          "outputs": {"point_quantities": ["XP", "YP", "HSIGN"]}})
    assert len(swan.run(cfg, force=True, verbose=False)) == 2
    with pytest.raises(ValueError):
        swan.load_config(project.file, {"active_variants": ["nope"]})
