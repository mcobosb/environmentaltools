"""Tests of environmentaltools.models.barrier (no BARRIER executable needed).

The module was validated against the Guadalquivir calibration of M. Cobos (2026-02-26,
806 sections, project PRO-2025_P_BARRIER_GUADALQUIVIR): ``fix_cross_sections`` reproduces
his corrected ``cross-sections-1.8.csv`` (max. difference 1e-12), ``extract_points`` and
``compare`` reproduce the series and RMSE of the project scripts, and a case written by
``create_case`` gives bit-identical BARRIER v0.13 results.
"""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from environmentaltools.models import barrier

matplotlib.use("Agg")


@pytest.fixture
def base_case(tmp_path):
    case = tmp_path / "base"
    case.mkdir()
    x = np.arange(0, 10001, 1000.0)
    along = pd.DataFrame({"x": x, "z": -4.2700000000000005 + 0 * x, "nmann": 0.03, "xutm": x, "yutm": 0.0,
                          "AngMd": 0.0, "AngMi": 0.0, "beta": 1.0, "Sf": 1.0, "kh": 20.0})
    barrier.write_csv(along, case / "along_channel_data.csv")
    rows = [(xi, -4.0, eta, 10 + eta, 10 * eta, 10 + 2 * eta, eta / 2, 10, -5 - eta, 5 + eta, 1)
            for xi in x for eta in np.arange(0, 5.0, 0.5)]
    barrier.write_csv(pd.DataFrame(rows, columns=barrier.CROSS_SECTION_COLUMNS), case / "cross-sections.csv")
    cfg = {"run": {"name": "test", "start_date": "2008-05-01T00:00:00", "end_date": "2008-05-03T00:00:00",
                   "timestep": 3600},
           "geometry": {"along_channel_file": "along_channel_data.csv",
                        "cross_sections_file": "cross-sections.csv"},
           "hydrodynamics": {"courant_number": 0.3,
                             "boundary_conditions": {"downstream": {"type": 2, "file": "tides.csv"}},
                             "forcing": {"tributaries_file": None}}}
    (case / "config.yaml").write_text(yaml.safe_dump(cfg))
    return case


def fake_output(n_time=72, n_x=11):
    t = np.arange(n_time) * 3600.0
    x = np.linspace(0, 10000, n_x)
    eta = np.sin(2 * np.pi * t[:, None] / (12.42 * 3600)) * (1 + x[None, :] / 1e4)
    sal = np.tile(x / 1e4 * 30, (n_time, 1))
    ds = xr.Dataset({"eta": (("time", "x"), eta), "S": (("time", "x"), sal), "Q": (("time", "x"), 10 * eta),
                     "A": (("time", "x"), 100 + eta)}, coords={"time": t, "x": x},
                    attrs={"time_offset": "2008-05-01 00:00:00"})
    return ds.assign_coords(date=("time", pd.Timestamp("2008-05-01") + pd.to_timedelta(t, unit="s")),
                            km=("x", barrier.distance_to_mouth(x)))


def test_csv_roundtrip_keeps_every_digit(base_case):
    along = barrier.read_csv(base_case / "along_channel_data.csv")
    assert list(along.columns)[:3] == ["x", "z", "nmann"]
    assert along["z"].iloc[0] == -4.2700000000000005      # rounding changes BARRIER results
    assert (base_case / "along_channel_data.csv").read_text().startswith(";x,z,")


def test_write_series_dates_to_seconds(tmp_path):
    s = pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2008-05-01", periods=3, freq="10min"))
    barrier.write_series(s, tmp_path / "tides.csv", start="2008-05-01", names=("time_seconds", "level_m"))
    df = barrier.read_csv(tmp_path / "tides.csv")
    assert list(df.columns) == ["time_seconds", "level_m"]
    np.testing.assert_allclose(df["time_seconds"], [0, 600, 1200])
    with pytest.raises(ValueError):
        barrier.write_series(s, tmp_path / "x.csv")


def test_write_tributaries(tmp_path):
    q = pd.Series([1.0, 2.0], index=[0.0, 86400.0])
    barrier.write_tributaries([{"name": "a", "x_utm": 1.0, "y_utm": 2.0, "discharge": q},
                               {"x_utm": 3.0, "y_utm": 4.0, "discharge": q}], tmp_path / "trib.csv")
    lines = [l for l in (tmp_path / "trib.csv").read_text().splitlines() if not l.startswith(";")]
    assert lines[0] == "2" and lines[1] == "1.000,2.000" and lines[2] == "0.0,1.0" and lines[4] == "3.000,4.000"


def test_section_profile():
    x = np.arange(0, 10001, 1000.0)
    np.testing.assert_allclose(barrier.section_profile(x, value=2), 2)
    np.testing.assert_allclose(barrier.section_profile(x, points=[(0, 0), (10000, 10)]), x / 1000)
    st = barrier.section_profile(x, steps=[(0, 1), (5000, 2)])
    assert st[4] == 1 and st[5] == 2 and st[-1] == 2
    sm = barrier.section_profile(x, steps=[(0, 1), (5000, 2)], sigma=1)
    assert 1 < sm[5] < 2 and sm[0] == pytest.approx(1, abs=1e-3)
    with pytest.raises(ValueError):
        barrier.section_profile(x)


def test_set_section_parameter(base_case):
    along = barrier.read_csv(base_case / "along_channel_data.csv")
    out = barrier.set_section_parameter(along, "nmann", scale=2)
    np.testing.assert_allclose(out["nmann"], 0.06)
    out = barrier.set_section_parameter(along, "kh", points=[(0, 10), (10000, 110)], add=1)
    np.testing.assert_allclose(out["kh"], 11 + along["x"] / 100)
    assert along["nmann"].iloc[0] == 0.03                 # original untouched


def test_check_and_fix_cross_sections(base_case):
    cs = barrier.read_csv(base_case / "cross-sections.csv")
    assert barrier.check_cross_sections(cs).empty
    bad = cs.copy()
    bad.loc[3, "A"] = bad.loc[2, "A"] - 1                  # area decreases with the depth
    bad.loc[4, "xl"] = bad.loc[3, "xl"] + 1                 # left bank moves the wrong way
    viol = barrier.check_cross_sections(bad)
    assert set(viol["variable"]) == {"A", "xl"}
    fixed, n = barrier.fix_cross_sections(bad, step=1e-3)
    assert n >= 2 and barrier.check_cross_sections(fixed).empty
    assert fixed.loc[3, "A"] == pytest.approx(bad.loc[2, "A"] + 1e-3)


def test_create_case(base_case, tmp_path):
    new = barrier.create_case(tmp_path / "new", base_case, config={"hydrodynamics": {"courant_number": 0.2}},
                              along_channel={"nmann": {"value": 0.02}})
    cfg = barrier.read_config(new)
    assert cfg["hydrodynamics"]["courant_number"] == 0.2
    assert cfg["hydrodynamics"]["boundary_conditions"]["downstream"]["type"] == 2   # merged, not replaced
    np.testing.assert_allclose(barrier.read_csv(new / "along_channel_data.csv")["nmann"], 0.02)
    assert barrier.read_csv(base_case / "along_channel_data.csv")["nmann"].iloc[0] == 0.03
    assert set(barrier.input_files(new)) == {"geometry.along_channel_file", "geometry.cross_sections_file",
                                             "hydrodynamics.boundary_conditions.downstream.file"}
    with pytest.raises(FileExistsError):
        barrier.create_case(tmp_path / "new", base_case)


def test_project_prepare(base_case, tmp_path):
    (tmp_path / "tides_2.csv").write_text(";t,h\n0,1\n")
    project = barrier.load_project({"base_case": str(base_case), "cases_dir": str(tmp_path / "cases"),
                                    "results_dir": str(tmp_path / "results"),
                                    "variants": {"ref": {}, "rough": {"along_channel": {"nmann": {"scale": 1.5}}},
                                                 "tide": {"files": {"tides.csv": str(tmp_path / "tides_2.csv")}}}})
    manifest = barrier.prepare(project, verbose=False)
    assert list(manifest["case"]) == ["ref", "rough", "tide"]
    assert (tmp_path / "cases" / "tide" / "tides.csv").read_text() == ";t,h\n0,1\n"
    np.testing.assert_allclose(barrier.read_csv(tmp_path / "cases/rough/along_channel_data.csv")["nmann"], 0.045)
    assert json.loads((tmp_path / "cases/rough/variant.json").read_text())["along_channel"]
    assert barrier.prepare(project, verbose=False)["status"].eq("kept").all()
    with pytest.raises(ValueError):
        barrier.load_project({"base_case": "b", "variants": {"v": {"bad": 1}}})


def test_template_loads():
    assert barrier.load_project(barrier.TEMPLATE).case_names[0] == "reference"


def test_missing_executable(tmp_path, monkeypatch):
    monkeypatch.delenv("BARRIER_EXE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        barrier.barrier_executable(tmp_path / "nothing")


def test_read_output_and_points(tmp_path):
    fake_output().drop_vars(["date", "km"]).to_netcdf(tmp_path / "out.nc")
    assert barrier.has_time_steps(tmp_path / "out.nc")
    ds = barrier.read_output(tmp_path)
    assert ds["date"].values[1] == np.datetime64("2008-05-01T01:00")
    assert ds["km"].values[0] == 10 and ds["km"].values[-1] == 0
    points, series = barrier.extract_points(ds, {"up": 9.6, "mouth": 0.2})
    assert list(points["name"]) == ["mouth", "up"]          # sorted from the mouth
    assert list(points["section"]) == [10, 0]
    assert list(series["S"].columns) == ["mouth (km 0)", "up (km 10)"]
    np.testing.assert_allclose(series["S"].iloc[0], [30, 0])


def test_tidal_mean_removes_tide():
    t = np.arange(24 * 10)
    sig = 2 + np.sin(2 * np.pi * t / 12.42) + np.sin(2 * np.pi * t / 23.93)
    m = barrier.tidal_mean(sig)
    assert np.isnan(m[0]) and np.isnan(m[-1])
    assert np.nanmax(np.abs(m - 2)) < 0.1


def test_compare():
    ref = fake_output()
    new = ref.copy(deep=True)
    new["eta"] = new["eta"] + 0.01
    summary, sections = barrier.compare(ref, new)
    row = summary.set_index("variable").loc["eta"]
    assert row["rmse"] == pytest.approx(0.01) and not row["within_tolerance"]
    assert summary.set_index("variable").loc["S", "within_tolerance"]
    np.testing.assert_allclose(sections["rmse_eta"], 0.01)


def test_plots(tmp_path, base_case):
    ds = fake_output()
    points, series = barrier.extract_points(ds, {"a": 0, "b": 5})
    barrier.plot_point_series(series["eta"], "level (m)", "t", file_name=tmp_path / "p.png")
    barrier.plot_along_channel(ds, "S", points, file_name=tmp_path / "a.png")
    barrier.plot_comparison(ds, ds, stations_km=[0, 5], file_name=tmp_path / "c.png")
    barrier.plot_comparison(ds, ds, stations_km=[0, 5], window_hours=None, file_name=tmp_path / "c_raw.png")
    barrier.plot_section_parameters(barrier.read_csv(base_case / "along_channel_data.csv"),
                                    file_name=tmp_path / "s.png")
    assert all((tmp_path / f).exists() for f in ("p.png", "a.png", "c.png", "s.png"))


def test_run_case_in_local_work_dir(tmp_path, base_case):
    """BARRIER writes in a local work folder and its files end up in out_dir."""
    import sys
    exe = tmp_path / "fake_barrier"
    exe.write_text(f"#!{sys.executable}\n"
                   "import numpy as np, xarray as xr\n"
                   "xr.Dataset({'eta': (('time', 'x'), np.zeros((3, 4)))},\n"
                   "           coords={'time': [0., 3600., 7200.], 'x': np.arange(4.)}).to_netcdf('sim.nc')\n"
                   "print('Run successfully end')\n")
    exe.chmod(0o755)
    work, out = tmp_path / "local", tmp_path / "synced" / "case"
    work.mkdir()
    nc = barrier.run_case(base_case, out, exe, work_dir=work, verbose=False)
    assert nc == out / "sim.nc" and barrier.is_readable(nc)
    assert (out / "barrier_console.txt").exists()
    assert list(work.iterdir()) == []          # the temporary run folder is removed


def fake_barrier(tmp_path):
    """Executable that writes a small BARRIER-like output (11 sections, 2 days) in its working folder."""
    import sys
    exe = tmp_path / "fake_barrier"
    exe.write_text(f"#!{sys.executable}\n"
                   "import numpy as np, xarray as xr\n"
                   "t = np.arange(49) * 3600.0; x = np.linspace(0, 10000, 11)\n"
                   "eta = np.sin(2 * np.pi * t[:, None] / 44712) * (1 + x / 1e4)\n"
                   "v = lambda a, u, n: (('time', 'x'), a, {'units': u, 'long_name': n})\n"
                   "xr.Dataset({'eta': v(eta, 'm', 'free surface elevation'),\n"
                   "            'S': v(np.tile(x / 1e4 * 30, (49, 1)), 'psu', 'salinity'),\n"
                   "            'Q': v(10 * eta, 'm3/s', 'water flow'), 'A': v(100 + eta, 'm2', 'Area')},\n"
                   "           coords={'time': t, 'x': x}, attrs={'time_offset': '2008-05-01 00:00:00'}\n"
                   "           ).to_netcdf('test_barrier_sim.nc')\n")
    exe.chmod(0o755)
    return exe


def test_workflow_prepare_run_postprocess(base_case, tmp_path, monkeypatch):
    """prepare → run → postprocess from one project file; only what is pending is done again."""
    import time

    from environmentaltools.models.barrier.__main__ import main

    monkeypatch.delenv("BARRIER_EXE", raising=False)
    exe = fake_barrier(tmp_path)
    pd.DataFrame({"punto": ["Mouth", "Up"], "dist": [0.0, 9.0]}).to_excel(tmp_path / "points.xlsx", index=False)
    cfg = {"base_case": str(base_case), "cases_dir": "cases", "results_dir": "results",
           "postprocess_dir": "post", "work_dir": True,
           "executable": [str(tmp_path / "missing_barrier"), str(exe)],       # the first that exists is used
           "cases": {"ref": {}, "rough": {"along_channel": {"nmann": {"scale": 2}}}},
           "postprocess": {"points": "points.xlsx", "points_columns": {"name": "punto", "km": "dist"},
                           "reference": {"case": "ref"}, "stations_km": [0, 5]}}
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump(cfg))

    main(["all", str(f)])
    project = barrier.load_project(f)
    for case in ("ref", "rough"):
        assert (tmp_path / "cases" / case / "section_parameters.png").exists()
        assert (tmp_path / "results" / case / "done.json").exists()
        assert (tmp_path / "post" / case / "timeseries.xlsx").exists()
        assert (tmp_path / "post" / case / "along_channel_S.png").exists()
    assert (tmp_path / "post" / "rough" / "vs_ref.xlsx").exists()
    assert not (tmp_path / "post" / "ref" / "vs_ref.xlsx").exists()   # no comparison of the reference with itself
    series = pd.read_excel(tmp_path / "post" / "rough" / "timeseries.xlsx", sheet_name=None)
    assert {"eta", "S", "points", "legend"} <= set(series)
    assert list(series["S"].columns) == ["date", "Mouth (km 0)", "Up (km 9)"]
    overview = pd.read_excel(project.overview_file, sheet_name=None)
    stats = overview["section_parameters"].set_index(["parameter", "statistic"])
    assert stats.loc[("nmann", "mean"), "rough"] == pytest.approx(2 * stats.loc[("nmann", "mean"), "base"])

    # nothing pending: kept, not run, up to date
    assert set(barrier.prepare(project, verbose=False)["status"]) == {"kept"}
    assert len(barrier.run(project, verbose=False)) == 0
    assert set(barrier.postprocess(project, verbose=False)["status"]) == {"up to date"}

    # a case changed in the list is created and run again, and postprocessed again
    time.sleep(0.05)
    cfg["cases"]["rough"] = {"along_channel": {"nmann": {"scale": 3}}}
    f.write_text(yaml.safe_dump(cfg))
    project = barrier.load_project(f)
    status = barrier.prepare(project, verbose=False).set_index("case")["status"]
    assert status["rough"] == "updated" and status["ref"] == "kept"
    assert not (tmp_path / "results" / "rough" / "done.json").exists()
    assert list(barrier.run(project, verbose=False)["case"]) == ["rough"]
    post = barrier.postprocess(project, verbose=False).set_index("case")["status"]
    assert post["rough"] == "done"


def test_project_reference_file_and_validation(base_case, tmp_path):
    fake_output().drop_vars(["date", "km"]).to_netcdf(tmp_path / "other.nc")
    project = barrier.load_project({"base_case": str(base_case), "cases": {"a": {}},
                                    "postprocess_dir": str(tmp_path / "post"), "results_dir": str(tmp_path / "res"),
                                    "postprocess": {"points": {"Mouth": 0},
                                                    "reference": {"file": str(tmp_path / "other.nc"), "name": "old"}}})
    out = barrier.postprocess(project, verbose=False).set_index("case")["status"]
    assert out["old"] == "done (reference)" and out["a"] == "no result"
    assert (tmp_path / "post" / "old" / "timeseries_eta.png").exists()
    with pytest.raises(ValueError):
        barrier.load_project({"base_case": "b", "cases": {"a": {}}, "postprocess": {"reference": {"case": "zz"}}})
