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


def test_prepare_warns_without_tvd_limiter(base_case, tmp_path, capsys):
    """Without the TVD limiter BARRIER does not change the interior of the channel: prepare warns, also quietly."""
    assert barrier.check_config(barrier.read_config(base_case))                 # the fixture has no tvd_limiter
    assert barrier.check_config({"hydrodynamics": {"tvd_limiter": {"enabled": True}}}) == []
    tvd = {"config": {"hydrodynamics": {"tvd_limiter": {"enabled": True}}}}
    off = {"config": {"hydrodynamics": {"tvd_limiter": {"enabled": False}}}}
    project = barrier.load_project({"base_case": str(base_case), "cases_dir": str(tmp_path / "cases"),
                                    "cases": {"base": {}, "tvd": tvd, "off": off}})
    barrier.prepare(project, overview=False, figures=False, verbose=False)
    warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Warning:")]
    assert len(warnings) == 1 and "tvd_limiter" in warnings[0] and warnings[0].endswith("(cases: base, off)")


def test_project_overrides_add_cases(base_case, tmp_path):
    """The cases of the overrides are added to those of the project file (they were ignored)."""
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": str(base_case),
                                 "cases": {"a": {}, "b": {"config": {"hydrodynamics": {"courant_number": 0.2}}}}}))
    project = barrier.load_project(f, {"results_dir": "other", "cases": {"c": {},
                                                                         "b": {"along_channel": {"nmann": {"scale": 2}}}}})
    assert project.case_names == ["a", "b", "c"] and project.results_dir == tmp_path / "other"
    assert project["variants"]["b"] == {"config": {"hydrodynamics": {"courant_number": 0.2}},
                                        "along_channel": {"nmann": {"scale": 2}}}


def test_template_loads():
    project = barrier.load_project(barrier.TEMPLATE)
    assert project.cases_file.name == "cases.xlsx" and project.case_names == []


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


def test_boundary_forcings():
    ds = fake_output()
    f = barrier.boundary_forcings(ds)
    assert list(f.columns) == ["Q_upstream", "eta_mouth"]
    np.testing.assert_allclose(f["Q_upstream"], ds["Q"].isel(x=0).values)    # upstream boundary: section 0
    np.testing.assert_allclose(f["eta_mouth"], ds["eta"].isel(x=-1).values)   # mouth: last section
    assert f.index[1] == pd.Timestamp("2008-05-01 01:00")


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


def test_compare_runs_without_salinity():
    """Runs without salinity (e.g. a calibration of levels) are compared on the variables both outputs have."""
    ref = fake_output().drop_vars("S")
    summary, sections = barrier.compare(ref, ref.copy(deep=True))
    assert list(summary["variable"]) == ["eta", "Q", "A"] and "rmse_S" not in sections
    with pytest.raises(ValueError, match="No variable to compare"):
        barrier.compare(ref, ref, tolerances={"S": 0.05})
    fig = barrier.plot_comparison(ref, ref, stations_km=[0, 5])          # level row only
    assert len(fig.axes) == 2


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


def test_executable_for_another_system(tmp_path, base_case, monkeypatch):
    """An executable that the system cannot start (e.g. a Linux build on macOS) fails the case, not the batch."""
    monkeypatch.delenv("BARRIER_EXE", raising=False)
    exe = tmp_path / "linux_barrier"
    exe.write_bytes(b"\x7fELF not for this system")
    exe.chmod(0o755)
    with pytest.raises(RuntimeError, match="could not be started"):
        barrier.run_case(base_case, tmp_path / "out", exe, verbose=False)

    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": str(base_case), "executable": str(exe),
                                 "cases": {"a": {}, "b": {}}}))
    project = barrier.load_project(f)
    barrier.prepare(project, overview=False, figures=False, verbose=False)
    log = barrier.run(project, verbose=False)
    assert list(log["case"]) == ["a", "b"] and not log["ok"].any()
    assert log["error"].str.contains("could not be started").all()


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
    assert {"eta", "S", "forcings", "points", "legend"} <= set(series)
    assert list(series["forcings"].columns) == ["date", "Q_upstream", "eta_mouth"]
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


def test_cases_table(base_case, tmp_path):
    """Cases from an Excel table: one row per case, one column per change; empty cell = base case."""
    (tmp_path / "inputs").mkdir()
    pd.DataFrame({"x": [0, 10000], "kh": [10, 110]}).to_csv(tmp_path / "inputs" / "kh.csv", index=False)
    (tmp_path / "inputs" / "tides_2.csv").write_text(";t,h\n0,1\n")
    table = pd.DataFrame({
        "case": ["ref", "rough", "kh_file", "other"],
        "description": ["base", "n × 2", "", "faster"],
        "along_channel.nmann.scale": [None, 2, None, None],
        "along_channel.kh.file": [None, None, "kh.csv", None],
        "hydrodynamics.courant_number": [None, None, None, 0.2],
        "run.end_date": [None, None, None, "2008-05-02T00:00:00"],
        "files.tides.csv": [None, None, None, "tides_2.csv"],
        "_my_notes": ["x", None, None, None]})
    table.to_excel(tmp_path / "inputs" / "cases.xlsx", sheet_name="cases", index=False)
    cases = barrier.read_cases_table(tmp_path / "inputs" / "cases.xlsx")
    assert cases["ref"] == {}
    assert cases["rough"] == {"along_channel": {"nmann": {"scale": 2}}}          # 2.0 in Excel → 2
    assert cases["kh_file"] == {"along_channel": {"kh": {"points": [[0.0, 10.0], [10000.0, 110.0]]}}}
    assert cases["other"]["config"] == {"hydrodynamics": {"courant_number": 0.2}, "run": {"end_date": "2008-05-02T00:00:00"}}

    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": str(base_case), "cases_file": "inputs/cases.xlsx"}))
    project = barrier.load_project(f)
    assert project.case_names == ["ref", "rough", "kh_file", "other"]
    assert project["variants"]["other"]["files"] == {"tides.csv": "inputs/tides_2.csv"}   # relative to barrier.yaml
    barrier.prepare(project, verbose=False)
    other = tmp_path / "cases" / "other"
    assert (other / "tides.csv").read_text() == ";t,h\n0,1\n"
    assert barrier.read_config(other)["hydrodynamics"]["courant_number"] == 0.2
    np.testing.assert_allclose(barrier.read_csv(tmp_path / "cases/kh_file/along_channel_data.csv")["kh"],
                               10 + np.arange(0, 10001, 1000.0) / 100)
    assert set(barrier.prepare(project, verbose=False)["status"]) == {"kept"}

    # YAML cases → table → same cases
    back = tmp_path / "back.xlsx"
    barrier.write_cases_table(back, project["variants"])
    again = barrier.read_cases_table(back)
    assert {k: v for k, v in again.items() if k != "other"} == {k: v for k, v in cases.items() if k != "other"}


def test_cases_table_template_created(base_case, tmp_path):
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": str(base_case), "cases_file": "cases.xlsx"}))
    manifest = barrier.prepare(barrier.load_project(f), verbose=False)
    assert manifest.empty and (tmp_path / "cases.xlsx").exists()
    project = barrier.load_project(f)                       # the template has two example cases
    assert project.case_names == ["reference", "manning_x1.1"]
    assert project["variants"]["manning_x1.1"] == {"along_channel": {"nmann": {"scale": 1.1}}}
    with pytest.raises(ValueError):
        pd.DataFrame({"case": ["a", "a"]}).to_excel(tmp_path / "dup.xlsx", index=False)
        barrier.read_cases_table(tmp_path / "dup.xlsx")


def test_case_executable(base_case, tmp_path, monkeypatch):
    """A case can run with another BARRIER build (a name of `executables` or a path), also from the table."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    exe_a, exe_b = fake_barrier(tmp_path / "a"), fake_barrier(tmp_path / "b")
    monkeypatch.setenv("BARRIER_EXE", str(exe_a))              # the project build; ignored by cases with their own
    pd.DataFrame({"case": ["alt"], "executable": ["other"]}).to_excel(tmp_path / "cases.xlsx", index=False)
    cfg = {"base_case": str(base_case), "cases_dir": "cases", "results_dir": "results", "executable": str(exe_a),
           "executables": {"other": [str(tmp_path / "missing_barrier"), str(exe_b)]},
           "cases_file": "cases.xlsx", "cases": {"ref": {}}}
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump(cfg))
    project = barrier.load_project(f)
    assert project.case_executable("ref")[0] is None
    assert project.case_executable("alt") == ("other", [tmp_path / "missing_barrier", exe_b])
    barrier.prepare(project, verbose=False, figures=False)
    params = pd.read_excel(project.overview_file, sheet_name="parameters").set_index("parameter")
    assert params.loc["(executable)", "alt"] == "other" and params.loc["(executable)", "ref"] == "(project)"
    log = barrier.run(project, verbose=False).set_index("case")
    assert log["ok"].all()
    assert log.loc["ref", "executable"] == str(exe_a) and log.loc["alt", "executable"] == str(exe_b)
    assert json.loads((tmp_path / "results" / "alt" / "done.json").read_text())["executable"] == str(exe_b)

    # changing the build of a case makes it pending again
    pd.DataFrame({"case": ["alt"], "executable": [str(exe_a)]}).to_excel(tmp_path / "cases.xlsx", index=False)
    project = barrier.load_project(f)
    assert barrier.prepare(project, verbose=False, figures=False).set_index("case")["status"]["alt"] == "updated"
    assert list(barrier.run(project, verbose=False)["case"]) == ["alt"]

    cfg["cases"]["bad"] = {"executable": "nowhere"}
    f.write_text(yaml.safe_dump(cfg))
    project = barrier.load_project(f)
    barrier.prepare(project, verbose=False, figures=False)
    log = barrier.run(project, verbose=False).set_index("case")
    assert not log.loc["bad", "ok"] and "not found" in log.loc["bad", "error"]


def test_set_section_parameter_km_and_scale_list():
    along = pd.DataFrame({"x": np.arange(0.0, 10001.0, 1000.0), "kh": np.full(11, 100.0)})
    # x = 10000 m is the mouth (km 0); x = 0 is km 10
    out = barrier.set_section_parameter(along, "kh", scale=[[0, 0.5], [4, 0.5], [6, 2.0], [10, 2.0]], coordinate="km")
    km = (10000 - along["x"]) / 1000
    assert np.allclose(out["kh"], 100 * np.interp(km, [0, 4, 6, 10], [0.5, 0.5, 2.0, 2.0]))
    same = barrier.set_section_parameter(along, "kh", points=[[0, 1], [10, 2]], coordinate="km")
    assert np.allclose(same["kh"], np.interp(km, [0, 10], [1, 2]))
    assert np.array_equal(barrier.set_section_parameter(along, "kh", scale=1.6)["kh"], along["kh"] * 1.6)
    with pytest.raises(ValueError):
        barrier.set_section_parameter(along, "kh", scale=2, coordinate="pk")


def test_read_write_csv_round_trip(tmp_path):
    # números que el lector rápido de pandas no lee exactos
    values = [0.07750888407555664, 1.046186575066534, 21.54459501553473, -4.2700000000000005, 0.1 + 0.2]
    f = tmp_path / "along_channel_data.csv"
    barrier.write_csv(pd.DataFrame({"x": np.arange(len(values), dtype=float), "nmann": values}), f)
    text = f.read_text()
    barrier.write_csv(barrier.read_csv(f), f)
    assert f.read_text() == text
    assert barrier.read_csv(f)["nmann"].tolist() == values



# ------------------------------------------------------------------ geometry from a DEM and an axis
def trapezoid_points(length=10000.0):
    """xyz points of a straight channel along the x axis: bed -5 m 100 m wide, slopes 1:10 up to +3 m."""
    def zprof(y):
        a = np.abs(y)
        return np.where(a <= 50, -5.0, np.where(a <= 130, -5 + (a - 50) / 10, 3.0))
    X, Y = np.meshgrid(np.arange(-500, length + 501, 500.0), np.arange(-300, 301, 2.0))
    return pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": zprof(Y.ravel())})


def test_section_lines_follow_barrier_angles():
    """Left bank looking downstream at AngMi, right bank at AngMd (radians from the east), as BARRIER places them."""
    sec = barrier.section_lines([[0, 1000], [0, 0]], dx=300)            # flowing south
    np.testing.assert_allclose(sec["x"], [0, 1000 / 3, 2000 / 3, 1000])  # equal steps closest to dx, ends included
    np.testing.assert_allclose(sec["AngMi"], 0.0, atol=1e-12)            # left looking south = east
    np.testing.assert_allclose(np.abs(sec["AngMd"]), np.pi, atol=1e-12)  # right = west
    x, y = barrier.section_points(sec, [-10.0, 10.0])
    np.testing.assert_allclose(x[0], [10.0, -10.0], atol=1e-9)


def test_build_geometry_trapezoid_is_exact(capsys):
    cs, along, report = barrier.build_geometry(trapezoid_points(), [[0, 0], [10000, 0]], dx=1000, half_width=250,
                                               top_level=2, profile_step=1, level_step=0.5, kh=50)
    t = cs[cs["x"] == 5000]
    h = t["eta"].to_numpy()
    np.testing.assert_allclose(t["A"], 100 * h + 10 * h ** 2, atol=1e-9)
    np.testing.assert_allclose(t["P"][h > 0], 100 + 2 * h[h > 0] * np.sqrt(101), atol=1e-9)
    np.testing.assert_allclose(t["B"][h > 0], 100 + 20 * h[h > 0], atol=1e-9)
    np.testing.assert_allclose(t["xr"] - t["xl"], t["B"])
    assert list(along.columns) == barrier.ALONG_COLUMNS                  # the order BARRIER reads
    assert along["kh"].eq(50).all() and along["Sf"].eq(1).all() and len(along) == 11
    assert barrier.check_cross_sections(cs).empty
    assert not report["open_end"].any() and not report["crosses_next"].any()
    assert "Warning" not in capsys.readouterr().out


def test_section_table_dip_behind_higher_ground():
    """A deeper dip behind a bank does not count until the water tops the bank; tables stay monotonic."""
    s = np.arange(-100, 101, 1.0)
    z = np.where(np.abs(s) <= 20, -3.0, 1.0)                             # channel 40 m wide, banks at +1 m
    z = np.where((s > 50) & (s < 80), -4.0, z)                           # a deeper pond behind the right bank
    t = barrier.section_table(s, z, top_level=2.0, level_step=0.5, bed_half_width=30)
    below, above = t[t["z"] + t["eta"] <= 1.0], t[t["z"] + t["eta"] > 1.0]
    assert t["z"].iloc[0] == -3.0                                        # the bed is the channel, not the pond
    assert below["xr"].max() <= 21 and above["xr"].min() > 80
    assert barrier.check_cross_sections(t).empty


def test_build_geometry_from_raster(tmp_path, capsys):
    import rasterio
    from rasterio.transform import from_origin

    xs, ys = np.arange(-200, 4201, 2.0), np.arange(300, -301, -2.0)
    X, Y = np.meshgrid(xs, ys)
    z = np.where(np.abs(Y) <= 50, -5.0, np.where(np.abs(Y) <= 130, -5 + (np.abs(Y) - 50) / 10, 3.0))
    with rasterio.open(tmp_path / "dem.tif", "w", driver="GTiff", width=len(xs), height=len(ys), count=1,
                       dtype="float64", transform=from_origin(-201, 301, 2, 2), nodata=-9999) as dst:
        dst.write(z, 1)
    cs, along, report = barrier.build_geometry(tmp_path / "dem.tif", [[0, 0], [4000, 0]], dx=1000, half_width=250,
                                               top_level=2, out_dir=tmp_path / "case")
    a = cs[cs["x"] == 2000].set_index("eta")["A"]
    assert abs(a.iloc[-1] - (100 * 7 + 10 * 49)) / 1190 < 0.02         # nearest cell of a 2 m raster
    assert barrier.read_csv(tmp_path / "case" / "cross-sections.csv").shape == cs.shape
    assert list(barrier.read_csv(tmp_path / "case" / "along_channel_data.csv").columns) == barrier.ALONG_COLUMNS[:9]

    barrier.build_geometry(tmp_path / "dem.tif", [[0, 0], [4000, 0]], dx=1000, half_width=100, top_level=2,
                           verbose=False)                                # profiles shorter than the channel
    assert "water reaches the end of the profile" in capsys.readouterr().out


def test_smooth_sections_keeps_equal_sections():
    cs, _, _ = barrier.build_geometry(trapezoid_points(), [[0, 0], [10000, 0]], dx=1000, half_width=250, top_level=2,
                                      profile_step=1, level_step=0.5, verbose=False)
    sm = barrier.smooth_sections(cs, 2000, level_step=0.5)
    assert barrier.check_cross_sections(sm).empty
    for col in ("A", "B", "P", "z"):
        np.testing.assert_allclose(sm[col].to_numpy(), cs.sort_values(["x", "eta"])[col].to_numpy(), atol=1e-9)


def test_crossing_sections_in_a_sharp_bend(capsys):
    """Long sections in a sharp bend cross each other; smoothing the axis directions does not hide a real crossing."""
    t = np.linspace(0, np.pi, 50)
    axis = np.c_[300 * np.cos(t), 300 * np.sin(t)]                        # half circle, radius 300 m
    X, Y = np.meshgrid(np.arange(-900, 901, 5.0), np.arange(-600, 901, 5.0))
    ring = np.hypot(X, Y)
    pts = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(),
                        "z": np.where(np.abs(ring.ravel() - 300) <= 250, -5.0, 3.0)})   # 500 m wide channel
    _, _, report = barrier.build_geometry(pts, axis, dx=100, half_width=400, top_level=2, profile_step=5,
                                          verbose=False)
    assert report["crosses_next"].any()
    assert "cross the next one" in capsys.readouterr().out


def test_read_xyz_with_and_without_header(tmp_path):
    (tmp_path / "a.xyz").write_text("1 2 -3.5\n4 5 1.0e+00\n")
    (tmp_path / "b.csv").write_text("X,Y,Z\n1,2,-3.5\n4,5,1\n")
    for f in ("a.xyz", "b.csv"):
        df = barrier.read_xyz(tmp_path / f)
        assert list(df.columns) == ["x", "y", "z"] and df["z"].tolist() == [-3.5, 1.0]


def test_prismatic_geometry_matches_the_dem_one():
    """The analytic trapezoid gives the tables taken from a DEM of the same channel (above the bed row)."""
    kw = dict(dx=1000, top_level=2, level_step=0.5, verbose=False)
    dem, _, _ = barrier.build_geometry(trapezoid_points(), [[0, 0], [10000, 0]], half_width=250, profile_step=1, **kw)
    cs, along, report = barrier.prismatic_geometry(10000, bottom_width=100, side_slope=10, bed=-5, **kw)
    assert cs.shape == dem.shape and list(along.columns) == barrier.ALONG_COLUMNS[:9]
    wet = cs["eta"] > 0
    for col in ("x", "z", "eta", "A", "P", "B", "Rh", "xl", "xr"):
        np.testing.assert_allclose(cs.loc[wet, col], dem.loc[wet, col], atol=1e-9)
    assert barrier.check_cross_sections(cs).empty and not report["crosses_next"].any()


def test_prismatic_geometry_along_the_channel():
    """Bed by km to the mouth, exponential convergence of the width, rectangular sections; errors."""
    cs, along, report = barrier.prismatic_geometry(
        60000, dx=1000, bottom_width=3000, side_slope=0, bed={"points": [[0, -8], [60, -3]], "coordinate": "km"},
        top_level=3, convergence_length=20000, kh=100, check_level=0, verbose=False)
    assert along["z"].iloc[-1] == -8 and along["z"].iloc[0] == -3     # mouth deeper than the upstream end
    np.testing.assert_allclose(report["width"].iloc[[-1, 0]], [3000, 3000 * np.exp(-3)])
    mouth = cs[cs["x"] == 60000]
    np.testing.assert_allclose(mouth["A"], 3000 * mouth["eta"])
    np.testing.assert_allclose(mouth["P"], 3000 + 2 * mouth["eta"])
    assert along["kh"].eq(100).all()
    with pytest.raises(ValueError, match="above the bed"):
        barrier.prismatic_geometry(1000, dx=100, bottom_width=10, bed=-1, top_level=-2)
    with pytest.raises(ValueError, match="not both 0"):
        barrier.prismatic_geometry(1000, dx=100, bottom_width=0, side_slope=0, bed=-1, top_level=2)


def test_prepare_geometry_only_when_pending(base_case, tmp_path, capsys):
    """The 'geometry' section builds the base case files once, and again when its settings or the files change."""
    geo = {"type": "prismatic", "length": 10000, "dx": 1000, "bottom_width": 100, "bed": -5, "top_level": 2}
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": str(base_case), "geometry": geo, "cases": {"a": {}}}))
    report = barrier.prepare_geometry(barrier.load_project(f), verbose=False)
    assert len(report) == 11
    cs = barrier.read_csv(base_case / "cross-sections.csv")                 # names of the base config.yaml
    np.testing.assert_allclose(cs.loc[cs["x"] == 0, "A"], 100 * cs.loc[cs["x"] == 0, "eta"])
    for name in ("geometry.png", "geometry_report.xlsx", "geometry_signature.json"):
        assert (tmp_path / "base_report" / name).exists()

    assert barrier.prepare_geometry(barrier.load_project(f), verbose=False) is None        # nothing changed
    assert barrier.prepare_geometry(barrier.load_project(f, {"geometry": {"bed": -6}}), verbose=False) is not None
    assert barrier.read_csv(base_case / "along_channel_data.csv")["z"].eq(-6).all()
    with open(base_case / "cross-sections.csv", "a") as fh:                                # edited by hand
        fh.write("10000,-6,9,1,1,1,1,1,-1,1,1\n")
    project = barrier.load_project(f, {"geometry": {"bed": -6}})
    assert barrier.prepare_geometry(project) is not None
    assert "edited by hand" in capsys.readouterr().out
    assert (tmp_path / "base_report" / "edited" / "cross-sections.csv").read_text().endswith("1,1,1\n")
    assert barrier.prepare_geometry(project, force=True, verbose=False) is not None


def test_prepare_geometry_from_xyz_and_axis_files(tmp_path):
    """Files of the section are relative to the project file; a new DEM builds the geometry again."""
    import os

    (tmp_path / "inputs").mkdir()
    trapezoid_points(4000).to_csv(tmp_path / "inputs" / "dem.xyz", sep=" ", index=False, header=False)
    pd.DataFrame({"x": [0, 4000], "y": [0, 0]}).to_csv(tmp_path / "inputs" / "axis.csv", index=False)
    geo = {"type": "dem", "dem": "inputs/dem.xyz", "axis": "inputs/axis.csv", "dx": 1000, "half_width": 250,
           "top_level": 2, "profile_step": 1, "level_step": 0.5}
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": "base", "geometry": geo, "cases": {"a": {}}}))
    report = barrier.prepare_geometry(barrier.load_project(f), verbose=False)
    assert len(report) == 5 and (tmp_path / "base" / "cross-sections.csv").exists()   # usual names, new folder
    assert barrier.prepare_geometry(barrier.load_project(f), verbose=False) is None
    st = (tmp_path / "inputs" / "dem.xyz").stat()
    os.utime(tmp_path / "inputs" / "dem.xyz", ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert barrier.prepare_geometry(barrier.load_project(f), verbose=False) is not None


def test_geometry_section_errors(base_case, tmp_path):
    with pytest.raises(ValueError, match="needs 'type'"):
        barrier.load_project({"base_case": str(base_case), "cases": {"a": {}}, "geometry": {"dx": 100}})
    project = barrier.load_project({"base_case": str(base_case), "cases": {"a": {}},
                                    "geometry": {"type": "prismatic", "lenght": 1000}})
    with pytest.raises(ValueError, match="unknown keys \\['lenght'\\]"):
        barrier.prepare_geometry(project)


def test_all_builds_the_geometry_first(base_case, tmp_path, monkeypatch):
    from environmentaltools.models.barrier.__main__ import main

    monkeypatch.delenv("BARRIER_EXE", raising=False)
    geo = {"type": "prismatic", "length": 10000, "dx": 1000, "bottom_width": 100, "bed": -5, "top_level": 2}
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": str(base_case), "geometry": geo, "executable": str(fake_barrier(tmp_path)),
                                 "cases": {"ref": {}}}))
    main(["all", str(f)])
    case_cs = barrier.read_csv(tmp_path / "cases" / "ref" / "cross-sections.csv")
    assert case_cs["B"].max() == 100                                        # the case has the new geometry
    assert (tmp_path / "results" / "ref" / "done.json").exists()


# ------------------------------------------------------------------ starter base case
def test_new_base_case_writes_a_case_that_runs(tmp_path, capsys):
    """config.yaml with the settings BARRIER needs (none of check_config), series covering the run."""
    case = tmp_path / "base"
    barrier.prismatic_geometry(10000, dx=1000, bottom_width=100, bed=-5, top_level=3, out_dir=case, verbose=False)
    barrier.new_base_case(case, "2020-01-01", "2020-01-03", tide={"amplitude": 1.0}, discharge=50)
    cfg = barrier.read_config(case)
    assert barrier.check_config(cfg) == []
    assert cfg["hydrodynamics"]["initial_conditions"]["type"] == "calm" and "forcing" in cfg["hydrodynamics"]
    assert cfg["transport"]["salinity"]["enabled"] is True and "S" in cfg["run"]["output_variables"]
    assert cfg["transport"]["salinity"]["beta"] == 0.00076            # the salt changes the density
    assert all(f.exists() for f in barrier.input_files(case, cfg).values())
    tide = barrier.read_csv(case / "tides.csv")
    assert tide.iloc[0, 1] == 0.0 and tide.iloc[:, 1].max() == pytest.approx(1.0, abs=1e-3)   # starts at the mean
    assert tide.iloc[-1, 0] == 2 * 86400
    q = barrier.read_csv(case / "discharge.csv")
    assert q.iloc[:, 1].eq(50).all() and q.iloc[-1, 0] == 2 * 86400
    assert "still missing" not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        barrier.new_base_case(case, "2020-01-01", "2020-01-03", tide=0.0, discharge=50)


def test_new_base_case_options(tmp_path, capsys):
    """No salinity, dated series converted to seconds, warnings for short series and missing geometry."""
    dates = pd.date_range("2020-01-01", "2020-01-02", freq="h")
    tide = pd.Series(np.linspace(0, 1, len(dates)), index=dates)
    case = barrier.new_base_case(tmp_path / "c", "2020-01-01", "2020-01-03", tide=tide, discharge=10.0,
                                 salinity=None, courant_number=0.15)
    cfg = barrier.read_config(case)
    assert cfg["transport"]["salinity"]["enabled"] is False and "S" not in cfg["run"]["output_variables"]
    assert cfg["hydrodynamics"]["courant_number"] == 0.15
    assert barrier.read_csv(case / "tides.csv").iloc[-1, 0] == 86400
    out = capsys.readouterr().out
    assert "tides.csv does not cover the whole run" in out and "still missing" in out
    with pytest.raises(ValueError, match="Series or a number"):
        barrier.new_base_case(tmp_path / "d", "2020-01-01", "2020-01-03", tide=0, discharge={"amplitude": 1})


def test_read_series_formats(tmp_path):
    (tmp_path / "dated.csv").write_text("date,level\n2020-01-01 00:00,0.5\n2020-01-01 01:00,0.7\n")
    (tmp_path / "seconds.csv").write_text(";time_seconds,discharge_m3s\n0,10\n3600,12\n")
    (tmp_path / "spaces.txt").write_text("0 1.5\n600 1.6\n")
    dated = barrier.read_series(tmp_path / "dated.csv")
    assert isinstance(dated.index, pd.DatetimeIndex) and dated.tolist() == [0.5, 0.7]
    assert barrier.read_series(tmp_path / "seconds.csv").index.tolist() == [0.0, 3600.0]
    assert barrier.read_series(tmp_path / "spaces.txt").tolist() == [1.5, 1.6]


def test_prepare_simulation_only_when_pending(tmp_path, capsys):
    import os

    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "tide.csv").write_text("date,level\n2020-01-01,0.0\n2020-01-03,0.0\n")
    sim = {"start": "2020-01-01", "end": "2020-01-03", "tide": "inputs/tide.csv", "discharge": 20,
           "salinity": None, "config": {"hydrodynamics": {"manning_eta": True}}}
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({"base_case": "base", "simulation": sim, "cases": {"a": {}}}))
    assert barrier.prepare_simulation(barrier.load_project(f), verbose=False)
    cfg = barrier.read_config(tmp_path / "base")
    assert cfg["hydrodynamics"]["manning_eta"] is True and cfg["hydrodynamics"]["tvd_limiter"]["enabled"] is True
    assert cfg["transport"]["salinity"]["enabled"] is False
    assert not barrier.prepare_simulation(barrier.load_project(f), verbose=False)          # up to date

    st = (tmp_path / "inputs" / "tide.csv").stat()                                      # a new tide file
    os.utime(tmp_path / "inputs" / "tide.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert barrier.prepare_simulation(barrier.load_project(f), verbose=False)
    with open(tmp_path / "base" / "config.yaml", "a") as fh:                            # edited by hand
        fh.write("# my change\n")
    assert barrier.prepare_simulation(barrier.load_project(f))
    assert "edited by hand" in capsys.readouterr().out
    assert (tmp_path / "base_report" / "edited" / "config.yaml").read_text().endswith("# my change\n")
    assert not (tmp_path / "base" / "config.yaml").read_text().endswith("# my change\n")

    off = barrier.load_project(f, {"simulation": {"config": {"hydrodynamics": {"tvd_limiter": {"enabled": False}}}}})
    barrier.prepare_simulation(off, verbose=False)
    assert "tvd_limiter.enabled is not true" in capsys.readouterr().out


def test_simulation_section_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown keys \\['tides'\\]"):
        barrier.load_project({"base_case": "b", "cases": {"a": {}}, "simulation": {"tides": 1}})
    project = barrier.load_project({"base_case": str(tmp_path / "b"), "cases": {"a": {}},
                                    "simulation": {"end": "2020-01-02", "tide": 0, "discharge": 1}})
    with pytest.raises(ValueError, match="missing \\['start'\\]"):
        barrier.prepare_base(project)


def test_project_from_scratch(tmp_path, monkeypatch):
    """A project file alone (no base case folder): 'all' writes the base case, creates the case and runs it."""
    from environmentaltools.models.barrier.__main__ import main

    monkeypatch.delenv("BARRIER_EXE", raising=False)
    f = tmp_path / "barrier.yaml"
    f.write_text(yaml.safe_dump({
        "base_case": "base_case", "executable": str(fake_barrier(tmp_path)),
        "geometry": {"type": "prismatic", "length": 10000, "dx": 1000, "bottom_width": 100, "bed": -5,
                     "top_level": 2},
        "simulation": {"start": "2008-05-01", "end": "2008-05-03", "tide": {"amplitude": 1.0}, "discharge": 50},
        "cases": {"ref": {}}}))
    main(["all", str(f)])
    assert barrier.check_config(barrier.read_config(tmp_path / "cases" / "ref")) == []
    assert (tmp_path / "results" / "ref" / "done.json").exists()
    assert (tmp_path / "postprocess" / "ref" / "timeseries.xlsx").exists()
    assert sorted(p.name for p in (tmp_path / "base_report").iterdir() if p.suffix == ".json") == [
        "geometry_signature.json", "simulation_signature.json"]
