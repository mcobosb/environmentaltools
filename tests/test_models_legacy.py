"""Models moved from common/processes/graphics: new locations, deprecated aliases and the SWAN → COPLA workflow."""

import os
import platform
import stat
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from environmentaltools.models import coastalme, cshore, delft3d, swan_copla
from environmentaltools.processes import kobayashi_transport

KOBAYASHI = dict(fb=0.015, wf=0.03, gamma=0.7, rho=1025, br=0.1, eb=0.005, ef=0.01, S=2.65, aK=0.2,
                 d50=0.0003, Shic=0.05, bK=0.002, phi=0.63, local_angle=20, tburst=3600)


@pytest.mark.parametrize("old, new", [
    ("environmentaltools.processes.read_swan", swan_copla.read_swan),
    ("environmentaltools.processes.compute.nesting", swan_copla.nesting),
    ("environmentaltools.processes.load.extract_delft_wave_dat", delft3d.extract_wave_dat),
    ("environmentaltools.processes.write.write_cshore", cshore.write_input),
    ("environmentaltools.processes.run_coastalme", coastalme.run),
    ("environmentaltools.common.read_copla", swan_copla.read_copla),
    ("environmentaltools.common.write.create_project_directory", swan_copla.create_project_directory),
    ("environmentaltools.common.cshore_config", cshore.default_config),
    ("environmentaltools.graphics.plot_delft_grids", delft3d.plot_grids),
    ("environmentaltools.graphics.spatial.plot_delft_wave_field", delft3d.plot_wave_field),
])
def test_deprecated_alias_points_to_new_function(old, new):
    module, name = old.rsplit(".", 1)
    alias = getattr(__import__(module, fromlist=[name]), name)
    assert alias.__wrapped__ is new


def test_deprecated_alias_warns_and_renames_keywords(tmp_path):
    from environmentaltools import processes

    (tmp_path / "OBPROF").write_text("1 3\n0.0 -2.0\n1.0 -1.0\n2.0 0.5\n9.0 9.0\n")
    with pytest.warns(DeprecationWarning, match="models.cshore.read_output"):
        df = processes.read_cshore(file_="bprof", path=str(tmp_path))
    assert df["z"].tolist() == [-2.0, -1.0, 0.5]


def test_unknown_attribute_still_raises():
    from environmentaltools import common, processes

    with pytest.raises(AttributeError):
        processes.not_a_function
    with pytest.raises(AttributeError):
        common.not_a_function


def test_legacy_kobayashi_matches_pure_function():
    from environmentaltools.processes import sediment_transport_Kobayashi

    rng = np.random.default_rng(0)
    shape = (6, 8)
    grid = {"depth": rng.uniform(0.5, 10, shape), "Setup": rng.uniform(-0.2, 0.3, shape),
            "Hs": rng.uniform(0.2, 2, shape), "kp": rng.uniform(0.02, 0.5, shape), "DirM": rng.uniform(0, 360, shape),
            "Qb": rng.uniform(0, 1, shape), "U": rng.uniform(0, 1.2, shape), "DirU": rng.uniform(0, 360, shape),
            "Sb": rng.uniform(-0.05, 0.05, shape)}
    data = pd.DataFrame({"Hs": [1.8], "Tp": [9.0]})
    new = kobayashi_transport(grid["depth"], grid["Setup"], grid["Hs"], 9.0, grid["kp"], grid["DirM"], grid["Qb"],
                              grid["U"], grid["DirU"], grid["Sb"], KOBAYASHI)
    with pytest.warns(DeprecationWarning):
        old = sediment_transport_Kobayashi(0, data, dict(grid), KOBAYASHI)
    for key in new:
        np.testing.assert_array_equal(old[key], new[key])
    assert np.isfinite(new["qc"]).all() and np.isfinite(new["ql"]).all()


def test_cshore_input_file(tmp_path):
    props = cshore.default_config()
    props.update(header="test", dx=1.0, d50=0.0003, wf=0.03, Tp=8.0, Hrms=1.0, angle=0.0,
                 x=np.arange(3.0), zb=np.array([-2.0, -1.0, 0.5]), fw=np.full(3, 0.015))
    cshore.write_input(props, str(tmp_path))
    lines = (tmp_path / "infile").read_text().splitlines()
    assert lines[1].strip() == "test" and "->NBINP" in lines[-4]
    assert [float(v) for v in lines[-1].split()] == [2.0, 0.5, 0.015]


def test_run_executable_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="CSHORE executable not found"):
        cshore.run(str(tmp_path / "cshore.exe"), str(tmp_path))


def write_copla_velocity(path, nx, ny, rng):
    lines = ["header"] * 7
    k = 0
    for x in range(nx):
        for y in range(ny):
            k += 1
            lines.append(f"{k} {x * 10.0} {y * 10.0} {rng.uniform(-1, 1):.4f} {rng.uniform(-1, 1):.4f}")
    path.write_text("\n".join(lines) + "\n")


def test_read_copla_keeps_existing_grid(tmp_path):
    write_copla_velocity(tmp_path / "vel.001", 4, 6, np.random.default_rng(1))
    grid = swan_copla.read_copla(tmp_path / "vel.001", {"depth": np.ones((6, 8))})
    assert "depth" in grid and grid["U"].shape == (6, 8)
    assert np.allclose(grid["U"], np.hypot(grid["u"], grid["v"]))


FAKE_SWAN = """
import os, numpy as np
from scipy.io import savemat
case = os.path.basename(os.getcwd())
assert os.path.exists("swaninit") and os.path.exists(f"input_local_{case}.swn")
ny, nx = 6, 8
x, y = np.meshgrid(np.arange(nx) * 10.0, np.arange(ny) * 10.0)
rng = np.random.default_rng(2)
savemat(f"{case}.mat", {"Xp": x, "Yp": y, "Depth": 2 + 0.1 * x, "Qb": rng.uniform(0, 1, x.shape),
                        "Wlen": np.full(x.shape, 50.0), "Setup": np.full(x.shape, 0.05),
                        "Hsig": np.full(x.shape, 1.0), "Dir": np.full(x.shape, 200.0)})
"""

FAKE_COPLA = """
import os, numpy as np
case = os.path.basename(os.getcwd())
assert os.path.exists(f"{case}in.dat") and os.path.exists(f"{case}out.dat")
rng = np.random.default_rng(3)
lines = ["header"] * 7
k = 0
for x in range(4):
    for y in range(6):
        k += 1
        lines.append(f"{k} {x * 10.0} {y * 10.0} {rng.uniform(-1, 1):.4f} {rng.uniform(-1, 1):.4f}")
open(f"{case}vel.001", "w").write("\\n".join(lines) + "\\n")
"""


def fake_executable(path, code):
    path.write_text(f"#!{sys.executable}\n{code}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.mark.skipif(platform.system() == "Windows", reason="fake executables are scripts")
def test_nesting_runs_swan_copla_and_kobayashi(tmp_path):
    swan_exe = fake_executable(tmp_path / "swan", FAKE_SWAN)
    copla_exe = fake_executable(tmp_path / "copla", FAKE_COPLA)
    project = tmp_path / "project"
    (project / "0001").mkdir(parents=True)
    params = dict(KOBAYASHI, cwd="", directory=str(project), project_name="test", local_coords_x=0.0,
                  local_coords_y=0.0, local_length_x=70.0, local_length_y=50.0, local_nodes_x=8, local_nodes_y=6,
                  local_inc_x=10.0, local_inc_y=10.0, swan_executable=swan_exe, copla_executable=copla_exe)
    data = pd.DataFrame({"Hs": [1.5], "Tp": [8.0], "DirM": [200.0], "eta": [0.3], "Vv": [5.0], "DirV": [180.0]})

    grid = swan_copla.nesting(0, 0, params, data)

    case = project / "0001"
    assert {"0001.mat", "0001vel.001", "0001in.dat", "CLAVE.DAT"} <= set(os.listdir(case))
    assert grid["depth"].shape == grid["U"].shape == grid["qc"].shape == (6, 8)
    assert np.isfinite(grid["qc"]).all() and np.isfinite(grid["ql"]).all()

    # SWAN is not run again when its result exists
    os.remove(swan_exe)
    swan_copla.nesting(0, 0, params, data)


def test_delft_plot_grids(tmp_path):
    pytest.importorskip("cartopy")
    import matplotlib

    matplotlib.use("Agg")
    import environmentaltools.graphics  # noqa: F401  (sets matplotlib rc on import)

    x, y = np.meshgrid(300000 + np.arange(5) * 1000.0, 4.07e6 + np.arange(4) * 1000.0)
    lines = ["Coordinate System = Cartesian", "Missing Value = -999", "       5       4", " 0 0 0"]
    for arr in (x, y):
        for j in range(4):
            lines.append(f" ETA=    {j + 1}   " + "   ".join(f"{v:.3f}" for v in arr[j]))
    (tmp_path / "g.grd").write_text("\n".join(lines))
    with matplotlib.rc_context({"text.usetex": False}):
        ax = delft3d.plot_grids({"g": tmp_path / "g.grd"}, fname=str(tmp_path / "grids.png"))
    assert (tmp_path / "grids.png").exists() and ax is not None
