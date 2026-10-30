"""Execution of BARRIER cases (one case or every variant of a project, resumable)."""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..common.execution import describe_return_code, run_process



def has_time_steps(nc):
    """True if the NetCDF ``nc`` can be opened and has at least one time step
    (BARRIER may end "successfully" without simulating, leaving an empty file)."""
    import xarray as xr

    try:
        with xr.open_dataset(nc) as ds:
            return ds.sizes.get("time", 0) > 0
    except (OSError, ValueError):
        return False


def is_readable(nc):
    """True if every variable of the NetCDF ``nc`` can be read in full.

    Catches outputs damaged while being written (e.g. HDF5 chunks corrupted when the file sits in a
    folder that a sync client reads at the same time), which open normally and fail only when the
    damaged time steps are read.
    """
    import netCDF4

    try:
        with netCDF4.Dataset(nc) as ds:
            for v in ds.variables.values():
                v[:]
        return True
    except (OSError, RuntimeError):
        return False


def barrier_executable(executable=None):
    """BARRIER executable: ``executable``, else the environment variable ``BARRIER_EXE``, else ``barrier`` on the PATH.

    Raises:
        FileNotFoundError: If none exists.
    """
    candidates = [executable, os.environ.get("BARRIER_EXE"), shutil.which("barrier")]
    for c in candidates:
        if c and Path(c).expanduser().exists():
            return Path(c).expanduser()
    raise FileNotFoundError(
        "BARRIER executable not found: pass 'executable', set the environment variable BARRIER_EXE "
        "or put 'barrier' on the PATH (build it from https://github.com/mcobosb/Barrier)")


def run_case(case_dir, out_dir, executable=None, timeout=None, console="barrier_console.txt", work_dir=None,
             verbose=True):
    """Run BARRIER for one case folder.

    BARRIER reads the inputs from the folder of ``config.yaml`` and writes the NetCDF
    and the log in the working folder, which here is ``out_dir``. The screen output
    goes to ``out_dir/<console>``. The run is accepted only if it writes a
    new NetCDF with at least one time step that can be read in full.

    With ``work_dir``, BARRIER runs in a new subfolder of that local folder and its files are
    moved to ``out_dir`` when it finishes. Use it when ``out_dir`` is in a synced folder (Synology,
    OneDrive…) or, from WSL, on the Windows disk (``/mnt/c``): writing the NetCDF there while the
    sync client reads it can corrupt it.

    Args:
        case_dir (str | Path): Case folder with ``config.yaml``.
        out_dir (str | Path): Output folder (created if needed).
        executable (str | Path | None): BARRIER executable (see :func:`barrier_executable`).
        timeout (float | None): Maximum run time in seconds.
        console (str): Name of the file with the screen output.
        work_dir (str | Path | None): Local folder where BARRIER writes while it runs
            (e.g. ``tempfile.gettempdir()``); ``None`` writes directly in ``out_dir``.

    Returns:
        Path: The NetCDF output.

    Raises:
        RuntimeError: If BARRIER fails or writes no valid output.
    """
    exe = barrier_executable(executable)
    config = Path(case_dir).resolve() / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(f"No config.yaml in {case_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="barrier_", dir=work_dir)) if work_dir else out_dir
    if verbose:
        print(f"BARRIER: {exe}\n  case:    {case_dir}\n  outputs: {out_dir}\n  screen:  {out_dir / console}"
              + (f"\n  running in: {run_dir}" if work_dir else ""))
    started = time.time()
    try:
        rc = run_process([exe, config], run_dir, log_file=run_dir / console, timeout=timeout)
    finally:
        if work_dir:   # move everything BARRIER wrote, also after a failure (for the screen output)
            for f in run_dir.iterdir():
                (out_dir / f.name).unlink(missing_ok=True)
                shutil.move(str(f), out_dir / f.name)
            shutil.rmtree(run_dir, ignore_errors=True)
    console = out_dir / console
    new = [p for p in out_dir.glob("*.nc") if p.stat().st_mtime >= started and has_time_steps(p)]
    if rc != 0 or not new:
        raise RuntimeError(f"BARRIER wrote no valid output (return code {describe_return_code(rc)}); see {console}")
    nc = max(new, key=os.path.getmtime)
    if not is_readable(nc):
        raise RuntimeError(f"BARRIER output {nc.name} is damaged (some time steps cannot be read). If it was written "
                           "in a synced folder or on /mnt/c from WSL, run with a local work_dir")
    if verbose:
        print(f"  done in {(time.time() - started) / 60:.1f} min: {nc.name}")
    return nc


def is_done(project, case):
    return (project.results_dir / case / "done.json").exists()


def run(project, cases=None, force=False, verbose=True):
    """Run the cases of a project that are not finished yet.

    Each case writes its outputs in ``results_dir/<case>``; a ``done.json`` marks it
    as finished and ``results_dir/run_log.csv`` keeps one line per run.

    Args:
        project (BarrierProject): Project configuration (cases created with :func:`prepare`).
        cases (list[str] | None): Run only these cases.
        force (bool): Run again finished cases (their previous results are removed).

    Returns:
        pd.DataFrame: Log of this execution.
    """
    exe = barrier_executable(project.path(project["executable"]))
    timeout = project["timeout_hours"] * 3600 if project["timeout_hours"] else None
    todo = [c for c in project.case_names if (project.cases_dir / c / "config.yaml").exists()]
    if cases:
        todo = [c for c in todo if c in set(cases)]
    pending = todo if force else [c for c in todo if not is_done(project, c)]
    if verbose:
        print(f"BARRIER: {exe}\nCases: {len(todo)} | pending: {len(pending)}")
    log = []
    for k, case in enumerate(pending, 1):
        res = project.results_dir / case
        if res.exists():
            shutil.rmtree(res)
        row = {"case": case, "start": datetime.now().isoformat(timespec="seconds")}
        t0 = time.time()
        try:
            nc = run_case(project.cases_dir / case, res, exe, timeout, work_dir=project.path(project["work_dir"]),
                          verbose=False)
            row.update(ok=True, output=nc.name)
            (res / "done.json").write_text(json.dumps({**row, "minutes": round((time.time() - t0) / 60, 2)},
                                                      indent=1))
        except RuntimeError as exc:
            row.update(ok=False, error=str(exc))
        row["minutes"] = round((time.time() - t0) / 60, 2)
        log.append(row)
        if verbose:
            print(f"[{k}/{len(pending)}] {case}: {'OK' if row['ok'] else 'FAILED – ' + row['error']} "
                  f"({row['minutes']} min)")
    df = pd.DataFrame(log)
    if len(df):
        f = project.results_dir / "run_log.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, mode="a", header=not f.exists(), index=False)
    return df
