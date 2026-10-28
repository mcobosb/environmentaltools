"""Execution of BARRIER cases (one case or every variant of a project, resumable)."""

import json
import os
import shutil
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


def run_case(case_dir, out_dir, executable=None, timeout=None, console="barrier_console.txt", verbose=True):
    """Run BARRIER for one case folder.

    BARRIER reads the inputs from the folder of ``config.yaml`` and writes the NetCDF
    and the log in the working folder, which here is ``out_dir``. The screen output
    goes to ``out_dir/<console>``. The run is accepted only if it writes a
    new NetCDF with at least one time step.

    Args:
        case_dir (str | Path): Case folder with ``config.yaml``.
        out_dir (str | Path): Output folder (created if needed).
        executable (str | Path | None): BARRIER executable (see :func:`barrier_executable`).
        timeout (float | None): Maximum run time in seconds.
        console (str): Name of the file with the screen output.

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
    console = out_dir / console
    if verbose:
        print(f"BARRIER: {exe}\n  case:    {case_dir}\n  outputs: {out_dir}\n  screen:  {console}")
    started = time.time()
    rc = run_process([exe, config], out_dir, log_file=console, timeout=timeout)
    new = [p for p in out_dir.glob("*.nc") if p.stat().st_mtime >= started and has_time_steps(p)]
    if rc != 0 or not new:
        raise RuntimeError(f"BARRIER wrote no valid output (return code {describe_return_code(rc)}); see {console}")
    nc = max(new, key=os.path.getmtime)
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
            nc = run_case(project.cases_dir / case, res, exe, timeout, verbose=False)
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
