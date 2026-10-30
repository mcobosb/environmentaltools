"""Batch execution of prepared SWAN cases (resumable; nested grids run in sequence)."""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..common.execution import describe_return_code, library_environment, run_process
from .cases import COMMON

EXECUTABLES = ["swan_omp.exe", "swan.exe", "swan_omp", "swan"]


def swan_executable(cfg):
    """SWAN executable: the environment variable ``SWAN_EXE``, else the configured path (or list of paths,
    e.g. one per computer: the first that exists), else, with ``executable: auto``, the first
    ``swan_omp.exe``/``swan.exe`` found in ``search_dirs`` (Delft3D ships SWAN in ``…\\x64\\swan\\bin``
    or ``kernels\\x64\\bin``) or on the PATH."""
    from ..common.execution import resolve_executable

    exe = cfg["executable"]
    given = [cfg.path(e) for e in exe] if isinstance(exe, (list, tuple)) else (
        None if exe in (None, "auto") else cfg.path(exe))
    if given is not None and not isinstance(given, list) and not given.exists() and not os.environ.get("SWAN_EXE"):
        raise FileNotFoundError(f"SWAN executable not found: {given}")      # a single configured path must exist
    return resolve_executable(given, "SWAN_EXE", EXECUTABLES, cfg["search_dirs"], model="SWAN")


def is_done(cfg, case):
    return (cfg.results_dir / case / "done.json").exists()


def run(cfg, filter=None, cases=None, threads=None, force=False, verbose=True):
    """Run the prepared cases that are not finished yet.

    For each case: copy the case and the ``_common`` files to ``run_dir`` (local
    disk), run every grid in order (``INPUT_<k>_<grid>`` → ``INPUT``; ``PRINT`` →
    ``PRINT_<grid>``), stop if a grid fails, and copy the outputs (tables,
    spectra, PRINT, Errfile, screen logs, INPUT files) to ``results_dir/<case>``.
    Maps with ``store: archive`` are moved to ``archive_dir/<case>`` instead (heavy
    files kept only on the computing machine). With ``outputs: figures:``, a PNG of
    every saved map is drawn on the computing machine and copied to the results.
    A ``done.json`` marks finished cases; ``results_dir/run_log.csv`` keeps one
    line per case.

    Args:
        cfg (SwanConfig): Configuration.
        filter (str | None): Run only cases whose name contains this text.
        cases (list[str] | None): Run only these cases.
        threads (int | None): OpenMP threads (default: ``threads`` of the YAML or all cores).
        force (bool): Run again cases that are already finished (``done.json`` is replaced).

    Returns:
        pd.DataFrame: Log of this execution.
    """
    exe = swan_executable(cfg)
    env = library_environment(exe, threads=threads or cfg["threads"])
    root, res_root = cfg.cases_dir, cfg.results_dir
    run_root = cfg.path(cfg["run_dir"]) if cfg["run_dir"] else res_root.parent / "_run"
    timeout = cfg["timeout_hours"] * 3600 if cfg["timeout_hours"] else None
    todo = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != COMMON and (d / "case.json").exists())
    if filter:
        todo = [c for c in todo if filter in c]
    if cases:
        todo = [c for c in todo if c in set(cases)]
    pending = todo if force else [c for c in todo if not is_done(cfg, c)]
    if verbose:
        print(f"SWAN: {exe}\nCases: {len(todo)} | pending: {len(pending)} | run folder: {run_root}")

    log = []
    for k, case in enumerate(pending, 1):
        meta = json.loads((root / case / "case.json").read_text())
        wd = run_root / case
        if wd.exists():
            shutil.rmtree(wd)
        shutil.copytree(root / case, wd)
        for f in (root / COMMON).iterdir():
            shutil.copy2(f, wd / f.name)
        row = {"case": case, "start": datetime.now().isoformat(timespec="seconds")}
        ok = True
        for st in meta["stages"]:
            g = st["grid"]
            shutil.copy2(wd / st["input"], wd / "INPUT")
            for f in ("PRINT", "Errfile", "norm_end"):
                (wd / f).unlink(missing_ok=True)
            t0 = time.time()
            rc = run_process(exe, wd, env, wd / f"screen_{g}.log", timeout)
            row[f"min_{g}"] = round((time.time() - t0) / 60, 2)
            row[f"rc_{g}"] = describe_return_code(rc)
            for f in ("PRINT", "Errfile"):
                if (wd / f).exists():
                    (wd / f).replace(wd / f"{f}_{g}")
            expected = [v for v in meta["outputs"][g].values() if v]
            ok = all((wd / f).exists() for f in expected) and (wd / f"PRINT_{g}").exists()
            if not ok:
                row["error"] = f"grid {g}: missing outputs (return code {row[f'rc_{g}']})"
                break
        res = res_root / case
        if res.exists():                                        # a re-run replaces all previous results
            shutil.rmtree(res)
        res.mkdir(parents=True)
        archived = {f"map_{g}.tab" for g in (cfg["outputs"].get("maps") or {}) if cfg.map_store(g) == "archive"}
        for f in wd.iterdir():
            if f.name in archived:
                continue
            if f.name.endswith((".tab", ".sp2", ".log")) and not f.name.startswith("nest_") or \
                    f.name.startswith(("PRINT_", "Errfile_", "INPUT_")) or f.name == "case.json":
                shutil.copy2(f, res / f.name)
        if ok and archived:
            arch = cfg.archive_dir / case
            arch.mkdir(parents=True, exist_ok=True)
            for name in archived:
                if (wd / name).exists():
                    shutil.copy2(wd / name, arch / name)
        if ok and cfg["outputs"].get("figures"):
            try:
                from .plot import plot_case_maps
                import matplotlib
                matplotlib.use("Agg")
                plot_case_maps(cfg, case, folder=wd, file_name=res / f"maps_{case}.png")
            except Exception as exc:                            # a figure never stops the batch
                row["figure_error"] = str(exc)[:200]
        row["ok"] = ok
        if ok:
            (res / "done.json").write_text(json.dumps(row, indent=1))
            if not cfg["keep_run_dir"]:
                shutil.rmtree(wd, ignore_errors=True)
        log.append(row)
        if verbose:
            times = ", ".join(f"{s['grid']} {row.get('min_' + s['grid'], '-')} min" for s in meta["stages"])
            print(f"[{k}/{len(pending)}] {case}: {'OK' if ok else 'FAILED – ' + row['error']} ({times})")
        if not ok and k == 1:
            print(f"The first case failed: check screen_*.log and PRINT_* in {wd}")
            break
    df = pd.DataFrame(log)
    if len(df):
        f = res_root / "run_log.csv"
        df.to_csv(f, mode="a", header=not f.exists(), index=False)
    return df
