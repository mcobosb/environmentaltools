"""Creation of BARRIER cases: a copy of a base case with changes, one by one or as project variants."""

import json
import shutil
from pathlib import Path

import pandas as pd

from ..swan.config import deep_merge
from .config import read_config, write_config
from .inputs import read_csv, set_section_parameter, write_csv


def create_case(case_dir, base, config=None, along_channel=None, files=None, overwrite=False):
    """Create a case folder from a base case, optionally changing its configuration and inputs.

    Args:
        case_dir (str | Path): New case folder.
        base (str | Path): Base case folder (``config.yaml`` + input files).
        config (dict | None): Changes to ``config.yaml``, merged into it
            (e.g. ``{"hydrodynamics": {"courant_number": 0.2}}``).
        along_channel (dict | None): Changes to the along-channel parameters,
            ``{column: kwargs of set_section_parameter}`` (e.g.
            ``{"nmann": {"scale": 1.1}, "kh": {"points": [[0, 10], [50000, 300]], "sigma": 50}}``).
        files (dict | None): Input files replaced by others, ``{name in the case: source file}``.
        overwrite (bool): Replace ``case_dir`` if it exists.

    Returns:
        Path: The case folder.
    """
    case_dir, base = Path(case_dir), Path(base)
    if not (base / "config.yaml").exists():
        raise FileNotFoundError(f"No config.yaml in the base case {base}")
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{case_dir} already exists (overwrite=True to replace it)")
        shutil.rmtree(case_dir)
    shutil.copytree(base, case_dir)
    if config:
        write_config(deep_merge(read_config(case_dir), config), case_dir)
    for name, src in (files or {}).items():
        shutil.copy2(src, case_dir / name)
    if along_channel:
        f = case_dir / read_config(case_dir)["geometry"]["along_channel_file"]
        table = read_csv(f)
        for column, kwargs in along_channel.items():
            table = set_section_parameter(table, column, **(kwargs or {}))
        write_csv(table, f)
    return case_dir


def prepare(project, overwrite=False, verbose=True):
    """Create one case folder per variant of a project in ``cases_dir``.

    Args:
        project (BarrierProject): Project configuration.
        overwrite (bool): Rewrite existing case folders (otherwise they are kept).

    Returns:
        pd.DataFrame: Manifest (case, folder, changes), also written as ``cases_dir/cases.csv``.
    """
    root = project.cases_dir
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, v in project["variants"].items():
        v = v or {}
        case_dir = root / name
        if case_dir.exists() and not overwrite:
            status = "kept"
        else:
            files = {k: project.path(f) for k, f in (v.get("files") or {}).items()}
            create_case(case_dir, project.base_case, v.get("config"), v.get("along_channel"), files, overwrite)
            (case_dir / "variant.json").write_text(json.dumps(v, indent=1, default=str))
            status = "created"
        rows.append({"case": name, "folder": str(case_dir), "status": status,
                     "changes": json.dumps(v, default=str)})
        if verbose:
            print(f"{name}: {status}")
    df = pd.DataFrame(rows)
    df.to_csv(root / "cases.csv", index=False)
    return df
