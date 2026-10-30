"""Creation of BARRIER cases: a copy of a base case with changes, one by one or as project variants."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ..swan.config import deep_merge
from .config import input_files, read_config, write_config
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


#: Files written by :func:`prepare` in each case folder (not part of the case itself).
CASE_EXTRAS = ("variant.json", "base_signature.txt", "section_parameters.png")
#: Columns of ``along_channel_data`` summarised by :func:`cases_overview`.
SECTION_COLUMNS = ["nmann", "kh", "Sf", "beta"]


def folder_signature(folder, skip=CASE_EXTRAS):
    """SHA-1 of the names and contents of the files of a case folder, without the files added by :func:`prepare`."""
    from ..common.tables import folder_signature as signature

    return signature(folder, skip)


def _changed(case_dir, variant, base_sig):
    """Why an existing case folder no longer matches its entry of the list (``None`` if it does)."""
    saved = case_dir / "variant.json"
    if not saved.exists() or json.loads(saved.read_text()) != json.loads(json.dumps(variant, default=str)):
        return "changed in the list"
    sig = case_dir / "base_signature.txt"
    if not sig.exists() or sig.read_text() != base_sig:
        return "the base case changed"
    return None


def _along_channel(case_dir):
    return read_csv(Path(case_dir) / read_config(case_dir)["geometry"]["along_channel_file"])


#: Columns of a new cases table (examples of each kind of change).
TEMPLATE_COLUMNS = ["case", "description", "along_channel.nmann.scale", "along_channel.kh.scale",
                    "hydrodynamics.courant_number", "run.end_date", "files.tides.csv"]


def cases_to_table(cases):
    """Cases as written in a project file (``{case: {config, along_channel, files}}``) → cases table.

    Returns:
        pd.DataFrame: One row per case, one column per change (see
        :func:`~environmentaltools.models.barrier.config.read_cases_table`).
    """
    def flat(d, prefix):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) and v:
                out.update(flat(v, key + "."))
            else:
                out[key] = json.dumps(v) if isinstance(v, (list, dict)) else v
        return out

    rows = []
    for name, c in cases.items():
        row = {"case": name, **flat(c.get("config"), ""), **flat(c.get("along_channel"), "along_channel.")}
        row.update({f"files.{k}": v for k, v in (c.get("files") or {}).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    return df if len(df.columns) > 1 else df.reindex(columns=["case"])


def write_cases_table(file_name, cases=None, descriptions=None):
    """Write a cases table (Excel) to fill in: the given cases, or an example to start from.

    Args:
        file_name (str | Path): Output ``.xlsx``.
        cases (dict | None): Cases as in a project file; ``None`` writes an example row and the
            usual columns (:data:`TEMPLATE_COLUMNS`).
        descriptions (dict | None): ``{case: text}`` for the ``description`` column.

    Returns:
        Path: The written file.
    """
    from ._xlsx import write_tables

    if cases:
        table = cases_to_table(cases)
        table.insert(1, "description", [(descriptions or {}).get(c, "") for c in table["case"]])
    else:
        table = pd.DataFrame([{"case": "reference", "description": "base case without changes"},
                              {"case": "manning_x1.1", "description": "Manning n of every section × 1.1",
                               "along_channel.nmann.scale": 1.1}], columns=TEMPLATE_COLUMNS)
    return write_tables(Path(file_name), {"cases": table}, legend={
        "use": "One row per case: the base case with the changes of its row. An empty cell means 'as in the base "
               "case'. Add columns for other changes. After editing, run prepare (it only redoes what changed)",
        "case": "Name of the case (also the name of its folder)",
        "description": "Free text (ignored), as any column starting with '_'",
        "along_channel.<column>.<option>": "Section parameters (column: nmann, kh, Sf, beta; x in m from the upstream "
                                           "boundary). Options: value (constant), scale (×), add (+), points "
                                           "([[x, value], …], interpolated), steps ([[x_from, value], …], by "
                                           "stretches), sigma (Gaussian smoothing, in sections), file (csv x, value "
                                           "next to this table, used as points)",
        "files.<name in the case>": "File that replaces that input of the base case (relative to this table), e.g. "
                                    "files.tides.csv = tides_2010.csv",
        "<key.of.config.yaml>": "Any key of BARRIER's config.yaml written with dots, e.g. hydrodynamics.courant_number, "
                                "run.end_date (2010-03-01T00:00:00), transport.salinity.beta",
    })


def prepare(project, overwrite=False, overview=True, figures=True, verbose=True):
    """Create (or update) one case folder per case of a project in ``cases_dir``.

    Each case is a copy of ``base_case`` with the changes of its entry. A case whose
    entry or base case changed since it was created is created again and its result is
    marked as pending (its ``done.json`` is removed), so :func:`run` runs it again.
    Unchanged cases are kept as they are.

    In each case folder it also writes ``variant.json`` (its changes),
    ``base_signature.txt`` (fingerprint of the base case it comes from) and, with
    ``figures``, ``section_parameters.png`` (n, Kh, Sf and β of the case against the base).
    With ``overview``, the base case and all the cases side by side are written to
    ``project.overview_file`` (see :func:`cases_overview`) to review them before running.

    Args:
        project (BarrierProject): Project configuration.
        overwrite (bool): Create every case again, changed or not.
        overview (bool): Write the overview Excel file.
        figures (bool): Draw ``section_parameters.png`` in new cases.

    Returns:
        pd.DataFrame: Manifest (``case``, ``folder``, ``status``: created / updated / kept,
        ``changes``), also written as ``cases_dir/cases.csv``.
    """
    if project.cases_file is not None and not project.cases_file.exists():
        write_cases_table(project.cases_file)
        if verbose:
            print(f"Created the cases table {project.cases_file}: fill it in and run prepare again")
        if not project["variants"]:
            return pd.DataFrame(columns=["case", "folder", "status", "changes"])
    root = project.cases_dir
    root.mkdir(parents=True, exist_ok=True)
    base_sig = folder_signature(project.base_case)
    base_along = _along_channel(project.base_case) if figures else None
    rows = []
    for name, v in project["variants"].items():
        case_dir = root / name
        status = "created"
        if case_dir.exists():
            why = "overwrite" if overwrite else _changed(case_dir, v, base_sig)
            if why is None:
                status = "kept"
            else:
                status = "updated"
                (project.results_dir / name / "done.json").unlink(missing_ok=True)
                if verbose:
                    print(f"{name}: {why} → created again; its result is pending")
        if status != "kept":
            files = {k: project.path(f) for k, f in (v.get("files") or {}).items()}
            create_case(case_dir, project.base_case, v.get("config"), v.get("along_channel"), files, overwrite=True)
            (case_dir / "variant.json").write_text(json.dumps(v, indent=1, default=str))
            (case_dir / "base_signature.txt").write_text(base_sig)
        if figures and (status != "kept" or not (case_dir / "section_parameters.png").exists()):
            import matplotlib.pyplot as plt

            from .plot import plot_section_parameters

            plot_section_parameters({"base case": base_along, name: _along_channel(case_dir)},
                                    file_name=case_dir / "section_parameters.png")
            plt.close("all")
        rows.append({"case": name, "folder": str(case_dir), "status": status,
                     "changes": json.dumps(v, default=str)})
        if verbose:
            print(f"{name}: {status}")
    others = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in project["variants"])
    if others and verbose:
        print(f"Folders no longer in the list (not removed): {', '.join(others)}")
    df = pd.DataFrame(rows)
    df.to_csv(root / "cases.csv", index=False)
    if overview:
        write_overview(project)
        if verbose:
            print(f"Review before running: {project.overview_file}")
    return df


def _flatten(d):
    from ..common.tables import flatten

    return {k: "" if v == "" else str(v) for k, v in flatten(d).items()}


def _file_hashes(case_dir):
    return {k: hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else None
            for k, p in input_files(case_dir).items()}


def cases_overview(project):
    """The base case and every prepared case side by side, to review them before running.

    Returns:
        dict: Three DataFrames with a ``base`` column and one column per case:
        ``parameters`` (every value of ``config.yaml``, one row per key), ``files`` (input
        files; ``(DIFFERENT)`` when the content is not that of the base case) and
        ``section_parameters`` (min, mean and max along the channel of n, Kh, Sf and β).
    """
    folders = {"base": project.base_case, **{n: project.cases_dir / n for n in project.case_names
                                             if (project.cases_dir / n / "config.yaml").exists()}}
    flat = {n: _flatten(read_config(d)) for n, d in folders.items()}
    keys = list(dict.fromkeys(k for f in flat.values() for k in f))
    parameters = pd.DataFrame({"parameter": keys, **{n: [f.get(k, "") for k in keys] for n, f in flat.items()}})

    base_hashes = _file_hashes(project.base_case)
    rows = []
    for key, path in input_files(project.base_case).items():
        row = {"file": key, "base": path.name}
        for n, d in list(folders.items())[1:]:
            p = input_files(d).get(key)
            row[n] = ("(none)" if p is None else p.name if _file_hashes(d).get(key) == base_hashes[key]
                      else f"{p.name} (DIFFERENT)")
        rows.append(row)
    files = pd.DataFrame(rows, columns=["file", *folders])

    tables = {n: _along_channel(d) for n, d in folders.items()}
    stats = []
    for col in [c for c in SECTION_COLUMNS if c in tables["base"]]:
        for stat in ("min", "mean", "max"):
            stats.append({"parameter": col, "statistic": stat,
                          **{n: round(float(getattr(t[col], stat)()), 6) for n, t in tables.items()}})
    return {"parameters": parameters, "files": files,
            "section_parameters": pd.DataFrame(stats, columns=["parameter", "statistic", *folders])}


def write_overview(project, file_name=None):
    """Write :func:`cases_overview` to Excel; what differs from the base case is filled in orange.

    Returns:
        Path: The written file (``project.overview_file`` by default).
    """
    from ._xlsx import write_tables

    file_name = Path(file_name or project.overview_file)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    tables = cases_overview(project)
    cases = [c for c in tables["parameters"].columns[2:]]
    highlight = {k: ("base", cases[0]) for k in tables} if cases else None
    return write_tables(file_name, tables, highlight=highlight, legend={
        "sheet parameters": "Every value of config.yaml: one row per key, one column for the base case and one per "
                            "case. In orange, what differs from the base case",
        "sheet files": "Input files named by config.yaml. '(DIFFERENT)': the content is not that of the base case "
                       "(replaced with files: or changed with along_channel:)",
        "sheet section_parameters": "Min, mean and max along the channel of the section parameters "
                                    "(along_channel_data): nmann (Manning n), kh (dispersion, m²/s), Sf (lateral "
                                    "storage factor), beta. Full profiles: section_parameters.png in each case folder",
        "base": f"Base case: {project.base_case}",
        "cases": f"Case folders: {project.cases_dir}",
    })
