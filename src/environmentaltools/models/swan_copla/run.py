"""Execution of the SWAN → COPLA → Kobayashi workflow."""

import os
import shutil

from environmentaltools.processes.coastal import bed_slope, kobayashi_transport

from ..common.execution import run_executable
from .inputs import write_copla_input, write_swan_input
from .read import read_copla, read_swan


def run_swan(swan_executable_path, working_directory):
    """Run SWAN (Simulating WAves Nearshore) in ``working_directory``.

    Args:
        swan_executable_path (str): Path to the SWAN executable.
        working_directory (str): Folder with ``swaninit`` and the SWAN command file.

    Returns:
        int: Return code of the process.

    Raises:
        FileNotFoundError: If the executable is not found.
    """
    return run_executable(swan_executable_path, working_directory, "SWAN",
                          "https://swanmodel.sourceforge.io/download/download.htm")


def run_copla(copla_executable_path, working_directory):
    """Run COPLA (wave-induced currents) in ``working_directory``.

    Args:
        copla_executable_path (str): Path to the COPLA executable.
        working_directory (str): Folder with the COPLA input files.

    Returns:
        int: Return code of the process.

    Raises:
        FileNotFoundError: If the executable is not found.
    """
    return run_executable(copla_executable_path, working_directory, "COPLA")


def nesting(k, l, params, data, swan_executable=None, copla_executable=None):
    """Run the local SWAN and COPLA models of one sea state and compute the sediment transport.

    Writes the local SWAN input and runs SWAN (unless ``{case}.mat`` already
    exists), writes the COPLA input from the SWAN results and runs COPLA, reads
    both results, computes the bed slope and the Kobayashi et
    al. (2008) transport with the local SWAN wave height and direction.

    Args:
        k (int): Sea state index (0-based); the case folder is ``str(k + 1).zfill(4)``.
        l: Index label of the sea state in ``data``.
        params (dict): Model parameters: ``cwd`` and ``directory`` (project folder),
            the local grid (``local_*``), the Kobayashi constants (see
            :func:`environmentaltools.processes.kobayashi_transport`) and, if not
            given as arguments, ``swan_executable`` and ``copla_executable``.
        data (pd.DataFrame): Boundary conditions with ``Hs``, ``Tp``, ``DirM``,
            ``eta``, ``Vv`` and ``DirV``.
        swan_executable (str, optional): Path to the SWAN executable.
        copla_executable (str, optional): Path to the COPLA executable.

    Returns:
        dict: Fields of SWAN (``depth``, ``Hs``, ``Setup``, ...), COPLA (``U``,
        ``DirU``), the slope ``Sb`` and the transport (``Qst``, ``Qbt``, ``qc``, ``ql``, ...).
    """
    case_id = str(k + 1).zfill(4)
    case_dir = os.path.join(params["cwd"], params["directory"], case_id)
    swan_executable = swan_executable or params["swan_executable"]
    copla_executable = copla_executable or params["copla_executable"]

    write_swan_input(k, l, case_id, data, params, mesh="local")
    if not os.path.exists(os.path.join(case_dir, case_id + ".mat")):
        run_swan(swan_executable, case_dir)

    write_copla_input(k, l, case_id, data, params, mesh="local")
    run_copla(copla_executable, case_dir)

    grid = read_copla(os.path.join(case_dir, case_id + "vel.001"))
    grid = read_swan(os.path.join(case_dir, case_id + ".mat"), grid)

    # SWAN (Dir) and COPLA (DirU, rotated in read_copla) directions share the SWAN Cartesian frame
    grid["Sbx"], grid["Sby"] = bed_slope(grid["x"], grid["y"], grid["depth"])
    grid.update(kobayashi_transport(grid["depth"], grid["Setup"], grid["Hs"], data.loc[l, "Tp"], grid["kp"],
                                    grid["DirM"], grid["Qb"], grid["U"], grid["DirU"], grid["Sbx"], grid["Sby"],
                                    params))
    return grid


def clean(params):
    """Remove the project folder if ``params["delete_folder"]`` is True.

    Args:
        params (dict): Model parameters with ``delete_folder``, ``cwd`` and ``directory``.
    """
    if params["delete_folder"]:
        shutil.rmtree(os.path.join(params["cwd"], params["directory"]), ignore_errors=True)
