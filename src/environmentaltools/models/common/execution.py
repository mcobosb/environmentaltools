"""Helpers to locate model executables and run them as external processes."""

import os
import platform
import subprocess
from pathlib import Path


def find_executable(names, search_dirs=(), exclude=("guis",)):
    """Find the first executable called one of ``names`` inside ``search_dirs``.

    Args:
        names (list[str]): Executable names in order of preference
            (e.g. ``["swan_omp.exe", "swan.exe"]``).
        search_dirs (list[str | Path]): Folders searched recursively.
        exclude (tuple[str]): Path parts to skip (Delft3D ``guis`` folders ship
            other versions of the same DLLs).

    Returns:
        Path | None: Executable path, or None.
    """
    for name in names:
        for d in search_dirs:
            d = Path(d)
            if not d.exists():
                continue
            found = sorted(p for p in d.rglob(name) if not set(exclude) & set(p.parts))
            if found:
                return found[0]
    return None


def library_environment(exe, extra_dirs=(), threads=None, exclude=("guis",)):
    """Environment to run ``exe`` with its shared libraries on the PATH.

    On Windows, adds the executable folder and every folder with DLLs under its
    parent (e.g. Delft3D ``kernels\\x64\\lib``), excluding ``exclude`` parts.

    Args:
        exe (Path): Executable.
        extra_dirs (list): More folders to prepend to the PATH.
        threads (int | None): ``OMP_NUM_THREADS``; all cores when None.

    Returns:
        dict: Environment variables.
    """
    exe = Path(exe)
    env = os.environ.copy()
    dirs = [str(exe.parent)] + [str(d) for d in extra_dirs]
    if platform.system() == "Windows":
        dirs += sorted({str(f.parent) for f in exe.parent.parent.rglob("*.dll")
                        if not set(exclude) & set(f.parts)} - {str(exe.parent)})
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    env["OMP_NUM_THREADS"] = str(threads or os.cpu_count())
    return env


def run_process(cmd, cwd, env=None, log_file=None, timeout=None):
    """Run a command and write its screen output to ``log_file``.

    Returns:
        int | str: Return code, or ``"timeout"``.
    """
    cmd = [str(c) for c in (cmd if isinstance(cmd, (list, tuple)) else [cmd])]
    out = open(log_file, "w") if log_file else subprocess.DEVNULL
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        return "timeout"
    finally:
        if log_file:
            out.close()


def run_executable(executable, working_directory, model="model", download=None):
    """Run a model executable in ``working_directory`` and wait for it to finish.

    Args:
        executable (str | Path): Path to the executable.
        working_directory (str | Path): Folder with the model input files.
        model (str): Model name used in the error message.
        download (str | None): Where to get the model, added to the error message.

    Returns:
        int: Return code of the process.

    Raises:
        FileNotFoundError: If ``executable`` does not exist.
    """
    if not os.path.isfile(executable):
        hint = f"Please download {model} from: {download}\n" if download else ""
        raise FileNotFoundError(
            f"{model} executable not found at: {executable}\n{hint}"
            f"Provide the correct path to the {model} executable."
        )
    return subprocess.run(str(executable), cwd=working_directory).returncode


def describe_return_code(rc):
    """Readable Windows status codes (e.g. missing or conflicting DLLs)."""
    if not isinstance(rc, int) or rc == 0:
        return str(rc)
    code = f"0x{rc & 0xFFFFFFFF:08X}"
    hints = {"0xC0000135": "a DLL is missing", "0xC0000139": "DLL entry point not found (conflicting DLLs)",
             "0xC0000005": "access violation"}
    return f"{rc} ({code}{': ' + hints[code] if code in hints else ''})"
