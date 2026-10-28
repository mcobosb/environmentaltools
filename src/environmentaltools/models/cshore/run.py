"""Execution of CSHORE."""

from ..common.execution import run_executable


def run(executable_path, working_directory):
    """Run CSHORE (Coastal Storm Hindcast of Reshaping and Erosion) in ``working_directory``.

    Args:
        executable_path (str): Path to the CSHORE executable.
        working_directory (str): Folder with the ``infile`` written by :func:`write_input`.

    Returns:
        int: Return code of the process.

    Raises:
        FileNotFoundError: If the executable is not found.
    """
    return run_executable(executable_path, working_directory, "CSHORE", "https://github.com/erdc/cshore")
