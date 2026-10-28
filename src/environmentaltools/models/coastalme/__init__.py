"""CoastalME coastal morphology model."""

from ..common.execution import run_executable


def run(executable_path, working_directory):
    """Run CoastalME in ``working_directory``.

    Args:
        executable_path (str): Path to the CoastalME executable.
        working_directory (str): Folder where CoastalME is executed.

    Returns:
        int: Return code of the process.

    Raises:
        FileNotFoundError: If the executable is not found.
    """
    return run_executable(executable_path, working_directory, "CoastalME", "https://github.com/coastalme/")


__all__ = ["run"]
