"""Functions moved from ``processes`` to ``environmentaltools.models`` (deprecated aliases)."""

SWAN_COPLA = "environmentaltools.models.swan_copla"
CSHORE = "environmentaltools.models.cshore"
DELFT3D = "environmentaltools.models.delft3d"
COASTALME = "environmentaltools.models.coastalme"

LOAD = {
    "create_mesh_dictionary": (SWAN_COPLA, "create_mesh_dictionary", {"fname": "file_name", "uf": "sheet_name"}),
    "read_cshore": (CSHORE, "read_output", {"file_": "file_type"}),
    "read_copla": (SWAN_COPLA, "read_copla", {"fname": "file_name"}),
    "read_swan": (SWAN_COPLA, "read_swan", {"fname": "file_name", "vars_": "variables"}),
    "delft_raw_files_point": (DELFT3D, "raw_files_point", {"vars_": "variables", "nocases": "num_cases"}),
    "delft_raw_files": (DELFT3D, "raw_files", {"vars_": "variables", "case_id_": "case_id"}),
    "extract_delft_wave_dat": (DELFT3D, "extract_wave_dat"),
}

WRITE = {
    "write_cshore": (CSHORE, "write_input", {"folder": "output_folder"}),
    "write_swan": (SWAN_COPLA, "write_swan_input", {"i": "case_index", "index_": "time_index"}),
    "write_copla": (SWAN_COPLA, "write_copla_input", {"i": "case_index", "index_": "time_index"}),
    "directory": (SWAN_COPLA, "create_project_directory"),
}

COMPUTE = {
    "create_db": (SWAN_COPLA, "create_db"),
    "create_mesh": (SWAN_COPLA, "create_mesh"),
    "create_xarray": (SWAN_COPLA, "create_xarray"),
    "save_db": (SWAN_COPLA, "save_db"),
    "clean": (SWAN_COPLA, "clean"),
    "nesting": (SWAN_COPLA, "nesting"),
    "run_swan": (SWAN_COPLA, "run_swan"),
    "run_copla": (SWAN_COPLA, "run_copla"),
    "run_cshore": (CSHORE, "run", {"cshore_executable_path": "executable_path"}),
    "run_coastalme": (COASTALME, "run", {"coastalme_executable_path": "executable_path"}),
}
