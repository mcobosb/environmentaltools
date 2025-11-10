import drone_missions
from pathlib import Path


"""Main function with configuration dictionary instead of command line arguments."""

# Configuration dictionary
config = {
    "path": "output/scan_pattern",
    "polygon_no": "007",  # número de polígono como string para el nombre del archivo
    "template_folder": Path("data/template_207BCB3D-29F8-466B-AF1C-55CEDC044786.kmz"),
    "chunk_size": 50,
    "take": "first",  # choices: ["first", "last"]
    "out_dir": "output/missions",  # Optional output directory for KMZ files
    "missions_csv": None,  # CSV with names to use for KMZ files (one per line)
}

# Derived paths based on configuration
csv_path = Path(config["path"]) / f"waypoints_dji_poly_{config['polygon_no']}.csv"

# Call the main function with configuration parameters
drone_missions.create(
    config["template_folder"],
    csv_path,
    chunk_size=config["chunk_size"],
    take=config["take"],
    out_dir=config["out_dir"],
    polygon_no=config["polygon_no"],
)