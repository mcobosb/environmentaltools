import copernicusmarine
import os

def descargar_copernicus(dataset_id, dataset_version, variables, 
                        coords, fechas, output_path):
    """Función estable para descarga de datos de oleaje.

    Args:
        dataset_id (str): Identifier of the Copernicus dataset.
        dataset_version (str): Version of the dataset to be downloaded.
        variables (list): List of variables to include in the subset.
        coords (dict): Dictionary containing spatial limits ('lon_min', 'lon_max', 'lat_min', 'lat_max').
        fechas (dict): Dictionary with time range ('inicio', 'fin').
        output_path (str): Directory path where the file will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    filename = f"oleaje_{fechas['inicio'][:10]}.nc"
    
    try:
        print(f"Iniciando descarga de {dataset_id}...")
        copernicusmarine.subset(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            variables=variables,
            start_datetime=fechas['inicio'],
            end_datetime=fechas['fin'],
            minimum_longitude=coords['lon_min'],
            maximum_longitude=coords['lon_max'],
            minimum_latitude=coords['lat_min'],
            maximum_latitude=coords['lat_max'],
            coordinates_selection_method="strict-inside",
            netcdf_compression_level=1,
            output_directory=output_path,
            output_filename=filename,
            force_download=True
        )
        print(f"✅ Archivo guardado: {filename}")
    except Exception as e:
        print(f"❌ Error en la descarga: {e}")