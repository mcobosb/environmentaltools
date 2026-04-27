import cdsapi
import zipfile
import xarray as xr
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Any, Optional

class ProjectionDataConfig:
    """Configuración estructurada para Proyecciones Climáticas."""
    def __init__(self, config: dict):
        self.dataset_name = "sis-ocean-wave-timeseries"
        
        # Nombres de parámetros exactos para la API CDS-Beta
        self.variables = config.get('variables', ["Significant wave height"])
        self.experiment = config.get('experiment', "RCP8.5")
        self.years = config.get('years', ["2041"])
        
        # Gestión de directorios
        self.output_directory = Path(config.get('output_directory', "./data_projections"))
        self.output_directory.mkdir(parents=True, exist_ok=True)

class ProjectionDownloader:
    """Manejador de descargas para el Climate Data Store."""
    def __init__(self, config: ProjectionDataConfig):
        self.config = config
        self.client = cdsapi.Client()

    def download(self, filename: str) -> Optional[Path]:
        """Ejecuta la descarga desde el servidor de Copernicus."""
        target_path = self.config.output_directory / filename
        
        request = {
            "variable": self.config.variables,
            "experiment": self.config.experiment,
            "year": self.config.years
        }

        try:
            logger.info(f"Conectando a CDS para descargar: {self.config.years}")
            self.client.retrieve(self.config.dataset_name, request).download(str(target_path))
            logger.success(f"Descarga completa: {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"Error en la petición: {e}")
            return None

import os
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from loguru import logger

class ProjectionProcessor:
    """
    Clase para procesar, filtrar y visualizar proyecciones climáticas de oleaje.
    Maneja archivos ZIP, NetCDF con mallas no estructuradas y visualización cartográfica.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.temp_dir = self.config.output_directory / "temp_nc_files"

    def _prepare_files(self, file_path: Path) -> str:
        """Gestiona la descompresión y verifica que los archivos existan."""
        file_path = Path(file_path)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Si es un ZIP, verificamos si ya ha sido extraído comparando si hay archivos .nc
        if zipfile.is_zipfile(file_path):
            nc_existentes = list(self.temp_dir.glob("*.nc"))
            
            if not nc_existentes:
                logger.info(f"Extrayendo contenido de {file_path.name} en {self.temp_dir}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
            else:
                logger.info(f"Usando archivos .nc ya extraídos en {self.temp_dir}")
            
            return str(self.temp_dir / "*.nc")
        
        return str(file_path)

    def plot_data_coverage(self, file_path: Path, extent: list = [-8.0, -1.0, 35.0, 39.0], target_coords: list = None):
        """
        Genera figura de la malla, resalta el punto objetivo y calcula la distancia real en km.
        target_coords: [longitud, latitud]
        """
        path_to_open = self._prepare_files(file_path)
        
        try:
            # Abrir primer archivo para obtener la malla
            files = list(Path(self.temp_dir).glob("*.nc")) if "*.nc" in path_to_open else [Path(path_to_open)]
            ds = xr.open_dataset(files[0], engine='netcdf4')
            
            # 1. Extraer y normalizar coordenadas del modelo
            lon_grid = ds['station_x_coordinate'].values
            lat_grid = ds['station_y_coordinate'].values
            # Normalizamos longitudes a rango [-180, 180]
            lon_grid_norm = (lon_grid + 180) % 360 - 180

            fig = plt.figure(figsize=(10, 7))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)

            # Dibujar malla base
            ax.scatter(lon_grid_norm, lat_grid, s=1.5, c='red', alpha=0.3, 
                       transform=ccrs.PlateCarree(), label='Malla del modelo')

            if target_coords:
                t_lon, t_lat = target_coords
                # Normalizar longitud del punto buscado
                t_lon_norm = (t_lon + 180) % 360 - 180

                # 2. CÁLCULO DE DISTANCIA HAVERSINE (en km)
                # Convertimos todo a radianes
                phi1, lam1 = np.radians(lat_grid), np.radians(lon_grid_norm)
                phi2, lam2 = np.radians(t_lat), np.radians(t_lon_norm)
                
                dphi = phi2 - phi1
                dlam = lam2 - lam1
                
                a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                dist_km = 6371 * c # Radio de la Tierra en km
                
                # Encontrar el índice del mínimo
                idx_closest = dist_km.argmin()
                min_dist = dist_km[idx_closest]
                
                closest_lon = lon_grid_norm[idx_closest]
                closest_lat = lat_grid[idx_closest]

                # 3. Representación visual
                # Tu punto (Estrella)
                ax.scatter(t_lon_norm, t_lat, s=150, c='gold', marker='*', edgecolors='black', 
                           zorder=10, transform=ccrs.PlateCarree(), 
                           label=f'Objetivo: {t_lon}, {t_lat}')
                
                # Nodo más cercano (Círculo)
                ax.scatter(closest_lon, closest_lat, s=100, facecolors='none', edgecolors='blue', 
                           linewidth=2, zorder=11, transform=ccrs.PlateCarree(),
                           label=f'Más cercano (Dist: {min_dist:.2f} km)')

                logger.info(f"Punto más cercano ID {idx_closest}: {closest_lon:.4f}, {closest_lat:.4f}")
                logger.info(f"Distancia real al nodo: {min_dist:.3f} km")

            plt.legend(loc='lower right', frameon=True, fontsize='small')
            plt.title(f"Verificación de Malla y Distancia\n{self.config.experiment}", pad=20)
            
            output_img = self.config.output_directory / "mapa_verificacion_distancia.png"
            plt.savefig(output_img, dpi=300, bbox_inches='tight')
            logger.success(f"Mapa generado con distancia calculada.")
            plt.show()

        except Exception as e:
            logger.error(f"Error en el cálculo de cercanía: {e}")

    
    def process_to_dataframe(self, file_path: Path, lat_bounds: list = None, lon_bounds: list = None) -> Optional[pd.DataFrame]:
        """
        Convierte los archivos NetCDF a un DataFrame de Pandas filtrado por coordenadas.
        """
        path_to_open = self._prepare_files(file_path)
        
        try:
            logger.info("Cargando y uniendo archivos NetCDF...")
            ds = xr.open_mfdataset(path_to_open, combine='by_coords', engine='netcdf4')
            
            # Convertimos a DataFrame
            logger.info("Transformando a DataFrame (esto consume RAM)...")
            df = ds.to_dataframe().reset_index()

            # Identificar nombres de columnas de coordenadas
            lon_col = next((c for c in ['station_x_coordinate', 'longitude', 'lon'] if c in df.columns), None)
            lat_col = next((c for c in ['station_y_coordinate', 'latitude', 'lat'] if c in df.columns), None)

            if lat_bounds and lon_bounds and lat_col and lon_col:
                logger.info(f"Filtrando zona: Lat {lat_bounds}, Lon {lon_bounds}")
                mask = (df[lat_col] >= lat_bounds[0]) & (df[lat_col] <= lat_bounds[1]) & \
                       (df[lon_col] >= lon_bounds[0]) & (df[lon_col] <= lon_bounds[1])
                df = df[mask]

            # Limpieza de valores nulos (tierra) y renombrado
            df = df.dropna()
            if lon_col and lat_col:
                df = df.rename(columns={lon_col: 'longitude', lat_col: 'latitude'})

            logger.success(f"Procesamiento finalizado. Filas: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Fallo en el procesamiento: {e}")
            return None