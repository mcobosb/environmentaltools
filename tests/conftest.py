# conftest.py
"""Pytest configuration and fixtures for environmentaltools testing.

This file contains shared fixtures and configuration for testing
all modules in the environmentaltools package.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fixture para datos temporales simulados (legacy - mantenido para compatibilidad)
@pytest.fixture
def sample_temporal_df():
    """Legacy fixture for backward compatibility."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    values = np.random.rand(5)
    return pd.DataFrame({"timestamp": dates, "value": values})

# Nuevas fixtures para testing comprehensivo del módulo temporal
@pytest.fixture
def sample_time_series_simple():
    """Simple time series for basic testing."""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.random.normal(10, 2, len(dates))
    return pd.DataFrame({
        "date": dates,
        "value": values
    })

@pytest.fixture
def sample_environmental_data():
    """Realistic environmental data with multiple variables."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    n = len(dates)
    
    # Simulate realistic environmental variables
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n)
    wave_height = np.abs(np.random.weibull(2, n) * 1.5 + 0.3)
    wind_speed = np.abs(np.random.gamma(2, 2, n))
    
    return pd.DataFrame({
        "date": dates,
        "temperature": temperature,
        "wave_height": wave_height,
        "wind_speed": wind_speed
    })

@pytest.fixture
def sample_extreme_events_data():
    """Data specifically designed for extreme value analysis."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=2000, freq="6H")
    
    # Generate data with realistic extreme structure
    baseline = np.random.gamma(2, 1, len(dates))
    extremes_mask = np.random.random(len(dates)) < 0.05  # 5% extreme events
    extremes = np.random.pareto(1.5, len(dates)) * 3
    values = baseline + extremes_mask * extremes
    
    return pd.DataFrame({
        "date": dates,
        "value": values
    })

@pytest.fixture
def sample_multivariate_correlated():
    """Multivariate data with known correlation structure."""
    np.random.seed(42)
    n = 500
    
    # Create correlated variables using Cholesky decomposition
    correlation_matrix = np.array([
        [1.0, 0.7, -0.3],
        [0.7, 1.0, 0.2],
        [-0.3, 0.2, 1.0]
    ])
    
    L = np.linalg.cholesky(correlation_matrix)
    uncorrelated = np.random.normal(0, 1, (n, 3))
    correlated = uncorrelated @ L.T
    
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    return pd.DataFrame({
        "date": dates,
        "var1": correlated[:, 0],
        "var2": correlated[:, 1], 
        "var3": correlated[:, 2]
    })

@pytest.fixture
def sample_seasonal_data():
    """Time series with clear seasonal patterns."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=1095, freq="D")  # 3 years
    t = np.arange(len(dates))
    
    # Multiple seasonal components
    annual = 5 * np.sin(2 * np.pi * t / 365.25)
    semi_annual = 2 * np.sin(4 * np.pi * t / 365.25)
    trend = 0.01 * t
    noise = np.random.normal(0, 1, len(dates))
    
    values = 20 + annual + semi_annual + trend + noise
    
    return pd.DataFrame({
        "date": dates,
        "value": values
    })

# Fixtures para datos espaciales
@pytest.fixture
def sample_spatial_points():
    """Simple spatial points for spatial module testing."""
    from shapely.geometry import Point
    return [Point(x, y) for x, y in zip(range(5), range(5))]

@pytest.fixture
def sample_spatial_grid():
    """Regular spatial grid for testing."""
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    xx, yy = np.meshgrid(x, y)
    
    return {
        "x": xx.flatten(),
        "y": yy.flatten(),
        "z": np.random.random(len(xx.flatten()))
    }

# Fixtures para configuración gráfica
@pytest.fixture
def sample_graphics_config():
    """Configuration for graphics module testing."""
    return {
        "dpi": 300,
        "color_scheme": "viridis",
        "output_format": "png",
        "figsize": (10, 6),
        "style": "seaborn"
    }

@pytest.fixture
def sample_plot_parameters():
    """Parameters for plotting functions."""
    return {
        "xlabel": "Time",
        "ylabel": "Value", 
        "title": "Test Plot",
        "grid": True,
        "legend": True
    }

# Fixtures para simulación de drones y datos de campo
@pytest.fixture
def drone_config():
    """Configuration for drone-based data collection."""
    return {
        "depth_sensor": "sonarX",
        "gps_enabled": True,
        "sampling_rate": 1.0,
        "altitude": 10.0,
        "speed": 5.0
    }

@pytest.fixture
def field_measurement_data():
    """Simulated field measurement data."""
    np.random.seed(42)
    n_points = 50
    
    return pd.DataFrame({
        "x": np.random.uniform(0, 100, n_points),
        "y": np.random.uniform(0, 100, n_points),
        "depth": np.random.uniform(0, 20, n_points),
        "temperature": np.random.normal(18, 3, n_points),
        "salinity": np.random.normal(35, 2, n_points)
    })

# Fixtures para testing de modelos y configuraciones
@pytest.fixture
def basic_model_config():
    """Basic configuration for numerical models."""
    return {
        "model_type": "swan",
        "grid_resolution": 100,
        "time_step": 3600,
        "boundary_conditions": "default",
        "output_variables": ["Hsig", "Dir", "Period"]
    }

@pytest.fixture
def temporary_directory(tmp_path):
    """Temporary directory for file I/O tests."""
    return tmp_path

# Configuration for all tests
@pytest.fixture(autouse=True)
def configure_test_environment():
    """Configure testing environment (runs before every test)."""
    # Suppress warnings during testing
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Configure matplotlib for headless testing
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend