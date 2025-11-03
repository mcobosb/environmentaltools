"""Practical tests for environmentaltools.temporal module.

These tests use the actual functions available in the temporal module
and can be run without pytest, using only standard Python.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress warnings
warnings.filterwarnings("ignore")

def create_sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    
    # Create realistic environmental data
    t = np.arange(len(dates))
    seasonal = 2 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle
    trend = 0.01 * t  # Small trend
    noise = np.random.normal(0, 1, len(dates))
    values = 5 + seasonal + trend + noise
    
    return pd.DataFrame({
        "date": dates,
        "Hs": np.abs(values)  # Wave height should be positive
    })

def create_wave_data():
    """Create realistic wave height data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=2000, freq="6H")
    
    # Generate Weibull-distributed wave heights (realistic for ocean waves)
    values = np.random.weibull(2, len(dates)) * 2 + 0.3
    
    return pd.DataFrame({
        "date": dates,
        "Hs": values
    })

def test_stationary_analysis():
    """Test the stationary_analysis function from temporal.core."""
    print("Testing stationary_analysis...")
    
    try:
        from environmentaltools.temporal.core import stationary_analysis
        import scipy.stats as st
        
        # Create test data with datetime index
        df = create_sample_data()
        df.set_index('date', inplace=True)
        
        # Define parameters for stationary analysis (matching function signature)
        param = {
            'var': 'Hs',
            'basis_function': {'order': 0},  # 0 for stationary
            'reduction': False,
            'no_fun': 1,  # Single distribution
            'fun': {0: st.norm},  # Normal distribution
            'ws_ps': [],  # Empty for single distribution
            'piecewise': False,
            'fix_percentiles': True
        }
        
        # Run the analysis
        result = stationary_analysis(df, param)
        
        # Check results - returns (dataframe, params_tuple, mode_list)
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 3, "Result should have 3 elements (df, params, mode)"
        
        df_out, params, mode = result
        assert isinstance(df_out, pd.DataFrame), "First element should be DataFrame"
        assert isinstance(params, tuple), "Parameters should be a tuple"
        assert isinstance(mode, list), "Mode should be a list"
        assert len(params) > 0, "Should have fitted parameters"
        
        print("✓ stationary_analysis test passed")
        return True
        
    except Exception as e:
        print(f"✗ stationary_analysis test failed: {e}")
        return False

def test_fit_distribution():
    """Test the fit_distribution function."""
    print("Testing fit_distribution...")
    
    try:
        from environmentaltools.temporal.core import fit_distribution
        import scipy.stats as st
        
        # Create test data - fit_distribution expects data series, bins, and model
        df = create_wave_data()
        data_series = df["Hs"]
        
        # Test fitting a distribution (correct signature: data, bins, model)
        result = fit_distribution(data_series, bins=25, model=st.weibull_min)
        
        # Check that we get numpy array with parameters
        assert result is not None, "Should return a result"
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert len(result) > 0, "Should have fitted parameters"
        
        print("✓ fit_distribution test passed")
        return True
        
    except Exception as e:
        print(f"✗ fit_distribution test failed: {e}")
        return False

def test_nonstationary_analysis():
    """Test the nonstationary_analysis function."""
    print("Testing nonstationary_analysis...")
    
    try:
        from environmentaltools.temporal.core import nonstationary_analysis
        import scipy.stats as st
        
        # Create test data with datetime index
        df = create_sample_data()
        df.set_index('date', inplace=True)
        
        # Add normalized time column required for non-stationary analysis
        import datetime
        df["n"] = np.fmod(
            (df.index - datetime.datetime(df.index[0].year, 1, 1, 0)).total_seconds().values
            / (1 * 365.25 * 24 * 3600),  # Annual period
            1,
        )
        
        # Parameters for non-stationary analysis
        param = {
            'var': 'Hs',
            'basis_function': {
                'method': 'trigonometric',
                'order': 2,
                'no_terms': 2,
                'periods': [1, 0.5]
            },
            'basis_period': [1],
            'reduction': False,
            'no_fun': 1,
            'fun': {0: st.norm},
            'ws_ps': [],
            'piecewise': False,
            'fix_percentiles': False,
            'non_stat_analysis': True,
            'initial_parameters': {'make': False},
            'optimization': {
                'method': 'SLSQP',
                'eps': 1e-3,
                'maxiter': 100,
                'ftol': 1e-3,
                'bounds': 0.5
            }
        }
        
        # Run non-stationary analysis
        result = nonstationary_analysis(df, param)
        
        # Check results - returns updated param dict
        assert isinstance(result, dict), "Result should be a dict"
        assert 'par' in result or 'parameters' in result, "Should have parameters"
        
        print("✓ nonstationary_analysis test passed")
        return True
        
    except Exception as e:
        print(f"✗ nonstationary_analysis test failed: {e}")
        return False

def test_storm_series():
    """Test the storm_series function from temporal.analysis."""
    print("Testing storm_series...")
    
    try:
        from environmentaltools.temporal.analysis import storm_series
        
        # Create wave data with datetime index
        df = create_wave_data()
        df.set_index('date', inplace=True)
        
        # Define columns and info parameters (matching function signature)
        cols = ['Hs']
        info = {
            'threshold': 2.0,  # Threshold for storm detection
            'time_step': 6,    # Time step in hours 
            'min_duration': 12,  # Minimum duration in hours
            'min_interarrival_time': 24,  # Minimum time between storms
            'interpolation': False,  # Required parameter
            'inter_time': 24  # Alternative parameter name
        }
        
        # Detect storms
        result = storm_series(df, cols, info)
        
        # Check that we get a result
        assert result is not None, "Should return storm data"
        
        print("✓ storm_series test passed")
        return True
        
    except Exception as e:
        print(f"✗ storm_series test failed: {e}")
        return False

def test_temporal_utils():
    """Test utility functions from temporal.utils."""
    print("Testing temporal utilities...")
    
    try:
        from environmentaltools.temporal.utils import extreme_events
        
        # Create test data with datetime index
        df = create_wave_data()
        df.set_index('date', inplace=True)
        
        # Parameters for extreme_events (matching function signature)
        var_name = 'Hs'
        threshold = df[var_name].quantile(0.9)  # 90th percentile
        minimum_interarrival_time = pd.Timedelta(hours=12)
        minimum_cycle_length = pd.Timedelta(hours=6)
        
        # Find extreme events
        result = extreme_events(
            df, var_name, threshold, 
            minimum_interarrival_time, minimum_cycle_length
        )
        
        # Check result
        assert result is not None, "Should return extreme events"
        
        print("✓ temporal utilities test passed")
        return True
        
    except Exception as e:
        print(f"✗ temporal utilities test failed: {e}")
        return False

def test_basic_imports():
    """Test that we can import the main temporal module functions."""
    print("Testing basic imports...")
    
    try:
        # Test core imports
        from environmentaltools.temporal.core import stationary_analysis, nonstationary_analysis
        from environmentaltools.temporal.analysis import storm_series
        from environmentaltools.temporal.utils import extreme_events
        
        # Test that they are callable
        assert callable(stationary_analysis), "stationary_analysis should be callable"
        assert callable(nonstationary_analysis), "nonstationary_analysis should be callable"
        assert callable(storm_series), "storm_series should be callable"
        assert callable(extreme_events), "extreme_events should be callable"
        
        print("✓ Basic imports test passed")
        return True
        
    except ImportError as e:
        print(f"✗ Basic imports test failed: {e}")
        return False

def test_data_handling():
    """Test basic data handling and preparation."""
    print("Testing data handling...")
    
    try:
        # Create and validate test data
        df = create_sample_data()
        
        assert isinstance(df, pd.DataFrame), "Should create DataFrame"
        assert "Hs" in df.columns, "Should have Hs column"
        assert "date" in df.columns, "Should have date column"
        assert len(df) == 1000, "Should have 1000 rows"
        assert not df["Hs"].isna().any(), "Should not have NaN values"
        assert (df["Hs"] >= 0).all(), "Wave heights should be non-negative"
        
        print("✓ Data handling test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data handling test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("RUNNING ENVIRONMENTALTOOLS.TEMPORAL TESTS")
    print("="*60)
    
    tests = [
        test_basic_imports,
        test_data_handling,
        test_stationary_analysis,
        test_fit_distribution,
        test_nonstationary_analysis,
        test_storm_series,
        test_temporal_utils,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            failed += 1
        print("-" * 40)
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return passed, failed

if __name__ == "__main__":
    run_all_tests()