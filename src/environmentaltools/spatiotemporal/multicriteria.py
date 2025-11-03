"""
Multi-Criteria Decision Analysis (MCDA) for Salt Marsh Restoration Suitability
=============================================================================

This module performs comprehensive multi-criteria decision analysis using the TOPSIS
(Technique for Order of Preference by Similarity to Ideal Solution) method to identify
optimal locations for salt marsh restoration based on multiple environmental and
vegetation factors.

The module provides:
- Data integration from multiple suitability scoring layers
- TOPSIS multi-criteria decision analysis with multiple weighting schemes
- Comprehensive visualization of results including ranking maps and isolines
- Statistical analysis and sensitivity assessment
- Professional outputs for scientific publication

TOPSIS Methodology:
1. Normalizes decision matrix to eliminate units of measurement
2. Calculates weighted normalized decision matrix using various weighting methods
3. Determines positive and negative ideal solutions
4. Calculates distances to ideal solutions
5. Computes closeness coefficient (TOPSIS score)
6. Ranks alternatives based on TOPSIS scores

Evaluation Criteria:
- Environmental Factors: Topography, Land Use, Distance to existing marshes
- Vegetation Indices (Mean): SAVI, LSWI, MSI average values
- Vegetation Indices (Trends): SAVI, LSWI, MSI temporal trends

Weighting Methods:
- Variance-based: Higher weight to criteria with more variation
- Equal weights: All criteria weighted equally (optional)
- Entropy-based: Based on information content (optional)

Author: Salt Marsh Monitoring Project
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

from typing import Dict, List

# Multi-criteria decision analysis libraries
from pymcdm.methods import TOPSIS
from pymcdm import weights as w
from pymcdm.helpers import rankdata
from pymcdm import visuals


def run_topsis_mcda(combined_df: pd.DataFrame, weighting_name: str, output_dir: Path):
    """
    Execute TOPSIS multi-criteria decision analysis on the combined suitability scores.
    
    This function performs the complete TOPSIS analysis workflow:
    1. Prepares the decision matrix from suitability scores
    2. Applies multiple weighting schemes
    3. Calculates TOPSIS scores and rankings
    4. Creates comprehensive visualizations
    5. Saves results in multiple formats
    
    Args:
        combined_df (pd.DataFrame): Combined suitability scores dataset
        output_dir (Path): Output directory for results
        
    Returns:
        pd.DataFrame: Final TOPSIS results with scores and rankings, or None if error
        
    Raises:
        Exception: For critical errors in TOPSIS analysis
    """
    print(f"\n🎯 TOPSIS MULTI-CRITERIA DECISION ANALYSIS")
    print("=" * 70)
    
    # Prepare data for TOPSIS analysis
    score_cols = [col for col in combined_df.columns if col not in ['longitude', 'latitude']]
    
    # Filter points with complete data across all criteria
    complete_mask = combined_df[score_cols].notna().all(axis=1)
    complete_data = combined_df[complete_mask].copy()
    
    if len(complete_data) == 0:
        print("❌ No points with complete data for TOPSIS analysis")
        return None
    
    print(f"📊 TOPSIS Analysis Setup:")
    print(f"   📍 Total points: {len(combined_df):,}")
    print(f"   ✅ Complete data points: {len(complete_data):,} "
          f"({len(complete_data)/len(combined_df)*100:.1f}%)")
    print(f"   📈 Evaluation criteria: {len(score_cols)}")
    
    # Extract decision matrix (criteria scores only)
    decision_matrix = complete_data[score_cols].values
    
    # Display criteria information
    _display_criteria_information(decision_matrix, score_cols)
    
    # Define criteria types (all are benefit criteria - higher is better)
    criteria_types = np.ones(len(score_cols))  # 1 = benefit, -1 = cost
    
    # Execute TOPSIS with different weighting methods
    print(f"\n⚖️  TOPSIS WEIGHT CALCULATION AND ANALYSIS:")
    
    topsis_results = {}
    weight_results = {}
    
    # Create output directory
    mcda_dir = output_dir / "multicriteria_analysis"
    mcda_dir.mkdir(exist_ok=True)
    # WEIGHTING METHODS CONFIGURATION
    # WEIGHTING METHODS CONFIGURATION
    WEIGHTING_METHODS = {
        'Variance': w.variance_weights,
        'Equal': w.equal_weights,
        'Entropy': w.entropy_weights,
        'Standard_Deviation': w.standard_deviation_weights,
        'Gini': w.gini_weights
    }

    
    
    # Apply each weighting method
    print(f"\n   🔸 Weighting Method: {weighting_name}")
    weighting_method = WEIGHTING_METHODS.get(weighting_name)
    
    # Calculate weights using specified method
    weights = weighting_method(decision_matrix)
    weight_results[weighting_name] = weights
    
    print(f"      Calculated weights: {[f'{w:.3f}' for w in weights]}")
    
    # Apply TOPSIS method
    topsis = TOPSIS()
    preferences = topsis(decision_matrix, weights, criteria_types)
    rankings = topsis.rank(preferences)
    
    # Store results
    result_data = complete_data.copy()
    result_data[f'topsis_score_{weighting_name.lower()}'] = preferences
    result_data[f'topsis_rank_{weighting_name.lower()}'] = rankings
    
    topsis_results = {
        'preferences': preferences,
        'rankings': rankings,
        'data': result_data
    }
    
    # Display method statistics
    print(f"      📊 TOPSIS scores: {preferences.min():.4f} - {preferences.max():.4f}")
    print(f"      🏆 Top 5 indices: {np.argsort(preferences)[-5:][::-1].tolist()}")
    
    # Create comprehensive visualizations
    print(f"\n🎨 Creating comprehensive visualizations...")
    _create_all_visualizations(topsis_results, weight_results, score_cols, mcda_dir)
    
    # Prepare and save final results
    final_results = _prepare_final_results(complete_data, topsis_results, score_cols)
    _save_final_results(final_results, mcda_dir)
    
    # Display final summary
    _display_topsis_summary(final_results, topsis_results, score_cols, mcda_dir)
    
    return final_results



def _display_criteria_information(decision_matrix: np.ndarray, score_cols: List[str]) -> None:
    """Display detailed information about evaluation criteria."""
    print(f"\n📋 EVALUATION CRITERIA ANALYSIS:")
    
    for i, col in enumerate(score_cols):
        values = decision_matrix[:, i]
        print(f"   {i+1:2d}. {col:>25}: {values.min():.3f} - {values.max():.3f} "
              f"(μ={values.mean():.3f}, σ={values.std():.3f})")


def _create_all_visualizations(topsis_results: Dict,
                              weight_results: Dict,
                              score_cols: List[str],
                              output_dir: Path) -> None:
    """Create all TOPSIS visualization outputs."""
    # 1. Weights comparison visualization
    create_weights_visualization(weight_results, score_cols, output_dir)
    
    # 2. TOPSIS scores maps
    create_topsis_maps(topsis_results, output_dir)
    
    # 3. Ranking map with isolines
    create_topsis_ranking_map(topsis_results, output_dir)
    
    # 4. Comparative statistics
    create_comparative_statistics(topsis_results, score_cols, output_dir)
    return



def _prepare_final_results(complete_data: pd.DataFrame,
                          topsis_results: Dict,
                          score_cols: List[str]) -> pd.DataFrame:
    """Prepare final combined results dataset."""
    # Start with base data (coordinates + criteria scores)
    final_results = complete_data[['longitude', 'latitude'] + score_cols].copy()
    
    # Add results from each weighting method
    for weight_name, results in topsis_results.items():
        final_results[f'topsis_score_{weight_name.lower()}'] = results['preferences']
        final_results[f'topsis_rank_{weight_name.lower()}'] = results['rankings']
    
    # Calculate average TOPSIS score across all methods
    topsis_score_cols = [col for col in final_results.columns if 'topsis_score_' in col]
    final_results['topsis_score_mean'] = final_results[topsis_score_cols].mean(axis=1)
    final_results['topsis_rank_mean'] = rankdata(final_results['topsis_score_mean'], reverse=True)
    
    return final_results


def create_weights_visualization(weight_results: Dict, 
                               criteria_names: List[str], 
                               output_dir: Path) -> None:
    """
    Create visualization comparing different weighting schemes.
    
    Args:
        weight_results (dict): Dictionary containing weight arrays for each method
        criteria_names (list): Names of evaluation criteria
        output_dir (Path): Output directory for saving figures
    """
    print(f"   📊 Creating weights comparison visualization...")
    
    # Prepare weight data
    weight_sets = list(weight_results.values())
    method_names = list(weight_results.keys())
    
    # Create weights comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='white')
    
    # Use pymcdm's built-in weights visualization
    visuals.weights_plot(weight_sets,
                        xticklabels=method_names,
                        legend_ncol=3,
                        ax=ax)
    
    # Customize plot appearance
    ax.set_xlabel('Weighting Methods', fontweight='bold', fontsize=12)
    ax.set_ylabel('Weight Value', fontweight='bold', fontsize=12)
    ax.tick_params(labelsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    weights_file = output_dir / 'topsis_weights_comparison.png'
    plt.savefig(weights_file, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"      ✅ Weights visualization saved: {weights_file}")
    
    # Save weights table
    weights_df = pd.DataFrame(weight_results, index=criteria_names)
    weights_csv = output_dir / 'topsis_weights_table.csv'
    weights_df.to_csv(weights_csv, float_format='%.4f')
    print(f"      ✅ Weights table saved: {weights_csv}")
    return 



def create_topsis_maps(topsis_results: Dict, output_dir: Path) -> None:
    """
    Create spatial maps of TOPSIS scores for different weighting methods.
    
    Args:
        topsis_results (dict): Dictionary containing TOPSIS results for each method
        output_dir (Path): Output directory for saving figures
    """
    print(f"   🗺️  Creating TOPSIS score maps...")
    
    n_methods = len(topsis_results)
    
    if n_methods == 1:
        # Single method visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
        
        method_name, results = list(topsis_results.items())[0]
        data = results['data']
        
        # Extract coordinates and scores
        lons = data['longitude'].values
        lats = data['latitude'].values
        scores = results['preferences']
        
        # Create scatter plot
        scatter = ax.scatter(lons, lats, c=scores, cmap='RdYlGn', 
                           s=0.5, alpha=0.8, rasterized=True)
        
        # Configure axes
        ax.set_xlabel('Longitude (°)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontweight='bold', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=10)
        
        # Add colorbar
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('TOPSIS Suitability Score', fontweight='bold', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
    else:
        # Multiple methods comparison
        n_cols = min(2, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows), 
                                sharex=True, sharey=True, facecolor='white')
        
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        last_scatter = None
        
        for i, (method_name, results) in enumerate(topsis_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = results['data']
            
            # Extract coordinates and scores
            lons = data['longitude'].values
            lats = data['latitude'].values
            scores = results['preferences']
            
            # Create scatter plot
            scatter = ax.scatter(lons, lats, c=scores, cmap='RdYlGn', 
                               s=0.5, alpha=0.8, rasterized=True)
            
            # Configure axes
            ax.set_title(f'{method_name} Weighting', fontweight='bold', fontsize=12)
            ax.set_xlabel('Longitude (°)', fontweight='bold')
            ax.set_ylabel('Latitude (°)', fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=9)
            
            last_scatter = scatter
        
        # Hide unused axes
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        # Add shared colorbar
        if last_scatter is not None:
            plt.tight_layout(rect=[0, 0.12, 1, 1])
            cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
            cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('TOPSIS Suitability Score', fontweight='bold', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
    
    # Save figure
    maps_file = output_dir / 'topsis_suitability_maps.png'
    plt.savefig(maps_file, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"      ✅ TOPSIS maps saved: {maps_file}")
    return