import matplotlib.pyplot as plt
import numpy as np
from environmentaltools.spatiotemporal import covariance
from matplotlib.pyplot import cm


def covariance_comparison(covDistanceS, covDistanceT, covEmpST, tLag, res, family, type):
    """Plot empirical and theoretical spatiotemporal covariance comparison.

    Visualizes the fit between empirical spatiotemporal covariance and the
    theoretical covariance model, displaying both as 3D surfaces or 2D contours.

    Args:
        covDistanceS (numpy.ndarray): Spatial distance grid for covariance evaluation.
        covDistanceT (numpy.ndarray): Temporal distance grid for covariance evaluation.
        covEmpST (numpy.ndarray): Empirical spatiotemporal covariance values.
        tLag (array-like): Temporal lag values.
        res (scipy.optimize.OptimizeResult): Optimization result containing fitted
            covariance parameters in res.x.
        family (str): Covariance family name for theoretical model.
        type (str): Plot type - "3d" for 3D surface plot, other for 2D contour plot.

    Returns:
        None: Displays the plots.

    Note:
        For "3d" type, displays wireframe theoretical covariance with empirical
        surface overlay, plus contour plot of residuals. For 2D type, shows
        side-by-side contour comparisons of theoretical and empirical covariances.
    """
    if type == "3d":
        fig = plt.figure(figsize=(5, 4))
        ax = fig.gca(projection="3d")

        x, y = np.meshgrid(
            np.linspace(0, np.amax(covDistanceS), 20),
            np.linspace(0, np.amax(covDistanceT), 20),
        )
        covTh = covariance.calculate_theoretical_covariance(family, [x, y], res.x)
        ax.plot_wireframe(x, y, covTh, rstride=1, cstride=1, color="k", lw=0.5)
        ax.plot_surface(
            covDistanceS, covDistanceT, covEmpST, cmap=cm.autumn_r, alpha=0.6
        )

        ax.scatter(covDistanceS, covDistanceT, covEmpST, marker="o", color="k")

        ax.set_xlabel(r"$\mathbf{S_{lag}}$")
        ax.set_ylabel(r"$\mathbf{T_{lag}}$")
        ax.set_ylim(np.max(tLag), np.min(tLag))
        ax.set_zlabel("cov", fontweight="bold")

        covTh = covariance.covariance(family, [covDistanceS, covDistanceT], res.x)
        plt.figure(figsize=(5, 4))
        error = covEmpST - covTh
        # e_mda = np.sum(np.abs(error))/np.size(covEmpST)
        e_mse = np.sqrt(np.sum(error ** 2)) / np.size(covEmpST)
        CS3 = plt.contour(covDistanceS, covDistanceT, error, 5, colors="k")
        # cbar = plt.colorbar()
        plt.clabel(CS3, inline=1, fontsize=8)
        plt.ylim([np.max(tLag), np.min(tLag)])
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        textstr = (
            r"$\mathbf{\varepsilon_{RMSE}}$="
            + " "
            + str(np.round(e_mse, 2))
            + "\n"
            + r"$\mathbf{\varepsilon_{max}}$="
            + " "
            + str(np.round(np.max(error), 2))
            + "\n"
            + r"$\mathbf{\varepsilon_{min}}$="
            + " "
            + str(np.round(np.min(error), 2))
        )
        plt.text(
            0.75,
            0.25,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.xlabel(r"$\mathbf{S_{lag}}$")
        plt.ylabel(r"$\mathbf{T_{lag}}$")
    else:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        x, y = np.meshgrid(
            np.linspace(0, np.amax(covDistanceS), 20),
            np.linspace(0, np.amax(covDistanceT), 20),
        )
        covTh = covariance.covariance(family, [x, y], res.x)
        CS1 = plt.contour(
            x, y, covTh, 10, cmap=cm.autumn_r, alpha=0.6, label=r"$c_{th}$"
        )
        plt.clabel(CS1, inline=1, fontsize=10)
        CS2 = plt.contour(
            covDistanceS,
            covDistanceT,
            covEmpST,
            10,
            cmap=cm.autumn_r,
            alpha=0.6,
            label=r"$c_{emp}$",
        )
        for c in CS2.collections:
            c.set_dashes([(0, (2.0, 2.0))])
        plt.clabel(CS2, inline=1, fontsize=10)

        plt.xlabel(r"$\mathbf{S_{lag}}$")
        plt.ylabel(r"$\mathbf{T_{lag}}$")
        plt.ylim([np.max(tLag), np.min(tLag)])
        plt.legend([r"$c_{th}$", r"$c_{emp}$"], loc="best")

        covTh = covariance.covariance(family, [covDistanceS, covDistanceT], res.x)
        plt.subplot(1, 2, 2)
        CS3 = plt.contour(covDistanceS, covDistanceT, np.abs(covEmpST - covTh), 5)
        # cbar = plt.colorbar()
        plt.clabel(CS3, inline=1, fontsize=10)
        plt.ylim([np.max(tLag), np.min(tLag)])

        plt.xlabel(r"$\mathbf{S_{lag}}$")
        plt.ylabel(r"$\mathbf{T_{lag}}$")
    plt.show()


def anisotropic_spatiotemporal_covariance(covdist, covdistd, covdistt, empcovang, slag, type):
    """Plot anisotropic spatiotemporal covariance as polar or 3D visualization.

    Displays directional (angular) spatiotemporal covariance patterns to identify
    anisotropy in the spatial correlation structure.

    Args:
        covdist (numpy.ndarray): Spatial distance values for covariance.
        covdistd (numpy.ndarray): Angular direction values (degrees) for covariance.
        covdistt (numpy.ndarray): Temporal distances for each covariance slice.
        empcovang (numpy.ndarray): Empirical angular covariance values
            (shape: [n_distances, n_angles, n_times]).
        slag (float): Spatial lag distance for scaling.
        type (tuple): Two-element tuple specifying visualization:
            - type[0]: "polar" for polar plots, "3d" for 3D surface plots
            - type[1]: "variogram" for variogram display, other for covariance

    Returns:
        None: Displays the plots.

    Note:
        Angular values are extended by 360° to create continuous polar plots.
        For variogram mode, displays gamma(h) = c(0) - c(h).
    """
    covdistd = np.radians(np.vstack((covdistd, covdistd + 180, covdistd[0, :] + 360)))
    covdist = np.vstack((covdist, covdist, covdist[0, :]))

    if type[0] == "polar":
        # Polar plot generation
        for i in range(np.shape(empcovang)[2]):
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
            if type[1] == "variogram":
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
                empcov = empcov[0, 0] - empcov
            else:
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
            CS = ax.contourf(covdistd, covdist, empcov)
            ax.contour(covdistd, covdist, empcov, color="k")
            plt.colorbar(CS)
            ax.grid(True)
            ax.set_title("t = " + str(covdistt[i]))

    elif type[0] == "3d":
        covdistx, covdisty = slag * np.cos(covdistd), slag * np.sin(covdistd)

        for i in range(np.shape(empcovang)[2]):
            if type[1] == "variogram":
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
                empcov = empcov[0, 0] - empcov
                title = r"$\mathbf{\gamma}$"
            else:
                title = r"$\mathbf{c}$"
                empcov = np.vstack(
                    (empcovang[:, :, i].T, empcovang[:, :, i].T, empcovang[:, 0, i].T)
                )
            fig = plt.figure(figsize=(5, 4))
            ax = fig.gca(projection="3d")
            ax.plot_surface(covdistx, covdisty, empcov, alpha=0.6, cmap=cm.autumn_r)

            ax.set_xlabel(r"$\mathbf{S_{x}}$", fontweight="bold", labelpad=20)
            axis = np.hstack((-covdistx[0, ::-1], covdistx[0, :]))
            ax.set_xticks(axis)
            ax.set_xticklabels(np.round(np.abs(axis), decimals=2), rotation=45)
            ax.set_ylabel(r"$\mathbf{S_{y}}$", fontweight="bold", labelpad=20)
            ax.set_yticks(axis)
            ax.set_yticklabels(np.round(np.abs(axis), decimals=2), rotation=-45)
            ax.set_title(
                r"$\mathbf{T}$ =  " + str(np.round(covdistt[i], decimals=2)),
                fontweight="bold",
            )
            ax.set_zlabel(title)
            ax.set_zlim([0, np.nanmax(empcovang)])

    plt.show()
    return


def era5_time_series_plot(
    data,
    variable_name: str = 'swh',
    variable_label: str = 'Significant Wave Height',
    variable_units: str = 'm',
    start_year: int = None,
    end_year: int = None,
    output_path: str = None,
    file_name: str = None
) -> str:
    """Create a comprehensive time series plot of ERA5 data.
    
    Creates a multi-panel visualization including:
    - Complete time series
    - Monthly statistics (mean, IQR, max)
    - Distribution histogram with mean and 95th percentile
    
    Args:
        data (pd.DataFrame): Data with datetime index and variable column.
        variable_name (str, optional): Column name in dataframe. Defaults to 'swh'.
        variable_label (str, optional): Label for plots. Defaults to 'Significant Wave Height'.
        variable_units (str, optional): Units for axis labels. Defaults to 'm'.
        start_year (int, optional): Start year for plot title. If None, extracted from data.
        end_year (int, optional): End year for plot title. If None, extracted from data.
        output_path (str, optional): Directory to save plot. If None, uses current directory.
        file_name (str, optional): Custom filename. If None, auto-generated.
    
    Returns:
        str: Path to the saved plot file.
    
    Raises:
        ValueError: If plot creation fails.
    
    Example:
        >>> import pandas as pd
        >>> from environmentaltools.graphics.spatiotemporal import era5_time_series_plot
        >>> 
        >>> # For wave height
        >>> plot_path = era5_time_series_plot(
        ...     data, 
        ...     variable_name='swh',
        ...     variable_label='Significant Wave Height',
        ...     variable_units='m',
        ...     output_path='./plots'
        ... )
        >>> 
        >>> # For wind speed
        >>> plot_path = era5_time_series_plot(
        ...     data, 
        ...     variable_name='u10', 
        ...     variable_label='10m U Wind', 
        ...     variable_units='m/s'
        ... )
    """
    import pandas as pd
    import matplotlib.dates as mdates
    from datetime import datetime
    from pathlib import Path
    
    try:
        # Extract years from data if not provided
        if start_year is None:
            start_year = data.index.min().year
        if end_year is None:
            end_year = data.index.max().year
        
        # Set up the figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(
            f'ERA5 {variable_label} Time Series Analysis\n'
            f'{start_year}-{end_year}', 
            fontsize=16, 
            fontweight='bold'
        )
        
        # Main time series plot
        ax1.plot(
            data.index, 
            data[variable_name], 
            color='steelblue', 
            linewidth=0.5, 
            alpha=0.7
        )
        ax1.set_ylabel(f'{variable_label} ({variable_units})', fontsize=12)
        ax1.set_title(
            f'Complete Time Series ({end_year - start_year + 1} years)', 
            fontsize=14
        )
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Monthly statistics (box plot style)
        monthly_data = data.groupby(data.index.month)[variable_name].describe()
        months = range(1, 13)
        ax2.plot(months, monthly_data['mean'], 'o-', color='darkblue', linewidth=2, label='Mean')
        ax2.fill_between(
            months, 
            monthly_data['25%'], 
            monthly_data['75%'], 
            alpha=0.3, 
            color='lightblue', 
            label='IQR (25%-75%)'
        )
        ax2.plot(months, monthly_data['max'], '^-', color='red', alpha=0.7, label='Max')
        ax2.set_ylabel(f'{variable_label} ({variable_units})', fontsize=12)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_title(f'Monthly {variable_label} Statistics', fontsize=14)
        ax2.set_xticks(months)
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Histogram
        ax3.hist(
            data[variable_name], 
            bins=50, 
            density=True, 
            alpha=0.7, 
            color='lightcoral', 
            edgecolor='black'
        )
        ax3.axvline(
            data[variable_name].mean(), 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Mean: {data[variable_name].mean():.2f} {variable_units}'
        )
        ax3.axvline(
            data[variable_name].quantile(0.95), 
            color='orange', 
            linestyle='--', 
            linewidth=2,
            label=f'95th percentile: {data[variable_name].quantile(0.95):.2f} {variable_units}'
        )
        ax3.set_xlabel(f'{variable_label} ({variable_units})', fontsize=12)
        ax3.set_ylabel('Probability Density', fontsize=12)
        ax3.set_title(f'{variable_label} Distribution', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Dataset Statistics:
Total records: {len(data):,}
Data period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}
Mean {variable_label}: {data[variable_name].mean():.3f} {variable_units}
Max {variable_label}: {data[variable_name].max():.3f} {variable_units}
Std {variable_label}: {data[variable_name].std():.3f} {variable_units}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)
        
        # Determine output path
        if output_path is None:
            output_dir = Path.cwd() / 'results'
        else:
            output_dir = Path(output_path) / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            variable_short = variable_name.replace('_', '')
            file_name = f"timeseries_{variable_short}_{start_year}_{end_year}_{timestamp}.png"
        
        plot_path = output_dir / file_name
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(plot_path)
        
    except Exception as e:
        raise ValueError(f"Plot creation failed: {str(e)}")


def plot_presence_boundary(contours, mean_map, threshold=None, title=None, 
                          figsize=(10, 8), cmap='viridis', output_path=None):
    """
    Visualize mean presence boundary with contours and background heatmap.
    
    Creates a visualization showing the spatial mean map as a heatmap with
    overlaid contour lines indicating the presence boundary threshold.
    
    Parameters
    ----------
    contours : list
        List of contour coordinates from mean_presence_boundary function.
    mean_map : np.ndarray
        2D array of temporal mean values for each spatial location.
    threshold : float, optional
        Threshold value used to define presence boundary. If None, uses
        the mean of mean_map.
    title : str, optional
        Plot title. If None, uses default title.
    figsize : tuple, optional
        Figure size as (width, height). Default is (10, 8).
    cmap : str, optional
        Colormap for the heatmap. Default is 'viridis'.
    output_path : str or Path, optional
        Path to save the figure. If None, displays the plot.
    
    Returns
    -------
    None or str
        If output_path is provided, returns the path where figure was saved.
        Otherwise, displays the plot and returns None.
    
    Examples
    --------
    >>> from environmentaltools.spatiotemporal import indicators
    >>> from environmentaltools.graphics import spatiotemporal
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> data_cube = np.random.random((365, 50, 50))
    >>> contours, mean_map = indicators.mean_presence_boundary(data_cube, threshold=0.6)
    >>> 
    >>> # Visualize
    >>> spatiotemporal.plot_presence_boundary(contours, mean_map, threshold=0.6)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean map as heatmap
    im = ax.imshow(mean_map, cmap=cmap, origin='lower', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Temporal Mean Value')
    
    # Plot contours - handle different formats
    if len(contours) > 0:
        for i, contour in enumerate(contours):
            # Handle both 1D and 2D contour arrays
            contour = np.atleast_2d(contour)
            if contour.shape[1] == 2:
                # Standard format: (n_points, 2) with [row, col]
                label = 'Presence Boundary' if i == 0 else None
                ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, label=label)
            elif contour.shape[0] == 2:
                # Transposed format: (2, n_points) with [row; col]
                label = 'Presence Boundary' if i == 0 else None
                ax.plot(contour[1, :], contour[0, :], 'r-', linewidth=2, label=label)
    
    # Add threshold line to colorbar if provided
    if threshold is not None:
        cbar.ax.axhline(threshold, color='red', linewidth=2, linestyle='--')
        cbar.ax.text(0.5, threshold, f' {threshold:.3f}', 
                    va='center', ha='left', color='red', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    if title is None:
        if threshold is not None:
            title = f'Mean Presence Boundary (threshold = {threshold:.3f})'
        else:
            title = 'Mean Presence Boundary'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend (only one entry even if multiple contours)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    
    # Save or show
    if output_path is not None:
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return str(output_path)
    else:
        plt.show()
        return None
