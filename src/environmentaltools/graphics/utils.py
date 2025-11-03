import matplotlib.pyplot as plt
import shutil


def enable_latex_rendering():
    """Enable LaTeX rendering for matplotlib plots."""
    if shutil.which("latex") is None:
        raise RuntimeError(
            "LaTeX is required to render text in figures. Supported distributions:\n"
            "  - MiKTeX (https://miktex.org/download) - Windows\n"
            "  - TeX Live (https://tug.org/texlive/) - Cross-platform\n"
            "  - TinyTeX (https://yihui.org/tinytex/) - Lightweight"
        )
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r'\usepackage{amsmath}'
    })
    return


def show(file_name: str = None, res: int = 600):
    """Save figure to file or display on screen.

    Saves the current matplotlib figure to a file with specified resolution
    or displays it interactively.

    Args:
        file_name (str, optional): Path to save the plot. If None, displays
            interactively. If "to_axes", returns without action. Defaults to None.
        res (int): Resolution in DPI for saved figure. Defaults to 600.

    Returns:
        None: Either displays the figure or saves it to file.
    """

    if not file_name:
        plt.show()
    elif file_name == ("to_axes"):
        pass
    else:
        if "png" or "pdf" in file_name:
            plt.savefig(f"{file_name}", dpi=res, bbox_inches="tight")
        else:
            plt.savefig(f"{file_name}" + ".png", dpi=res, bbox_inches="tight")
        plt.close()
    return


def handle_axis(
    ax,
    row_plots: int = 1,
    col_plots: int = 1,
    dim: int = 2,
    figsize: tuple = (5, 5),
    projection=None,
    kwargs: dict = {},
):
    """Create matplotlib axis for figure if needed.

    Creates a new matplotlib axis/figure if none provided, supporting 2D, 3D,
    and projection-based plots with customizable layout.

    Args:
        ax (matplotlib.axes.Axes): Existing axis for the plot, or None to create new.
        row_plots (int): Number of subplot rows. Defaults to 1.
        col_plots (int): Number of subplot columns. Defaults to 1.
        dim (int): Number of dimensions (2 for 2D, 3 for 3D plots). Defaults to 2.
        figsize (tuple): Figure size as (width, height) in inches. Defaults to (5, 5).
        projection: Projection type for the axis (e.g., 'polar', CRS objects).
            Defaults to None.
        kwargs (dict): Additional keyword arguments passed to plt.subplots.
            Defaults to {}.

    Returns:
        tuple: (fig, ax) where fig is the Figure object (or None if ax was provided)
            and ax is the Axes object(s). If multiple subplots, ax is flattened array.
    """

    fig = None
    if not ax:
        if dim == 2:
            fig, ax = plt.subplots(row_plots, col_plots, figsize=figsize, **kwargs)
        elif not projection is None:
            ax = plt.axes(
                projection=projection
            )  # project using coordinate reference system (CRS) of street map
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        if row_plots + col_plots > 2:
            ax = ax.flatten()

    return fig, ax


def labels(variable):
    """Gives the labels and units for plots

    Args:
        * variable (string): name of the variable

    Returns:
        * units (string): label with the name of the variable and the units
    """

    units = {
        "calms": r"$\delta$ (h)",
        "depth": "Depth (m)",
        "dm": r"$\theta_m$ (deg)",
        "dv": r"$\theta_v$ (deg)",
        "DirM": r"$\theta_m$ (deg)",
        "DirU": r"$\theta_U$ (deg)",
        "DirV": r"$\theta_v$ (deg)",
        "Dmd": r"$\theta_m$ (deg)",
        "Dmv": r"$\theta_v$ (deg)",
        "dur": r"d (hr)",
        "dur_calms": r"$\Delta_0$ (hr)",
        "dur_storm": r"$d_0$ (hr)",
        "eta": r"$\eta$ (m)",
        "hs": r"$H_{s}$ (m)",
        "Hm0": r"$H_{m0}$ (m)",
        "Hs": r"$H_{s}$ (m)",
        "lat": "Latitude (deg)",
        "lon": "Longitude (deg)",
        "ma": r"$M_{ast}$ (m)",
        "mm": r"$M_{met}$ (m)",
        "pr": r"P (mm/day)",
        "Q": r"Q (m$^3$/s)",
        "Qd": r"$Q_d$ (m$^3$/s)",
        "S": r"S (psu)",
        "slr": r"$\Delta\eta$ (m)",
        "storm": r"d (h)",
        "surge": r"$\eta_s$ (m)",
        "swh": r"$H_{s}$ (m)",
        "t": "t (s)",
        "Tm0": r"$T_{m0}$ (s)",
        "tp": r"$T_p$ (s)",
        "Tp": r"$T_p$ (s)",
        "U": r"U (m/s)",
        "V": r"V (m/s)",
        "VelV": "[m/s]",
        "vv": r"$V_v$ (m/s)",
        "Vv": r"$V_v$ (m/s)",
        "Wd": r"$W_d$ (deg)",
        "Wv": r"$W_v$ (m/s)",
        "x": "x (m)",
        "y": "y (m)",
        "z": "z (m)",
        "None": "None",
    }

    if isinstance(variable, str):
        if not variable in units.keys():
            labels_ = ""
        else:
            labels_ = units[variable]
    elif isinstance(variable, list):
        labels_ = list()
        for var_ in variable:
            if not var_ in units.keys():
                labels_.append("")
            else:
                labels_.append(units[var_])
    else:
        raise ValueError
    return labels_
