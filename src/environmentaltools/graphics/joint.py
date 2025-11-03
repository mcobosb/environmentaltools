# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from environmentaltools.graphics.utils import handle_axis, labels, show
from matplotlib import gridspec
from scipy.stats import genextreme, lognorm, norm

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)
params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}


def dscatter(x, y, *varargin):
    """Create density scatter plot with smoothed 2D histogram coloring.

    Generates a scatter plot where point colors represent local density,
    computed using 2D histogram binning and smoothing.

    Args:
        x (array-like): X-coordinates of data points.
        y (array-like): Y-coordinates of data points.
        *varargin: Additional variable arguments (currently unused).

    Note:
        This function uses a smoothing parameter lambda_=20 and limits bins to 200.
    """
    lambda_ = 20

    minx, maxx, miny, maxy = np.min(x), np.max(x), np.min(y), np.max(y)

    nbins = np.array(
        [np.min((np.size(np.unique(x)), 200)), np.min((np.size(np.unique(y)), 200))]
    )

    edges1 = np.linspace(minx, maxx, nbins[0] + 1)
    edges1 = np.hstack((-np.inf, edges1[1:-1], np.inf))
    edges2 = np.linspace(miny, maxy, nbins[0] + 1)
    edges2 = np.hstack((-np.inf, edges2[1:-1], np.inf))

    n = nbins[0] + 1
    bin_ = np.zeros([n, 2])
    # Reverse the columns to put the first column of X along the horizontal
    # axis, the second along the vertical.
    dum, bin_[:, 1] = np.histogram(x, bins=edges1)
    dum, bin_[:, 0] = np.histogram(y, bins=edges2)
    h = np.bincount(bin_, weights=nbins([1, 0])) / n
    g = smooth1D(h, nbins[1] / lambda_)
    f = smooth1D(g.T, nbins[0] / lambda_).T

    f = f / np.max(f)
    ind = np.ravel_multi_index(np.shape(f), bin_[:, 0], bin[:, 1], order="F")
    col = f[ind]
    h = plt.scatter(x, y, col, "s", "filled")


def smooth1D(y, lambda_):
    """Apply 1D smoothing to data using regularization matrix.

    Implements smoothing through matrix operations with penalty terms
    based on first and second differences.

    Args:
        y (array-like): Input data to smooth (m x n array).
        lambda_ (float): Smoothing parameter controlling penalty strength.

    Returns:
        numpy.ndarray: Smoothed data with same shape as input.
    """
    m, n = np.shape(y)
    e = np.eye(m)
    d1 = np.diff(e)
    d2 = np.diff(d1)
    p = lambda_ ** 2 * d2.T * d2 + 2 * lambda_ * d1.T * d1
    z = (e + p) / y
    return z


def plot_conditional_regime(
    xpar,
    xparaux,
    xlim,
    ylim,
    alpha,
    dist,
    param,
    yest,
    reg,
    xg,
    yg,
    pdfg,
    ci,
    x,
    y,
    xlab,
    ylab,
    yrlab,
    valmed,
):
    """Plot joint regime results from conditional distribution function.

    Visualizes the relationship between two variables through conditional
    probability distributions and regression models with confidence intervals.

    Args:
        xpar (array-like): Primary conditioning parameter values.
        xparaux (array-like): Auxiliary conditioning parameter values.
        xlim (tuple): X-axis limits (min, max).
        ylim (tuple): Y-axis limits (min, max).
        alpha (float): Significance level for confidence intervals.
        dist (str): Distribution type ('GEV', 'Lognormal', 'Normal', etc.).
        param (array-like): Distribution parameters for each regime.
        yest (array-like): Estimated response values.
        reg (int): Regression type identifier (1=polynomial1, 2=polynomial2,
            3=power1, 4=power2).
        xg (array-like): Grid x-coordinates for PDF contours.
        yg (array-like): Grid y-coordinates for PDF contours.
        pdfg (array-like): Probability density function values on grid.
        ci (array-like): Confidence interval bounds.
        x (array-like): Observed x data points.
        y (array-like): Observed y data points.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        yrlab (str): Right y-axis label (for marginal distribution).
        valmed (float): Median value for reference line.

    Returns:
        None: Displays the plot.
    """

    from environmentaltools.temporal import poli1, poli2, pote1, pote2

    xpar2 = np.asarray(xparaux)

    ndat = len(yest)

    # Reduce data points if exceeding maximum for visualization performance
    ndat_max = 10000
    if ndat > ndat_max:
        ids_r = np.linspace(0, ndat - 1, ndat_max, dtype=int)
        xpar = xpar[ids_r, :]
        xpar2 = xpar2[ids_r, :]
        yest = yest[ids_r, :]
        x = x[ids_r]
        y = y[ids_r]

    plt.style.use("ggplot")
    xaux = np.linspace(np.min(x), np.max(x), 1000)
    yaux = np.linspace(np.min(y), np.min(y), 1000)
    d_dist, d_reg = dict(), dict()
    d_dist["lognormal"], d_dist["normal"], d_dist["gev"] = lognorm, norm, genextreme
    d_reg["poli1"], d_reg["poli2"], d_reg["pote1"], d_reg["pote2"] = (
        poli1,
        poli2,
        pote1,
        pote2,
    )

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 6)

    # Panel 1: P-value vs conditioning variable
    ax1 = fig.add_subplot(gs[0, 0:2])
    id1 = np.where(xpar[:, -2] >= alpha)
    id2 = np.where(xpar[:, -2] < alpha)
    ax1.plot(xpar[id1, 0], xpar[id1, -2], "b.")
    ax1.plot(xpar[id2, 0], xpar[id2, -2], "r.")
    ax1.plot((np.min(xpar[:, 0]), np.max(xpar[:, 0])), (alpha, alpha), "-r")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("p-value")
    ax1.tick_params(axis="x", labelsize=10)

    npar = len(xparaux[0])

    # Panel 2: Distribution parameters vs conditioning variable
    if npar == 2:
        ini = 0
        for i in range(npar):
            ax2 = fig.add_subplot(gs[1, ini : ini + 3])
            ax2.plot(xpar[:, 0], xpar2[:, i], ".", color="gray")
            ax2.plot(xaux, param[i], "-r", lw=2)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(dist + [" - location", " - scale"][i])
            ax2.tick_params(axis="x", labelsize=8)
            ini += 3

    elif npar == 3:
        ini = 0
        for i in range(npar):
            ax2 = fig.add_subplot(gs[1, ini : ini + 2])
            ax2.plot(xpar[:, 0], xpar2[:, i], ".", color="gray")
            ax2.plot(xaux, param[i], "-r", lw=2)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(dist + [" - shape", " - location", " - scale"][i])
            ax2.tick_params(axis="x", labelsize=8)
            ini += 2

    # Panel 3: Joint distribution and regression
    ax3 = fig.add_subplot(gs[0, 2:-2])
    id_ = np.where(np.isnan(yest) | np.isinf(yest))[0]
    if any(id_):
        print("Warning: Reconstructed variable values contain NaN or Inf")
    id_ = np.where(~np.isnan(yest) | ~np.isinf(yest))[0]
    ax3.scatter(xpar[id_, 1], yest[id_])
    ax3.plot(xpar[id_, 1], xpar[id_, 1], "k", lw=2)
    ax3.set_xlabel(ylab)
    ax3.set_ylabel(yrlab)
    ax3.tick_params(axis="x", labelsize=8)

    # Panel 4: Joint PDF with confidence intervals
    ax4 = fig.add_subplot(gs[0, 4:])
    ax4.contour(xg, yg, pdfg)
    ax4.scatter(x, y, 10, "gray", label="data")
    ax4.plot(xaux, valmed, "k", lw=2, label="fit")
    ax4.plot(xaux, ci[0, :], "--r", lw=2, label="CI")
    ax4.plot(xaux, ci[1, :], "--r", lw=2)
    ax4.legend(loc="best", scatterpoints=1, fontsize=8)
    ax4.set_xlim([np.min(x), np.max(x)])
    ax4.set_xlabel(xlab)
    ax4.set_ylabel(ylab)
    ax4.tick_params(axis="x", labelsize=8)

    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.4)


def plot_copula(copula, ax=None, labels=[], file_name: str = None, log: bool = False):
    """Plot copula function with contours and parameters.

    Generates a 2D visualization of the copula probability density function
    with contour lines and displays copula parameters.

    Args:
        copula: Copula object with CDF and parameter methods.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates new figure
            if None. Defaults to None.
        labels (list): List of two strings for x and y axis labels. Defaults to [].
        file_name (str, optional): Path to save the plot. If None, displays
            interactively. Defaults to None.
        log (bool): If True, uses logarithmic scale for color mapping. Defaults
            to False.

    Returns:
        None: Displays or saves the plot.
    """

    nlen, nlent = 1000, 1000

    fig, ax = handle_axis(ax)

    data1 = copula.X
    data2 = copula.Y

    x, y = [], []
    xt = np.linspace(1 / len(data1), 1 - 1 / len(data1), nlent)
    u, v = np.linspace(1 / len(data1), 1 - 1 / len(data1), nlen), np.linspace(
        1 / len(data1), 1 - 1 / len(data1), nlent
    )
    copula.generate_C(u, v)
    
    # Generate contours from copula CDF
    for j in xt:
        copula.U = xt
        copula.V = np.ones(nlent) * j
        copula.generate_xy()
        if copula.X1.size == 0:
            x.append(copula.U * 0)
        else:
            x.append(copula.X1)

        if copula.Y1.size == 0:
            y.append(copula.U * 0)
        else:
            y.append(copula.Y1)

    if log:
        x, y = np.log10(x), np.log10(y)
    cs = ax.contour(np.asarray(x), np.asarray(y), copula.C, 8, linestyles="dashed")

    # Compute empirical copula from data
    f, xedges, yedges = np.histogram2d(data1, data2, bins=nlen)
    Fe = np.cumsum(np.cumsum(f, axis=0), axis=1) / (np.sum(f) + 1)
    xmid, ymid = (xedges[0:-1] + xedges[1:]) / 2, (yedges[0:-1] + yedges[1:]) / 2
    xe, ye = np.meshgrid(xmid, ymid)
    if log:
        xe, ye = np.log10(xe), np.log10(ye)
    cs = ax.contour(xe, ye, np.flipud(np.rot90(Fe)), 8, linestyles="solid")

    ax.clabel(cs, cs.levels, inline=True, fontsize=10)
    
    # Display copula parameters
    ax.text(
        0.6,
        0.8,
        r"$\theta$ = " + str(np.round(copula.theta, decimals=4)),
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.6,
        0.75,
        r"$\tau$ = " + str(np.round(copula.tau, decimals=4)),
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    show(file_name)

    return ax
