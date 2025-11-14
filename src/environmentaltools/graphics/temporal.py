import datetime
from itertools import product

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LogNorm
from environmentaltools.graphics.utils import handle_axis, labels, show, enable_latex_rendering
from environmentaltools.temporal.analysis import storm_properties
from environmentaltools.temporal import core
from environmentaltools.common import utils
from pandas.plotting import register_matplotlib_converters
from windrose import WindroseAxes

"""This file is part of environmentaltools.

environmentaltools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

environmentaltools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with environmentaltools.  If not, see <https://www.gnu.org/licenses/>.
"""

enable_latex_rendering()


register_matplotlib_converters()
cmp_g = cmocean.cm.haline_r


def mda(data, cases, variables, title=None, ax=None, fname=None):
    """Plots a 3D scatter figure of the MDA cases over the data cloud of points

    Args:
        * data (pd.DataFrame): time series of the data.
        * cases (pd.DataFrame): mda cases.
        * variables (list): the name of the variables.
        * title (string): text for the figure title.
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if len(variables) == 1:
        _, ax = handle_axis(ax)
        ax.plot(data[variables[0]].values, marker=".", alpha=0.05)
        ax.plot(cases[variables[0]].values, color="k", marker="o")
        ax.set_xlabel("time")
        ax.set_ylabel(labels(variables[0]))

    elif len(variables) == 2:
        _, ax = handle_axis(ax)

    elif len(variables) == 3:
        _, ax = handle_axis(ax, dim=3)

        ax.scatter(
            data[variables[0]].values,
            data[variables[1]].values,
            data[variables[2]].values,
            marker=".",
            alpha=0.05,
        )
        ax.scatter(
            cases[variables[0]].values,
            cases[variables[1]].values,
            cases[variables[2]].values,
            color="k",
            marker="o",
        )
        ax.set_xlabel(labels(variables[0]))
        ax.set_ylabel(labels(variables[1]))
        ax.set_zlabel(labels(variables[2]))
    else:
        comb = [x[::-1] for x in product(range(0, len(variables)), repeat=2)]
        removed = []
        for i in comb:
            if i[1] <= i[0]:
                removed.append(i)

        _, ax = plt.subplots(len(variables), len(variables), figsize=(20, 20))
        for i in comb:
            if i in removed:
                ax[i[0], i[1]].axis("off")
            else:
                ax[i[0], i[1]].scatter(
                    data[variables[i[1]]].values,
                    data[variables[i[0]]].values,
                    marker=".",
                    alpha=0.05,
                )
                ax[i[0], i[1]].scatter(
                    cases[variables[i[1]]].values,
                    cases[variables[i[0]]].values,
                    color="k",
                    marker=".",
                )

            if i[0] + 1 == i[1]:
                ax[i[0], i[1]].set_ylabel(labels(variables[i[0]]))
                ax[i[0], i[1]].set_xlabel(labels(variables[i[1]]))

    if title is not None:
        ax.set_title(title)

    show(fname)

    return ax


def timeseries(data: pd.DataFrame, variable: str, ax=None, file_name: str = None):
    """Plots the time series of the variable

    Args:
        * data (pd.DataFrame): time series
        * variable (string): variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, ax = handle_axis(ax)
    ax.plot(data.loc[:, variable])
    try:
        ax.set_ylabel(labels(variable))
    except:
        ax.set_ylabel(variable)

    show(file_name)
    return ax


def storm_timeseries(
    df_sim: pd.DataFrame, df_obs: pd.DataFrame, variables: list, file_name: str = None
):
    """Plots the time series of simulation and observations (from differents RCM) for the choosen variables

    Args:
        * df_sim (pd.DataFrame): simulated time series
        * df_obs (dict): any key of the dict should be a pd.DataFrame of observed time series
        * variables (list): name of the variables
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if not isinstance(df_sim, pd.DataFrame):
        df_sim = df_sim.to_frame()

    _, ax = plt.subplots(
        len(variables), 1, figsize=(12, len(variables) * 2), sharex=True
    )

    for i, j in enumerate(variables):
        if isinstance(df_obs, dict):
            for key in df_obs.keys():
                ax[i].plot(
                    df_obs[key][j], ".", ms=2, alpha=0.5, label=key.split("_")[0]
                )
            # ax[i].set_xlim([df_obs[key].index[0], df_obs[key].index[-1]])
        else:
            if not isinstance(df_obs, pd.DataFrame):
                df_obs = df_obs.to_frame()
            ax[i].plot(df_obs[j], label=j)
            # ax[i].set_xlim([df_obs[j].index[0], df_obs[j].index[-1]])
        ax[i].set_ylabel(labels(j))
        ax[i].plot(df_sim[j], ".", ms=2, alpha=0.5, label="Simulation")

        if i == 0:
            ax[i].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0 + 0.1 * len(variables)),
                ncol=4,
            )

    show(file_name)
    return ax


def test_normality(data, params, ax=None, fname=None):
    """Test and visualize data normality with QQ-plot.

    Performs normality test on transformed data and displays a probability plot
    to assess Gaussian distribution fit.

    Args:
        data (array-like): Raw data values to test.
        params (dict): Transformation parameters for data normalization.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.

    Note:
        Prints chi-squared statistic and p-value from D'Agostino-Pearson test.
        Null hypothesis: data comes from a normal distribution (α=0.05).
    """

    data, params = core.transform(data, params)
    ax, fname = handle_axis(ax)

    _, ax = plt.subplots(1, 1, figsize=(8, 4))

    st.probplot(data, dist=st.norm, plot=ax)

    ax.set_title("Normality test")
    ax.set_ylabel("")

    k2, p = st.normaltest(data)
    print("\nChi-squared statistic = %.3f, p = %.3f" % (k2, p))

    alpha = 0.05
    if p > alpha:
        print(
            "\nThe transformed data is Gaussian (fails to reject the null hypothesis)"
        )
    else:
        print(
            "\nThe transformed data does not look Gaussian (reject the null hypothesis)"
        )
    show(fname)

    return ax


def cadency(data, label="", ax=None, legend=False, fname=None):
    """Plot temporal differences (cadency) of time series.

    Displays the time intervals between consecutive data points to identify
    gaps or irregular sampling in the time series.

    Args:
        data (pd.DataFrame): Time series with datetime index.
        label (str): Label for the plotted line. Defaults to "".
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        legend (bool): If True, displays legend. Defaults to False.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    _, ax = handle_axis(ax)
    ax.plot(
        data.index[1:], (data.index[1:] - data.index[:-1]).seconds / 3600, label=label
    )
    ax.set_ylabel("Cadency time (hr)")

    if legend:
        ax.legend()

    show(fname)
    return ax


def spectra(
    data,
    semilog=True,
    title=None,
    fname=None,
    ax=None,
    xlabel="Periods $T$ (years)",
    label="LombScargle",
):
    """Plot power spectral density from spectral analysis.

    Displays the power spectrum with frequency or period on x-axis and
    power on y-axis, highlighting statistically significant peaks.

    Args:
        data (pd.DataFrame): Spectral data with 'psd' column for power values
            and 'significant' boolean column. Index contains frequencies/periods.
        semilog (bool): If True, uses semilog y-axis. Defaults to True.
        title (str, optional): Plot title. Defaults to None.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        xlabel (str): X-axis label. Defaults to "Periods $T$ (years)".
        label (str): Legend label for method type ('LombScargle', 'FFT', etc.).
            Defaults to 'LombScargle'.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if semilog:
        ax.semilogy(data["psd"], label=label)
    else:
        ax.plot(data["psd"], label=label)
    ax.set_xlabel(xlabel)
    if label == "LombScargle":
        ax.set_ylabel("Normalized Lombscargle Periodogram")
    else:
        ax.set_ylabel("Fast Fourier Transform")
    ax.plot(data.loc[data["significant"], "psd"], "bo", label="significant")

    # for index in data.loc[data['significant'], 'psd'].index:
    #     ax.annotate('{:.2f}'.format(index), (index, data.loc[index, 'psd']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.legend(loc="best")
    # ax.set_xlim(left=0.05)
    ax.grid()
    if title is not None:
        ax.set_title(title)
    show(fname)

    return ax


def cdf(
    data: pd.DataFrame,
    var: str,
    ax=None,
    file_name: str = None,
    seaborn: bool = False,
    legend: str = False,
    label: str = None,
):
    """Plot cumulative distribution function (CDF) of data.

    Displays the empirical CDF either using custom ECDF calculation or
    seaborn's distribution plot with cumulative option.

    Args:
        data (pd.DataFrame): Time series data.
        var (str): Name of the variable column to plot.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        file_name (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.
        seaborn (bool): If True, uses seaborn.distplot for CDF visualization.
            Defaults to False.
        legend (bool): If True, displays legend. Defaults to False.
        label (str, optional): Custom label for the plot. If None, uses variable
            name with units. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    _, ax = handle_axis(ax)
    if not isinstance(label, str):
        label = labels(var)

    if seaborn:
        sns.distplot(
            data[var],
            hist_kws={"cumulative": True},
            kde_kws={"cumulative": True},
            ax=ax,
        )
    else:
        emp = utils.ecdf(data, var)
        ax.plot(emp[var].values, emp.index, label=label)

    if legend:
        ax.legend()

    # show(file_name)
    return ax


def boxplot(data: pd.DataFrame, variable: str, ax=None, file_name: str = None):
    """Draw monthly box plot of variable.

    Creates box plots showing the distribution of a variable across
    calendar months.

    Args:
        data (pd.DataFrame): Time series with datetime index.
        variable (str): Name of the variable column to plot.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        file_name (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    data["date"] = data.index
    data["month"] = data["date"].dt.strftime("%b")
    sns.boxplot(x="month", y=variable, data=data, ax=ax)

    return ax


def qq(
    data,
    prob_model="norm",
    marker=".",
    color="k",
    line="45",
    ax=None,
    label=None,
    fname=None,
):
    """Draw QQ-plot comparing data against theoretical distribution.

    Creates a quantile-quantile plot to assess how well data matches
    a specified probability distribution.

    Args:
        data (pd.DataFrame): Time series data to plot.
        prob_model (str): Name of scipy.stats probability distribution.
            Defaults to 'norm'.
        marker (str): Matplotlib marker style. Defaults to '.'.
        color (str): Color for data points. Defaults to 'k' (black).
        line (str): Reference line style ('45', 's', 'r', 'q', or None).
            Defaults to '45'.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        label (str, optional): Label for the data. Defaults to None.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    ax = sm.qqplot(
        data.values,
        dist=getattr(st, prob_model),
        fit=True,
        line=line,
        marker=marker,
        color=color,
        ax=ax,
        label=label,
    )
    show(fname)

    return ax


def probplot(data, ax=None, fit=True, fname=None):
    """Plot probability plot using statsmodels.

    Creates a probability plot to assess distributional assumptions
    with optional distribution fit.

    Args:
        data (pd.DataFrame or array-like): Data values to plot.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        fit (bool): If True, fits a distribution to the data. Defaults to True.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    probpl = sm.ProbPlot(data, fit=fit)
    probpl.probplot(ax=ax)
    show(fname)

    return ax


def nonstationary_percentiles(
    data: pd.DataFrame, variable: str, fun: str, pars=None, ax=None, fname=None
):
    """Plot monthly CDFs in normalized space.

    Displays cumulative distribution functions for each month in normalized
    (Gaussian) space to assess seasonal nonstationarity in the distribution.

    Args:
        data (pd.DataFrame): Time series with datetime index.
        variable (str): Name of the variable column to analyze.
        fun (str): Name of scipy.stats probability distribution (e.g., 'norm',
            'lognorm', 'genextreme').
        pars (tuple, optional): Distribution parameters. If None, fits parameters
            to data. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.

    Note:
        Each month is represented by a different color. Deviations from the
        45° line indicate departures from the annual distribution.
    """

    _, ax = handle_axis(ax)
    data["n"] = (
        (data.index.dayofyear + data.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": data.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    monthly_window = 1 / 12
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    if not pars:
        pars = getattr(st, fun).fit(data[variable])

    t, l = 0, 0
    while t < 1.0:
        aux = data.loc[((data["n"] > t) & (data["n"] < t + monthly_window))][variable]
        # try:
        n = len(aux)
        sorted_ = np.sort(aux)

        x = np.linspace(0, aux.max(), 100)
        cdf = getattr(st, fun).cdf(x, *pars)
        cdfe = np.interp(x, sorted_, np.linspace(0, 1, n))
        plt.plot(st.norm.ppf(cdfe), st.norm.ppf(cdf), label=months[l])
        # except:
        #     pass
        t += monthly_window
        l += 1

    ax.set_xticks(st.norm.ppf([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    ax.set_xticklabels([1, 5, 10, 25, 50, 75, 90, 95, 99])
    ax.set_yticks(st.norm.ppf([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    ax.set_yticklabels([1, 5, 10, 25, 50, 75, 90, 95, 99])
    ax.set_xlabel(r"Empirical percentiles")
    ax.set_ylabel(r"Theoretical percentiles")
    ax.grid(True)
    ax.legend()
    ax.plot([-3, 3], [-3, 3], "k")
    show(fname)
    return


def nonstationary_qq_plot(
    data: pd.DataFrame, var_: str, prob_model: str = "norm", fname=None
):
    """Draw monthly QQ-plots to assess seasonal distribution changes.

    Creates a 3x4 grid of QQ-plots, one for each calendar month, to visualize
    how the distribution varies throughout the year.

    Args:
        data (pd.DataFrame): Time series with datetime index.
        var_ (str): Name of the variable column to analyze.
        prob_model (str): Name of scipy.stats probability distribution.
            Defaults to 'norm'.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        None: Displays the plot.

    Note:
        Each subplot shows data for one calendar month. Deviations from the
        reference line indicate non-normality for that month.
    """

    _, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(10, 8))
    axs = axs.flatten()

    data["n"] = (
        (data.index.dayofyear + data.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": data.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    monthly_window = 1 / 12
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    t, l = 0, 0
    while t < 1.0:
        aux = data.loc[((data["n"] > t) & (data["n"] < t + monthly_window)), var_]
        try:
            qq(aux, prob_model, ax=axs[l], fname="to_axes")
        except:
            pass

        axs[l].text(0.05, 0.85, months[l], transform=axs[l].transAxes, weight="bold")
        t += monthly_window
        l += 1

    show(fname)

    return


def scatter_error_dependencies(
    df_dt: dict, variables: list, label: str, ax=None, file_name: str = None
):
    """Plot scatter comparison before and after temporal dependency correction.

    Displays side-by-side scatter plots showing data before and after removing
    temporal dependencies to assess the effect of the transformation.

    Args:
        df_dt (dict): Dictionary containing temporal dependency parameters with keys:
            - 'data_before': Original time series DataFrame
            - 'data_after': Decorrelated time series DataFrame
        variables (list): List of two variable names to plot [var_x, var_y].
        label (str): Title label for the plot.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        file_name (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    _, ax = handle_axis(ax)

    if isinstance(variables, str):
        variables = [variables]

    # _, ax = plt.subplots(1, 1)
    for i, j in enumerate(variables):
        if len(variables) == 1:
            dfy = df_dt["y"][i]
            dfy_ = df_dt["y*"][i]
        else:
            dfy = df_dt["y"][i]
            dfy_ = df_dt["y*"][i]

        ax.plot(
            dfy,
            dfy_,
            ".",
            label=label[i],
        )
    ax.grid()
    ax.set_xlabel(r" Observed ($\Phi^{-1}(F_{\zeta}(\zeta))$")
    ax.set_ylabel(r" Modeled ($\Phi^{-1}(F_{\zeta}(\zeta))$")
    ax.legend(loc=4, title=r"$\zeta$")
    show(file_name)
    return ax


def scatter(df1, df2, variables, names=["Observed", "Modeled"], fname=None, ax=None):
    """Plot scatter comparison between two datasets.

    Creates scatter plots comparing corresponding variables between two
    datasets (e.g., observed vs. modeled values).

    Args:
        df1 (pd.DataFrame): First dataset (typically observed data).
        df2 (pd.DataFrame): Second dataset (typically modeled data).
        variables (list): List of variable names to plot.
        names (list): Labels for [x-axis, y-axis] data sources.
            Defaults to ["Observed", "Modeled"].
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.

    Returns:
        list: List of axes objects for each subplot.
    """

    _, axs = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 5))
    if len(variables) != 1:
        axs.flatten()
    else:
        axs = list(axs)

    for index, variable in enumerate(variables):
        axs[index].plot(df1[variable], df2[variable], ".", label=variable)
        axs[index].grid()
        axs[index].set_xlabel(r" " + names[0])
    axs[0].set_ylabel(r" " + names[1])
    axs[0].legend(loc=4, title=r"$\zeta$")
    show(fname)
    return axs


def look_models(data, variable, params, num=10, fname=None):
    """Plot CDFs of multiple distribution fits for model comparison.

    Overlays empirical CDF with theoretical CDFs from top-ranked fitted
    distributions to visually compare goodness of fit.

    Args:
        data (pd.DataFrame): Time series data.
        variable (str): Name of the variable column.
        params (pd.DataFrame): DataFrame with fitted distribution parameters
            containing columns: distribution name, SSE, and parameter values.
            Should be sorted by fit quality (best first).
        num (int): Maximum number of distributions to plot. Defaults to 10.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.

    Note:
        Plots up to 'num' best-fitting distributions based on sum of squared errors.
    """

    _, ax = plt.subplots(1, 1)
    emp = utils.ecdf(data, variable)
    plt.plot(emp[variable], emp["prob"], label="empirical cdf")

    x = np.linspace(data[variable].min(), data[variable].max(), 1000)

    num = np.min([num, len(params)])
    for i in range(num):
        prob_model = getattr(st, params.iloc[i, 0])
        parameters = params.iloc[i, 2 : prob_model.numargs + 4].values
        try:
            ax.plot(
                x,
                prob_model.cdf(x, *parameters),
                label=params.iloc[i, 0].replace("_", " "),
            )
        except:
            continue
    ax.set_xlabel(labels(variable))
    ax.set_ylabel("prob")
    ax.legend(ncol=2)
    show(fname)

    return ax


def crosscorr(xy, xys, variable, lags=48, fname=None):
    """Plot cross-correlation between observed and simulated variables.

    Displays the cross-correlation function comparing two time series
    to assess temporal relationships and model performance.

    Args:
        xy (tuple): Tuple of two arrays (x, y) for observed data.
        xys (tuple): Tuple of two arrays (x, y) for simulated data.
        variable (str): Name of variable for plot labeling.
        lags (int): Maximum lag time in hours. Defaults to 48.
        fname (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.

    Note:
        Cross-correlation is normalized by standard deviations and means.
    """

    _, ax = plt.subplots(1, 1)
    tiempo = np.arange(-lags, lags, 3 / 24.0)
    i1, i2 = len(xy[0]) / 2.0 - len(tiempo) / 2.0, len(xy[0]) / 2.0 + len(tiempo) / 2.0
    i1s, i2s = (
        len(xys[0]) / 2.0 - len(tiempo) / 2.0,
        len(xys[0]) / 2.0 + len(tiempo) / 2.0,
    )
    ccf = np.correlate(xy[0], xy[1], mode="same")
    ccf = ccf / len(ccf) - np.mean(xy[0]) * np.mean(xy[1])
    ccf = ccf / (np.std(xy[0]) * np.std(xy[1]))
    ccf2 = np.correlate(xys[0], xys[1], mode="same")
    ccf2 = ccf2 / len(ccf2) - np.mean(xys[0]) * np.mean(xys[1])
    ccf2 = ccf2 / (np.std(xys[0]) * np.std(xys[1]))

    plt.figure()
    plt.plot(tiempo, ccf[i1:i2], ".", color="gray")
    plt.plot(tiempo, ccf2[i1s:i2s], "k")
    plt.ylim(ymax=1)

    plt.legend(("Observed", "Simulated"), loc="best")
    plt.xlabel("Time [days]")
    plt.ylabel("Cross-correlation of " + variable)
    plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
    show(fname)

    return ax


def corr(data: pd.DataFrame, lags: int = 24, ax=None, file_name: str = None):
    """Plot autocorrelation function of time series.

    Displays the normalized autocorrelation as a function of time lag
    to identify temporal dependencies and periodicity.

    Args:
        data (pd.DataFrame): Time series data (single column or Series).
        lags (int): Maximum lag time in hours. Defaults to 24.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.
        file_name (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    _, ax = handle_axis(ax)
    ax.acorr(data, usevlines=False, maxlags=lags, normed=True, lw=2)

    ax.set_xlabel(r"Lags (hr)")
    ax.set_ylabel(r"Normalized autocorrelation")
    ax.grid(True)
    ax.legend()
    show(file_name)

    return ax


def joint_plot(data: pd.DataFrame, varx: str, vary: str, ax=None):
    """Plot joint probability distribution of two variables.

    Creates a joint plot with marginal distributions using seaborn to visualize
    the bivariate relationship and individual distributions.

    Args:
        data (pd.DataFrame): Time series containing both variables.
        varx (str): Name of variable for x-axis.
        vary (str): Name of variable for y-axis.
        ax (matplotlib.axes.Axes, optional): Axis for the plot. Creates new if None.
            Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    sns.jointplot(x=varx, y=vary, data=data, ax=ax)

    return ax


def bivariate_ensemble_pdf(
    df_sim: pd.DataFrame, df_obs: dict, varp: list, file_name: str = None
):
    """Plot ensemble of bivariate PDFs comparing simulation with multiple observations.

    Displays a grid of contour plots showing the bivariate probability density
    function for a simulation and multiple observational datasets.

    Args:
        df_sim (pd.DataFrame): Simulated time series data.
        df_obs (dict): Dictionary where each key contains an observed time series
            DataFrame. Each will be plotted in a separate subplot.
        varp (list): List of two variable names [var1, var2] to plot.
        file_name (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.

    Returns:
        matplotlib.axes.Axes: Array of axes with the plots.

    Note:
        Creates 3x3 grid with simulation in first subplot and observations following.
    """

    _, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))
    row, column = 0, 0
    H, x, y = np.histogram2d(
        df_sim[varp[0]], df_sim[varp[1]], bins=[25, 25], density=True
    )
    x, y = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    y, x = np.meshgrid(y, x)
    levels = np.linspace(np.max(H) / 8, np.max(H), 8)
    ax[row, column].contourf(x, y, H, alpha=0.25, levels=np.append(0, levels))
    CS = ax[row, column].contour(x, y, H, levels=levels)
    ax[row, column].clabel(CS, inline=1, fontsize=10)
    ax[row, column].set_title("SIMULATION")
    ax[row, column].set_ylabel(labels(varp[1]))
    ax[row, column].grid(True)
    ax[0, 1].axis("off")
    column += 2

    for key in df_obs.keys():
        Ho, xo, yo = np.histogram2d(
            df_obs[key][varp[0]], df_obs[key][varp[1]], bins=[25, 25], density=True
        )
        xo, yo = (xo[:-1] + xo[1:]) / 2, (yo[:-1] + yo[1:]) / 2
        yo, xo = np.meshgrid(yo, xo)
        ax[row, column].contourf(xo, yo, Ho, alpha=0.25, levels=np.append(0, levels))
        CS = ax[row, column].contour(x, y, H, levels=levels)
        ax[row, column].clabel(CS, inline=1, fontsize=10)
        ax[row, column].set_title(key.split("_")[0])
        ax[row, column].grid(True)
        if column == 0:
            ax[row, column].set_ylabel(labels(varp[1]))

        if row == 2:
            ax[row, column].set_xlabel(labels(varp[0]))
        column += 1

        if column >= 3:
            column = 0
            row += 1
    show(file_name)
    return


def bivariate_pdf(
    df_sim: pd.DataFrame,
    df_obs: pd.DataFrame,
    variables: list,
    bins: int = None,
    levels: list = None,
    ax=None,
    file_name: str = None,
    logx: str = False,
    logy: str = False,
    contour: bool = False,
):
    """Plot bivariate PDF comparison between observed and simulated data.

    Creates side-by-side contour or image plots of the bivariate probability
    density functions for visual model validation.

    Args:
        df_sim (pd.DataFrame): Simulated time series data.
        df_obs (pd.DataFrame): Observed time series data.
        variables (list): List of two variable names [var1, var2] to plot.
        bins (int or list, optional): Number of bins for histogram. If None,
            uses [25, 25]. Defaults to None.
        levels (list, optional): Contour levels. If None, auto-computed.
            Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes for the plot. Creates new if None.
            Defaults to None.
        file_name (str, optional): File path to save the plot. If None, displays
            interactively. Defaults to None.
        logx (bool): If True, applies log transform to first variable.
            Defaults to False.
        logy (bool): If True, applies log transform to second variable.
            Defaults to False.
        contour (bool): If True, uses contour plot instead of image. 
            Defaults to False.

    Returns:
        matplotlib.axes.Axes: Array of axes with the plots.
    """

    _, ax = handle_axis(ax, col_plots=2)

    if bins is None:
        bins = [25, 25]
    if logy:
        df_obs[variables[0]] = np.log(df_obs[variables[0]])
        df_sim[variables[0]] = np.log(df_sim[variables[0]])

    if logx:
        df_obs[variables[1]] = np.log(df_obs[variables[1]])
        df_sim[variables[1]] = np.log(df_sim[variables[1]])

    nn, nx_, ny_ = np.histogram2d(
        df_obs[variables[0]], df_obs[variables[1]], bins=bins, density=True
    )
    ny, nx = np.meshgrid(ny_[:-1] + np.diff(ny_) / 2.0, nx_[:-1] + np.diff(nx_) / 2.0)

    if levels is None:
        levels = np.linspace(np.max(nn) / 12, np.max(nn), 12)

    min_, max_ = np.min(nn), np.max(nn)

    ax[0].imshow(
        np.flipud(nn),
        extent=[np.min(ny), np.max(ny), np.min(nx), np.max(nx)],
        vmin=min_,
        vmax=max_,
        cmap="viridis",
        aspect="auto",
    )

    labelx = labels(variables[1])
    labely = labels(variables[0])
    if logx:
        labelx = "log " + labelx

    if logy:
        labely = "log " + labely

    ax[0].set_xlabel(labelx)
    ax[0].set_ylabel(labely)

    nn_, nx, ny = np.histogram2d(
        df_sim[variables[0]], df_sim[variables[1]], bins=[nx_, ny_], density=True
    )
    ny, nx = np.meshgrid(ny[:-1] + np.diff(ny) / 2.0, nx[:-1] + np.diff(nx) / 2.0)
    cs = ax[1].imshow(
        np.flipud(nn_),
        extent=[np.min(ny), np.max(ny), np.min(nx), np.max(nx)],
        vmin=min_,
        vmax=max_,
        cmap="viridis",
        aspect="auto",
    )
    ax[1].set_xlabel(labelx)
    ax[1].set_yticklabels([])

    show(file_name)

    # R2 = 1 - np.sum((np.ravel(nn) - np.ravel(nn_)) ** 2) / np.sum(
    #     (np.ravel(nn_) - np.mean(np.ravel(nn_))) ** 2
    # )
    # print(R2)

    return ax


def nonstationary_cdf(
    data: pd.DataFrame,
    variable: str,
    param: dict = None,
    daysWindowsLength: int = 14,
    equal_windows: bool = False,
    ax=None,
    log: bool = False,
    file_name: str = None,
    label: str = None,
    lst="-",
    legend: bool = True,
    legend_loc: str = "right",
    title: str = None,
    date_axis: bool = False,
    pemp: list = None,
    emp: bool = True,
):
    """Plots the time variation of given percentiles of data and theoretical function if provided

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable to be adjusted
        * param (dict, optional): the parameters of the the theoretical model if they are also plotted.
        * daysWindowsLength (int, optional): period of windows length for making the non-stationary empirical distribution function. Defaults to 14 days.
        * equal_windows (bool): use the windows for the ecdf of total data and timestep
        * ax: matplotlib.ax
        * log: logarhitmic scale
        * file_name (string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * label: string with the label
        * lst (string, optional): linestyle for theoretical distribution.
        * legend: plot the legend
        * legend_loc: locate the legend
        * title: draw the title
        * date_axis: create a secondary axis with time
        * pemp: list with percentiles to be plotted
        * emp (bool, optional): if True plot the empirical nonst distribution

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    T = 1
    if param is not None:
        if param["basis_period"] is not None:
            T = np.max(param["basis_period"])

    data["n"] = np.fmod(
        (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    dt = 366
    n = np.linspace(0, 1, dt)
    if emp:
        xp, pemp = utils.nonstationary_ecdf(
            data,
            variable,
            wlen=daysWindowsLength / (365.25 * T),
            equal_windows=equal_windows,
            pemp=pemp,
        )

    _, ax = handle_axis(ax)

    ax.set_prop_cycle("color", [plt.cm.winter(i) for i in np.linspace(0, 1, len(pemp))])
    if emp:
        col_per = list()

        if len(xp.index.unique()) > 60:
            marker, ms, markeredgewidth = ".", 8, 1.5
        else:
            marker, ms, markeredgewidth = "+", 4, 1.5

        for j, i in enumerate(pemp):
            if isinstance(param, dict):
                if param["transform"]["plot"]:
                    xp[i], _ = core.transform(xp[[i]], param)
                    xp[i] -= param["transform"]["min"]
                    if "scale" in param:
                        xp[i] = xp[i] / param["scale"]
            if log:
                p = ax.semilogy(
                    xp[i],
                    marker=marker,
                    ms=ms,
                    markeredgewidth=markeredgewidth,
                    lw=0,
                    label=str(i),
                )
            else:
                p = ax.plot(
                    xp[i],
                    marker=marker,
                    ms=ms,
                    markeredgewidth=markeredgewidth,
                    lw=0,
                    label=str(i),
                )
            col_per.append(p[0].get_color())

    if isinstance(param, dict):
        if param["status"] == "Distribution models fitted succesfully":
            param = utils.string_to_function(param, None)

            for i, j in enumerate(pemp):
                df = pd.DataFrame(np.ones(dt) * pemp[i], index=n, columns=["prob"])
                df["n"] = n
                if (param["non_stat_analysis"] == True) | (param["no_fun"] > 1):
                    res = core.ppf(df, param)
                else:
                    res = pd.DataFrame(
                        param["fun"][0].ppf(df["prob"], *param["par"]),
                        index=df.index,
                        columns=[variable],
                    )

                # Transformed timeserie
                if (not param["transform"]["plot"]) & param["transform"]["make"]:
                    if "scale" in param:
                        res[param["var"]] = res[param["var"]] * param["scale"]

                    res[param["var"]] = res[param["var"]] + param["transform"]["min"]
                    res[param["var"]] = core.inverse_transform(
                        res[[param["var"]]], param
                    )
                elif ("scale" in param) & (not param["transform"]["plot"]):
                    res[param["var"]] = res[param["var"]] * param["scale"]

                if log:
                    if emp:
                        ax.semilogy(
                            res[param["var"]].index,
                            res[param["var"]].values,
                            color=col_per[i],
                            ls=lst,
                            lw=2,
                            label=str(j),
                        )
                    else:
                        ax.plot(
                            res[param["var"]].index.values,
                            res[param["var"]].values,
                            ls=lst,
                            lw=2,
                            label=str(j),
                        )
                else:
                    if param["type"] == "circular":
                        if emp:
                            ax.plot(
                                res[param["var"]].index,
                                np.rad2deg(res[param["var"]].values),
                                color=col_per[i],
                                ls=lst,
                                lw=2,
                                label=str(j),
                            )
                        else:
                            ax.plot(
                                res[param["var"]].index,
                                np.rad2deg(res[param["var"]].values),
                                ls=lst,
                                lw=2,
                                label=str(j),
                            )
                    else:
                        if emp:
                            ax.plot(
                                res[param["var"]].index,
                                res[param["var"]].values,
                                color=col_per[i],
                                ls=lst,
                                lw=2,
                                label=str(j),
                            )
                        else:
                            ax.plot(
                                res[param["var"]].index,
                                res[param["var"]].values,
                                ls=lst,
                                lw=2,
                                label=str(j),
                            )
        else:
            raise ValueError(
                "Model was not fit successfully. Look at the marginal fit."
            )

    ax.grid()

    box = ax.get_position()
    if legend:
        # Shrink current axis
        if param:
            if legend_loc == "bottom":
                ax.set_position([box.x0, box.y0, box.width, box.height])
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(-0.2, 0.0),
                    ncol=len(pemp),
                    title="Percentiles",
                )
                if param["type"] == "circular":
                    ax.set_yticks([0, 90, 180, 270, 360])
            else:
                ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                # Put a legend to the right of the current axis
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=2,
                    title="Percentiles",
                )
                if param["type"] == "circular":
                    ax.set_yticks([0, 90, 180, 270, 360])
        else:
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, title="Percentiles"
            )

    if isinstance(title, str):
        ax.set_title(title, color="k", fontweight="bold")

    if not label:
        label = labels(variable)

    if log:
        label = "log " + label
    ax.set_ylabel(label)
    ax.set_xlabel("Normalized period")
    if date_axis:
        ax2 = ax.twiny()
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        ax2.set_xticks(
            np.array(
                [
                    0.6 / 13,
                    1.5 / 13,
                    2.5 / 13,
                    3.5 / 13,
                    4.5 / 13,
                    5.5 / 13,
                    6.5 / 13,
                    7.5 / 13,
                    8.5 / 13,
                    9.5 / 13,
                    10.5 / 13,
                    11.5 / 13,
                    12.4 / 13,
                ]
            ),
            minor=False,
        )
        # 16 is a slight approximation since months differ in number of days.
        # ax2.xaxis.set_minor_locator(np.array([1/13, 2/13, 3/13, 4/13, 5/13, 6/13, 7/13, 8/13, 9/13, 10/13, 11/13, 12/13]))
        # ax2.xaxis.set_major_formatter(ticker.NullFormatter())

        # Hide major tick labels
        ax2.set_xticklabels("")

        # Customize minor tick labels
        ax2.set_xticks(
            np.array(
                [
                    1 / 13,
                    2 / 13,
                    3 / 13,
                    4 / 13,
                    5 / 13,
                    6 / 13,
                    7 / 13,
                    8 / 13,
                    9 / 13,
                    10 / 13,
                    11 / 13,
                    12 / 13,
                ]
            ),
            minor=True,
        )
        ax2.set_xticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], minor=True
        )
        ax2.tick_params(
            axis="x",  # changes apply to the x-axis
            which="minor",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True,
        )

        ax2.set_xlabel(r"Normal Year")
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width * 0.6, box.height * 0.9]
        )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    show(file_name)

    return ax


def nonstat_cdf_ensemble(
    data: pd.DataFrame,
    variable: str,
    param: dict = None,
    models: list = None,
    daysWindowsLength: int = 14,
    equal_windows: bool = False,
    ax=None,
    log: bool = False,
    file_name: str = None,
    label: str = None,
    legend: bool = True,
    legend_loc: str = "right",
    title: str = None,
    date_axis: bool = False,
    pemp: list = None,
):
    """Plots the time variation of given percentiles of data and theoretical function if provided

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable to be adjusted
        * param (dict, optional): the parameters of the the theoretical model if they are also plotted.
        * daysWindowsLength (int, optional): period of windows length for making the non-stationary empirical distribution function. Defaults to 14 days.
         * equal_windows (bool): use the windows for the ecdf of total data and timestep
        * ax: matplotlib.ax
        * log: logarhitmic scale
        * file_name (string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * label: string with the label
        * legend: plot the legend
        * legend_loc: locate the legend
        * title: draw the title
        * date_axis: create a secondary axis with time
        * pemp: list with percentiles to be plotted

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    T = 1
    if param is not None:
        if param[variable][models[0]]["basis_period"] is not None:
            T = np.max(param[variable][models[0]]["basis_period"])

    data["n"] = np.fmod(
        (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    dt = 366
    n = np.linspace(0, 1, dt)
    xp, pemp = utils.nonstationary_ecdf(
        data,
        variable,
        wlen=daysWindowsLength / (365.25 * T),
        equal_windows=equal_windows,
        pemp=pemp,
    )

    _, ax = handle_axis(ax)

    ax.set_prop_cycle("color", [plt.cm.winter(i) for i in np.linspace(0, 1, len(pemp))])
    col_per = list()

    if len(xp.index.unique()) > 60:
        marker, ms, markeredgewidth = ".", 8, 1.5
    else:
        marker, ms, markeredgewidth = "+", 4, 1.5

    for j, i in enumerate(pemp):
        if isinstance(param, dict):
            if param[variable][models[0]]["transform"]["plot"]:
                xp[i], _ = core.transform(xp[[i]], param[variable][models[0]])
                xp[i] -= param[variable][models[0]]["transform"]["min"]
                if "scale" in param:
                    xp[i] = xp[i] / param[variable][models[0]]["scale"]
        if log:
            p = ax.semilogy(
                xp[i],
                marker=marker,
                ms=ms,
                markeredgewidth=markeredgewidth,
                lw=0,
                label=str(i),
            )
        else:
            p = ax.plot(
                xp[i],
                marker=marker,
                ms=ms,
                markeredgewidth=markeredgewidth,
                lw=0,
                label=str(i),
            )
        col_per.append(p[0].get_color())

    if isinstance(param, dict):
        if (
            param[variable][models[0]]["status"]
            == "Distribution models fitted succesfully"
        ):
            for i, j in enumerate(pemp):
                df = pd.DataFrame(np.ones(dt) * pemp[i], index=n, columns=["prob"])
                df["n"] = n
                if (param[variable][models[0]]["non_stat_analysis"] == True) | (
                    param[variable][models[0]]["no_fun"] > 1
                ):
                    res = core.ensemble_ppf(df, param, "pr", nodes=[4383, 900])
                else:
                    res = pd.DataFrame(
                        param[models[0]]["fun"][0].ppf(
                            df["prob"], *param[models[0]]["par"]
                        ),
                        index=df.index,
                        columns=[variable],
                    )

                # Transformed timeserie
                if (not param[variable][models[0]]["transform"]["plot"]) & param[
                    variable
                ][models[0]]["transform"]["make"]:
                    if "scale" in param:
                        res[param[variable][models[0]]["var"]] = (
                            res[param[variable][models[0]]["var"]]
                            * param[variable][models[0]]["scale"]
                        )

                    res[param[variable][models[0]]["var"]] = (
                        res[param[variable][models[0]]["var"]]
                        + param[variable][models[0]]["transform"]["min"]
                    )
                    res[param[variable][models[0]]["var"]] = core.inverse_transform(
                        res[[param[variable][models[0]]["var"]]],
                        param[variable][models[0]],
                    )
                elif ("scale" in param) & (
                    not param[variable][models[0]]["transform"]["plot"]
                ):
                    res[param[variable][models[0]]["var"]] = (
                        res[param[variable][models[0]]["var"]]
                        * param[variable][models[0]]["scale"]
                    )

                if log:
                    ax.semilogy(
                        res[param[variable][models[0]]["var"]].index,
                        res[param[variable][models[0]]["var"]].values,
                        color=col_per[i],
                        lw=2,
                        label=str(j),
                    )
                else:
                    if param[variable][models[0]]["type"] == "circular":
                        ax.plot(
                            res[param[variable][models[0]]["var"]].index,
                            np.rad2deg(res[param[variable][models[0]]["var"]].values),
                            color=col_per[i],
                            lw=2,
                            label=str(j),
                        )
                    else:
                        ax.plot(
                            res[param[variable][models[0]]["var"]].index,
                            res[param[variable][models[0]]["var"]].values,
                            color=col_per[i],
                            lw=2,
                            label=str(j),
                        )
        else:
            raise ValueError(
                "Model was not fit successfully. Look at the marginal fit."
            )

    ax.grid()

    if legend:
        # Shrink current axis
        box = ax.get_position()
        if param:
            if legend_loc == "bottom":
                ax.set_position([box.x0, box.y0, box.width, box.height])
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(-0.2, 0.0),
                    ncol=len(pemp),
                    title="Percentiles",
                )
                if param[variable][models[0]]["type"] == "circular":
                    ax.set_yticks([0, 90, 180, 270, 360])
            else:
                ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                # Put a legend to the right of the current axis
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=2,
                    title="Percentiles",
                )
                if param[variable][models[0]]["type"] == "circular":
                    ax.set_yticks([0, 90, 180, 270, 360])
        else:
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, title="Percentiles"
            )

    if isinstance(title, str):
        ax.set_title(title, color="k", fontweight="bold")

    if not label:
        label = labels(variable)
    ax.set_ylabel(label)
    ax.set_xlabel("Normalized period")
    if date_axis:
        ax2 = ax.twiny()
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        ax2.set_xticks(
            np.array(
                [
                    0.6 / 13,
                    1.5 / 13,
                    2.5 / 13,
                    3.5 / 13,
                    4.5 / 13,
                    5.5 / 13,
                    6.5 / 13,
                    7.5 / 13,
                    8.5 / 13,
                    9.5 / 13,
                    10.5 / 13,
                    11.5 / 13,
                    12.4 / 13,
                ]
            ),
            minor=False,
        )
        # 16 is a slight approximation since months differ in number of days.
        # ax2.xaxis.set_minor_locator(np.array([1/13, 2/13, 3/13, 4/13, 5/13, 6/13, 7/13, 8/13, 9/13, 10/13, 11/13, 12/13]))
        # ax2.xaxis.set_major_formatter(ticker.NullFormatter())

        # Hide major tick labels
        ax2.set_xticklabels("")

        # Customize minor tick labels
        ax2.set_xticks(
            np.array(
                [
                    1 / 13,
                    2 / 13,
                    3 / 13,
                    4 / 13,
                    5 / 13,
                    6 / 13,
                    7 / 13,
                    8 / 13,
                    9 / 13,
                    10 / 13,
                    11 / 13,
                    12 / 13,
                ]
            ),
            minor=True,
        )
        ax2.set_xticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], minor=True
        )
        ax2.tick_params(
            axis="x",  # changes apply to the x-axis
            which="minor",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True,
        )

        ax2.set_xlabel(r"Normal Year")
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width * 0.6, box.height * 0.9]
        )
    show(file_name)

    return ax


def soujourn(data, variable, info, case="above", ax=None, fname=None):
    """Plots the distribution function of soujourn above or below a given threshold

    Args:
        data (_type_): _description_
        variable (_type_): _description_
        info (_type_): _description_
            time_step = "1H"
            min_duration = 3
            inter_time = 3
            threshold = threshold
            interpolation = True
        ax (_type_, optional): _description_. Defaults to None.
        case (str, optional): _description_. Defaults to "above".
        fname (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    info = storm_properties(data, variable, info)

    if case == "above":
        var_ = "dur_storm"
    elif case == "below":
        var_ = "dur_calms"
    else:
        raise ValueError("Case options are above or below. {} given.".format(case))

    _, ax = handle_axis(ax)
    ax = cdf(
        info,
        var_,
        ax=ax,
        file_name="to_axes",
        seaborn=False,
        legend=False,
        label=None,
    )

    show(fname)
    return ax


def nonstationary_cdf_ensemble(
    data: pd.DataFrame, variable: str, ax=None, marker: str = ".", file_name: str = None
):
    """Plots the time variation of given percentiles of data and the theoretical function if provided

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable to be adjusted
        * ax (matplotlib.axis): axis for the plot
        * marker (string): symbol feature of the plot
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot
    """

    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    T = 1
    data["n"] = np.fmod(
        (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    xp, pemp = utils.nonstationary_ecdf(data, variable)
    cols = [
        "blue",
        "orange",
        "green",
        "red",
        "brown",
        "cyan",
        "purple",
        "black",
        "gray",
        "yellow",
    ]
    for j, i in enumerate(pemp):
        ax.plot(xp.loc[:, i], marker, color=cols[j], alpha=0.25)

    ax.grid()

    ax.set_ylabel(labels(variable))
    ax.set_xlabel("Normalized year")
    show(file_name)

    return ax


def pdf_n_i(
    df_obs: pd.DataFrame,
    ns: list,
    param: dict = None,
    variable: str = None,
    nbins: int = 12,
    wlen: float = 14 / 365.25,
    file_name: str = None,
    stationary: bool = False,
):
    """Compute the pdf at n-times

    Args:
        df_obs (pd.DataFrame): input data
        ns (list): [description]
        param (dict): [description]
        variable (str, optional): [description]. Defaults to None.
        nbins (int): number of bins. Defaults to 12.
        wlen (float): length of the moving windows in days. Defautls 14/365.25
        file_name (str, optional): [description]. Defaults to None.

    Returns:
        matplotlib.ax: the figure
    """

    _, axs = plt.subplots(2, 1, sharex=True)
    axs = axs.flatten()
    colors = ["deepskyblue", "cyan", "darkblue", "royalblue", "b"]

    if param is not None:
        # Make the plot for stationary theoretical distributions
        if stationary:
            ns = [0.5]
        # Make the plot for non-stationary theoretical distributions at a given n
        for i, j in enumerate(ns):
            df = core.numerical_cdf_pdf_at_n(j, param, variable)

            if all:
                label_ = "F"
            else:
                label_ = "F(n: " + str(j) + ")"
            axs[0].plot(df["cdf"], color=colors[i], label=label_ + "-theoretical")
            axs[1].plot(df["pdf"], color=colors[i], label=label_ + "-theoretical")

    if df_obs is not None:
        # Make the plot for stationary empirical distributions
        if stationary:
            x = np.linspace(0, 1, nbins)
            emp = df_obs[variable].quantile(q=x).values

            axs[0].plot(emp, x, ".", color=colors[i], label=label_ + "-empirical")

            axs[1].plot(
                emp[1:],
                np.diff(x) / np.diff(emp),
                ".",
                color=colors[i],
                label=label_ + "-empirical",
            )

        # Make the plot for non-stationary empirical distributions at a given n
        else:

            df_obs["n"] = np.fmod(
                (df_obs.index - datetime.datetime(df_obs.index[0].year, 1, 1, 0))
                .total_seconds()
                .values
                / (param["basis_period"][0] * 365.25 * 24 * 3600),
                1,
            )

            for i, j in enumerate(ns):
                or_ = False
                min_ = j - wlen
                if min_ < 0:
                    min_ = 1 - wlen
                    or_ = True
                max_ = j + wlen
                if or_:
                    mask = (df_obs["n"] <= max_) | (df_obs["n"] >= min_)
                else:
                    mask = (df_obs["n"] <= max_) & (df_obs["n"] >= min_)

                if isinstance(param, dict):
                    if param["transform"]["plot"]:
                        df_obs[variable], _ = core.transform(df_obs[variable], param)
                        df_obs[variable] -= param["transform"]["min"]
                        if "scale" in param:
                            df_obs[variable] = df_obs[variable] / param["scale"]
                if stationary:
                    label_ = "F"
                else:
                    label_ = "F(n: " + str(j) + ")"
                x = np.linspace(0, 1, nbins)

                emp = df_obs[variable].loc[mask].quantile(q=x).values

                axs[0].plot(emp, x, ".", color=colors[i], label=label_ + "-empirical")

                axs[1].plot(
                    emp[1:],
                    np.diff(x) / np.diff(emp),
                    ".",
                    color=colors[i],
                    label=label_ + "-empirical",
                )

    axs[0].grid()
    axs[1].grid()

    axs[0].legend()
    axs[0].set_ylabel("probability")
    axs[1].set_ylabel("probability")
    axs[1].set_xlabel(labels(variable))

    show(file_name)

    return axs


def wrose(
    wd: np.ndarray,
    ws: np.ndarray,
    legend_title: str = "Wave rose",
    fig_title: str = None,
    var_name: str = "Wave height (m)",
    bins: list = [0, 0.25, 0.5, 1.5, 2.5],
    calm_limit=0,
    file_name: str = None,
):
    """Draws a wind or wave rose

    Args:
        * wd (pd.DataFrame): time series with the circular variable
        * ws (pd.DataFrame): time series with the linear variable
        * legend_title (str, optional): set the title of the rose. Defaults to 'Wave rose'.
        * fig_title (str, optional): set the title of the figure. Defaults to None.
        * var_name (str, optional): name of the mean variable. Default 'Wave height (m)'
        * bins: value of segments for variable
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        wd,
        ws,
        nsector=16,
        edgecolor="white",
        normed=True,
        cmap=plt.cm.viridis,
        opening=1,
        calm_limit=calm_limit,
        bins=bins,
    )
    fig.subplots_adjust(top=0.8)

    ax.set_xticklabels(["E", "NE", "N", "NW", "W", "SW", "S", "SE"])

    if isinstance(legend_title, str):
        ax.text(
            0.5,
            1.2,
            legend_title,
            fontsize=12,
            horizontalalignment="center",
            transform=ax.transAxes,
        )

    ax.set_legend(title=var_name, loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
    if isinstance(fig_title, str):
        plt.rc("text", usetex=False)
        ax.text(-0.1, 1, fig_title, transform=ax.transAxes, fontweight="bold")

    show(file_name)
    return ax


def seasonalbox(data, variable, fname=None):
    """Draws a boxplot

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, ax = plt.subplots()
    box, median = [], []
    box = [data.loc[data.index.month == i].values[:, 0] for i in range(1, 13)]
    median = data.groupby(data.index.month).median()

    bp = plt.boxplot(
        box,
        notch=1,
        sym="+",
        patch_artist=True,
        widths=0.3,
        showmeans=False,
        showfliers=False,
    )
    for k in bp["boxes"]:
        k.set(color="brown", linewidth=1, alpha=0.5)

    plt.plot(range(1, 13), median, "r", label="median")
    plt.plot(range(1, 13), np.mean(median) * np.ones(12), color="gray", label="mean")
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    plt.ylabel(labels(variable))

    plt.legend()
    show(fname)

    return ax


def ensemble_acorr(
    lags: list,
    lagsim: list,
    corr_: list,
    corrsim_: list,
    vars_: list,
    ax=None,
    file_name: str = None,
):
    """Plots the correlation bands of RCMs and a given simulation

    Args:
        lags (list): lags of RCM data from correlation functions (x-axis)
        lagsim (list): lags of simulation data from correlation functions (x-axis)
        corr_ (list): correlation of RCM data
        corrsim_ (list): correlation of simulations
        vars_ (list): variables to be plotted
        ax ([matplotlib.ax], optional): Axis where to plot the figure. Defaults to None.
        file_name (str, optional): A file name for saving the figure. Defaults to None.

    Returns:
        ax.matplotlib: the figure
    """

    _, ax = handle_axis(ax)

    color = ["royalblue", "lightsteelblue", "lightgrey", "darkgoldenrod", "forestgreen"]

    for ind_, var_ in enumerate(vars_):
        if ind_ == 0:
            ax.fill_between(
                lags[var_][0],
                np.min(corr_[var_], axis=0),
                np.max(corr_[var_], axis=0),
                alpha=0.25,
                color=color[ind_],
                label="RCMs band",
            )
        else:
            ax.fill_between(
                lags[var_][0],
                np.min(corr_[var_], axis=0),
                np.max(corr_[var_], axis=0),
                alpha=0.25,
                color=color[ind_],
            )
        ax.plot(
            lagsim[var_], corrsim_[var_], lw=2, color=color[ind_], label=labels(var_)
        )

    ax.set_xlabel(r"$\mathrm{\mathbf{Lags\quad  (hours)}}$")
    ax.set_ylabel(r"$\mathrm{\mathbf{Normalized\quad autocorrelation}}$")
    ax.grid(True)
    ax.legend()
    show(file_name)
    return ax


def heatmap(
    data: np.ndarray,
    param: dict,
    cmap: str = "bwr_r",
    type_: str = None,
    file_name: str = None,
    ax=None,
    minmax=False,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        * data (np.ndarray): A 2D array of shape (N, M).
        * param (dict): A list or array of length N with the labels for the rows.
        * type_ (str): type of variable to be plotted. B stands for the parameter
            matrix and Q for the covariance matrix
        * file_name: name of the oputput file
    """
    _, ax = handle_axis(ax)

    # Plot the heatmap
    if minmax == "minimax":
        im = ax.imshow(data, cmap=cmap, vmin=np.min(data), vmax=np.max(data))
    elif isinstance(minmax, list):
        im = ax.imshow(data, cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    elif minmax == "log":
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=0.001, vmax=10))
    else:
        im = ax.imshow(data, cmap=cmap)

    # We want to show all ticks ...
    ax.set_xticks(np.arange(np.asarray(data).shape[1]))
    ax.set_yticks(np.arange(np.asarray(data).shape[0]))
    # ... and label them with the respective list entries.

    if type_ == "B":
        column_labels, j = ["mean"], 0
        for i in range(np.asarray(data).shape[1] - 1):
            if not i % len(param["vars"]):
                j += 1
            column_labels.append(
                labels(param["vars"][i % len(data)]) + " (t-" + str(j) + ")"
            )
        row_labels = []
        for key_ in param["vars"]:
            row_labels.append(labels(key_))
    else:
        column_labels = param["vars"]
        row_labels = param["vars"]

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(np.asarray(data).shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(np.asarray(data).shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_heatmap(im, valfmt="{x:.2f}")

    # fig.tight_layout()
    show(file_name)

    return


def qqplot(
    df1, df2, variable, noperc, ax=None, fname=None, label=None, legend=True, title=None
):
    """[summary]

    Args:
        df1 ([type]): [description]
        df2 ([type]): [description]
        variable ([type]): [description]
        noperc ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
    """
    cdf1 = utils.ecdf(df1, variable, noperc)
    cdf2 = utils.ecdf(df2, variable, noperc)

    _, ax = handle_axis(ax)

    if not isinstance(label, str):
        label = labels(variable)

    ax.plot(cdf1, cdf2, marker="*", label=label)
    ax.axline([0, 0], [1, 1], color="red", lw=2)
    ax.set_xlabel("Quantiles \n Modeled " + labels(variable))
    ax.set_ylabel("Quantiles \n Observed " + labels(variable))
    ax.grid(True)

    if isinstance(title, str):
        ax.set_title(title, color="black", fontweight="bold")

    if legend:
        ax.legend()

    show(fname)
    return ax


def line_ci(ppfs, var_, keys=["mean", "std"], ax=None, fname=None, title=None):
    """[summary]

    Args:
        ppfs ([type]): [description]
        var_ ([type]): [description]
        keys (list, optional): [description]. Defaults to ['mean', 'std'].
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        title ([type], optional): [description]. Defaults to None.
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.set_prop_cycle(
        "color", [cmocean.cm.matter(i) for i in np.linspace(0, 1, len(ppfs.keys()))]
    )

    for ind_, key in enumerate(ppfs.keys()):
        if var_.lower().startswith("d"):
            ax.plot(np.rad2deg(ppfs[key][keys[0]]))
            ax.fill_between(
                ppfs[key].index,
                np.rad2deg(ppfs[key][keys[0]] - ppfs[key][keys[1]]),
                np.rad2deg(ppfs[key][keys[0]] + ppfs[key][keys[1]]),
                alpha=0.2,
            )
        else:
            ax.plot(ppfs[key][keys[0]])
            ax.fill_between(
                ppfs[key].index,
                ppfs[key][keys[0]] - ppfs[key][keys[1]],
                ppfs[key][keys[0]] + ppfs[key][keys[1]],
                alpha=0.2,
            )
    ax.set_xlabel("Normalized Year", fontweight="bold")
    # ax.set_ylabel(labels(var_))
    ax.grid(True)

    if isinstance(title, str):
        ax.set_title(title, color="black", fontweight="bold")

    show(fname)
    return


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = threshold
    else:
        threshold = np.max(
            [abs(np.percentile(data.data, 10)), np.percentile(data.data, 90)]
        )

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center", size=8)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(abs(data.data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def peaks_over_threshold(event, df, umb):
    """Plot pairwise scatter plots of variables with peaks over threshold.

    Displays two-by-two scatter plots of variables with annual extreme events
    and values exceeding specified thresholds highlighted.

    Args:
        event (pd.DataFrame): DataFrame with annual peaks of the main variable
            and values of accompanying variables.
        df (pd.DataFrame): DataFrame with all data for main and accompanying variables.
        umb (array-like): Series of threshold values to plot.

    Returns:
        matplotlib.figure.Figure: Figure containing the scatter plots.
    """

    plt.style.use('ggplot')
    df = df.dropna()
    naux, numb = np.shape(df)[1]-1, len(umb)
    ncol, nfil = 1, 1
    nven = ncol*nfil
    nomb = df.columns
    event, df = event.values, df.values
    while nven < naux:
        if ncol == nfil:
            ncol += 1
        else:
            nfil += 1
        nven = ncol*nfil

    col = ['k', 'b', 'g', 'purple', 'brown']
    fig = plt.figure()
    for i in range(1, naux+1):
        fig.add_subplot(nfil, ncol, i)
        plt.plot(df[:, 0], df[:, i], '.', color='gray')

        for j in range(0, numb):
            id_ = np.where(event[:, 0] > umb[j])[0]
            plt.plot(event[id_, 0], event[id_, i], '.', color=col[j], label='threshold '+str(umb[j]))

        plt.xlim(np.min(df[:, 0]), np.max(df[:, 0]))
        plt.ylim(np.min(df[:, i]), np.max(df[:, i]))
        plt.xlabel(nomb[0])
        plt.ylabel(nomb[i])


def threshold_fits(event, df, ajuste, umb, comb, fun, ejes):
    """Plot pairwise scatter plots with regression fits over threshold.

    Displays two-by-two scatter plots with fitted regression curves
    for values exceeding different thresholds, including confidence intervals.

    Args:
        event (list): List of values exceeding each threshold (length equals
            number of thresholds).
        df (pd.DataFrame): DataFrame containing all time series.
        ajuste (dict): Dictionary with fit values (x, upper_ci, mean, lower_ci)
            for each threshold. Keys follow format: 'x'+comb+fun+threshold,
            'y'+comb+fun+threshold, 'ysup'+comb+fun+threshold, 'yinf'+comb+fun+threshold.
            For directional variables (dh, dv), includes directional sector keys.
        umb (array-like): Threshold values to evaluate.
        comb (list): Variable combinations to plot.
        fun (list): Function names used for fitting.
        ejes (list): Axis labels for each variable.

    Returns:
        None: Displays the plots.

    Note:
        Directional variables (dh, dv) are handled specially with modulo 360
        for angular wrapping and may include multiple directional sectors.
    """

    plt.style.use('ggplot')
    numb = len(umb)
    nfun = len(fun)
    nombres = df.columns
    ncomb = len(comb)
    col = ['b', 'y', 'purple', 'brown']
    for k in range(ncomb):
        plt.figure(figsize=(3*nfun + 2, 3))
        plt.title(comb[k])
        for i in range(0, nfun):
            ax = plt.subplot(1, nfun, i+1)
            ax.set_title('Ajuste con '+fun[i])
            plt.plot(df[nombres[0]], df[nombres[k+1]], '.', color='gray', markersize=7)
            for j in range(numb):
                plt.plot(event[j][nombres[0]], event[j][nombres[k+1]], '.', color=col[j], markersize=7)

                if ((comb[k] == 'dh') | (comb[k] == 'dv')):

                    if ajuste['dirN' + comb[k - 1] + fun[i] + str(umb[j])][0]:
                        for kk in range(ajuste['ndirp' + comb[k] + fun[i] + str(umb[j])][0]):
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['y' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)] %360,
                                     color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['ysup' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)] %360, '.',
                                     markersize=0.5, color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['yinf' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)] %360, '.',
                                     markersize=0.5, color=col[j])
                    else:
                        for kk in range(ajuste['ndirp' + comb[k] + fun[i] + str(umb[j])][0]):
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['y' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['ysup' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)], '--',
                                     markersize=0.8, color=col[j])
                            plt.plot(ajuste['x' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)],
                                     ajuste['yinf' + comb[k] + fun[i] + str(umb[j]) + 'dir' + str(kk)], '--',
                                     markersize=0.8, color=col[j])

                    plt.xlim((0, np.max(ajuste['x' + comb[k] + fun[i] + str(umb[j])  + 'dir0'])))
                    plt.ylim((0, 360))

                else:
                    plt.plot(ajuste['x'+comb[k]+fun[i]+str(umb[j])], ajuste['y'+comb[k]+fun[i]+str(umb[j])], color=col[j])
                    plt.plot(ajuste['x'+comb[k]+fun[i]+str(umb[j])], ajuste['ysup'+comb[k]+fun[i]+str(umb[j])], '--',
                             color=col[j])
                    plt.plot(ajuste['x'+comb[k]+fun[i]+str(umb[j])], ajuste['yinf'+comb[k]+fun[i]+str(umb[j])], '--',
                             color=col[j])

                    plt.xlim((0, np.max(ajuste['x' + comb[k] + fun[i] + str(umb[j])])))
                    plt.ylim((0, 1.5 * np.max(df[nombres[k + 1]])))

            if i > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.ylabel(ejes[k+1])


            plt.xlabel(ejes[0])
        plt.gcf().subplots_adjust(bottom=0.2, left=0.15)


def dimensionless_fits(event, df, ajuste, fun, ejes):
    """Plot dimensionless scatter plots for wave parameters.

    Generates scatter plots for fitting Hs and Tp, limiting values according
    to fully developed sea conditions using dimensionless wave parameters.

    Args:
        event (list): List of values exceeding each threshold (length equals
            number of thresholds).
        df (pd.DataFrame): DataFrame with all time series including 'hs', 'tp', 'vv'.
        ajuste (dict): Dictionary with fit values (x, upper_ci, mean, lower_ci)
            for each threshold. Keys include:
            - 'x'+variable+function: x-values for the fit
            - 'y'+variable+function: mean fit values
            - 'ysup'+variable+function: upper confidence interval
            - 'yinf'+variable+function: lower confidence interval
            - 'hs_aux', 't_aux': Auxiliary curve for fully developed conditions
        fun: Function identifier used for fitting.
        ejes (list): Axis labels [x-label, y-label-tp, y-label-vv].

    Returns:
        None: Displays the plots.

    Note:
        Uses dimensionless parameters: Hs*g/(1.1*Vv²) and Tp*g/(1.1*Vv).
        The plot includes the theoretical curve for fully developed sea.
    """

    plt.style.use('ggplot')
    nombres = ['hs', 'tp', 'vv']
    nomb = ['tp', 'vv', 'adim']
    col = ['b', 'y', 'purple']
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.plot(df[nombres[0]], df[nombres[1]], '.', color='gray', markersize=7)
    plt.plot(event[0][nombres[0]], event[0][nombres[1]], '.', color=col[0], markersize=7)
    plt.plot(ajuste['x'+nomb[0]+str(fun)], ajuste['y'+nomb[0]+str(fun)], color=col[0])
    plt.plot(ajuste['x'+nomb[0]+str(fun)], ajuste['ysup'+nomb[0]+str(fun)], '--', color=col[0])
    plt.plot(ajuste['x'+nomb[0]+str(fun)], ajuste['yinf'+nomb[0]+str(fun)], '--', color=col[0])
    plt.plot(ajuste['hs_aux'], ajuste['t_aux'], 'k', lw=2)
    plt.xlabel(ejes[0])
    plt.ylabel(ejes[1])

    plt.subplot(1, 3, 2)
    plt.plot(df[nombres[0]]*9.81/(1.1*df[nombres[2]]**2), df[nombres[1]]*9.81/(1.1*df[nombres[2]]), '.', color='gray',
             markersize=7)
    plt.plot(event[0][nombres[0]]*9.81/(1.1*event[0][nombres[2]]**2), event[0][nombres[1]]*9.81/(
        1.1*event[0][nombres[2]]), '.', color=col[0], markersize=7)
    plt.plot(ajuste['x'+nomb[2]+str(fun)], ajuste['y'+nomb[2]+str(fun)], color='b')
    plt.plot(ajuste['x'+nomb[2]+str(fun)], ajuste['ysup'+nomb[2]+str(fun)], '--', color='b')
    plt.plot(ajuste['x'+nomb[2]+str(fun)], ajuste['yinf'+nomb[2]+str(fun)], '--', color='b')
    plt.plot(np.array([0.2541, 0.2541, 0]), np.array([0, 7.944, 7.944]), '--k')
    plt.xlim([0, 0.3])
    plt.ylim([0, 10])
    plt.xlabel(r'$\frac{H_sg}{v^2}$')
    plt.ylabel(r'$\frac{T_pg}{v}$')

    plt.subplot(1, 3, 3)
    plt.plot(df[nombres[0]], df[nombres[2]], '.', color='gray', markersize=7)
    plt.plot(event[0][nombres[0]], event[0][nombres[2]], '.', color=col[0], markersize=7)
    plt.plot(ajuste['x'+nomb[1]+str(fun)], ajuste['y'+nomb[1]+str(fun)], color='b')
    plt.plot(ajuste['x'+nomb[1]+str(fun)], ajuste['ysup'+nomb[1]+str(fun)], '--', color='b')
    plt.plot(ajuste['x'+nomb[1]+str(fun)], ajuste['yinf'+nomb[1]+str(fun)], '--', color='b')
    plt.xlabel(ejes[0])
    plt.ylabel(ejes[2])
    plt.tight_layout(pad=0.5)



def pot_lmom(
    info,
    nvar,
    file_name: str = None,
):
    """Plot Peaks Over Threshold analysis results using L-moments method.

    Generates a comprehensive visualization of POT analysis including parameter
    stability, goodness-of-fit statistics, and return period estimates across
    different threshold values.

    Args:
        info (dict): Dictionary containing POT analysis results with keys:
            - 'thresholds': Array of threshold values tested
            - 'mean_value_lmom': Mean parameter estimates (location, shape, scale)
            - 'upper_lim', 'lower_lim': Confidence interval bounds
            - 'au2_lmom': Anderson-Ruiz squared statistic values
            - 'au2pv_lmom': Complementary p-values (1 - p_value)
            - 'tr_eval': Return periods evaluated
            - 'nyears': Number of years in dataset
        nvar: Variable identifier (currently unused).
        file_name (str, optional): Path to save the figure. If None, displays
            interactively. Defaults to None.

    Returns:
        None: Displays or saves the plot.

    Note:
        The optimal threshold is automatically selected as the one minimizing
        the complementary p-value (au2pv_lmom).
    """
    selected_value = info["thresholds"][np.argmin(info["au2pv_lmom"])]

    xlim = (np.floor(np.min(info["thresholds"])), np.ceil(np.max(info["thresholds"])))
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(left=0.2, wspace=0.4)

    ax = []
    ax.append(plt.subplot2grid((3, 3), (0, 0)))
    ax.append(plt.subplot2grid((3, 3), (1, 0)))
    ax.append(plt.subplot2grid((3, 3), (2, 0)))

    # Location parameter plot using L-moments
    ax[0].set_title(r"\textbf{PARAMETERS USING L-MOMENTS}", fontsize=10)
    ax[0].plot(info["thresholds"], info["mean_value_lmom"][:, 3], "k-", lw=2)
    ax[0].plot(
        info["thresholds"],
        info["upper_lim"][:, 3],
        "k--",
        info["thresholds"],
        info["lower_lim"][:, 3],
        "k--",
    )
    ax[0].axvline(x=selected_value, color="r")
    ax[0].set_ylabel(r"\textbf{Location parameters ($\mathbf{\nu}$)}")
    ax[0].grid()
    ax[0].set_xlim(xlim)
    ax[0].get_xaxis().set_ticklabels([])
    ax[0].yaxis.set_label_coords(-0.2, 0.5)

    # Shape parameter plot
    ax[1].plot(info["thresholds"], info["mean_value_lmom"][:, 0], "k-", lw=2)
    ax[1].plot(
        info["thresholds"],
        info["upper_lim"][:, 0],
        "k--",
        info["thresholds"],
        info["lower_lim"][:, 0],
        "k--",
    )
    ax[1].axvline(x=selected_value, color="r")
    ax[1].set_ylabel(r"\textbf{Shape parameter (k)}")
    ax[1].grid()
    ax[1].set_xlim(xlim)
    ax[1].get_xaxis().set_ticklabels([])
    ax[1].yaxis.set_label_coords(-0.2, 0.5)

    # Modified scale parameter plot
    ax[2].plot(info["thresholds"], info["mean_value_lmom"][:, 4], "k-", lw=2)
    ax[2].plot(
        info["thresholds"],
        info["upper_lim"][:, 4],
        "k--",
        info["thresholds"],
        info["lower_lim"][:, 4],
        "k--",
    )
    ax[2].axvline(x=selected_value, color="r")
    ax[2].set_ylabel(r"\textbf{Modified scale parameter ($\mathbf{\sigma^*}$)}")
    ax[2].grid()
    ax[2].set_xlabel(r"\textbf{Threshold}")
    ax[2].set_xlim(xlim)
    ax[2].yaxis.set_label_coords(-0.2, 0.5)

    # Return period quantile plots
    for i in range(0, np.size(info["tr_eval"])):
        ax.append(plt.subplot2grid((3, 3), (0 + i, 1)))
        ax[3 + i].plot(
            info["thresholds"], info["mean_value_lmom"][:, 6 + i], "k-", lw=2
        )
        ax[3 + i].plot(
            info["thresholds"],
            info["upper_lim"][:, 6 + i],
            "k--",
            info["thresholds"],
            info["lower_lim"][:, 6 + i],
            "k--",
        )
        ax[3 + i].axvline(x=selected_value, color="r")
        ax[3 + i].set_ylabel(
            r"\textbf{Return period of " + str(int(info["tr_eval"][i])) + " yr}"
        )
        ax[3 + i].grid()
        ax[3 + i].set_xlim(xlim)
        ax[3 + i].set_ylim(
            (
                np.floor(np.min(info["lower_lim"][:, 6 + i])),
                np.ceil(np.max(info["upper_lim"][:, 6 + i])),
            )
        )
        if i+1 < np.size(info["tr_eval"]):
            ax[3 + i].get_xaxis().set_ticklabels([])

    ax[3].set_title(r"\textbf{VALUES FOR SEVERAL RETURN PERIODS}", fontsize=10)
    ind_ = 3 + np.size(info["tr_eval"])
    ax[ind_ - 1].set_xlabel(r"\textbf{Threshold}")

    # Goodness-of-fit statistics and p-value plots
    ax.append(plt.subplot2grid((3, 3), (0, 2), rowspan=2))
    ax[ind_].plot(info["thresholds"], info["au2_lmom"], "k-", lw=2)
    ax[ind_].axvline(x=selected_value, color="r")
    ax[ind_].grid()
    ax[ind_].set_ylabel(r"\textbf{$\mathbf{A_R^2}$ statistic}")

    ax.append(ax[ind_].twinx())
    ax[ind_ + 1].plot(info["thresholds"], info["au2pv_lmom"], "k--")
    ax[ind_ + 1].axvline(x=selected_value, color="r")
    ax[ind_ + 1].plot(
        selected_value,
        info["au2pv_lmom"][np.argmin(info["au2pv_lmom"])],
        "or",
        label="Selected",
    )
    ax[ind_ + 1].set_ylabel(r"\textbf{$\mathbf{1-p_{value}}$}")
    ax[ind_ + 1].grid()
    ax[ind_ + 1].set_xlim(xlim)
    ax[ind_ + 1].legend()

    # Annual events and total peaks plot
    ax.append(plt.subplot2grid((3, 3), (2, 2)))
    ax[ind_ + 2].plot(info["thresholds"], info["mean_value_lmom"][:, 5], "k-", lw=2)
    ax[ind_ + 2].axvline(x=selected_value, color="r")
    ax[ind_ + 2].set_ylabel(r"\textbf{No. of annual maxima}", fontweight="bold")
    ax[ind_ + 2].set_xlabel(r"\textbf{Threshold}", fontweight="bold")
    ax[ind_ + 2].grid()
    ax[ind_ + 2].set_xlim(xlim)

    ax.append(ax[ind_ + 2].twinx())
    nl = len(ax[ind_ + 2].get_yticks())
    nmax = np.ceil(
        np.ceil(np.max(info["mean_value_lmom"][:, 5])) * info["nyears"] / (nl - 1)
    )
    n2l = np.linspace(0, nmax * (nl - 1), nl)

    ax[ind_ + 3].set_ylabel(r"\textbf{No. of peaks}", fontweight="bold")
    ax[ind_ + 3].set_ylim(
        [0, np.ceil(np.max(info["mean_value_lmom"][:, 5])) * info["nyears"]]
    )
    ax[ind_ + 3].set_xlim(xlim)
    ax[ind_ + 3].set_yticks(n2l)
    show(file_name)
    return


def annual_maxima_analysis(
    boot,
    orig,
    ci,
    tr,
    peaks,
    npeaks,
    eventanu,
    func,
    flabel=r"H$_{m0}$ (m)",
    tr_plot=100,
):
    """Plot annual maxima extreme value analysis with bootstrap results.

    Generates a comprehensive visualization including parameter distributions,
    return period estimates with confidence intervals, and goodness-of-fit
    assessment for extreme value distributions.

    Args:
        boot (tuple): Bootstrap results tuple (boot_params, boot_alternative).
            boot_params: Array of bootstrap parameter estimates (n_boot x n_params).
            boot_alternative: Optional alternative distribution bootstrap results.
        orig (tuple): Original fitted parameters (orig_params, orig_alternative).
        ci (tuple): Confidence intervals (ci_params, ci_alternative).
            Format: [lower_bounds, upper_bounds].
        tr (array-like): Return periods for evaluation.
        peaks (array-like): Peak values extracted from the time series.
        npeaks (array-like): Empirical frequencies for plotting positions.
        eventanu (array-like): Annual event values.
        func (str): Distribution function name ('genpareto', 'genextreme', etc.).
        flabel (str, optional): Label for the variable with LaTeX formatting.
            Defaults to r"H$_{m0}$ (m)".
        tr_plot (int, optional): Specific return period to highlight in histogram.
            Defaults to 100 years.

    Returns:
        None: Displays the plot.

    Note:
        The function displays parameter stability through histograms and
        evaluates model fit using return period curves with confidence bands.
    """

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.2, wspace=0.4)

    # Shape parameter (k) histogram
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    n0, c0 = np.histogram(boot[0][0][:, 0], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )
    if boot[1]:
        plt.plot(np.array([0, 0]), ax1.get_ylim(), "r--")

    plt.plot(orig[0][0][0] * np.array([1, 1]), ax1.get_ylim(), "b--")
    plt.xlabel(" Shape (k)")
    plt.ylabel("Frequency")
    ax1.get_yaxis().set_ticks([])

    # Scale parameter (sigma) histogram
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    n0, c0 = np.histogram(boot[0][0][:, 2], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )

    if boot[1]:
        n1, c1 = np.histogram(boot[1][0][:, 2], bins=20)
        n1 = n1 / np.sum(n1) / (c1[1] - c1[0])
        plt.step(c1[:-1], n1, "r", where="post")
        plt.plot(orig[1][0][2] * np.array([1, 1]), ax2.get_ylim(), "r--")

    plt.plot(orig[0][0][2] * np.array([1, 1]), ax2.get_ylim(), "b--")
    plt.xlabel(r"Scale ($\sigma$)")
    plt.ylabel("Frequency")
    ax2.get_yaxis().set_ticks([])

    # Location parameter (mu) histogram
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    n0, c0 = np.histogram(boot[0][0][:, 1], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )
    if boot[1]:
        n1, c1 = np.histogram(boot[1][0][:, 1], bins=20)
        n1 = n1 / np.sum(n1) / (c1[1] - c1[0])
        plt.step(c1[:-1], n1, "r", where="post")
        plt.plot(orig[1][0][1] * np.array([1, 1]), ax3.get_ylim(), "r--")

    plt.plot(orig[0][0][1] * np.array([1, 1]), ax3.get_ylim(), "b--")
    plt.xlabel(r"Position ($\mu$)")
    plt.ylabel("Frequency")
    ax3.get_yaxis().set_ticks([])

    # Return period value histogram for specific Tr

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.68, 0.05),
        0.25,
        0.42,
        fill=False,
        color="r",
        lw=2,
        zorder=1000,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.extend([rect])

    ax4 = plt.subplot2grid((2, 3), (1, 2))

    tri = np.hstack((np.arange(1, 11), np.arange(2, 11) * 10, np.arange(2, 11) * 100))
    tri = np.hstack((np.arange(1.1, 2 + 1e-6, 0.1), tri[tri > 2]))
    idx = np.where(np.abs(tri - tr_plot) == np.min(np.abs(tri - tr_plot)))[0][0]

    n0, c0 = np.histogram(boot[0][0][:, idx + 3], bins=20)
    n0 = n0 / np.sum(n0) / (c0[1] - c0[0])
    plt.bar(
        c0[:-1] + (c0[1] - c0[0]) / 2,
        n0,
        width=c0[1] - c0[0],
        color="blue",
        lw=0,
        alpha=0.5,
    )
    plt.plot(orig[0][0][idx + 3] * np.array([1, 1]), ax4.get_ylim(), "b--")
    plt.xlabel(flabel + r"$_{Tr " + str(int(tri[idx])) + " yr}$")
    plt.ylabel("Frequency")
    ax4.get_yaxis().set_ticks([])

    plt.axvline(x=ci[0][0][1, idx + 3], color="gray")
    plt.axvline(x=ci[0][0][0, idx + 3], color="gray")

    if boot[1]:
        n1, c1 = np.histogram(boot[1][0][:, idx], bins=20)
        n1 = n1 / np.sum(n1) / (c1[1] - c1[0])
        plt.step(c1[:-1], n1, "r", where="post")
        plt.plot(orig[1][0][idx] * np.array([1, 1]), ax4.get_ylim(), "r--")

    # Return period vs. quantile plot
    ax5 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax5.fill(
        np.hstack((tr, tr[::-1])),
        np.hstack((ci[0][0][0, 4:], ci[0][0][1, ::-1][:-4])),
        color="gray",
        alpha=0.5,
        lw=0,
        label="confidence band",
    )
    if func == "genpareto":
        ax5.plot(
            1.0 / ((1.0 - np.arange(1.0, npeaks + 1.0) / (npeaks + 1.0)) * eventanu),
            np.sort(peaks),
            "g.",
            markersize=8,
            label="data",
        )
    else:
        ax5.plot(
            1.0 / npeaks[::-1], np.sort(eventanu), "g.", markersize=8, label="data"
        )

    ax5.plot(tr, orig[0][0][4:], "b--", label=func)
    ax5.axvline(x=tri[idx], color="r")

    if boot[1]:
        label = "gumbel_r"
        if func == "genpareto":
            label = "expon"
        ax5.plot(tr, orig[1][0][-len(tr) :], "r--", label=label)
        ax5.plot(tr, ci[1][0][0, -len(tr) :], "r-.")
        ax5.plot(tr, ci[1][0][1, -len(tr) :], "r-.")

    ax5.legend()

    ax5.set_xlim([1, 1000])
    ax5.set_xscale("log", nonposx="clip")
    ax5.set_ylabel(flabel)
    ax5.set_xlabel("Return period (yr)")
    return


def serie_peaks(df_serie, df_peaks, ylab, nombre):
    """Plot time series with identified peak values.

    Displays the complete time series and overlays the extracted peak events
    for visual verification of peak detection algorithms.

    Args:
        df_serie (pd.DataFrame): Complete time series data with datetime index.
        df_peaks (pd.DataFrame): Subset containing only peak values with datetime index.
        ylab (str): Y-axis label describing the variable.
        nombre (str): Plot title.

    Returns:
        None: Displays the plot.
    """

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(df_serie.index, df_serie.iloc[:, 0], color="b")
    plt.plot(df_peaks.index, df_peaks.iloc[:, 0], "ro", color="orange")
    plt.ylabel(ylab)
    plt.xticks(rotation=30)
    plt.legend(("Timeseries", "peaks"), loc="best")
    plt.title(nombre)


def serie_peaks_umbral(df_serie, df_peaks, ylab, umbral, nombre):
    """Plot time series with peaks over threshold.

    Displays the time series and highlights values exceeding a specified
    threshold for Peaks Over Threshold (POT) analysis.

    Args:
        df_serie (pd.DataFrame): Complete time series data with datetime index.
        df_peaks (pd.DataFrame): Values exceeding the threshold with datetime index.
        ylab (str): Y-axis label describing the variable.
        umbral (float): Threshold value used for peak extraction.
        nombre (str): Plot title.

    Returns:
        None: Displays the plot.
    """

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(df_serie.index, df_serie.iloc[:, 0], color="b")
    plt.plot(df_peaks.index, df_peaks.iloc[:, 0], "ro", color="orange")
    plt.ylabel(ylab)
    plt.xticks(rotation=30)
    plt.legend(
        ("Timeseries", "peaks (threshold = {:.2f}".format(umbral) + ")"), loc="best"
    )

    plt.title(nombre)
