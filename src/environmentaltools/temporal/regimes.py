from loguru import logger

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.optimize import fsolve
from scipy.special import gamma

from environmentaltools.common import utils


def _fit_lmom_for_bca(data, tr, func, nyears):
    """Internal helper function for BCa jackknife fitting.
    
    Fits distribution using L-moments for BCa acceleration factor computation.
    Used internally by confidence_intervals with resample='bca'.
    
    Parameters
    ----------
    data : array-like
        Jackknife resampled data (with one observation removed)
    tr : array-like
        Return periods to evaluate
    func : str or scipy.stats distribution
        Distribution type:
        - For L-MOM: 'genpareto', 'expon', 'genextreme', 'gumbel_r'
        - For MLE: scipy.stats distribution object
    nyears : int
        Number of years in dataset
        
    Returns
    -------
    np.ndarray
        Parameter estimates and return period values
        
    Notes
    -----
    This function attempts to match the fitting method and distribution type
    from the original analysis to ensure consistent jackknife estimates.
    """
    # Handle L-moments distributions
    if isinstance(func, str):
        if (func == "genpareto") or (func == "expon"):
            return lmom_genpareto(data, tr, nyears)
        elif (func == "genextreme") or (func == "gumbel_r"):
            params = l_mom(data, func)
            pri = 1 - 1.0 / tr
            qtr = st.genextreme.ppf(pri, *params)
            return np.hstack((*params, 1, qtr))
        else:
            # Try to use probability_model_fit for unknown string distributions
            try:
                return probability_model_fit(data, tr, "L-MOM", func, nyears)
            except:
                # Fallback: assume GPD
                return lmom_genpareto(data, tr, nyears)
    
    # Handle scipy.stats distribution objects (MLE case)
    else:
        try:
            return probability_model_fit(data, tr, "MLE", func)
        except:
            # If MLE fails, try fitting as GPD
            return lmom_genpareto(data, tr, nyears)


def confidence_intervals(boot, alpha, resample, *args):
    """Calculate confidence intervals using percentile or BCa bootstrap methods.
    
    Computes confidence intervals for bootstrap resampling results using either
    standard percentile method or bias-corrected and accelerated (BCa) method.

    Parameters
    ----------
    boot : np.ndarray
        Bootstrap resampling matrix with shape (n_simulations, n_parameters)
    alpha : float
        Significance level (e.g., 0.05 for 95% confidence intervals)
    resample : {'standard', 'bca'}
        Method for estimating confidence intervals:
        
        - 'standard': Simple percentile method
        - 'bca': Bias-corrected and accelerated bootstrap
        
    *args : tuple, optional
        Additional arguments required for BCa method:
        (orig, nosim, peaks, tri, tipo, nyears) where:
        
        - orig : array-like
            Original parameter estimates
        - nosim : int
            Number of bootstrap simulations
        - peaks : array-like
            Original peak time series
        - tri : array-like
            Return periods
        - tipo : str or scipy.stats distribution
            Distribution type
        - nyears : int
            Number of years in the dataset

    Returns
    -------
    np.ndarray
        Confidence interval bounds with shape (2, n_parameters):
        
        - ci[0, :] : lower bounds
        - ci[1, :] : upper bounds

    Notes
    -----
    The BCa method adjusts for bias and skewness in the bootstrap distribution
    using acceleration (a) and bias-correction (z0) factors. It provides more
    accurate intervals than standard percentiles, especially for small samples
    or skewed distributions.
    
    The acceleration factor is computed using jackknife influence values:
    a = sum(U^3) / [6 * sum(U^2)^(3/2)]
    
    where U are the jackknife deviations.

    References
    ----------
    Efron, B. (1987). "Better Bootstrap Confidence Intervals". 
    Journal of the American Statistical Association, 82(397), 171-185.

    Examples
    --------
    >>> boot = np.random.randn(1000, 3)  # Bootstrap samples
    >>> ci = confidence_intervals(boot, alpha=0.05, resample='standard')
    >>> print(f"95% CI: [{ci[0, 0]:.3f}, {ci[1, 0]:.3f}]")
    """

    if resample == "standard":
        ci = np.percentile(boot, np.array((alpha / 2, 1 - alpha / 2)) * 100, axis=0)
    elif resample == "bca":
        orig, nosim, peaks, tri, tipo, nyears = args[0]
        npeaks, norig = np.size(peaks), np.size(orig)
        pz0 = np.zeros(norig)
        for i in range(0, norig):
            mask = boot[:, i] < orig[i]
            pz0[i] = np.sum(mask) / float(nosim)
        z0 = st.norm.ppf(pz0)
        u = np.zeros([npeaks, norig])
        id_ = np.arange(npeaks)
        for i in range(0, npeaks):
            # Jackknife: fit distribution with i-th observation removed
            # Compute influence values for BCa acceleration factor
            u[i, :] = _fit_lmom_for_bca(peaks[id_ != i], tri, tipo, nyears) - orig
        a = np.sum(u ** 3) / np.sum(u ** 2) ** (3 / 2) / 6
        zalpha = st.norm.ppf((alpha / 2, 1 - alpha / 2))
        prob = np.zeros([2, norig])
        prob[0, :] = st.norm.cdf(z0 + (z0 + zalpha[0]) / (1 - a * (z0 + zalpha[0])))
        prob[1, :] = st.norm.cdf(z0 + (z0 + zalpha[1]) / (1 - a * (z0 + zalpha[1])))
        ci = np.zeros([2, norig])
        for i in range(0, norig):
            if not (np.isnan(prob[0, i]) | np.isnan(prob[1, i])):
                ci[:, i] = np.percentile(boot[:, i], prob[:, i] * 100, axis=0)

    return ci


def bootstrapping(peaks, tri, method, func, nyears, nosim, resample):
    """Perform bootstrap resampling for return period estimation.
    
    Computes distribution parameters and return period values using bootstrap
    resampling with parametric or non-parametric methods.

    Parameters
    ----------
    peaks : array-like
        Annual maxima or peak over threshold time series
    tri : array-like
        Return periods to evaluate (in years)
    method : {'MLE', 'L-MOM'}
        Fitting method for the probability distribution:
        
        - 'MLE': Maximum Likelihood Estimation
        - 'L-MOM': L-moments method
        
    func : str or scipy.stats distribution
        Probability distribution name or object
    nyears : int
        Number of years in the dataset (for POT analysis)
    nosim : int
        Number of bootstrap simulations
    resample : {'parametric', 'non-parametric'}
        Resampling method:
        
        - 'parametric': Resample from fitted distribution
        - 'non-parametric': Resample with replacement from original data

    Returns
    -------
    boot : np.ndarray
        Bootstrap results with shape (nosim, n_params + 1 + len(tri))
        Contains parameters, nu, and return period values for each simulation
    orig : np.ndarray
        Original parameter estimates from the data

    Notes
    -----
    Parametric bootstrap:
    1. Fit distribution to original data
    2. Generate synthetic data from fitted distribution
    3. Refit distribution to synthetic data
    4. Repeat nosim times
    
    Non-parametric bootstrap:
    1. Resample with replacement from original data
    2. Fit distribution to resampled data
    3. Repeat nosim times
    
    The function handles special cases for GPD and GEV distributions when
    using L-moments.

    Examples
    --------
    >>> import numpy as np
    >>> peaks = np.random.weibull(1.5, 50)
    >>> tri = np.array([10, 50, 100])
    >>> boot, orig = bootstrapping(
    ...     peaks, tri, 'L-MOM', 'genextreme', 50, 1000, 'parametric'
    ... )
    """

    npeaks = np.size(peaks)
    orig = probability_model_fit(peaks, tri, method, func, nyears)
    boot = np.zeros([nosim, np.size(orig)])
    if resample == "parametric":
        if method == "MLE":
            params = orig[: func.numargs + 2]
            logger.info(
                "Computing bootstrapping of {} simulations using MLE and parametric methods for {} probability model. It will take a while.".format(
                    str(nosim), func.name
                )
            )
            for j in range(0, nosim):
                boot[j, :] = probability_model_fit(
                    func.rvs(*params, npeaks), tri, method, func
                )
        elif method == "L-MOM":
            logger.info(
                "Computing bootstrapping of {} simulations using L-MOM and parametric methods for {} probability model.".format(
                    str(nosim), func
                )
            )
            if (func == "expon") | (func == "genpareto"):
                for j in range(0, nosim):
                    boot[j, :] = probability_model_fit(
                        st.genpareto.rvs(orig[0], orig[1], orig[2], npeaks),
                        tri,
                        method,
                        func,
                        nyears,
                    )
            elif (func == "genextreme") | (func == "gumbel_r"):
                for j in range(0, nosim):
                    boot[j, :] = probability_model_fit(
                        st.genextreme.rvs(orig[0], orig[1], orig[2], npeaks),
                        tri,
                        method,
                        func,
                    )
    elif resample == "non-parametric":
        for j in range(0, nosim):
            logger.info(
                "Computing bootstrapping of {} simulations using {} and non-parametric methods for {} probability model. It will take a while.".format(
                    str(nosim), method, func.name
                )
            )
            boot[j, :] = probability_model_fit(
                np.random.choice(peaks, npeaks), tri, method, func, nyears
            )

    return boot, orig


def probability_model_fit(data, tr, method, func, *nyears):
    """Estimate probability distribution parameters and return period values.
    
    Fits a probability distribution to data using specified method and
    computes return period values for given return periods.

    Parameters
    ----------
    data : array-like
        Time series data (annual maxima or peaks over threshold)
    tr : array-like
        Return periods in years (e.g., [10, 50, 100, 1000])
    method : {'MLE', 'L-MOM'}
        Parameter estimation method:
        
        - 'MLE': Maximum Likelihood Estimation (uses scipy.stats.fit)
        - 'L-MOM': L-moments method (custom implementation)
        
    func : str or scipy.stats distribution
        Probability distribution:
        
        - For MLE: scipy.stats distribution object
        - For L-MOM: string in ['expon', 'genpareto', 'genextreme', 'gumbel_r']
        
    *nyears : int, optional
        Number of years in dataset (required for POT analysis with GPD)

    Returns
    -------
    np.ndarray
        Array containing [shape, loc, scale, nu, qtr_1, qtr_2, ...] where:
        
        - shape, loc, scale : distribution parameters
        - nu : average number of events per year
        - qtr_i : return level for return period tr[i]

    Notes
    -----
    For annual maxima (GEV, Gumbel):
    - nu = 1 (one event per year)
    - Return levels: x_T = F^(-1)(1 - 1/T)
    
    For peaks over threshold (GPD):
    - nu = n_peaks / n_years
    - Return levels: x_T = F^(-1)(1 - 1/(T*nu))
    
    L-moments method provides more robust parameter estimates for small
    samples compared to MLE.

    See Also
    --------
    l_mom : L-moments parameter estimation
    bootstrapping : Bootstrap resampling for uncertainty estimation

    Examples
    --------
    >>> import scipy.stats as st
    >>> data = st.genextreme.rvs(0.1, loc=2, scale=0.5, size=50)
    >>> tr = np.array([10, 50, 100])
    >>> params = probability_model_fit(data, tr, 'MLE', st.genextreme)
    >>> print(f"Shape: {params[0]:.3f}, Loc: {params[1]:.3f}, Scale: {params[2]:.3f}")
    """

    if method == "MLE":
        params = func.fit(data)
        pri = 1 - 1.0 / tr
        qtr = func.ppf(pri, *params)
        nu = 1

    elif method == "L-MOM":
        params = l_mom(data, func)
        if (func == "expon") | (func == "genpareto"):
            nu = np.size(data) / nyears[0]
            pri = 1 - 1 / (tr * nu)
            qtr = st.genpareto.ppf(pri, *params)
        elif (func == "genextreme") | (func == "gumbel_r"):
            pri = 1 - 1.0 / tr
            qtr = st.genextreme.ppf(pri, *params)
            nu = 1

    return np.hstack((*params, nu, qtr))


def l_mom(data, func):
    """Estimate distribution parameters using L-moments method.
    
    Computes distribution parameters from sample L-moments (linear combinations
    of order statistics). Provides robust estimates especially for small samples.

    Parameters
    ----------
    data : array-like
        Sample data to fit
    func : {'genpareto', 'expon', 'genextreme', 'gumbel_r'}
        Distribution type to fit:
        
        - 'genpareto': Generalized Pareto Distribution (3 parameters)
        - 'expon': Exponential distribution (2 parameters, k=0)
        - 'genextreme': Generalized Extreme Value distribution (3 parameters)
        - 'gumbel_r': Gumbel distribution (2 parameters, k=0)

    Returns
    -------
    list
        Distribution parameters [k, mu, sig] where:
        
        - k : float
            Shape parameter (0 for Exponential and Gumbel)
        - mu : float
            Location parameter
        - sig : float
            Scale parameter

    Notes
    -----
    L-moments are linear combinations of order statistics:
    
    - λ₁ = E[X] (mean)
    - λ₂ = E[X_{2:2} - X_{1:2}]/2 (scale)
    - λ₃ related to skewness
    
    L-moment ratios (τ₃ = λ₃/λ₂) are used to estimate shape parameters.
    
    For GEV, the shape parameter k is solved numerically from:
    2(1-3^(-k))/(1-2^(-k)) - 3 = τ₃
    
    Sample L-moments use plotting positions: p_i = (i-0.35)/n

    References
    ----------
    Hosking, J.R.M. (1990). "L-moments: Analysis and Estimation of 
    Distributions Using Linear Combinations of Order Statistics". 
    Journal of the Royal Statistical Society, Series B, 52(1), 105-124.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.exponential(2.0, 100)
    >>> k, mu, sig = l_mom(data, 'expon')
    >>> print(f"Exponential: k={k:.3f}, mu={mu:.3f}, sig={sig:.3f}")
    """

    data = np.sort(data)
    n = np.size(data)
    p = (np.arange(1, n + 1) - 0.35) / n
    lam1, lam2 = np.mean(data), np.mean((2 * p - 1) * data)
    if func == "genpareto":
        lam3 = np.mean((1 - 6 * p + 6 * p ** 2) * data)
        tau3 = lam3 / lam2
        k = (3 * tau3 - 1) / (1 + tau3)
        sig = lam2 * (1 - k) * (2 - k)
        mu = lam1 - sig / (1 - k)
    elif func == "expon":
        sig = 2 * lam2
        mu = lam1 - sig
        k = 0
    elif func == "genextreme":
        lam3 = np.mean((1 - 6 * p + 6 * p ** 2) * data)
        tau3 = lam3 / lam2

        def fun(x, tau3):
            return 2.0 * (1 - 3.0 ** -x) / (1 - 2.0 ** -x) - 3 - tau3

        c = 2 / (3 + tau3) - np.log(2) / np.log(3)
        k = 7.859 * c + 2.9554 * c ** 2
        k = fsolve(fun, k, args=(tau3))
        sig = lam2 * k / (1 - 2 ** (-k)) / gamma(1 + k)
        mu = lam1 - sig * (1 - gamma(1 + k)) / k
    elif func == "gumbel_r":
        sig = lam2 / np.log(2)
        mu = lam1 - 0.5772 * sig
        k = 0

    return [k, mu, sig]


def lmom_genpareto(data, tr, nyears):
    """Fit Generalized Pareto Distribution using L-moments with diagnostic metrics.
    
    Estimates GPD parameters via L-moments and computes return period values
    along with additional diagnostic statistics for POT analysis.

    Parameters
    ----------
    data : array-like
        Peak over threshold data (exceedances)
    tr : array-like
        Return periods in years to evaluate
    nyears : int
        Number of years in the dataset

    Returns
    -------
    np.ndarray
        Array containing [k, mu, sig, mrlp, sigmodif, nu, qtr_1, qtr_2, ...]
        
        - k : float
            Shape parameter
        - mu : float
            Location parameter (threshold)
        - sig : float
            Scale parameter
        - mrlp : float
            Mean residual life plot value: E[X - mu | X > mu]
        - sigmodif : float
            Modified scale: sig - k*mu
        - nu : float
            Average number of exceedances per year
        - qtr_i : float
            Return level for return period tr[i]

    Notes
    -----
    For peaks over threshold analysis with GPD:
    
    - Return levels: x_T = μ + (σ/ξ)[(Tλ)^ξ - 1]
    - Mean excess: E[X-u|X>u] = (σ + ξu)/(1-ξ)
    
    The modified scale parameter is used in some POT formulations.

    Examples
    --------
    >>> data = np.random.exponential(2.0, 100)
    >>> tr = np.array([10, 50, 100])
    >>> results = lmom_genpareto(data, tr, nyears=10)
    >>> k, mu, sig = results[0], results[1], results[2]
    """

    k, mu, sig = l_mom(data, "genpareto")
    mrlp = np.mean(data - mu)
    sigmodif = sig - k * mu
    nu = np.size(data) / nyears
    pri = 1 - 1 / (tr * nu)
    qtr = st.genpareto.ppf(pri, k, mu, sig)
    return np.hstack((k, mu, sig, mrlp, sigmodif, nu, qtr))


def pot_method(
    df: pd.DataFrame,
    var_: str,
    window_size: int,
    alpha: float = 0.05,
    sim_no: int = 10000,
    method: str ="nearest"
):
    """Perform Peaks Over Threshold (POT) analysis with multiple thresholds.
    
    Analyzes extreme values using Generalized Pareto Distribution fitted with
    L-moments. Tests multiple threshold values and provides bootstrap confidence
    intervals for return period estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with datetime index
    var_ : str
        Name of the variable column to analyze
    window_size : int
        Window size for extracting independent maxima events
    alpha : float, default=0.05
        Significance level for confidence intervals (0.05 gives 95% CI)
    sim_no : int, default=10000
        Number of bootstrap simulations
    method : str, default='nearest'
        Interpolation method for p-value table (currently commented out)

    Returns
    -------
    dict
        Results dictionary with keys:
        
        - 'mean_value_lmom' : np.ndarray
            Shape (n_thresholds, n_params+6+len(tr_eval))
            Mean parameter estimates and return levels for each threshold
        - 'upper_lim' : np.ndarray
            Upper confidence interval bounds
        - 'lower_lim' : np.ndarray
            Lower confidence interval bounds
        - 'au2_lmom' : np.ndarray
            Anderson-Darling test statistics (currently zeros)
        - 'au2pv_lmom' : np.ndarray
            Anderson-Darling p-values (currently zeros)
        - 'nyears' : int
            Number of years in the dataset
        - 'thresholds' : np.ndarray
            Threshold values tested (90th to 99.9th percentiles)
        - 'tr_eval' : np.ndarray
            Return periods evaluated [1, 50, 100, 1000] years

    Notes
    -----
    The POT method:
    
    1. Extracts independent events using moving window maxima
    2. Tests thresholds from 90th to 99.9th percentiles
    3. For each threshold:
       - Fits GPD using L-moments
       - Computes bootstrap confidence intervals
       - (Optionally) performs Anderson-Darling goodness-of-fit test
    
    Higher thresholds provide better asymptotic approximation but fewer data
    points. The optimal threshold balances bias-variance tradeoff.

    References
    ----------
    Coles, S. (2001). "An Introduction to Statistical Modeling of 
    Extreme Values". Springer.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Hs': np.random.weibull(1.5, 1000)
    ... }, index=pd.date_range('2000', periods=1000, freq='D'))
    >>> results = pot_method(df, 'Hs', window_size=48, alpha=0.05)
    >>> print(f"Number of thresholds tested: {len(results['thresholds'])}")
    """
    # Initialize evaluation return periods
    tr_eval = np.array([1,50, 100, 1000])

    thresholds = np.percentile(
        df[var_], np.hstack([np.linspace(90, 99, 10), np.linspace(99.1, 99.9, 9)])
    )

    no_thres = np.size(thresholds)

    results = {}
    results["mean_value_lmom"], results["upper_lim"], results["lower_lim"], results["au2_lmom"], results["au2pv_lmom"] = (
        np.zeros([no_thres, np.size(tr_eval) + 6]),
        np.zeros([no_thres, np.size(tr_eval) + 6]),
        np.zeros([no_thres, np.size(tr_eval) + 6]),
        np.zeros(no_thres),
        np.zeros(no_thres),
    )

    # Load p-value table for Anderson-Darling test and create the interpolator
    # data = loadmat(
    #     os.path.join(os.path.dirname(__file__), "../", "utils/misc/PVAL_AU2_LMOM.mat")
    # )

    # pvallmom, au2val, ko, ndata = (
    #     data["PVAL"],
    #     data["AU2VAL"][0],
    #     data["Ko"][0],
    #     data["NDATA"][0],
    # )

    # interpolator = RegularGridInterpolator(
    #     (ndata, ko, au2val),
    #     pvallmom,
    #     method=method,
    #     bounds_error=False,
    #     fill_value=None,
    # )

    # Compute the POT analysis
    logger.info(
        "Computing POT analysis for a window size of "
        + str(df.index[window_size] - df.index[0])
    )
    df_max_events = utils.max_moving(df[var_], window_size)
    event = df_max_events.loc[
        df_max_events[var_] > np.percentile(df[var_], 90), var_
    ].values[:]
    nyears = df.index.year.max() - df.index.year.min()

    # Perform analysis for each threshold
    for i in range(no_thres):
        logger.info(
            "Threshold "
            + str(i + 1)
            + " of "
            + str(no_thres)
            + " ("
            + str(np.round(thresholds[i], decimals=3))
            + ")."
        )
        events_for_threshold_i = event[event > thresholds[i]]
        n_evi = np.size(events_for_threshold_i)
        results["mean_value_lmom"][i, :] = lmom_genpareto(events_for_threshold_i, tr_eval, nyears)

        # Bootstrap resampling using L-moments for Generalized Pareto Distribution
        boot = np.zeros([sim_no, np.size(tr_eval) + 6])

        for j in range(0, sim_no):
            boot[j, :] = lmom_genpareto(
                np.random.choice(events_for_threshold_i, n_evi), tr_eval, nyears
            )

        results["upper_lim"][i, :] = np.percentile(boot, (1 - alpha / 2) * 100, axis=0)
        results["lower_lim"][i, :] = np.percentile(boot, (alpha / 2) * 100, axis=0)

        # Anderson-Darling Upper test without confidence intervals for L-moments
        # results["au2_lmom"][i] = au2(results["mean_value_lmom"][i, 0:3], events_for_threshold_i)
        # results["au2pv_lmom"][i] = interpolator(
        #     [
        #         np.max((10, np.min((500, n_evi)))),
        #         np.sign(results["mean_value_lmom"][i, 0])
        #         * np.min((np.abs(results["mean_value_lmom"][i, 0]), 0.5)),
        #         results["au2_lmom"][i],
        #     ]
        # )
    
    results["nyears"] = nyears
    results["thresholds"] = thresholds
    results["tr_eval"] = tr_eval

    return results


def au2(par, data):
    """Calculate Anderson-Darling test statistic for GPD goodness-of-fit.
    
    Computes the Anderson-Darling A² statistic to assess how well a 
    Generalized Pareto Distribution fits the data. Lower values indicate
    better fit.

    Parameters
    ----------
    par : array-like
        GPD parameters [shape, loc, scale]
    data : array-like
        Peak over threshold data to test

    Returns
    -------
    float
        Anderson-Darling A² test statistic

    Notes
    -----
    The Anderson-Darling statistic is:
    
    A² = n/2 - 2Σ[F(x_i)] - Σ[(2i-1)/n * log(1-F(x_i))]
    
    where F is the fitted CDF and x_i are the sorted data.
    
    This statistic gives more weight to tail deviations than KS test,
    making it particularly suitable for extreme value analysis.
    
    The CDF values are adjusted by 1e-6 to avoid log(0) issues.

    References
    ----------
    Anderson, T.W. and Darling, D.A. (1952). "Asymptotic theory of 
    certain goodness of fit criteria based on stochastic processes". 
    Annals of Mathematical Statistics, 23, 193-212.

    Examples
    --------
    >>> data = np.random.exponential(2.0, 100)
    >>> par = [0, 0, 2.0]  # Exponential is GPD with shape=0
    >>> ad_stat = au2(par, data)
    >>> print(f"Anderson-Darling statistic: {ad_stat:.3f}")
    """

    ndat = np.size(data)
    cdf = np.sort(st.genpareto.cdf(data, *par))
    cdf -= 1e-6  # Avoid log(0) when CDF=1.0
    estadist = (
        ndat / 2
        - 2 * np.sum(cdf)
        - np.sum((2 - (2 * np.arange(1, ndat + 1) - 1.0) / ndat) * np.log(1 - cdf))
    )

    return estadist


def fit_gpd_bootstrap(df_eventos, alpha, umb, param, bca, nyears):
    """Compute GPD parameters with bootstrap resampling and confidence intervals.
    
    Automatically fits Generalized Pareto Distribution using L-moments with
    bootstrap resampling for specified thresholds. Also tests if Exponential
    distribution is appropriate when shape parameter CI includes zero.

    Parameters
    ----------
    df_eventos : pd.DataFrame or np.ndarray
        Event time series data
    alpha : float
        Significance level for confidence intervals
    umb : array-like
        Threshold values to test
    param : {'parametric', 'non-parametric'}
        Bootstrap resampling method
    bca : {'standard', 'bca'}
        Confidence interval computation method
    nyears : int
        Number of years in the dataset

    Returns
    -------
    boot : list of lists
        Bootstrap results [GPD_results, Exponential_results (if applicable)]
        Each contains bootstrap parameter matrices for each threshold
    orig : list of lists
        Original parameter estimates [GPD_estimates, Exponential_estimates]
    ci : list of lists
        Confidence intervals [GPD_CI, Exponential_CI]
    tr : np.ndarray
        Return periods evaluated
    peaks : np.ndarray
        Peak values used (for last threshold processed)
    npeaks : int
        Number of peaks (for last threshold)
    eventanu : float
        Average events per year (for last threshold)

    Notes
    -----
    For each threshold:
    
    1. Extracts peaks exceeding threshold
    2. Fits GPD using L-moments with bootstrap
    3. Computes confidence intervals
    4. If CI for shape includes 0, also fits Exponential distribution
    
    Exponential is a special case of GPD with shape=0.

    Examples
    --------
    >>> umb = np.array([2.0, 2.5])
    >>> boot, orig, ci, tr, peaks, npeaks, eventanu = fit_gpd_bootstrap(
    ...     df['Hs'], alpha=0.05, umb=umb, param='parametric',
    ...     bca='standard', nyears=10
    ... )
    """
    tr = np.hstack((np.arange(1, 11), np.arange(2, 11) * 10, np.arange(2, 11) * 100))
    nosim = 10000

    # Extract peaks over threshold with PWM and bootstrap confidence intervals
    # Identify events
    numb = np.size(umb)
    event = df_eventos.values[:]

    boot, orig, ci = (
        [list() for i in range(2)],
        [list() for i in range(2)],
        [list() for i in range(2)],
    )
    for i in range(numb):
        peaks = event[event > umb[i]]
        npeaks = np.size(peaks)
        eventanu = npeaks / nyears

        # Central value and bootstrap with predefined threshold
        boot_s, orig_s = bootstrapping(peaks, tr, "L-MOM", "genpareto", nyears, nosim, param)
        ci_s = confidence_intervals(
            boot_s, alpha, bca, (orig_s, nosim, peaks, tr, "genpareto", nyears)
        )
        boot[0].append(boot_s), orig[0].append(orig_s), ci[0].append(ci_s)

        # If Exponential is appropriate (shape CI includes 0), compute it
        expo = (ci_s[0, 0] < 0) & (ci_s[1, 0] > 0)
        if expo:
            boot_e, orig_e = bootstrapping(peaks, tr, "L-MOM", "expon", nyears, nosim, param)
            ci_e = confidence_intervals(
                boot_e, alpha, bca, (orig_e, nosim, peaks, tr, "expon", nyears)
            )
            boot[1].append(boot_e), orig[1].append(orig_e), ci[1].append(ci_e)

    return boot, orig, ci, tr, peaks, npeaks, eventanu


def annual_maxima_method(df, alpha, method, func, resample, ci_method, tr = None):
    """Fit GEV or Gumbel distribution to annual maxima with bootstrap CI.
    
    Performs extreme value analysis using annual maxima approach. Fits
    Generalized Extreme Value (GEV) or Gumbel distribution with bootstrap
    resampling to estimate confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with datetime index
    alpha : float
        Significance level for confidence intervals (e.g., 0.05 for 95% CI)
    method : {'MLE', 'L-MOM'}
        Parameter estimation method:
        
        - 'MLE': Maximum Likelihood Estimation
        - 'L-MOM': L-moments method
        
    func : str or scipy.stats distribution
        Distribution to fit:
        
        - For L-MOM: 'genextreme', 'gumbel_r'
        - For MLE: scipy.stats distribution object (e.g., st.genextreme)
        
    resample : {'parametric', 'non-parametric'}
        Bootstrap resampling method
    ci_method : {'standard', 'bca'}
        Confidence interval computation method:
        
        - 'standard': Percentile method
        - 'bca': Bias-corrected and accelerated
        
    tr : array-like, optional
        Return periods to evaluate (in years). If None, uses default range
        from 1.1 to 1000 years

    Returns
    -------
    tr : np.ndarray
        Return periods evaluated
    pannumax : np.ndarray
        Plotting positions for annual maxima
    annumax : pd.Series
        Annual maxima values
    boot : list of lists
        Bootstrap results [GEV_results, Gumbel_results (if applicable)]
    orig : list of lists
        Original parameter estimates
    ci : list of lists
        Confidence intervals

    Notes
    -----
    The annual maxima method:
    
    1. Extracts one maximum value per year
    2. Fits GEV distribution: F(x) = exp{-[1+ξ(x-μ)/σ]^(-1/ξ)}
    3. If shape parameter CI includes 0, also fits Gumbel (ξ=0)
    
    Return levels are computed as: x_T = F^(-1)(1 - 1/T)
    
    Gumbel is a special case of GEV with shape parameter ξ=0.

    Raises
    ------
    ValueError
        If invalid method, function, resample, or ci_method specified

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Hs': np.random.weibull(1.5, 3650)
    ... }, index=pd.date_range('2010', periods=3650, freq='D'))
    >>> tr, p, maxima, boot, orig, ci = annual_maxima_method(
    ...     df, alpha=0.05, method='L-MOM', func='genextreme',
    ...     resample='parametric', ci_method='standard'
    ... )
    """

    if not ((method == "MLE") | (method == "L-MOM")):
        raise ValueError(
            'Fitting methods for probability models are "MLE" or "L-MOM". Given {}.'.format(
                method
            )
        )

    if method == "L-MOM":
        if not func in ["genpareto", "genextreme", "gumbel_r", "expon"]:
            raise ValueError(
                'Function {} is not included in L-MOM methods for fitting. Use "genpareto", "genextreme", "gumbel_r" or "expon".'.format(
                    method
                )
            )
    else:
        try:
            func = getattr(st, func)
        except:
            raise ValueError("Function {} is not included scipy.stats.".format(func))

    if not ((resample == "parametric") | (resample == "non-parametric")):
        raise ValueError(
            'Resampling methods are "parametric" or "non-parametric". Given {}'.format(
                resample
            )
        )

    if not ((ci_method == "standard") | (ci_method == "bca")):
        raise ValueError(
            'Confidence interval methods are "standard" or "bca". Given {}'.format(
                ci_method
            )
        )

    if tr is None:
        tr = np.hstack((np.arange(1, 11), np.arange(2, 11) * 10, np.arange(2, 11) * 100))
        tr = np.hstack((np.arange(1.1, 2 + 1e-6, 0.1), tr[tr > 2]))
    annumax = df.groupby(df.index.year).max()
    nyears = len(annumax)
    pannumax = np.arange(1.0, nyears + 1.0) / (nyears + 1.0)

    nosim = 10
    boot, orig, ci = (
        [list() for i in range(2)],
        [list() for i in range(2)],
        [list() for i in range(2)],
    )
    boot_a, orig_a = bootstrapping(annumax, tr, method, func, nyears, nosim, resample)
    ci_a = confidence_intervals(
        boot_a, alpha, ci_method, (orig_a, nosim, annumax, tr, func, nyears)
    )
    boot[0].append(boot_a), orig[0].append(orig_a), ci[0].append(ci_a)

    # Whether the parameters indicate that gumbel_r can be estimated, repeat the method with gumbel_r probability model
    if (ci_a[0, 0] < 0) & (ci_a[1, 0] > 0):
        if method == "MLE":
            func = st.gumbel_r
        else:
            func = "gumbel_r"
        boot_g, orig_g = bootstrapping(
            annumax, tr, method, func, nyears, nosim, resample
        )
        ci_g = confidence_intervals(
            boot_g, alpha, ci_method, (orig_g, nosim, annumax, tr, func, nyears)
        )
        boot[1].append(boot_g), orig[1].append(orig_g), ci[1].append(ci_g)

    return tr, pannumax, annumax, boot, orig, ci
