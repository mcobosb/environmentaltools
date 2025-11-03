import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from scipy.stats import kendalltau, pearsonr, scoreatpercentile, spearmanr
from sklearn.neighbors import KernelDensity


class Copula:
    """Copula-based bivariate dependence modeling.
    
    Estimates copula parameters for joint random variables and generates
    correlated samples. Supports Clayton, Frank, and Gumbel copula families.

    Parameters
    ----------
    X : np.ndarray
        Data of the first variable (1-dimensional)
    Y : np.ndarray
        Data of the second variable (1-dimensional, same size as X)
    family : str
        Copula family: 'clayton', 'frank', or 'gumbel'
    *args : tuple, optional
        Theoretical marginal distributions: (F1, p1, F2, p2) where
        F1, F2 are scipy.stats distributions and p1, p2 are parameter tuples

    Attributes
    ----------
    X : np.ndarray
        First variable data
    Y : np.ndarray
        Second variable data
    family : str
        Selected copula family
    theta : float
        Estimated copula parameter
    tau : float
        Kendall's tau correlation coefficient
    pr : tuple
        Pearson correlation coefficient and p-value
    sr : tuple
        Spearman correlation coefficient and p-value
    U : np.ndarray or None
        Generated uniform samples for first variable
    V : np.ndarray or None
        Generated uniform samples for second variable
    X1 : np.ndarray
        Generated samples in original scale for first variable
    Y1 : np.ndarray
        Generated samples in original scale for second variable

    Notes
    -----
    Based on the ambhas 0.4.0 module by Sat Kumar Tomer.
    Original: http://civil.iisc.ernet.in/~satkumar/
    
    Modifications include:
    - Conditional copulas for all families
    - Replaced statistics package with sklearn.neighbors.KernelDensity
    
    The copula parameter theta is estimated from Kendall's tau:
    - Clayton: theta = 2*tau / (1 - tau)
    - Frank: theta optimized via Debye function
    - Gumbel: theta = 1 / (1 - tau)

    Examples
    --------
    >>> from scipy.stats import norm, lognorm
    >>> x = np.random.normal(0, 1, 1000)
    >>> y = 2*x + np.random.normal(0, 0.5, 1000)
    >>> cop = Copula(x, y, 'clayton')
    >>> cop.generate_xy(500)
    >>> print(cop.theta, cop.tau)
    
    References
    ----------
    Nelsen, R.B. (2006). An Introduction to Copulas. Springer.
    
    Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman & Hall.
    """

    def __init__(self, X, Y, family, *args):
        """Initialize the Copula with data and family selection.

        Parameters
        ----------
        X : np.ndarray
            Data of the first variable (must be 1-dimensional)
        Y : np.ndarray
            Data of the second variable (must be 1-dimensional, same size as X)
        family : str
            Copula family: 'clayton', 'frank', or 'gumbel'
        *args : tuple, optional
            Theoretical marginal distributions as (F1, p1, F2, p2)

        Raises
        ------
        ValueError
            If X or Y are not 1-dimensional
            If X and Y have different sizes
            If family is not 'clayton', 'frank', or 'gumbel'

        Notes
        -----
        The initialization:
        1. Validates input dimensions and sizes
        2. Estimates Kendall's tau, Pearson's r, and Spearman's rho
        3. Computes copula parameter theta from tau
        4. Initializes U and V to None (generated later)
        """
        # check dimensions
        if not ((X.ndim == 1) and (Y.ndim == 1)):
            raise ValueError("The dimension should be one.")

        # check that both vectors have the same length
        if X.size != Y.size:
            raise ValueError("The size of both vectors must be the same.")

        # check that the family is correct
        copula_family = ["clayton", "frank", "gumbel"]
        if family not in copula_family:
            raise ValueError("Copulas available are clayton, frank or gumbel.")

        self.X = X
        self.Y = Y
        self.family = family

        # if theoretical marginals are provided
        if args:
            self.F1, self.F2 = args[0][0], args[0][2]
            self.p1, self.p2 = args[0][1], args[0][3]

        # estimate Kendall's correlation coefficient
        tau = kendalltau(self.X, self.Y)[0]
        self.tau = tau

        # estimate Pearson's R and Spearman's R
        self.pr = pearsonr(self.X, self.Y)
        self.sr = spearmanr(self.X, self.Y)

        # estimate the copula parameter
        self._get_parameter()

        # set U and V to None
        self.U = None
        self.V = None

    def _get_parameter(self):
        """Estimate the copula parameter theta.
        
        Computes theta from Kendall's tau using family-specific relationships:
        - Clayton: theta = 2*tau / (1 - tau)
        - Frank: theta optimized via Debye function minimization
        - Gumbel: theta = 1 / (1 - tau)

        Notes
        -----
        For Frank copula, uses numerical optimization (fmin) to solve:
        tau = 1 - 4/theta * [D_1(theta) - 1]
        where D_1 is the first-order Debye function.
        """

        if self.family == "clayton":
            self.theta = 2 * self.tau / (1 - self.tau)

        elif self.family == "frank":
            self.theta = -fmin(self._frank_fun, -5, disp=False)[0]

        elif self.family == "gumbel":
            self.theta = 1 / (1 - self.tau)

    def generate_uv(self, n=1000):
        """Generate random uniform variables (U, V) from the copula.

        Parameters
        ----------
        n : int, optional
            Number of copula samples to generate. Default is 1000.

        Raises
        ------
        ValueError
            For Clayton: if theta <= -1 or theta == 0
            For Frank: if theta == 0
            For Gumbel: if theta <= 1

        Notes
        -----
        Generation algorithms by copula family:
        
        **Clayton**: Uses conditional distribution method
        - U ~ Uniform(0,1)
        - V = U * (W^(-theta/(1+theta)) - 1 + U^theta)^(-1/theta)
        
        **Frank**: Uses conditional inverse method
        - Solves implicit equation involving exponentials
        
        **Gumbel**: Uses Marshall-Olkin algorithm
        - Generates stable distribution random variables
        - Uses rejection sampling for complex dependence structure
        
        Results stored in self.U and self.V attributes.

        Examples
        --------
        >>> cop = Copula(x, y, 'gumbel')
        >>> cop.generate_uv(500)
        >>> print(cop.U[:5], cop.V[:5])
        """
        # clayton copula
        if self.family == "clayton":
            U = np.random.uniform(size=n)
            W = np.random.uniform(size=n)

            if self.theta <= -1:
                raise ValueError(
                    "the parameter for clayton copula should be more than -1"
                )
            elif self.theta == 0:
                raise ValueError("The parameter for clayton copula should not be 0")

            if self.theta < sys.float_info.epsilon:
                V = W
            else:
                V = U * (
                    W ** (-self.theta / (1 + self.theta)) - 1 + U ** self.theta
                ) ** (-1 / self.theta)

        # frank copula
        elif self.family == "frank":
            U = np.random.uniform(size=n)
            W = np.random.uniform(size=n)

            if self.theta == 0:
                raise ValueError("The parameter for frank copula should not be 0")
            if abs(self.theta) > np.log(sys.float_info.max):
                V = (U < 0) + np.sign(self.theta) * U
            elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                V = (
                    -np.log(
                        (np.exp(-self.theta * U) * (1 - W) / W + np.exp(-self.theta))
                        / (1 + np.exp(-self.theta * U) * (1 - W) / W)
                    )
                    / self.theta
                )
            else:
                V = W

        # gumbel copula
        elif self.family == "gumbel":
            if self.theta <= 1:
                raise ValueError(
                    "The parameter for gumbel copula must be greater than one."
                )
            if self.theta < 1 + sys.float_info.epsilon:
                U = np.random.uniform(size=n)
                V = np.random.uniform(size=n)
            else:
                u = np.random.uniform(size=n)
                w = np.random.uniform(size=n)
                w1 = np.random.uniform(size=n)
                w2 = np.random.uniform(size=n)

                u = (u - 0.5) * np.pi
                u2 = u + np.pi / 2
                e = -np.log(w)
                t = np.cos(u - u2 / self.theta) / e
                gamma = (
                    (np.sin(u2 / self.theta) / t) ** (1 / self.theta) * t / np.cos(u)
                )
                s1 = (-np.log(w1)) ** (1 / self.theta) / gamma
                s2 = (-np.log(w2)) ** (1 / self.theta) / gamma
                U = np.array(np.exp(-s1))
                V = np.array(np.exp(-s2))

        self.U = U
        self.V = V
        return

    def generate_cond(self):
        """Generate conditional samples V given U using the conditional copula.
        
        Generates V from the conditional distribution C(V|U) where U is
        provided in self.U. Uses the conditional copula:
        C(v|u) = ∂C(u,v)/∂u

        Raises
        ------
        ValueError
            For Clayton: if theta <= -1 or theta == 0
            For Frank: if theta == 0
            For Gumbel: if theta <= 1

        Notes
        -----
        Conditional generation formulas:
        
        **Clayton**: 
            V = U^(-theta) * (U^(-theta) + W^(-theta) - 1)^(-1/theta) / 
                (U * (U^(-theta) + W^(-theta) - 1))
        
        **Frank**: 
            Uses exponential transformation of conditional inverse
        
        **Gumbel**: 
            Complex expression involving logarithms and power transformations
        
        Results stored in self.V. Requires self.U to be set beforehand.

        Examples
        --------
        >>> cop.U = np.array([0.2, 0.5, 0.8])
        >>> cop.generate_cond()
        >>> print(cop.V)
        """

        if isinstance(self.U, float):
            n = 1
        else:
            n = len(self.U)

        W = np.random.uniform(size=n)

        # clayton copula
        if self.family == "clayton":
            if self.theta <= -1:
                raise ValueError(
                    "the parameter for clayton copula should be more than -1"
                )
            elif self.theta == 0:
                raise ValueError("The parameter for clayton copula should not be 0")

            if self.theta < sys.float_info.epsilon:
                V = W
            else:
                V = (
                    self.U ** -self.theta
                    * (self.U ** -self.theta + W ** -self.theta - 1)
                    ** (-1 / self.theta)
                ) / (self.U * (self.U ** -self.theta + W ** -self.theta - 1))

        # frank copula
        elif self.family == "frank":
            if self.theta == 0:
                raise ValueError("The parameter for frank copula should not be 0")

            if abs(self.theta) > np.log(sys.float_info.max):
                V = (self.U < 0) + np.sign(self.theta) * self.U
            elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                tU, tW = (self.theta + 0j) ** self.U, (self.theta + 0j) ** W
                V = (
                    -tU
                    * (tW - 1)
                    / ((1 + (tU - 1) * (tW - 1) / (self.theta - 1)) * (self.theta - 1))
                )
            else:
                V = W

        # gumbel copula
        elif self.family == "gumbel":
            if self.theta <= 1:
                raise ValueError(
                    "The gumbel copula parameter must be greater than 1."
                )
            if self.theta < 1 + sys.float_info.epsilon:
                V = np.random.uniform(size=len(self.U))
            else:
                V = (
                    -np.log(self.U) ** self.theta
                    * ((-np.log(self.U)) ** self.theta + (-np.log(W)) ** self.theta)
                    ** (1 / self.theta)
                    * (
                        np.exp(
                            -(
                                (
                                    (-np.log(self.U)) ** self.theta
                                    + (-np.log(W)) * self.theta
                                )
                                ** (1 / self.theta)
                            )
                        )
                    )
                    / (
                        self.U
                        * np.log(self.U)
                        * ((-np.log(self.U)) ** self.theta + (-np.log(W)) ** self.theta)
                    )
                )

        self.V = np.absolute(V)
        return

    def generate_xy(self, n=0):
        """Generate random samples (X, Y) in original scale from the copula.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate. If 0 and U is not None, generates
            conditional samples. Default is 0.

        Notes
        -----
        The function:
        
        1. Generates (U, V) from copula if not already generated
        2. Transforms to original scale using:
        
           - Theoretical marginals (if provided in __init__)
           - Empirical inverse CDFs via kernel density estimation
           
        3. Stores results in self.X1 and self.Y1
        4. Resets self.U and self.V to None
        
        If U is already set, generates conditional V given U.
        Uses Epanechnikov kernel with bandwidth=0.1 for empirical CDFs.

        Examples
        --------
        >>> cop = Copula(x, y, 'clayton')
        >>> cop.generate_xy(1000)
        >>> plt.scatter(cop.X1, cop.Y1)
        """
        # estimate inverse cdf of x and y

        # if U and V are not already generated
        if self.U is None:
            self.generate_uv(n)
        elif n != 0:
            self.generate_cond()

        if hasattr(self, "p1"):
            X1 = self.F1.ppf(self.U, *self.p1)
            Y1 = self.F2.ppf(self.V, *self.p2)
        else:
            # if not hasattr(self, '_inv_cdf_x'):
            self._inverse_cdf()
            X1 = self._inv_cdf_x(self.U)
            Y1 = self._inv_cdf_y(self.V)
        self.X1 = X1
        self.Y1 = Y1

        # set U and V to None
        self.U = None
        self.V = None

        return

    def generate_C(self, u, v):
        """Compute copula values C(u, v) on a grid.

        Parameters
        ----------
        u : np.ndarray
            Grid points for first variable (uniform scale)
        v : np.ndarray
            Grid points for second variable (uniform scale)

        Notes
        -----
        Computes the copula CDF C(u,v) = P(U ≤ u, V ≤ v) on a meshgrid
        using family-specific formulas:
        
        **Clayton**: 
        C(u,v) = (u^(-theta) + v^(-theta) - 1)^(-1/theta)
        
        **Frank**: 
        C(u,v) = -1/theta * log(1 + (exp(-theta*u)-1)*(exp(-theta*v)-1)/(exp(-theta)-1))
        
        **Gumbel**: 
        C(u,v) = exp(-((−log u)^theta + (−log v)^theta)^(1/theta))
        
        Results stored in self.C as a 2D array.

        Examples
        --------
        >>> u = np.linspace(0.01, 0.99, 50)
        >>> v = np.linspace(0.01, 0.99, 50)
        >>> cop.generate_C(u, v)
        >>> plt.contourf(u, v, cop.C)
        """
        uq, vq = np.meshgrid(u, v)

        if self.family == "clayton":
            self.C = np.zeros(np.shape(uq))
            mask = (uq ** -self.theta + vq ** -self.theta - 1) ** -(1 / self.theta) > 0
            self.C[mask] = (uq[mask] ** -self.theta + vq[mask] ** -self.theta - 1) ** -(
                1 / self.theta
            )
        elif self.family == "frank":
            self.C = (
                -1
                / self.theta
                * np.log(
                    1
                    + (np.exp(-self.theta * uq) - 1)
                    * (np.exp(-self.theta ** vq) - 1)
                    / (np.exp(-self.theta) - 1)
                )
            )
        elif self.family == "gumbel":
            self.C = np.exp(
                -(
                    ((-np.log(uq)) ** self.theta + (-np.log(vq)) ** self.theta)
                    ** (1 / self.theta)
                )
            )

        return

    def estimate(self, data=None):
        """Estimate ensemble statistics for Y conditioned on X values.

        Parameters
        ----------
        data : np.ndarray, optional
            X values at which to estimate Y statistics. If None, uses self.X.

        Returns
        -------
        Y1_mean : np.ndarray
            Mean of Y for each X value
        Y1_std : np.ndarray
            Standard deviation of Y for each X value
        Y1_ll : np.ndarray
            25th percentile (lower quartile) of Y for each X value
        Y1_ul : np.ndarray
            75th percentile (upper quartile) of Y for each X value

        Notes
        -----
        The function:
        1. Generates copula ensemble if not already available (10000 samples)
        2. Bins X1 samples into 50 bins
        3. Computes statistics within each bin
        4. Interpolates statistics linearly to provided data points
        
        Useful for uncertainty quantification and conditional predictions.

        Examples
        --------
        >>> cop = Copula(x, y, 'frank')
        >>> x_new = np.linspace(x.min(), x.max(), 100)
        >>> y_mean, y_std, y_ll, y_ul = cop.estimate(x_new)
        >>> plt.plot(x_new, y_mean)
        >>> plt.fill_between(x_new, y_ll, y_ul, alpha=0.3)
        """
        nbin = 50
        # check if generate_xy has been called, otherwise call it now
        try:
            self.X1
            copula_ens = len(self.X1)
        except:
            copula_ens = 10000
            self.generate_xy(copula_ens)

        if data is None:
            data = self.X

        n_ens = copula_ens / nbin
        ind_sort = self.X1.argsort()
        x_mean = np.zeros((nbin,))
        y_mean = np.zeros((nbin,))
        y_ul = np.zeros((nbin,))
        y_ll = np.zeros((nbin,))
        y_std = np.zeros((nbin,))

        for ii in range(nbin):
            x_mean[ii] = self.X1[ind_sort[n_ens * ii : n_ens * (ii + 1)]].mean()
            y_mean[ii] = self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]].mean()
            y_std[ii] = self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]].std()
            y_ll[ii] = scoreatpercentile(
                self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]], 25
            )
            y_ul[ii] = scoreatpercentile(
                self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]], 75
            )

        foo_mean = interp1d(x_mean, y_mean, bounds_error=False)
        foo_std = interp1d(x_mean, y_std, bounds_error=False)
        foo_ll = interp1d(x_mean, y_ll, bounds_error=False)
        foo_ul = interp1d(x_mean, y_ul, bounds_error=False)

        Y1_mean = foo_mean(data)
        Y1_std = foo_std(data)
        Y1_ll = foo_ll(data)
        Y1_ul = foo_ul(data)

        return Y1_mean, Y1_std, Y1_ll, Y1_ul

    def _inverse_cdf(self):
        """Compute empirical inverse CDFs for X and Y using kernel density estimation.
        
        Creates interpolators that map from uniform [0,1] to original data scales.
        Used when theoretical marginal distributions are not provided.

        Notes
        -----
        Algorithm:
        1. Compute unique sorted values of X and Y
        2. Estimate PDF using Epanechnikov kernel (bandwidth=0.1)
        3. Numerically integrate PDF to get CDF
        4. Normalize CDF to [0, 1]
        5. Create linear interpolator from CDF to data values
        6. Use linear extrapolation outside data range
        
        Results stored in self._inv_cdf_x and self._inv_cdf_y interpolators.
        
        The Epanechnikov kernel provides optimal mean squared error for
        univariate density estimation.

        References
        ----------
        Silverman, B.W. (1986). Density Estimation for Statistics 
        and Data Analysis. Chapman & Hall.
        """

        x1 = np.unique(self.X)
        x2 = np.cumsum(
            np.exp(
                KernelDensity(kernel="epanechnikov", bandwidth=0.1)
                .fit(self.X[:, np.newaxis])
                .score_samples(x1[:, np.newaxis])
            )
        )
        x2 = x2 / np.max(x2)

        self._inv_cdf_x = interp1d(
            x2, x1, bounds_error=False, fill_value="extrapolate", kind="linear"
        )

        y1 = np.unique(self.Y)
        y2 = np.cumsum(
            np.exp(
                KernelDensity(kernel="epanechnikov", bandwidth=0.1)
                .fit(self.Y[:, np.newaxis])
                .score_samples(y1[:, np.newaxis])
            )
        )
        y2 = y2 / np.max(y2)

        self._inv_cdf_y = interp1d(
            y2, y1, bounds_error=False, fill_value="extrapolate", kind="linear"
        )
        return

    def _integrand_debye(self, t):
        """Integrand of first-order Debye function.
        
        Parameters
        ----------
        t : float
            Integration variable
            
        Returns
        -------
        float
            Value of t/(exp(t) - 1)
        """
        return t / (np.exp(t) - 1)

    def _debye(self, alpha):
        """First-order Debye function D_1(alpha).

        Parameters
        ----------
        alpha : float
            Argument of Debye function

        Returns
        -------
        float
            D_1(alpha) = (1/alpha) * integral_0^alpha [t/(exp(t)-1)] dt

        Notes
        -----
        The Debye function relates to Frank copula parameter via:
        tau = 1 - 4/alpha * [D_1(alpha) - 1]
        
        Used for parameter estimation in Frank copula.
        """
        return quad(self._integrand_debye, sys.float_info.epsilon, alpha)[0] / alpha

    def _frank_fun(self, alpha):
        """Objective function for Frank copula parameter optimization.

        Parameters
        ----------
        alpha : float
            Candidate theta value

        Returns
        -------
        float
            Squared difference between observed and theoretical tau

        Notes
        -----
        Minimizes: [(1 - tau)/4 - (D_1(-alpha) - 1)/alpha]^2
        to find optimal Frank copula parameter.
        """
        diff = (1 - self.tau) / 4.0 - (self._debye(-alpha) - 1) / alpha
        return diff ** 2
