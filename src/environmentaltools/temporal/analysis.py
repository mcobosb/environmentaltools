import datetime
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.vector_ar.var_model import VAR
from loguru import logger

from environmentaltools.common import utils, read, save
from environmentaltools.temporal import core
from environmentaltools.temporal.classification import class_storm_seasons
from environmentaltools.temporal.utils import extreme_events, events_duration

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


def show_init_message():
    message = (
        "\n"
        + "Copyright (C) 2026 Environmental Fluid Dynamics Group (University of Granada)\n"
        + "=============================================================================\n"
        + "This program is free software; you can redistribute it and/or modify it under\n"
        + "the terms of the GNU General Public License as published by the Free Software\n"
        + "Foundation; either version 3 of the License, or (at your option) any later \n"
        + "version.\n"
        + "This program is distributed in the hope that it will be useful, but WITHOUT \n"
        + "ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS\n"
        + "FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
        + "You should have received a copy of the GNU General Public License along with\n"
        + "this program; if not, write to the Free Software Foundation, Inc., 675 Mass\n"
        + "Ave, Cambridge, MA 02139, USA.\n"
        + "============================================================================="
    )
    return message


def fit_marginal_distribution(df: pd.DataFrame, parameters: dict, verbose: bool = False):
    """Fits a stationary (or not), simple or mixed probability model to data.
    
    Additional information can be found in Cobos et al., 2022, 'MarineTools.temporal: 
    A Python package to simulate Earth and environmental time series'. Environmental 
    Modelling and Software.

    Parameters
    ----------
    df : pd.DataFrame
        The raw time series
    parameters : dict
        The initial guess parameters of the probability models with the following keys:
        
        - 'var' : str
            Name of the variable
        - 'type' : str
            Defines circular or linear variables
        - 'fun' : list
            List of strings with the name of the probability model
        - 'non_stat_analysis' : bool
            False for stationary, True for non-stationary
        - 'ws_ps' : float or list
            Initial guess of percentiles or weights of PMs
        - 'basis_function' : dict or None
            GFS expansion specification:
            
            - 'method' : str
                Option for the GFS
            - 'no_terms' : int
                Number of terms of GFS
            - 'periods' : list
                Periods of oscillation for NS-PMs
        
        - 'transform' : dict or None
            Normalization options:
            
            - 'make' : bool
                Whether to apply transformation
            - 'method' : str
                'box-cox' or 'yeo-johnson'
            - 'plot' : bool
                Whether to plot
        
        - 'detrend' : dict or None
            Removing trends options:
            
            - 'make' : bool
                Whether to detrend
            - 'method' : str
                GFS option (commonly polynomial approaches)
        
        - 'optimization' : dict
            Parameters for scipy.optimize.minimize:
            
            - 'method' : str
                e.g., "SLSQP"
            - 'maxiter' : float
                Maximum iterations
            - 'ftol' : float
                Function tolerance
            - 'eps' : float
                Step size for numerical derivatives
            - 'bounds' : float
                Bounds for optimization
            - 'weighted' : bool
                For weighted data along time axis
        
        - 'giter' : int
            Number of global iterations
        - 'initial_parameters' : dict or None
            Initial parameters for unique optimization mode:
            
            - 'make' : bool
                Whether to use initial parameters
            - 'mode' : list
                Mode to be computed independently
            - 'par' : list
                Initial guess of parameters for given mode
        
        - 'file_name' : str, optional
            Path where analysis will be saved
    
    verbose : bool, optional
        If True, shows information of the fitting process. Default is False.

    Returns
    -------
    dict
        The fitting parameters

    Examples
    --------
    >>> param = {'Hs': {'var': 'Hs',
    ...                 'fun': {0: 'norm'},
    ...                 'type': 'linear',
    ...                 'non_stat_analysis': True,
    ...                 'basis_function': {"method": "trigonometric",
    ...                                   "no_terms": 4,
    ...                                   "periods": [1, 2, 4]},
    ...                 'ws_ps': 1,
    ...                 'transform': {"make": True,
    ...                              "plot": False,
    ...                              "method": "box-cox"},
    ...                 'detrend': {"make": True,
    ...                            "method": "polynomial"},
    ...                 'optimization': {'method': 'SLSQP',
    ...                                 'eps': 1e-7,
    ...                                 'ftol': 1e-4,
    ...                                 'maxiter': 1e2,
    ...                                 'bounds': 0.5},
    ...                 'giter': 10,
    ...                 'scale': False,
    ...                 'bic': True,
    ...                 'file_name': 'output.pkl'
    ...                 }
    ...         }
    """
    # Initial computational time
    start_time = time.time()
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if verbose:
        logger.info(show_init_message())
    logger.info("Current Time = %s\n" % current_time)

    # Remove nan in the input timeseries
    df = pd.DataFrame(df).dropna()

    # Check if negative and positive values are in the timeseries for fitting purpouses
    if df[df < 0].any().values[0]:
        if verbose:
            logger.info(
                "Dataset has negative values. Check that the chosen distribution functions adequately fit negative values."
            )

    if (parameters["type"] == "circular") & ("ws_ps" not in parameters.keys()):
        # Transform angles to radian
        df = np.deg2rad(df)
        # Compute the percentile of change between probability models
        ecdf = utils.ecdf(df, parameters["var"], no_perc=1000)
        # Smooth the ecdf
        ecdf["soft"] = utils.smooth_1d(ecdf[parameters["var"]], 100)
        # Compute the difference
        ecdf["dif"] = ecdf["soft"].diff()
        # Obtain the index of the max
        max_ = utils.max_moving_window(ecdf["dif"], 250)
        parameters["ws_ps"] = [max_.index[0]]

    parameters["verbose"] = verbose
    # Check that the input dictionary is well defined
    parameters = check_marginal_params(parameters)

    # Transform angles into radians
    if parameters["type"] == "circular":
        df[parameters["var"]] = np.deg2rad(df[parameters["var"]])

    # Normalized the data using one of the normalization method if it is required
    if parameters["transform"]["make"]:
        df, parameters = core.transform(df, parameters)
        parameters["transform"]["min"] = df.min().values[0] - 1e-2
        df -= parameters["transform"]["min"]

    # Scale and shift time series for ensuring the use of any PM
    parameters["range"] = float((df.max() - df.min()).values[0])
    if parameters["scale-shift"]:
        if parameters["range"] > 10:
            df = df / (parameters["range"] / 3)
            parameters["scale"] = parameters["range"] / 3
            # if parameters["piecewise"]:
            #     for ind_, val_ in enumerate(parameters["ws_ps"]):
            #         parameters["ws_ps"][ind_] = val_ / (parameters["range"] / 3)

    # Bound the variable with some reference values
    if parameters["type"] == "circular":
        parameters["minimax"] = [0, 2 * np.pi]
    else:
        parameters["minimax"] = [
            float(df[parameters["var"]].min()),
            float(
                np.max(
                    [df[parameters["var"]].max(), df[parameters["var"]].max() * 1.25]
                )
            ),
        ]

    # Calculate the normalize time along the reference oscillating period
    df["n"] = np.fmod(
        (df.index - datetime.datetime(df.index[0].year, 1, 1, 0)).total_seconds().values
        / (parameters["basis_period"][0] * 365.25 * 24 * 3600),
        1,
    )
    # Create and additional time line for detrending
    if parameters["detrend"]["make"]:
        df["n_detrend"] = (df.index - df.index[0]) / (df.index[-1] - df.index[0])

    # Compute the temporaL weights if it is required
    if parameters["weighted"]["make"]:
        if parameters["weighted"]["window"] == "month":
            counts = df.groupby("n").count()  # TODO: con pesos promedio mensuales.
        else:
            counts = df.groupby("n").count()
            nmean = counts.mean()
            weights = nmean / counts
            df["counts"] = np.nan
            for n in counts.index:
                nn = len(df.loc[df["n"] == n, "counts"])
                df.loc[df["n"] == n, "counts"] = np.ones(nn) * weights.loc[n].values
        parameters["weighted"]["values"] = df["counts"]
    else:
        parameters["weighted"]["values"] = 1

    if not parameters["non_stat_analysis"]:
        # Make the stationary analysis
        if verbose:
            logger.info("MARGINAL STATIONARY FIT")
            logger.info(
                "=============================================================================="
            )
        # Write the information about the variable, PMs and method
        term = (
            "Stationary fit of "
            + parameters["var"]
            + " to a "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += " - " + str(parameters["fun"][i].name)
        term += " - genpareto " * parameters["reduction"]
        term += " probability model"
        if verbose:
            logger.info(term)
        # Make the stationary analysis
        df, parameters["par"], parameters["mode"] = core.stationary_analysis(df, parameters)

        # Write the information about the variable, PMs and method
    elif parameters["non_stat_analysis"] & (
        not parameters["initial_parameters"]["make"]
    ):
        # Make the stationary analysis first
        df, parameters["par"], parameters["mode"] = core.stationary_analysis(df, parameters)
        # Make the non-stationary analysis
        if verbose:
            logger.info("MARGINAL NON-STATIONARY FIT")
            logger.info(
                "=============================================================================="
            )
        term = (
            "\nNon-stationary fit of "
            + parameters["var"]
            + " with the "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += " - " + str(parameters["fun"][i].name)
        if parameters["reduction"]:
            term += " - genpareto " * parameters["reduction"]
        term += " probability model"
        if verbose:
            logger.info(term)
            logger.info(
                "with the "
                + parameters["optimization"]["method"]
                + " optimization method."
            )
            logger.info(
                "=============================================================================="
            )

        # Make the non-stationary analysis
        parameters = core.nonstationary_analysis(df, parameters)

    else:
        # Make the non-stationary analysis of a given mode
        term = (
            "Non-stationary fit of "
            + parameters["var"]
            + " with the "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += "-" + str(parameters["fun"][i].name)
        term += " and mode:"
        for mode in parameters["initial_parameters"]["mode"]:
            term += " " + str(mode)
        if verbose:
            logger.info(term)
            logger.info(
                "=============================================================================="
            )
        # Make the non-stationary analysis
        parameters = core.nonstationary_analysis(df, parameters)

    # List all the parameters in a standar format
    if parameters["reduction"]:
        if not parameters["non_stat_analysis"]:
            pars, esc = core.get_params(df, parameters, parameters["par"], [0, 0, 0], 0)
        else:  # Los parámetros del no estacionario para la cola inferior no tienen amplitud y fase
            t_expans = core.params_t_expansion(
                parameters["mode"],
                parameters,
                df["n"],
            )
            pars, esc = core.get_params(
                df,
                parameters,
                parameters["par"],
                [parameters["mode"][0], parameters["mode"][0], parameters["mode"][0]],
                t_expans,
            )

        parameters_ = pars.iloc[0, 2:]
        parameters_["ps_0"] = esc[1]
        parameters_["ps_1"] = 1 - esc[2]
        parameters["standar_parameters"] = parameters_.to_dict()

    # Change the object function for its string names
    parameters["fun"] = {i: parameters["fun"][i].name for i in parameters["fun"].keys()}
    parameters["status"] = "Distribution models fitted succesfully"

    # Final computational time
    logger.info("End of the fitting process")
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    # Save the parameters in the file if "file_name" is given in params
    # utils.mkdir(parameters["folder_name"])

    if not "file_name" in parameters.keys():
        generate_outputfilename(parameters)

    if "folder_name" in parameters.keys():
        os.makedirs(parameters["folder_name"], exist_ok=True)
        parameters["file_name"] = os.path.join(
            parameters["folder_name"], parameters["file_name"]
        )
    else:
        pass

    del parameters["weighted"]["values"]
    os.makedirs(os.path.dirname(parameters["file_name"]), exist_ok=True)
    save.to_json(parameters, parameters["file_name"])

    # Return the dictionary with the parameters of the analysis
    return parameters


def check_marginal_params(param: dict):
    """Checks the input parameters and includes some required arguments for the computation

    Args:
        * param (dict): the initial guess parameters of the probability models (see the docs of marginalfit)

    Returns:
        * param (dict): checked and updated parameters
    """

    param["no_param"] = {}
    param["scipy"] = {}
    param["reduction"] = False
    param["no_tot_param"] = 0
    param["constraints"] = True

    logger.info("USER OPTIONS:")
    k = 1

    # Checking the transform parameters if any
    if not "transform" in param.keys():
        param["transform"] = {}
        param["transform"]["make"] = False
        param["transform"]["plot"] = False
    else:
        if param["transform"]["make"]:
            if not param["transform"]["method"] in ["box-cox", "yeo-johnson"]:
                raise ValueError(
                    "The power transformation methods available are 'yeo-johnson' and 'box-cox', {} given.".format(
                        param["transform"]["method"]
                    )
                )
            else:
                logger.info(
                    str(k)
                    + " - Data is previously normalized ("
                    + param["transform"]["method"]
                    + " method given)".format(str(k))
                )
                k += 1

    # Checking the detrend parameters if any
    if not "detrend" in param.keys():
        param["detrend"] = {}
        param["detrend"]["make"] = False
    else:
        if param["detrend"]["make"]:
            if not param["detrend"]["method"] in [
                "chebyshev",
                "legendre",
                "laguerre",
                "hermite",
                "ehermite",
                "polynomial",
            ]:
                raise ValueError(
                    "Methods available are:\
                    chebyshev, legendre, laguerre, hermite, ehermite or polynomial.\
                    Given {}.".format(
                        param["detrend"]["method"]
                    )
                )
            else:
                logger.info(
                    str(k)
                    + " - Detrend timeseries is appliedData is previously normalized ("
                    + param["detrend"]["method"]
                    + " method given)".format(str(k))
                )
                k += 1

    # Check if it can be reduced the number of parameters using Solari (2011) analysis
    if not "fun" in param.keys():
        raise ValueError("Probability models are required in a list in fun.")
    if len(param["fun"].keys()) == 3:
        if (param["fun"][0] == "genpareto") & (param["fun"][2] == "genpareto"):
            param["reduction"] = True
            logger.info(
                str(k)
                + " - The combination of PMs given enables the reduction"
                + " of parameters during the optimization"
            )
            k += 1

    # Check the number of probability models
    if len(param["fun"].keys()) > 3:
        raise ValueError(
            "No more than three probability models are allowed in this version"
        )

    if param["reduction"]:
        # Particular case where the number of parameters to be optimized is reduced
        param["fun"] = {
            0: getattr(st, param["fun"][0]),
            1: getattr(st, param["fun"][1]),
            2: getattr(st, param["fun"][2]),
        }
        param["no_fun"] = 2
        param["no_param"][0] = int(param["fun"][1].numargs + 2)
        if param["no_param"][0] > 5:
            raise ValueError(
                "Probability models with more than 3 parameters are not allowed in this version"
            )
        param["no_param"][1] = 1
        param["no_tot_param"] = int(param["fun"][1].numargs + 3)
    else:
        param["no_fun"] = len(param["fun"].keys())
        for i in range(param["no_fun"]):
            if isinstance(param["fun"][i], str):
                if param["fun"][i] == "wrap_norm":
                    param["fun"][i] = core.wrap_norm()
                    param["scipy"][i] = False
                    param["constraints"] = False
                else:
                    param["fun"][i] = getattr(st, param["fun"][i])
                    param["scipy"][i] = True
            param["no_param"][i] = int(param["fun"][i].numargs + 2)
            if param["no_param"][i] > 5:
                raise ValueError(
                    "Probability models with more than 3 parameters are not allowed in this version"
                )
            param["no_tot_param"] += int(param["fun"][i].numargs + 2)

    if param["non_stat_analysis"] == False:
        param["basis_period"] = None
        param["basis_function"] = {"method": "None", "order": 0, "no_terms": 0}

    if not "basis_period" in param:
        param["basis_period"] = [1]
    elif param["basis_period"] == None:
        param["order"] = 0
        if param["non_stat_analysis"] == False:
            param["basis_period"] = [1]
    elif isinstance(param["basis_period"], int):
        param["basis_period"] = list(param["basis_period"])

    if (not "basis_function" in param.keys()) & param["non_stat_analysis"]:
        raise ValueError("Basis function is required when non_stat_analysis is True.")

    if (not "method" in param["basis_function"]) & param["non_stat_analysis"]:
        raise ValueError("Method is required when non_stat_analysis is True.")
    elif param["non_stat_analysis"]:
        if ((not "no_terms") & (not "periods")) in param["basis_function"].keys():
            raise ValueError(
                "Number of terms higher than zero or list of periods with more at least one element is required when non_stat_analysis is True."
            )
        elif param["basis_function"]["method"] in [
            "trigonometric",
            "sinusoidal",
            "modified",
        ]:
            if ((not "no_terms") & (not "periods")) in param["basis_function"].keys():
                raise ValueError(
                    "Number of terms or periods are required for Fourier Series approximation."
                )
            else:
                if not "periods" in param["basis_function"]:
                    param["basis_function"]["periods"] = list(
                        1 / np.arange(1, param["basis_function"]["no_terms"] + 1)
                    )
                    param["basis_function"]["order"] = param["basis_function"][
                        "no_terms"
                    ]
                else:
                    param["basis_function"]["no_terms"] = len(
                        param["basis_function"]["periods"]
                    )
                    param["basis_function"]["order"] = param["basis_function"][
                        "no_terms"
                    ]
                # param["approximation"]["periods"].sort(reverse=True)
                if not "basis_period" in param:
                    param["basis_period"] = [param["basis_function"]["periods"][0]]
        else:
            if param["basis_function"]["method"] not in [
                "chebyshev",
                "legendre",
                "laguerre",
                "hermite",
                "ehermite",
                "polynomial",
            ]:
                raise ValueError(
                    "Method available are:\
                    trigonometric, modified, sinusoidal, \
                    chebyshev, legendre, laguerre, hermite, ehermite or polynomial.\
                    Given {}.".format(
                        param["basis_function"]["method"]
                    )
                )
            else:
                if not "degree" in param["basis_function"].keys():
                    raise ValueError("The polynomial methods require the degree")
                param["basis_function"]["order"] = param["basis_function"]["degree"]
                if not "basis_period" in param:
                    param["basis_period"] = [param["basis_function"]["periods"][0]]

        logger.info(
            str(k)
            + " - The basis function given is {}.".format(
                param["basis_function"]["method"]
            )
        )
        k += 1

        logger.info(
            str(k)
            + " - The number of terms given is {}.".format(
                param["basis_function"]["order"]
            )
        )
        k += 1

    # Check if initial parameters are given
    if not "initial_parameters" in param.keys():
        param["initial_parameters"] = {}
        param["initial_parameters"]["make"] = False
    elif "initial_parameters" in param.keys():
        if not "make" in param["initial_parameters"]:
            raise ValueError(
                "The evaluation of a certain mode requires that initial parameter 'make' set to True. Not given."
            )
        if not "par" in param["initial_parameters"].keys():
            param["initial_parameters"]["par"] = []
            logger.info(
                str(k)
                + " - Parameters of optimization not given. It will be applied the Fourier initialization."
            )
            k += 1
        else:
            param["par"] = param["initial_parameters"]["par"]
            logger.info(
                str(k)
                + " - Parameters of optimization given ({}).".format(
                    param["initial_parameters"]["par"]
                )
            )
            k += 1
        if not "mode" in param["initial_parameters"].keys():
            raise ValueError(
                "The evaluation of a mode requires the initial mode 'mode'. Give the mode."
            )
        else:
            param["mode"] = param["initial_parameters"]["mode"]
            logger.info(
                str(k)
                + " - Mode of optimization given ({}).".format(
                    param["initial_parameters"]["mode"]
                )
            )
            k += 1

    if not "optimization" in param.keys():
        param["optimization"] = {}
        param["optimization"]["method"] = "SLSQP"
        param["optimization"]["eps"] = 1e-3
        param["optimization"]["maxiter"] = 1e2
        param["optimization"]["ftol"] = 1e-3
    else:
        if param["optimization"] is None:
            param["optimization"] = {}
            param["optimization"]["method"] = "SLSQP"
            param["optimization"]["eps"] = 1e-3
            param["optimization"]["maxiter"] = 1e2
            param["optimization"]["ftol"] = 1e-3
        else:
            if not "eps" in param["optimization"].keys():
                param["optimization"]["eps"] = 1e-3
            if not "maxiter" in param["optimization"].keys():
                param["optimization"]["maxiter"] = 1e2
            if not "ftol" in param["optimization"].keys():
                param["optimization"]["ftol"] = 1e-3

    if not "giter" in param["optimization"].keys():
        param["optimization"]["giter"] = 10
    else:
        if not isinstance(param["optimization"]["giter"], int):
            raise ValueError("The number of global iterations should be an integer.")
        else:
            logger.info(
                "{} - Global iterations were given by user ({})".format(
                    str(k), str(param["optimization"]["giter"])
                )
            )
            k += 1

    if not "bounds" in param["optimization"].keys():
        if param["type"] == "circular":
            param["optimization"]["bounds"] = 0.1
        else:
            param["optimization"]["bounds"] = 0.5
    else:
        if not isinstance(param["optimization"]["bounds"], (float, int, bool)):
            raise ValueError("The bounds should be a float, integer or False.")
        else:
            logger.info(
                "{} - Bounds were given by user (bounds = {})".format(
                    str(k), str(param["optimization"]["bounds"])
                )
            )
            k += 1

    if "piecewise" in param:
        if not param["reduction"]:
            if param["piecewise"]:
                param["constraints"] = False
                logger.info(
                    str(k)
                    + " - Piecewise analysis of PMs defined by user. Piecewise is set to True."
                )
                k += 1
        else:
            logger.info(
                str(k)
                + " - Piecewise analysis is not recommended when reduction is applied. Piecewise is set to False."
            )
            param["piecewise"] = False
            k += 1
    else:
        param["piecewise"] = False

    if param["no_fun"] == 1:
        param["constraints"] = False

    if param["reduction"]:
        param["constraints"] = False

    if not "transform" in param.keys():
        param["transform"] = {"make": False, "method": None, "plot": False}

    if param["reduction"]:
        if len(param["ws_ps"]) != 2:
            raise ValueError(
                "Expected two percentiles for the analysis. Got {}.".format(
                    str(len(param["ws_ps"]))
                )
            )
    else:
        if (not "ws_ps" in param) & (param["no_fun"] - 1 == 0):
            param["ws_ps"] = []
        elif (not "ws_ps" in param) & (param["no_fun"] - 1 != 0):
            raise ValueError(
                "Expected {} weight\\s for the analysis. However ws_ps option is not given.".format(
                    str(param["no_fun"] - 1)
                )
            )

        if len(param["ws_ps"]) != param["no_fun"] - 1:
            raise ValueError(
                "Expected {} weight\\s for the analysis. Got {}.".format(
                    str(param["no_fun"] - 1), str(len(param["ws_ps"]))
                )
            )

    # Check if the variable is circular or linear
    if param["type"] == "circular":
        logger.info("{} - Type is set to 'circular'.".format(str(k)))
    else:
        logger.info("{} - Type is set to 'linear'.".format(str(k)))
    
    if param["type"] == "circular":
        for fun_ in param["fun"].values():
            if fun_.name not in ["vonmises", "wrap_cauchy", "wrap_norm", "norm"]:
                raise ValueError(
                    "For circular variables, only vonmises, wrap_cauchy, wrap_norm and norm PMs are allowed. Got {}.".format(
                        fun_.name
                    )
                )
    k += 1

    if (any(np.asarray(param["ws_ps"]) > 1) or any(np.asarray(param["ws_ps"]) < 0)) & (
        not param["piecewise"]
    ):
        raise ValueError(
            "Percentiles cannot be lower than 0 or bigger than one. Got {}.".format(
                str(param["ws_ps"])
            )
        )

    if not "guess" in param.keys():
        param["guess"] = False

    if not "bic" in param.keys():
        param["bic"] = False

    if param["constraints"]:
        if (not param["optimization"]["method"] == "SLSQP") & (param["no_fun"] > 1):
            raise ValueError(
                "Constraints are just available for SLSQP method in this version."
            )

    if "fix_percentiles" in param.keys():
        if param["fix_percentiles"]:
            logger.info(
                "{} - Percentiles are fixed. Fix_percentiles is set to True.".format(
                    str(k)
                )
            )
            k += 1
    elif not param["non_stat_analysis"]:
        param["fix_percentiles"] = True
        logger.info(
            "{} - Percentiles are fixed. Fix_percentiles is set to True.".format(str(k))
        )
        k += 1
    else:
        param["fix_percentiles"] = False

    if not "scale-shift" in param.keys():
        param["scale-shift"] = False

    if not param["scale-shift"]:
        param["scale"] = 1
        param["shift"] = 0

    if not "weighted" in param.keys():
        param["weighted"] = {}
        param["weighted"]["make"] = False
    else:
        if not "make" in param["weighted"].keys():
            param["weighted"]["make"] = False
        elif isinstance(param["weighted"]["make"], bool):
            logger.info("{} - Weighted data along time is set to True.".format(str(k)))
            k += 1
            if not "window" in param["weighted"]:
                param["weighted"]["window"] = "timestep"
            elif not (
                (param["weighted"]["window"] == "timestep")
                | (param["weighted"]["window"] == "month")
            ):
                raise ValueError("Weighted window options are 'timestep' or 'month'.")
            logger.info(
                "{} - Weighted window for every {}.".format(
                    str(k), param["weighted"]["window"]
                )
            )
            k += 1
        else:
            raise ValueError("Weighted options are True or False.")

    if param["verbose"]:
        logger.info("{} - Verbose is set to True.".format(str(k)))
        k += 1

    if k == 1:
        logger.info("None.")

    logger.info(
        "==============================================================================\n"
    )

    return param


# def init_fourier_coefs():
#     """Compute an estimation of the initial parameters for trigonometric expansions"""
#     timestep = 1 / 365.25
#     wlen = 14 / 365.25  # 14-days window
#     res = pd.DataFrame(
#         0, index=np.arange(0, 1, timestep), columns=["s", "loc", "scale"]
#     )
#     for ii, i in enumerate(res.index):
#         if i >= (1 - wlen):
#             final_offset = i + wlen - 1
#             mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
#                 data["n"] <= final_offset
#             )
#         elif i <= wlen:
#             initial_offset = i - wlen
#             mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
#                 data["n"] >= 1 + initial_offset
#             )
#         else:
#             mask = (data["n"] >= i - wlen) & (data["n"] <= i + wlen)

#         model = st.gamma
#         result = st.fit(
#             model,
#             data[station].loc[mask],
#             bounds=[(0, 5), bound, (0, 100)],
#         )
#         res.loc[i, :] = result.params.a, result.params.loc, result.params.scale

#     coefs = np.fft.fft(res.loc[:, paramName] - np.mean(res.loc[:, paramName]))

#     N = len(res.loc[:, paramName])
#     # Choose one side of the spectra
#     cn = np.ravel(coefs[0 : N // 2] / N)

#     an, bn = 2 * np.real(cn), -2 * np.imag(cn)

#     an = an[: index + 1]
#     bn = bn[: index + 1]

#     parameters = np.mean(res.loc[:, paramName])
#     for order_k in range(index):
#         parameters = np.hstack([parameters, an[order_k + 1], bn[order_k + 1]])
#     return


def add_noise_to_array(
    data: pd.DataFrame, variables: list, remove: bool = False, filter_: str = None
):
    """Adds small random noise to the selected variable(s) in a time series for better estimations.

    Args:
        data (pd.DataFrame): Raw time series.
        variable (list): Variable(s) to apply noise to.
        remove (bool): If True, rows filtered by `filter_` are removed from the output.
        filter_ (str, optional): Query string to filter the DataFrame before adding noise.

    Returns:
        pd.DataFrame: DataFrame with noise added to the selected variable(s).
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    df_out = data.copy()

    # Remove duplicate values
    df_out = df_out[~df_out.index.duplicated(keep="first")]

    # Multi-variable path (original loop)
    for var_ in variables:
        # Only operate on non-NaN values in the filtered subset
        idx_valid = df_out[var_].dropna().index
        if len(idx_valid) == 0:
            raise ValueError(
                f"Input time series for variable '{var_}' is empty after filtering."
            )
        unique_vals = np.sort(df_out.loc[idx_valid, var_].unique())
        if len(unique_vals) > 1:
            increments = st.mode(np.diff(unique_vals))[0]
        else:
            increments = 1e-6  # fallback small noise if all values are identical
        noise = np.random.rand(len(idx_valid)) * increments

        # More robust assignment to avoid broadcasting issues
        values_with_noise = df_out.loc[idx_valid, var_].values + noise
        df_out.loc[idx_valid, var_] = values_with_noise

    # Eliminar todos los NaNs
    df_out = df_out.dropna()

    return df_out


def look_models(data, variable, percentiles=[1], file_name="models_out", funcs="natural"):
    """Fit multiple probability models to data and rank by estimation quality.
    
    Tests various probability distributions from scipy.stats and ranks them by
    Sum of Squared Errors (SSE) to identify the best-fitting model for the data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw time series containing the variable to fit
    variable : str
        Name of the variable column to analyze
    percentiles : list, optional
        Values of CDF at transitions between different probability models
        for mixed distributions. Default: [1]
    file_name : str, optional
        Name of the output file to save fitted parameters. Default: 'models_out'
    funcs : str or list, optional
        Probability models to test. Options:
        - 'natural': Common environmental distributions (default)
        - None: All continuous distributions in scipy.stats
        - list: Custom list of distribution names

    Returns
    -------
    pd.DataFrame
        DataFrame with fitted parameters sorted by SSE (best fit first).
        Columns: 'models', 'sse', and distribution parameters ('a', 'b', 'c', etc.)

    Notes
    -----
    The 'natural' option includes distributions commonly used in environmental
    modeling: alpha, beta, expon, genpareto, genextreme, gamma, gumbel_r,
    gumbel_l, triang, lognorm, norm, rayleigh, weibull_min, weibull_max.
    
    The SSE is computed between the empirical CDF and the fitted CDF.
    Lower SSE indicates better fit.

    Examples
    --------
    >>> results = look_models(data, 'wave_height', file_name='wave_models')
    >>> print(results.head())  # Shows top 5 best-fitting models
    """

    # TODO: for mixed functions
    if not funcs:
        funcs = st._continuous_distns._distn_names
    elif funcs == "natural":
        funcs = [
            "alpha",
            "beta",
            "expon",
            "genpareto",
            "genextreme",
            "gamma",
            "gumbel_r",
            "gumbel_l",
            "triang",
            "lognorm",
            "norm",
            "rayleigh",
            "weibull_min",
            "weibull_max",
        ]

    results = dict()
    cw = np.hstack([0, np.cumsum(percentiles)])
    dfs = data.sort_values(variable, ascending=True)
    dfs["p"] = np.linspace(0, 1, len(dfs))
    # for i, j in enumerate(percentiles):
    # filt = ((dfs['p'] > cw[i]) & (dfs['p'] < j))
    # Create a table with the parameters of the best estimations and sse
    results = pd.DataFrame(
        0,
        index=np.arange(0, len(funcs)),
        columns=["models", "sse", "a", "b", "c", "d", "e", "f"],
    )
    results.index.name = "id"

    # Computeh the best estimations for the given models
    for k, name in enumerate(funcs):
        model = getattr(st, name)
        out = core.fit_distribution(dfs.loc[:, variable], 25, model)
        results.loc[k, "models"] = name
        results.iloc[k, 1 : len(out) + 1] = out
    results.sort_values(by="sse", inplace=True)
    results["position"] = np.arange(1, len(funcs) + 1)

    # for i, j in enumerate(percentiles):
    results.replace(0, "-", inplace=True)

    # Save to a xlsx file
    save.to_xlsx(results, file_name)

    return results


def storm_series(data, cols, info):
    """Extract storm events from time series data using threshold-based criteria.
    
    Identifies discrete storm events based on duration, inter-arrival time, and
    threshold criteria following the methodology of Lira-Loarca et al. (2020).

    Parameters
    ----------
    data : pd.DataFrame
        Raw time series with datetime index
    cols : list
        Names of required concomitant variables (first one used for threshold)
    info : dict
        Storm event criteria with keys:
        
        - threshold : float
            Minimum value to define storm event
        - min_duration : int
            Minimum storm duration (in time_step units)
        - inter_time : int
            Minimum inter-arrival time between storms (in time_step units)
        - time_step : str
            Time step unit: 'D' (days) or 'h' (hours)
        - interpolation : bool, optional
            Whether to interpolate missing data. Default: False
        - filename : str, optional
            Output file name for saving results

    Returns
    -------
    pd.DataFrame
        Storm time series with identified events and their properties

    Raises
    ------
    ValueError
        If any label in cols is not in data columns

    Notes
    -----
    Uses the extreme_events package to identify storms based on:
    
    - Exceedance of threshold value
    - Minimum event duration
    - Minimum separation between events
    
    References
    ----------
    Lira-Loarca, A., et al. (2020). A global classification of coastal flood
    hazard climates. Scientific Reports.

    Examples
    --------
    >>> info = {
    ...     'threshold': 2.0,
    ...     'min_duration': 6,
    ...     'inter_time': 24,
    ...     'time_step': 'h',
    ...     'interpolation': True
    ... }
    >>> storms = storm_series(data, ['wave_height', 'wind_speed'], info)
    """
    check_items = all(item in data.columns for item in cols)
    if not check_items:
        raise ValueError("Any label of cols is not in data")

    if info["time_step"] == "D":
        strTimeStep, intDur = " days", "D"
    else:
        strTimeStep, intDur = " hours", "h"
    min_duration_td = pd.Timedelta(str(info["min_duration"]) + strTimeStep)
    min_interarrival_td = pd.Timedelta(str(info["inter_time"]) + strTimeStep)

    cycles_ini, _, infoc = extreme_events(
        data,
        cols[0],
        info["threshold"],
        min_interarrival_td,
        min_duration_td,
        interpolation=info["interpolation"],
        truncate=True,
        extra_info=True,
    )

    times = pd.date_range(data.index[0], data.index[-1], freq=info["time_step"])
    df = pd.DataFrame(-99.99, index=times, columns=cols)
    df["nstorm"] = 0
    tini = times[0]
    for i, j in enumerate(cycles_ini):
        tini = j.index[0]
        if i == len(cycles_ini) - 1:
            tfin = data.index[-1]
        else:
            tfin = cycles_ini[i + 1].index[0]
        df.loc[tini:tfin, "nstorm"] = i + 1
        if not info["interpolation"]:
            df.loc[j.index[0] : j.index[-1], cols] = infoc["data_cycles"].loc[
                j.index[0] : j.index[-1], cols
            ]
        else:
            df.loc[j.index[1] : j.index[-2], cols] = infoc["data_cycles"].loc[
                j.index[1] : j.index[-2], cols
            ]

    df.index.name = "date"
    df.replace(-99.99, np.nan, inplace=True)

    if "filename" in info.keys():
        save.to_csv(df.dropna(), info["filename"])
    df.dropna(inplace=True)
    return df


def storm_properties(data, cols, info):
    """Extract and characterize individual storm event properties.
    
    Identifies storm events and computes their statistical properties including
    duration, peak values, integrated values, and seasonal classification.

    Parameters
    ----------
    data : pd.DataFrame
        Raw time series with datetime index
    cols : list
        Names of variables to analyze for each storm
    info : dict
        Storm event criteria and analysis parameters:
        
        - threshold : float
            Minimum value to define storm event
        - min_duration : int
            Minimum storm duration (in data_timestep units)
        - inter_time : int
            Minimum inter-arrival time between storms
        - data_timestep : str
            Time step of input data: 'D' (days) or 'H' (hours)
        - interpolation : bool, optional
            Whether to interpolate storm boundaries. Default: False
        - class_type : str, optional
            Season classification scheme: 'WSSF', 'WS', 'SF'. Default: None
        - filename : str, optional
            Output file name for saving results

    Returns
    -------
    dict
        Dictionary containing storm event DataFrames with computed properties:
        - Peak values for each variable
        - Integrated (cumulative) values
        - Storm duration
        - Temporal information (start, end, peak time)
        - Seasonal classification (if class_type specified)

    Notes
    -----
    For each storm event, computes:
    
    - Maximum values (peaks)
    - Integrated values (area under curve)
    - Duration
    - Timing information
    - Season assignment (optional)
    
    Examples
    --------
    >>> info = {
    ...     'threshold': 1.5,
    ...     'min_duration': 6,
    ...     'inter_time': 24,
    ...     'data_timestep': 'H',
    ...     'interpolation': True,
    ...     'class_type': 'WSSF'
    ... }
    >>> storms = storm_properties(data, ['Hs', 'Tp', 'Dir'], info)
    """
    if not "interpolation" in info:
        info["interpolation"] = False

    if not "class_type" in info:
        info["class_type"] = "WSSF"

    if info["time_step"] == "D":
        strTimeStep, intDur = " days", "D"
    else:
        strTimeStep, intDur = " hours", "h"
    min_duration_td = pd.Timedelta(str(info["min_duration"]) + strTimeStep)
    min_interarrival_td = pd.Timedelta(str(info["inter_time"]) + strTimeStep)

    cycles_ini, calms_ini, _ = extreme_events(
        data,
        cols[0],
        info["threshold"],
        min_interarrival_td,
        min_duration_td,
        interpolation=info["interpolation"],
        truncate=True,
        extra_info=True,
    )

    # TODO: quitar cuando Pedro haya arreglado los indices de las calmas
    if cycles_ini[-1].index[-1] == data.index[-1]:
        del cycles_ini[-1]
        del calms_ini[-1]

    for ii, _ in enumerate(cycles_ini):
        if ii == len(cycles_ini) - 1:
            calms_ini[ii] = calms_ini[ii].rename(
                {calms_ini[ii].index[0]: cycles_ini[ii].index[-1]}
            )
        else:
            calms_ini[ii] = calms_ini[ii].rename(
                {
                    calms_ini[ii].index[0]: cycles_ini[ii].index[-1],
                    calms_ini[ii].index[-1]: cycles_ini[ii + 1].index[0],
                }
            )

    # Duration of storm and interarrival time
    cycles, calms = cycles_ini, calms_ini
    dur_cycles = events_duration(list(cycles))
    dur_calms = events_duration(list(calms))

    dur_cycles = pd.DataFrame(
        dur_cycles.values, index=dur_cycles.index, columns=["dur_storm"]
    )
    dur_cycles = class_storm_seasons(dur_cycles, info["class_type"])

    dur_cycles["dur_storm"] = dur_cycles["dur_storm"] / np.timedelta64(1, intDur)
    dur_calms = dur_calms / np.timedelta64(1, intDur)

    durs_storm_calm = pd.DataFrame(
        -1,
        index=np.arange(len(dur_cycles)),
        columns=["dur_storm", "dur_calms", "season"],
    )
    durs_storm_calm["season"] = dur_cycles["season"].values
    durs_storm_calm["dur_storm"] = dur_cycles["dur_storm"].values
    durs_storm_calm["dur_calms"] = dur_calms.values

    maxs, medians, ini, end = [], [], [], []
    for event in cycles:
        maxs.append(event.max())
        medians.append(event.median())
        ini.append(event.index[0])
        end.append(event.index[-1])

    durs_storm_calm["max_value"] = maxs
    durs_storm_calm["median_value"] = medians
    durs_storm_calm["storm_ini"] = ini
    durs_storm_calm["storm_end"] = end

    if "filename" in info.keys():
        save.to_csv(durs_storm_calm, info["filename"])

    return durs_storm_calm


def dependencies(df: pd.DataFrame, param: dict):
    """Fit temporal dependency structure using Vector Autoregression (VAR) model.
    
    Estimates multivariate temporal dependencies between environmental variables
    following Solari & van Gelder (2011) and Solari & Losada (2011) methodology.

    Parameters
    ----------
    df : pd.DataFrame
        Raw time series with datetime index containing all variables
    param : dict
        Parameters for dependency analysis with nested structure:
        
        - TD : dict
            Temporal dependency parameters:
            
            * vars : list
                Names of variables to include in dependency analysis
            * method : str
                Dependency method ('VAR' for Vector Autoregression)
            * order : int
                Order of the VAR model (lag length)
            * mvar : str, optional
                Main variable for event-based analysis
            * threshold : float, optional
                Threshold of main variable for event identification
            * events : bool, optional
                If True, analyze only storm events. Default: False
            * not_save_error : bool, optional
                If True, exclude error time series from output. Default: False
            * file_name : str, optional
                Output file name for saving results
                
        - {var_name} : dict (for each variable in vars)
            Marginal distribution parameters from fit_marginal_distribution

    Returns
    -------
    dict
        Dictionary with fitted VAR model parameters including:
        - Autoregression coefficients
        - Model order and diagnostics
        - Variable transformations
        - Error statistics

    Notes
    -----
    The function:
    
    1. Transforms variables to uniform marginals using fitted CDFs
    2. Fits VAR model to transformed data
    3. Estimates temporal dependency structure
    4. Saves results to JSON file if file_name specified
    
    For circular variables (e.g., directions), converts to radians before analysis.
    
    References
    ----------
    Solari, S., & van Gelder, P. H. A. J. M. (2011). On the use of vector 
    autoregressive (VAR) and regime switching VAR models for the simulation 
    of sea and wind state parameters. Probabilistic Engineering Mechanics.
    
    Solari, S., & Losada, M. A. (2011). A unified statistical model for 
    hydrological variables including the selection of threshold for the peak 
    over threshold method. Water Resources Research.
    
    Lira-Loarca, A., et al. (2020). A global classification of coastal flood
    hazard climates. Scientific Reports.

    Examples
    --------
    >>> param = {
    ...     'TD': {
    ...         'vars': ['Hs', 'Tp', 'Dir'],
    ...         'method': 'VAR',
    ...         'order': 3,
    ...         'events': False,
    ...         'file_name': 'dependency_results'
    ...     },
    ...     'Hs': {...},  # from fit_marginal_distribution
    ...     'Tp': {...},  # from fit_marginal_distribution
    ...     'Dir': {...}  # from fit_marginal_distribution
    ... }
    >>> df_dt = dependencies(df, param)
    """
    logger.info(show_init_message())

    logger.info("UNI/MULTIVARIATE & TEMPORAL DEPENDENCY")
    logger.info(
        "=============================================================================="
    )

    # Remove nan in the input timeseries
    df = pd.DataFrame(df).dropna()

    # Remove nans
    df.dropna(inplace=True)

    # Check that the input dictionary is well defined
    param["TD"] = check_dependencies_params(param["TD"])

    # Compute: (1) the univariate and temporal analysis is one variable is given,
    #          (2) the multivariate and temporal analysis is more than one is given
    logger.info(
        "Computing the parameters of the stationary {} model up to {} order.".format(
            param["TD"]["method"], param["TD"]["order"]
        )
    )
    logger.info(
        "=============================================================================="
    )

    variables_ = df.columns
    # Compute the normalize time using the maximum period
    df["n"] = (
        (df.index.dayofyear + df.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": df.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    # Transform angles into radians
    for var_ in param["TD"]["vars"]:
        if param[var_]["type"] == "circular":
            df[var_] = np.deg2rad(df[var_])

    cdf_ = pd.DataFrame(index=df.index, columns=param["TD"]["vars"])

    for var_ in param["TD"]["vars"]:
        param[var_]["order"] = np.max(param[var_]["mode"])
        param = utils.string_to_function(param, var_)

        variable = pd.DataFrame(df[var_].values, index=df.index, columns=["data"])
        variable["n"] = df["n"].values
        # variable[var_] = df[var_].values

        # Transformed timeserie
        if param[var_]["transform"]["make"]:
            variable["data"], _ = core.transform(variable["data"], param[var_])
            variable["data"] -= param[var_]["transform"]["min"]

        if "scale" in param[var_]:
            variable["data"] = variable["data"] / param[var_]["scale"]

        # Compute the CDF using the estimated parameters
        cdf_[var_] = core.cdf(variable, param[var_])

        # Remove outlayers
        if any(cdf_[var_] >= 1 - 1e-6):
            logger.info(
                "Casting {} probs of {} next to one (F({}) > 1-1e-6).".format(
                    str(np.sum(cdf_[var_] >= 1 - 1e-6)), var_, var_
                )
            )
            cdf_.loc[cdf_[var_] >= 1 - 1e-6, var_] = 1 - 1e-6

        if any(cdf_[var_] <= 1e-6):
            logger.info(
                "Casting {} probs of {} next to zero (F({}) < 1e-6).".format(
                    str(np.sum(cdf_[var_] <= 1e-6)), var_, var_
                )
            )
            cdf_.loc[cdf_[var_] <= 1e-6, var_] = 1e-6

        # If "events" is True, the conditional analysis over threshold following the
        # steps given in Lira-Loarca et al (2019) is applied
        if (var_ == param["TD"]["mvar"]) & param["TD"]["events"]:
            logger.info(
                "Computing conditioned probability models to the threshold of the main variable"
            )
            cdfj = cdf_[var_].copy()
            variable = pd.DataFrame(
                np.ones(len(df["n"])) * param["TD"]["threshold"],
                index=df.index,
                columns=["data"],
            )
            variable[var_] = df[var_].values
            if param[var_]["transform"]["make"]:
                variable[var_], _ = core.transform(variable[var_], param[var_])
                variable[var_] -= param[var_]["transform"]["min"]
                variable["data"], _ = core.transform(variable["data"], param[var_])
            variable["n"] = df["n"].values
            cdfu = core.cdf(variable, param[var_])
            cdf_umbral = pd.DataFrame(cdfu)
            cdf_umbral["n"] = variable["n"]
            cdf_[var_] = (cdfj - cdfu) / (1 - cdfu)

    # Remove nans in CDF
    if any(np.sum(np.isnan(cdf_))):
        logger.info(
            "Some Nan ("
            + str(np.sum(np.sum(np.isnan(cdf_))))
            + " values) are founds before the normalization."
        )
        cdf_[np.isnan(cdf_)] = 0.5

    if param["TD"]["method"] == "VAR":
        z_ = pd.DataFrame(-1, columns=variables_, index=df.index)
        z = np.zeros(np.shape(cdf_))
        # Normalize the CDF of every variable
        for ind_, var_ in enumerate(cdf_):
            z[:, ind_] = st.norm.ppf(cdf_[var_].values)
            z_.loc[:, var_] = st.norm.ppf(cdf_[var_].values)

        # Save simulation file
        if param["TD"]["save_z"]:
            save.to_txt(
                z,
                "z_values" + ".csv",
            )

        # Fit the parameters of the AR/VAR(p) model
        df_dt = fit_var_model(z_, param["TD"]["order"])
        for key_ in param["TD"].keys():
            df_dt[key_] = param["TD"][key_]
    else:
        logger.info("No more methods are yet available.")

    # Save to json file
    if not "file_name" in param["TD"].keys():
        os.makedirs("dependency", exist_ok=True)
        filename = "dependency/"
        for var_ in param["TD"]["vars"]:
            filename += var_ + "_"
        filename += str(param["TD"]["order"]) + "_"
        filename += param["TD"]["method"]
    else:
        filename = param["TD"]["file_name"]

    param["TD"]["file_name"] = filename

    if param["TD"]["not_save_error"]:
        df_dt.pop("y", None)
        df_dt.pop("y*", None)

    save.to_json(df_dt, param["TD"]["file_name"], True)

    # Clean memory usage
    del cdf_, param

    return df_dt


def check_dependencies_params(param: dict):
    """Checks the input parameters and includes some required arguments for the computation of multivariate dependencies

    Args:
        * param (dict): the initial guess parameters of the probability models

    Returns:
        * param (dict): checked and updated parameters
    """

    logger.info("USER OPTIONS:")
    logger.info(
        "==============================================================================\n"
    )
    k = 1

    if not "method" in param.keys():
        param["method"] = "VAR"
        logger.info(str(k) + " - VAR method used")
        k += 1

    if not "not_save_error" in param.keys():
        param["not_save_error"] = True

    if not "events" in param.keys():
        param["events"] = False

    if not "mvar" in param.keys():
        param["mvar"] = None

    if not "save_z" in param.keys():
        param["save_z"] = False

    logger.info(
        "==============================================================================\n"
    )
    global text_warning
    text_warning = True

    return param


def fit_var_model(data: np.ndarray, order: int):
    """Computes the coefficients of the VAR(p) model and chooses the model with lowest BIC.

    Args:
        * data (np.ndarray): normalize data with its probability model
        * order (int): maximum order (p) of the VAR model

    Returns:
        * par_dt (dict): parameter of the temporal dependency using VAR model
    """
    # Create the list of output parameters
    data_ = data.values.T
    [dim, t] = np.shape(data_)
    t = t - order
    bic, r2adj = np.zeros(order), []

    par_dt = [list() for i in range(order)]
    for p in range(1, order + 1):
        # Create the matrix of input data for p-order
        y = data_[:, order:]
        z0 = np.zeros([p * dim, t])
        for i in range(1, p + 1):
            z0[(i - 1) * dim : i * dim, :] = data_[:, order - i : -i]
        z = np.vstack((np.ones(t), z0))
        # Estimated the parameters using the ordinary least squared error analysis
        par_dt[p - 1], bic[p - 1], r2a = varfit_OLS(y, z)
        r2adj.append(r2a)
        if dim == 1:
            # Use values only to avoid datetime index issues
            model = AR(data.iloc[:, 0].values, lags=p)
            res = model.fit()
        else:
            # Use values only to avoid datetime index issues
            model = VAR(data.values)
            res = model.fit(maxlags=p)
        # print(res.summary())

        # Computed using statmodels
        par_dt[p - 1]["B"] = res.params.T
        par_dt[p - 1]["U"] = y - np.dot(res.params.T, z)
        # Estimate de covariance matrix
        par_dt[p - 1]["Q"] = np.cov(par_dt[p - 1]["U"])
        bic[p - 1] = res.bic
        # lag_order = res.k_ar

    # Select the minimum BIC and return the parameter associated to it
    id_ = np.argmin(bic)
    par_dt = par_dt[id_]
    par_dt["id"] = int(id_)
    par_dt["bic"] = [float(bicValue) for bicValue in bic]
    par_dt["R2adj"] = r2adj[par_dt["id"]]
    logger.info(
        "Minimum BIC ("
        + str(par_dt["bic"][par_dt["id"]])
        + ") obtained for p-order "
        + str(par_dt["id"] + 1)  # Python starts at zero
        + " and R2adj: "
        + str(par_dt["R2adj"])
    )
    logger.info(
        "=============================================================================="
    )

    if id_ + 1 == order:
        logger.info("The lower BIC is in the higher order model. Increase the p-order.")

    return par_dt


def varfit_OLS(y, z):
    """Estimates the parameters of VAR using the RMSE described in Lutkepohl (ecs. 3.2.1 and 3.2.10)

    Args:
        * y: X Matrix in Lutkepohl
        * z: Z Matrix in Lutkepohl

    Returns:
        * df (dict): matrices B, Q, y U
        * bic (float): Bayesian Information Criteria
        * R2adj (float): correlation factor
    """

    df = dict()
    m1, m2 = np.dot(y, z.T), np.dot(z, z.T)

    # Estimate the parameters
    df["B"] = np.dot(m1, np.linalg.inv(m2))

    nel, df["dim"] = np.shape(df["B"].T)
    df["U"] = y - np.dot(df["B"], z)
    # Estimate de covariance matrix
    df["Q"] = np.cov(df["U"])
    df["y"] = y
    if df["dim"] == 1:
        error_ = np.random.normal(np.zeros(df["dim"]), df["Q"], z.shape[1]).T
    else:
        error_ = np.random.multivariate_normal(
            np.zeros(df["dim"]), df["Q"], z.shape[1]
        ).T
    df["y*"] = np.dot(df["B"], z) + error_

    # Estimate R2 and R2-adjusted parameters
    R2 = np.sum((df["y*"] - np.mean(y)) ** 2, axis=1) / np.sum(
        (y - np.mean(y)) ** 2, axis=1
    )
    R2adj = 1 - (1 - R2) * (len(z.T) - 1) / (len(z.T) - nel - 1)

    # rmse = np.sqrt(np.sum((st.norm.cdf(y) - st.norm.cdf(np.dot(df["B"], z))) ** 2, axis=1)/y.shape[1])
    # mae = np.sum(np.abs(st.norm.cdf(y) - st.norm.cdf(np.dot(df["B"], z))), axis=1)/y.shape[1]
    # logger.info(rmse, mae)

    # Compute the LLF
    multivariatePdf = st.multivariate_normal.pdf(
        df["U"].T, mean=np.zeros(df["dim"]), cov=df["Q"]
    )
    mask = multivariatePdf > 0

    global text_warning
    if len(multivariatePdf) != len(multivariatePdf[mask]):
        if text_warning:
            logger.info(
                "Casting {} zero-values of the multivariate pdf. Removed.".format(
                    str(np.sum(~mask))
                )
            )
            text_warning = False

        llf = np.sum(np.log(multivariatePdf[mask]))
    else:
        llf = np.sum(np.log(multivariatePdf))

    # aic = df['dim']*np.log(np.sum(np.abs(y - np.dot(df['B'], z)))) + 2*nel
    # Compute the BIC
    bic = -2 * llf + np.log(np.size(y)) * np.size(np.hstack((df["B"], df["Q"])))

    return df, bic, R2adj.tolist()


def ensemble_dt(models: dict, percentiles="equally"):
    """Compute ensemble of temporal dependency parameters from multiple models.
    
    Combines VAR model parameters from multiple Regional Climate Models (RCMs)
    or different model realizations using weighted or equal averaging.

    Parameters
    ----------
    models : dict
        Dictionary with model identifiers as keys and file paths as values.
        Each file should contain temporal dependency parameters (B, Q matrices)
    percentiles : str or list, optional
        Weighting scheme for ensemble:
        
        - 'equally': Equal weight for all models (default)
        - list of float: Weight for each model (must sum to 1)

    Returns
    -------
    dict
        Ensemble temporal dependency parameters containing:
        
        - B : np.ndarray
            Averaged autoregression coefficient matrix
        - Q : np.ndarray
            Averaged covariance matrix of residuals
        - id : int
            Model order

    Notes
    -----
    The ensemble is computed by:
    
    1. Reading B and Q matrices from each model
    2. Aligning matrix dimensions (padding with zeros if needed)
    3. Computing weighted average (or simple average if 'equally')
    
    For equal weights: B_ensemble = mean(B_i)
    
    For custom weights: B_ensemble = sum(w_i * B_i)

    Examples
    --------
    >>> models = {
    ...     'model1': 'path/to/model1.json',
    ...     'model2': 'path/to/model2.json',
    ...     'model3': 'path/to/model3.json'
    ... }
    >>> # Equal weighting
    >>> ensemble = ensemble_dt(models, percentiles='equally')
    >>> 
    >>> # Custom weighting
    >>> ensemble = ensemble_dt(models, percentiles=[0.5, 0.3, 0.2])
    """
    # Initialize matrices
    B, Q = [], []
    # Read the parameter of every ensemble model
    for model_ in models.keys:
        df_dt = read.rjson(models[model_], "td")
        B.append(df_dt["B"])
        Q.append(df_dt["Q"])

    nmodels = len(B)
    norders = np.max([np.shape(Bm) for Bm in B], axis=0)

    Bs = np.zeros([norders[0], norders[1], nmodels])

    # Compute the ensemble using percentiles
    for i in range(nmodels):
        if percentiles == "equally":
            Bs[:, : np.shape(B[i])[1], i] = B[i]
        else:
            Bs[:, : np.shape(B[i])[1], i] = B[i] * percentiles[i]

    # Compute the ensemble using percentiles
    if percentiles == "equally":
        B, Q = np.mean(Bs, axis=2), np.mean(Q, axis=0)
    else:
        B, Q = np.sum(Bs, axis=2), np.sum(Q, axis=0)

    # Create a dictionary with parameters of the ensemble
    df_dt_ensemble = dict()
    df_dt_ensemble["B"], df_dt_ensemble["Q"], df_dt_ensemble["id"] = (
        B,
        Q,
        int((norders[1] - 1) / norders[0]),
    )  # ord_

    # Create the fit directory and save the parameters to a json file
    os.makedirs("fit", exist_ok=True)
    save.to_json(df_dt_ensemble, "fit/ensemble_df_dt", True)
    return B, Q, int((norders[1] - 1) / norders[0])


def iso_indicators(
    indicators: str,
    reference: pd.DataFrame,
    variable: str,
    param: dict = None,
    data: pd.DataFrame = None,
    daysWindowsLength: int = 14,
    pemp: list = None,
):
    """Compute indicators from iso-probability lines of non-stationary CDF.
    
    Calculates climate change indicators by tracking iso-probability contours
    (e.g., 95th percentile) through time in non-stationary distributions.

    Parameters
    ----------
    indicators : str or list
        Indicator types to compute. Can be single string or list of strings.
        Examples: 'mean', 'p95', 'p99', 'max'
    reference : pd.DataFrame
        Reference time series with datetime index for baseline period
    variable : str
        Name of variable column to analyze
    param : dict, optional
        Non-stationary distribution parameters with keys:
        
        - basis_period : list or None
            Time periods for non-stationary analysis
            
        If None, uses empirical non-stationary CDF. Default: None
    data : pd.DataFrame, optional
        Additional data for empirical analysis. Default: None
    daysWindowsLength : int, optional
        Window length (days) for moving window CDF estimation. Default: 14
    pemp : list, optional
        Pre-computed empirical percentiles. If None, computed from data.
        Default: None

    Returns
    -------
    dict or pd.DataFrame
        Computed indicators with temporal evolution of iso-probability lines

    Notes
    -----
    Iso-probability lines track how a specific quantile (e.g., 95th percentile)
    changes over time in non-stationary conditions. Useful for:
    
    - Assessing climate change impacts
    - Detecting trends in extreme events
    - Comparing baseline vs future scenarios
    
    Uses moving window approach to estimate time-varying CDF, then extracts
    specified quantiles at each time step.

    Examples
    --------
    >>> indicators = iso_indicators(
    ...     indicators=['p95', 'p99'],
    ...     reference=historical_data,
    ...     variable='wave_height',
    ...     param=fitted_params,
    ...     daysWindowsLength=30
    ... )
    """

    if not isinstance(indicators, list):
        indicators = [indicators]

    if param is not None:
        if param["basis_period"] is not None:
            T = np.max(param["basis_period"])
        emp_non_st = False
    else:
        emp_non_st = True
        T = 1
        data["n"] = np.fmod(
            (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
            .total_seconds()
            .values
            / (T * 365.25 * 24 * 3600),
            1,
        )

    reference["n"] = np.fmod(
        (reference.index - datetime.datetime(reference.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    dt = 366
    n = np.linspace(0, 1, dt)
    if pemp is None:
        xp, pemp = utils.nonstationary_ecdf(
            reference, variable, wlen=daysWindowsLength / (365.25 * T), pemp=pemp
        )
    else:
        xp, pemp = utils.nonstationary_ecdf(
            reference, variable, wlen=daysWindowsLength / (365.25 * T)
        )

    # A empirical model
    if emp_non_st:
        data_check, _ = utils.nonstationary_ecdf(
            data, variable, wlen=daysWindowsLength / (365.25 * T), pemp=pemp
        )
    else:
        # A theoretical model
        # ----------------------------------------------------------------------------------
        for j, i in enumerate(pemp):
            if not emp_non_st:
                if param["transform"]["plot"]:
                    xp[i], _ = core.transform(xp[[i]], param)
                    xp[i] -= param["transform"]["min"]
                    if "scale" in param:
                        xp[i] = xp[i] / param["scale"]

                param = utils.string_to_function(param, None)
        data_check = pd.DataFrame(0, index=n, columns=pemp)

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
                res[param["var"]] = core.inverse_transform(res[[param["var"]]], param)
            elif ("scale" in param) & (not param["transform"]["plot"]):
                res[param["var"]] = res[param["var"]] * param["scale"]

            data_check[j] = res[param["var"]]

    # ----------------------------------------------------------------------------------
    results = pd.DataFrame(-1.0, index=pemp, columns=[indicators])
    for indicator in indicators:
        if indicator == "rmse":
            for j in pemp:
                results.loc[j, indicator] = utils.rmse(xp[j], data_check[j])
        elif indicator == "maximum_absolute_error":
            for j in pemp:
                results.loc[j, indicator] = utils.maximum_absolute_error(
                    xp[j], data_check[j]
                )
        elif indicator == "mean_absolute_error":
            for j in pemp:
                results.loc[j, indicator] = utils.mean_absolute_error(
                    xp[j], data_check[j]
                )

    return results


def confidence_bands(rmse, n, confidence_level):
    """Calculate confidence bands for model predictions using RMSE.
    
    Computes confidence intervals around model predictions based on
    Root Mean Square Error and sample size using Student's t-distribution.

    Parameters
    ----------
    rmse : float
        Root Mean Square Error of the model predictions
    n : int
        Number of data points used in the model
    confidence_level : float
        Confidence level for the interval (e.g., 0.95 for 95% confidence)

    Returns
    -------
    float
        Half-width of the confidence band (margin of error)

    Notes
    -----
    The confidence band is calculated as:
    
    margin = t_critical * (rmse / sqrt(n))
    
    where t_critical is the critical value from Student's t-distribution
    for the specified confidence level and n-1 degrees of freedom.
    
    The prediction interval is then: [prediction - margin, prediction + margin]

    Examples
    --------
    >>> margin = confidence_bands(rmse=0.5, n=100, confidence_level=0.95)
    >>> # Prediction interval: [y_pred - margin, y_pred + margin]
    """

    # Paso 2: Calcular el error estándar de los residuos
    ser = rmse / np.sqrt(n)

    # Paso 4: Calcular el valor crítico de la distribución t de Student
    degrees_of_freedom = n - 1
    alpha = 1 - confidence_level
    t_critical = st.t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Paso 5: Calcular el margen de error
    margin_of_error = t_critical * ser

    return margin_of_error


def generate_outputfilename(parameters):
    """_summary_

    Args:
        parameters (_type_): _description_
    """

    filename = parameters["var"] + "_" + str(parameters["fun"][0])
    for i in range(1, parameters["no_fun"]):
        filename += "_" + str(parameters["fun"][i])
    filename += "_genpareto" * parameters["reduction"]

    # for i in parameters["ws_ps"]:
    # filename += "_" + str(i)

    filename += "_st_" * (not parameters["non_stat_analysis"])
    filename += "_nonst" * parameters["non_stat_analysis"]

    filename += "_" + str(parameters["basis_period"][0])

    filename += "_" + parameters["basis_function"]["method"]
    if "no_terms" in parameters["basis_function"].keys():
        filename += "_" + str(parameters["basis_function"]["no_terms"])
    else:
        filename += "_" + str(parameters["basis_function"]["degree"])
    filename += "_" + parameters["optimization"]["method"]

    parameters["file_name"] = "marginalfit/" + filename + ".json"
    return
