import pandas as pd 
import numpy as np 
from environmentaltools.temporal import analysis
from environmentaltools.graphics import temporal
from environmentaltools.common import read


var_ = 'DirM'
  
# Load data
csv_path = 'src/environmentaltools/data/temporal/marginal_fit/directional_data.csv'
df_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

data_fit = analysis.add_noise_to_array(df_data, [var_])

# Testing three normal distributions
ws_ps = [0.49,0.86]
params = {
    "var": var_,
    "fun":  {0: "norm", 1: "norm", 2: "norm"},
    "type": "circular",
    "circular" : True,
    "non_stat_analysis": True,
    "basis_function": {"method": "trigonometric", "no_terms" : 5},
    "fix_percentiles":True,
    "ws_ps": ws_ps,
    'file_name': f'src/environmentaltools/data/temporal/marginal_fit/fit_params_{var_}_CNRM_norm_trunc.json',
}

params[var_] = analysis.marginalfit(data_fit, params)
params = read.read_json(f'src/environmentaltools/data/temporal/marginal_fit/fit_params_{var_}_CNRM_norm_trunc.json')
temporal.nonstationary_cdf(
    data_fit,
    var_,
    params,
    date_axis=True,
    pemp=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95]),
    file_name= 'src/environmentaltools/data/temporal/marginal_fit/norm_trunc.pdf',
)

# Testing two wrapped normal distribution
ws_ps = [0.49]
params = {
    "var": var_,
    "fun":  {0: "wrap_norm", 1: "wrap_norm"},
    "type": "circular",
    "non_stat_analysis": True,
    "basis_function": {"method": "trigonometric", "no_terms" : 4},
    # "fix_percentiles":True,
    "ws_ps": ws_ps,
    'file_name': f'src/environmentaltools/data/temporal/marginal_fit/fit_params_{var_}_CNRM_wrap_norm.json',
}

params[var_] = analysis.fit_marginal_distribution(data_fit, params, True)
params = read.read_json(f'src/environmentaltools/data/temporal/marginal_fit/fit_params_{var_}_CNRM_wrap_norm.json')
temporal.nonstationary_cdf(
    data_fit,
    var_,
    params,
    date_axis=True,
    pemp=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90]),
    file_name= 'src/environmentaltools/data/temporal/marginal_fit/wrap_norm.pdf',
)
