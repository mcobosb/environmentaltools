import numpy as np
import pandas as pd


def extreme_events(data, var_name, threshold, minimum_interarrival_time, minimum_cycle_length, interpolation=False,
                   interpolation_method='linear', interpolation_freq='1min', truncate=False, extra_info=False):
    """Extract storm and calm cycles from time series data using threshold-based detection.
    
    Identifies independent extreme events (storms) and calm periods by detecting
    threshold crossings, merging nearby events, and filtering short-duration events.
    Optionally interpolates crossing times for improved temporal resolution.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index
    var_name : str
        Name of the variable column to analyze
    threshold : float
        Threshold value separating storm (cycle) and calm periods
    minimum_interarrival_time : pd.Timedelta
        Minimum time between storms to ensure statistical independence
    minimum_cycle_length : pd.Timedelta
        Minimum storm duration to consider it valid
    interpolation : bool, default=False
        Enable interpolation to estimate precise threshold crossing times
    interpolation_method : str, default='linear'
        Interpolation method: 'linear', 'cubic', 'nearest', etc.
    interpolation_freq : str, default='1min'
        Frequency for interpolation grid (e.g., '1min', '10s', '1h')
    truncate : bool, default=False
        Clip storm values below threshold and calm values above threshold
        to the threshold value
    extra_info : bool, default=False
        Return additional diagnostic information

    Returns
    -------
    cycles : list of pd.Series
        List of storm cycles, each as a Series with values and timestamps
    calm_periods : list of pd.Series
        List of calm periods between storms
    info : dict, optional
        Additional information (only if extra_info=True):
        
        - 'data_cycles' : pd.DataFrame
            Data for all storm periods
        - 'data_calm_periods' : pd.DataFrame
            Data for all calm periods
        - 'cycles_indexes_clipped' : list
            Indices where storm values were clipped (if truncate=True)
        - 'calm_periods_indexes_clipped' : list
            Indices where calm values were clipped (if truncate=True)
        - 'interpolation_indexes' : np.ndarray
            Indices of interpolated threshold crossings (if interpolation=True)

    Notes
    -----
    The algorithm follows these steps:
    
    1. Identify initial threshold crossings
    2. Optionally interpolate to refine crossing times
    3. Merge nearby storms (within minimum_interarrival_time)
    4. Remove short storms (< minimum_cycle_length)
    5. Optionally truncate values to threshold
    
    Storm independence is crucial for extreme value analysis. The
    minimum_interarrival_time should be chosen based on the decorrelation
    time of the process (typically 3-5 times the characteristic timescale).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Hs': [1.2, 2.5, 3.1, 2.8, 1.5, 0.8, 1.0, 2.9, 3.5, 2.1]
    ... }, index=pd.date_range('2020', periods=10, freq='6h'))
    >>> cycles, calms = extreme_events(
    ...     df, 'Hs', threshold=2.0,
    ...     minimum_interarrival_time=pd.Timedelta('12h'),
    ...     minimum_cycle_length=pd.Timedelta('6h')
    ... )
    >>> print(f"Found {len(cycles)} storm events")
    """
    full_data = data.copy()
    # noinspection PyUnresolvedReferences
    var_data = full_data[var_name]
    cycles_start, cycles_end = values_over_threshold(var_data, threshold)

    # If data ends with a cycle, it is necessary to remove it
    # noinspection PyTypeChecker
    if len(var_data.index) == cycles_end[-1]:
        var_data = var_data[:cycles_start[-1]]
        cycles_start = cycles_start[:-1]
        cycles_end = cycles_end[:-1]

    new_cycles_limits_indexes = None
    if interpolation:
        new_cycles_limits_indexes = interpolation_boundaries_index(var_data, cycles_start, cycles_end,
                                                                               interpolation_freq, interpolation_method,
                                                                               threshold)
        new_cycles_limits = pd.DataFrame(threshold, index=new_cycles_limits_indexes, columns=[var_name])

        # noinspection PyUnresolvedReferences
        full_data = full_data.combine_first(new_cycles_limits)
        full_data = full_data.interpolate(interpolation_method)

        var_data = full_data[var_name]
        cycles_start, cycles_end = values_over_threshold(var_data, threshold)

    # Find near cycles (not independent cycles)
    near_cycles = near_events(var_data, cycles_start, cycles_end, minimum_interarrival_time, interpolation)

    while np.any(near_cycles):
        # Join near cycles
        cycles_start = cycles_start[np.append(True, ~near_cycles)]
        cycles_end = cycles_end[np.append(~near_cycles, True)]

        # Find if there are more near cycles
        near_cycles = near_events(var_data, cycles_start, cycles_end, minimum_interarrival_time,
                                              interpolation)

    # Remove short cycles
    cycles_length = var_data.index[cycles_end - 1] - var_data.index[cycles_start]
    short_cycles = cycles_length < minimum_cycle_length

    cycles_start = cycles_start[~short_cycles]
    cycles_end = cycles_end[~short_cycles]

    cycles_indexes_clipped = None
    calm_periods_indexes_clipped = None
    cycles_mask = extreme_indexes(var_data, cycles_start, cycles_end)
    if truncate:
        lower_values_clipped = var_data[cycles_mask].clip(lower=threshold)
        upper_values_clipped = var_data[~cycles_mask].clip(upper=threshold)

        cycles_indexes_clipped = lower_values_clipped != var_data[cycles_mask]
        calm_periods_indexes_clipped = upper_values_clipped != var_data[~cycles_mask]
        # noinspection PyUnresolvedReferences
        cycles_indexes_clipped = cycles_indexes_clipped[cycles_indexes_clipped].index.tolist()
        # noinspection PyUnresolvedReferences
        calm_periods_indexes_clipped = calm_periods_indexes_clipped[calm_periods_indexes_clipped].index.tolist()

        var_data[cycles_mask] = lower_values_clipped
        var_data[~cycles_mask] = upper_values_clipped

    # Split cycles and calm periods
    cross_indexes = np.sort(np.concatenate([cycles_start, cycles_end]))
    cross_indexes = cross_indexes[cross_indexes != 0]  # avoid splitting by 0 index

    data_splitted = np.split(var_data, cross_indexes)

    # Check if var_data starts with a cycle or a calm period
    if not np.any(cycles_start):
        cycles = []
        calm_periods = []

        cycles_indexes = []
        calm_periods_indexes = []

    elif cycles_start[0] == 0:
        cycles = data_splitted[0::2]
        calm_periods = data_splitted[1::2]

        cycles_indexes = var_data[cycles_mask].index
        calm_periods_indexes = var_data[~cycles_mask].index
    else:
        cycles = data_splitted[1::2]
        calm_periods = data_splitted[2::2]

        cycles_indexes = var_data[cycles_mask].index
        calm_periods_indexes = var_data[~cycles_mask].index.difference(data_splitted[0].index)

    if extra_info:
        info = dict()

        info['data_cycles'] = full_data.loc[cycles_indexes]
        info['data_calm_periods'] = full_data.loc[calm_periods_indexes]

        if truncate:
            info['cycles_indexes_clipped'] = cycles_indexes_clipped
            info['calm_periods_indexes_clipped'] = calm_periods_indexes_clipped

        if interpolation:
            info['interpolation_indexes'] = new_cycles_limits_indexes

        return cycles, calm_periods, info
    else:
        return cycles, calm_periods


def events_duration(events):
    """Calculate duration of each event (storm or calm period).
    
    Computes the time span of each event from its first to last timestamp.

    Parameters
    ----------
    events : list of pd.Series
        List of events where each element is a Series with values and 
        datetime index representing the event time evolution

    Returns
    -------
    pd.Series
        Duration of each event (as timedelta) with the event start time
        as index

    Examples
    --------
    >>> events = [
    ...     pd.Series([2.5, 3.0, 2.8], index=pd.date_range('2020-01-01', periods=3, freq='1h')),
    ...     pd.Series([2.2, 2.7], index=pd.date_range('2020-01-05', periods=2, freq='1h'))
    ... ]
    >>> durations = events_duration(events)
    >>> print(durations)
    """
    events_stacked = pd.DataFrame(events).T

    start = events_stacked.apply(pd.Series.first_valid_index)
    end = events_stacked.apply(pd.Series.last_valid_index)

    events_length = end - start

    duration = pd.Series(events_length.values, index=start.values)

    return duration


def values_over_threshold(data, threshold):
    """Identify indices where data exceeds threshold using sparse arrays.
    
    Efficiently detects contiguous blocks of values above threshold using
    pandas sparse arrays for memory efficiency.

    Parameters
    ----------
    data : pd.Series
        Time series data
    threshold : float
        Threshold value for event detection

    Returns
    -------
    cycles_start : np.ndarray
        Integer indices marking the start of each exceedance block
    cycles_end : np.ndarray
        Integer indices marking the end (exclusive) of each exceedance block

    Notes
    -----
    Uses pandas sparse block arrays for efficient storage and detection
    of contiguous regions above threshold.

    Examples
    --------
    >>> data = pd.Series([1.0, 2.5, 3.0, 2.8, 1.5, 0.8, 2.9, 3.1])
    >>> start, end = values_over_threshold(data, threshold=2.0)
    >>> print(f"Exceedance blocks: {list(zip(start, end))}")
    """
    # Find cycles starting and ending positions
    # noinspection PyTypeChecker
    cycles_data = pd.arrays.SparseArray(data.where(data >= threshold), kind='block')

    cycles_start = cycles_data.sp_index.blocs
    cycles_duration = cycles_data.sp_index.blengths
    cycles_end = cycles_start + cycles_duration

    return cycles_start, cycles_end


def interpolation_series(data, index_start, index_end):
    """Prepare data for interpolation between cycle boundaries.
    
    Creates a DataFrame with start and end values grouped for subsequent
    interpolation across threshold crossings.

    Parameters
    ----------
    data : pd.Series
        Time series data
    index_start : array-like
        Indices just before threshold crossings
    index_end : array-like
        Indices just after threshold crossings

    Returns
    -------
    pd.DataFrame
        Combined data with 'value' and 'group' columns for interpolation

    Notes
    -----
    Groups are created to interpolate each crossing independently.
    """
    start = data[index_start]
    end = data[index_end]

    group = start.index.asi8

    interpolation_start = start.to_frame('value')
    interpolation_start['group'] = group

    interpolation_end = end.to_frame('value')
    interpolation_end['group'] = group

    interpolation = pd.concat([interpolation_start, interpolation_end])

    return interpolation


def interpolation_boundaries_index(data, cycles_start, cycles_end, interpolation_freq, interpolation_method, threshold):
    """Find precise threshold crossing times through interpolation.
    
    Interpolates between data points surrounding threshold crossings to
    estimate the exact time when the threshold was crossed.

    Parameters
    ----------
    data : pd.Series
        Time series data with datetime index
    cycles_start : np.ndarray
        Indices of cycle start positions
    cycles_end : np.ndarray
        Indices of cycle end positions
    interpolation_freq : str
        Temporal frequency for interpolation grid (e.g., '1min', '10s')
    interpolation_method : str
        Interpolation method: 'linear', 'cubic', 'quadratic', etc.
    threshold : float
        Threshold value to find crossing times for

    Returns
    -------
    np.ndarray
        DatetimeIndex values of estimated threshold crossing times

    Notes
    -----
    The function:
    1. Creates dense time grid at interpolation_freq around crossings
    2. Interpolates data values onto this grid
    3. Finds timestamps closest to threshold value
    
    This provides sub-sample temporal resolution for event boundaries.
    """
    if cycles_start[0] == 0:
        interpolation_start = interpolation_series(data, cycles_start[1:]-1, cycles_start[1:])
    else:
        interpolation_start = interpolation_series(data, cycles_start-1, cycles_start)

    interpolation_end = interpolation_series(data, cycles_end-1, cycles_end)

    interpolation_df = pd.concat([interpolation_start, interpolation_end])
    interpolation_df.index.name = 'time'

    interpolation_groups = interpolation_df.groupby('group').resample(interpolation_freq).asfreq()
    interpolation = interpolation_groups['value'].interpolate(interpolation_method)

    new_cycles_limits_indexes = interpolation.reset_index('time').groupby(level='group', group_keys=False).apply(
        interpolation_nearest, threshold)['time'].values

    new_cycles_limits_indexes = np.unique(new_cycles_limits_indexes)

    return new_cycles_limits_indexes


def interpolation_nearest(group, threshold):
    """Find nearest interpolated value to threshold.
    
    Helper function to locate the timestamp where interpolated values
    are closest to the threshold value.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame with 'value' column containing interpolated data
    threshold : float
        Target threshold value

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with the value closest to threshold

    Notes
    -----
    Uses absolute difference to find the nearest match.
    Returns the row with minimum absolute difference (value - threshold).
    """
    return group.iloc[(group['value']-threshold).abs().argsort()[:1]]


def extreme_indexes(data, extreme_start, extreme_end):
    """Create boolean mask for extreme event periods.
    
    Generates a boolean array indicating which time steps fall within
    any extreme event period.

    Parameters
    ----------
    data : pd.Series
        Time series data (used for length reference)
    extreme_start : np.ndarray
        Array of event start indices
    extreme_end : np.ndarray
        Array of event end indices (exclusive)

    Returns
    -------
    np.ndarray
        Boolean array where True indicates time steps within extreme events

    Notes
    -----
    The function:
    1. Creates condition matrix checking if each index falls within any event
    2. Uses broadcasting: (n_events, n_timesteps) comparison
    3. Returns True if index >= start AND < end for any event
    
    Examples
    --------
    >>> data = pd.Series(range(100))
    >>> starts = np.array([10, 50])
    >>> ends = np.array([20, 60])
    >>> mask = extreme_indexes(data, starts, ends)
    >>> mask[15]  # Within first event
    True
    >>> mask[30]  # Between events
    False
    """
    indexes = np.arange(len(data))

    condition = (indexes >= extreme_start[:, np.newaxis]) & (indexes < extreme_end[:, np.newaxis])
    extreme = np.any(condition, axis=0)

    return extreme


def near_events(data, cycles_start, cycles_end, minimum_interarrival_time, interpolation):
    """Identify non-independent events for merging.
    
    Determines which consecutive events are separated by less than the
    minimum interarrival time, indicating they should be merged into
    single events.

    Parameters
    ----------
    data : pd.Series
        Time series data with datetime index
    cycles_start : np.ndarray
        Array of cycle start indices
    cycles_end : np.ndarray
        Array of cycle end indices
    minimum_interarrival_time : pd.Timedelta
        Minimum time gap required between independent events
    interpolation : bool
        Whether interpolation was used for threshold crossings

    Returns
    -------
    np.ndarray
        Boolean array indicating which events are too close to the previous event

    Notes
    -----
    Implements the independence criterion for extreme events:
    
    - Events separated by less than minimum_interarrival_time are considered
      part of the same storm system
    - Typically set based on decorrelation time scale of the process
    
    The interpolation flag adjusts index selection:
    
    - True: Uses exact crossing times (cycles_end - 1)
    - False: Uses data sample indices (cycles_start - 1, cycles_end)
    
    Examples
    --------
    >>> minimum_gap = pd.Timedelta(hours=72)
    >>> near = near_events(data, starts, ends, minimum_gap, interpolation=False)
    >>> # Merge events where near is True
    """
    if interpolation:
        near_cycles = data.index[cycles_start[1:]] - data.index[cycles_end[:-1] - 1] < minimum_interarrival_time
    else:
        near_cycles = data.index[cycles_start[1:] - 1] - data.index[cycles_end[:-1]] < minimum_interarrival_time

    return near_cycles
