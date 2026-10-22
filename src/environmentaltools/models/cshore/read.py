"""Reading of CSHORE output files (``O*``)."""

from pathlib import Path

import numpy as np
import pandas as pd

# Columns after the cross-shore distance x (m), from subroutine OUTPUT of CSHORE.
# Quantities with "sta" are normalized: by ρg (Sxx, τb, energy) or by the depth (σ*).
COLUMNS = {
    "bprof": ["zb"],                                  # bed elevation (m)
    "setup": ["eta", "h", "sigma"],                   # mean water level, mean depth, std of η (m)
    "param": ["T", "Qb", "sigma_sta"],                # period (s), fraction of breaking waves, σ/h
    "xmome": ["Sxx_sta", "taubx_sta"],                # radiation stress and bottom stress (cross-shore)
    "ymome": ["Sxy_sta", "tauby_sta"],                # radiation stress and bottom stress (longshore)
    "energ": ["EF_sta", "Db_sta", "Df_sta"],          # energy flux/T, breaking and friction dissipation
    "xvelo": ["U", "Ustd"],                           # mean cross-shore velocity and std (m/s)
    "yvelo": ["sin_theta", "V", "Vstd"],              # sin of the wave angle, mean longshore velocity and std
    "rolle": ["Rq"],                                  # roller volume flux (m2/s)
    "bsusl": ["Pb", "Ps", "Vs"],                      # bedload and suspension probabilities, suspended volume (m)
    "porus": ["Upstd", "Dp_sta"],                     # porous flow
    "cross": ["Qbx", "Qsx", "Qx"],                    # cross-shore bed, suspended and total transport (m2/s)
    "longs": ["Qby", "Qsy", "Qy"],                    # longshore bed, suspended and total transport (m2/s)
    "crvol": ["Vbx", "Vsx", "Vx"],                    # cumulative cross-shore volumes (m2)
    "lovol": ["Vby", "Vsy", "Vy"],                    # cumulative longshore volumes (m2)
    "swash": ["Pwet"],                                # wet probability
    "swase": ["he", "Ue", "Qe"],                      # swash depth, velocity and discharge
    "diker": ["Edike", "D_sta", "Dsum"],              # dike erosion
}
TIMESERIES = ["line", "time", "q0", "qbx_lw", "qsx_lw"]   # OTIMSE: overtopping and transport at the landward end


def read_output(file_type: str, path: str, time=None, line: int = 1):
    """Read a CSHORE output file ``O<FILE_TYPE>``.

    CSHORE writes one block per output time: a header ``line  n  time`` and
    ``n`` rows ``x  values...``.

    Args:
        file_type (str): Output type: one of :data:`COLUMNS` (``"bprof"``,
            ``"setup"``, ``"param"``, ``"xvelo"``, ``"yvelo"``, ``"bsusl"``,
            ``"cross"``, ``"longs"``, ...) or ``"timse"``.
        path (str): Folder with the CSHORE outputs.
        time (float | str | None): Output time (s), ``"first"`` or ``"last"``;
            None returns every time.
        line (int): Cross-shore line.

    Returns:
        pd.DataFrame: Indexed by ``(time, x)`` (or by ``x`` when ``time`` is
        given). For ``"timse"``, one row per time.
    """
    file_type = file_type.lower()
    lines = (Path(path) / f"O{file_type.upper()}").read_text().split("\n")

    if file_type == "timse":
        df = pd.DataFrame([[float(v) for v in s.split()] for s in lines if s.strip()])
        df = df.iloc[:, : len(TIMESERIES)]
        df.columns = TIMESERIES[: df.shape[1]]
        return df[df["line"] == line].drop(columns="line").set_index("time")

    names = COLUMNS[file_type]
    blocks, head, rows = [], None, []

    def close():
        if head is not None and int(head[0]) == line and rows:
            data = np.array(rows)
            block = pd.DataFrame(data[:, 1: 1 + len(names)], columns=names[: data.shape[1] - 1])
            block.insert(0, "x", data[:, 0])
            block.insert(0, "time", float(head[2]))
            blocks.append(block)

    # Headers are "line n time" with integer line and n; data rows always have decimals.
    # Blocks can be empty (e.g. OYVELO and OLONGS with normally incident waves).
    for s_ in lines:
        tokens = s_.split()
        if not tokens:
            continue
        if len(tokens) == 3 and "." not in tokens[0] and "." not in tokens[1]:
            close()
            head, rows = tokens, []
        else:
            rows.append([float(v) for v in tokens])
    close()
    if not blocks:
        raise ValueError(f"O{file_type.upper()} has no data for line {line} (e.g. OYVELO/OLONGS need oblique "
                         "waves, OBSUSL needs IPROFL = 1)")

    df = pd.concat(blocks).set_index(["time", "x"])
    if time is None:
        return df
    times = df.index.get_level_values("time").unique()
    t = {"first": times[0], "last": times[-1]}.get(time, time)
    return df.xs(t, level="time")
