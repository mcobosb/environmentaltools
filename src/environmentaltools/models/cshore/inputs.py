"""Writing of the CSHORE input file (``infile``)."""

from pathlib import Path

import numpy as np


def _flag(fid, value, name):
    fid.write(f"{int(value):<42d}->{name}\n")


def _values(fid, values, names=""):
    line = "".join(f"{v:11.4f}" for v in values) + " "
    fid.write(f"{line:<42s}{'->' + names if names else ''}\n")


def write_input(properties: dict, output_folder: str):
    """Write the CSHORE input file ``infile``.

    The lines follow the reading order of subroutine ``INPUT`` of CSHORE
    (https://github.com/erdc/cshore): optional flags are only written when
    CSHORE reads them (e.g. ``IWTRAN``, ``IPOND`` and ``INFILT`` with
    ``IOVER = 1``). Supported: one cross-shore line, sand bottom
    (``IPERM = 0``, ``ISEDAV = 0``), no vegetation (``IVEG = 0``) and boundary
    conditions with ``ILAB = 1`` (one line per time interval).

    Args:
        properties (dict): Parameters (start from :func:`default_config`):

            - header (str | list[str]): Title lines.
            - iline, iprofl, isedav, iperm, iover, iwtran, ipond, infilt,
              iwcint, iroll, iwind, itide, iveg: Model options.
            - dx (float): Grid spacing (m).
            - gamma (float): Breaker ratio.
            - d50 (float): Median grain size (**mm**), wf (m/s), sg: sediment
              (``IPROFL = 1``).
            - effb, efff, slp, slpot: Suspension efficiencies and suspended
              load parameters (``slpot`` with ``IOVER = 1``).
            - tanphi, blp: Friction angle and bed load parameter.
            - rwh (float): Runup wire height (m) (``IOVER = 1``).
            - ilab, nwave, nsurg: Boundary conditions; ``nwave`` intervals.
            - timebc_wave (float | array): End time (s) of each interval; a
              scalar ``dt`` gives ``dt, 2 dt, ...``.
            - Tp, Hrms, Wsetup, swlbc, angle (float | array): Peak period (s),
              root-mean-square wave height (m), wave setup (m), still water
              level (m) and wave angle (degrees from the shore normal) at the
              seaward boundary, per interval.
            - x, zb, fw (array): Cross-shore distance (m, positive onshore,
              starting at 0), bed elevation (m, positive upwards) and bottom
              friction factor.
            - VelV, DirV (``IWIND = 1``) and slgradient (``ITIDE = 1``).

        output_folder (str): Folder where ``infile`` is written.

    Raises:
        NotImplementedError: For options outside the supported ones.
    """
    p = properties
    if p["iline"] != 1 or p["iperm"] != 0 or p.get("isedav", 0) != 0 or p["iveg"] != 0:
        raise NotImplementedError("Only ILINE = 1, IPERM = 0, ISEDAV = 0 and IVEG = 0 are supported.")
    if p["ilab"] != 1:
        raise NotImplementedError("Only ILAB = 1 boundary conditions are supported.")
    iprofl = int(p["iprofl"])
    iover = int(p["iover"])

    header = p["header"]
    header = header.splitlines() if isinstance(header, str) else list(header)

    nwave = int(p["nwave"])
    times = np.atleast_1d(np.asarray(p["timebc_wave"], float))
    if times.size == 1 and nwave > 1:
        times = times[0] * np.arange(1, nwave + 1)
    boundary = [np.broadcast_to(np.atleast_1d(np.asarray(p[k], float)), (nwave,))
                for k in ("Tp", "Hrms", "Wsetup", "swlbc", "angle")]

    with open(Path(output_folder) / "infile", "w") as fid:
        fid.write(f"{len(header)}\n")
        for line in header:
            fid.write(f"{line}\n")
        _flag(fid, p["iline"], "ILINE")
        _flag(fid, iprofl, "IPROFL")
        if iprofl == 1:
            _flag(fid, p.get("isedav", 0), "ISEDAV")
        _flag(fid, p["iperm"], "IPERM")
        _flag(fid, iover, "IOVER")
        if iover == 1:
            _flag(fid, p["iwtran"], "IWTRAN")
            if p["iwtran"] == 0:
                _flag(fid, p["ipond"], "IPOND")
            if iprofl == 1:
                _flag(fid, p["infilt"], "INFILT")
        for name in ("iwcint", "iroll", "iwind", "itide", "iveg"):
            _flag(fid, p[name], name.upper())
        _values(fid, [p["dx"]], "DX")
        _values(fid, [p["gamma"]], "GAMMA")
        if iprofl == 1:
            _values(fid, [p["d50"], p["wf"], p["sg"]], "D50 WF SG")
            if iover == 1:
                _values(fid, [p["effb"], p["efff"], p["slp"], p["slpot"]], "EFFB EFFF SLP SLPOT")
            else:
                _values(fid, [p["effb"], p["efff"], p["slp"]], "EFFB EFFF SLP")
            _values(fid, [p["tanphi"], p["blp"]], "TANPHI BLP")
        if iover == 1:
            _values(fid, [p["rwh"]], "RWH")
        _flag(fid, p["ilab"], "ILAB")
        _flag(fid, nwave, "NWAVE")
        _flag(fid, p.get("nsurg", nwave), "NSURGE")
        for i in range(nwave):
            fid.write(f"{times[i]:11.2f}" + "".join(f"{b[i]:11.4f}" for b in boundary) + "\n")

        _flag(fid, len(p["x"]), "NBINP")
        for x, z, f in zip(p["x"], p["zb"], np.broadcast_to(p["fw"], np.shape(p["x"]))):
            fid.write(f"{x:11.4f}{z:11.4f}{f:11.4f}\n")

        if p["iwind"]:
            fid.write("1\n")
            fid.write(f"{0:11.1f}{p['VelV']:11.4f}{p['DirV']:11.4f}\n")
            fid.write(f"{times[-1]:11.1f}{p['VelV']:11.4f}{p['DirV']:11.4f}\n")
        if p["itide"]:
            fid.write("1\n")
            fid.write(f"{0:11.1f}{p['slgradient']:13.8f}\n")
            fid.write(f"{times[-1]:11.1f}{p['slgradient']:13.8f}\n")
