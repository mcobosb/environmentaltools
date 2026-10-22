"""CSHORE cross-shore profile model: configuration, input file, execution and outputs.

When to use: to study cross-shore profiles of beaches that are roughly
uniform alongshore (each profile is solved independently, ∂/∂y = 0): surf and
swash zone, undertow, runup, overtopping and profile change (erosion, bars)
during storms. It does not capture refraction/diffraction in plan, structures
or circulation cells; for those use
:mod:`environmentaltools.models.swan_copla`.

Typical use::

    from environmentaltools.models import cshore

    props = cshore.default_config()
    props.update(header="profile 1", dx=1.0, x=x, zb=zb, fw=fw, ...)
    cshore.write_input(props, "run_01")
    cshore.run("/path/to/cshore", "run_01")
    profile = cshore.read_output("bprof", "run_01")
"""

from .config import default_config
from .inputs import write_input
from .read import read_output
from .run import run

__all__ = ["default_config", "write_input", "read_output", "run"]
