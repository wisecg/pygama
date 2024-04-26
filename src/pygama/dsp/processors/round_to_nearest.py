from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        f"void({t}, {t}, {t}[:])"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    "(),()->()",
    nopython=True,
    **nb_kwargs,
)
def round_to_nearest(val: np.ndarray, to_nearest: int | float, out: np.ndarray) -> None:
    """
    Round value to nearest multiple of to_nearest.

    Parameters
    ----------
    val
        value to be rounded
    to_nearest
        round to multiple of this
    out
        rounded value

    JSON Configuration Example
    --------------------------

    Note: this processor is aliased using `round` in ProcessingChain.
    The following two examples are equivalent.

    .. code-block :: json

        "t_rounded": {
            "function": "round_to_nearest",
            "module": "dspeed.processors",
            "args": ["t_in", "1*us", "t_rounded"]
            "unit": ["ns"]
        },
        "t_rounded": "round(t_in, 1*us)"
    """

    if np.isnan(val):
        out[:] = np.nan
    else:
        out[:] = to_nearest * round(val / to_nearest)