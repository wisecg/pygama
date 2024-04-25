from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs,
)
def rc_cr2(w_in: np.array, t_tau: float, w_out: np.array) -> None:
    """
    Apply a RC-CR^2 filter with the provided time
    constant to the waveform. Useful for determining pileup
    and trigger times. The filter was computed using a matched z-transform
    to keep the poles/zeroes of the analog transfer function in the same location.

    Parameters
    ----------
    w_in
        the input waveform.
    t_tau
        the time constant of the integration and differentiaion
    w_out
        the RC-CR^2 filtered waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_RC_CR2": {
            "function": "rc_cr2",
            "module": "dspeed.processors",
            "args": ["wf_bl", "300*ns", "wf_RC_CR2"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau):
        return

    if len(w_in) <= 3:
        raise DSPFatal(
            "The length of the waveform must be larger than 3 for the filter to work safely"
        )

    w_out[0] = w_in[0]
    w_out[1] = w_in[1]
    w_out[2] = w_in[2]

    # Make a temporary buffer at higher precision so that the recursive filter doesn't compound any float truncations
    w_tmp = np.zeros(4, dtype=np.float64)
    w_tmp[0] = w_in[0]
    w_tmp[1] = w_in[1]
    w_tmp[2] = w_in[2]

    a = np.exp(-1 / t_tau)

    denom_1 = 1
    denom_2 = -3 * a
    denom_3 = 3 * a**2
    denom_4 = -(a**3)

    num_1 = 1
    num_2 = -2
    num_3 = 1

    for i in range(3, len(w_in)):
        w_tmp[3] = (
            -denom_2 * w_tmp[2]
            - denom_3 * w_tmp[1]
            - denom_4 * w_tmp[0]
            + num_1 * w_in[i]
            + num_2 * w_in[i - 1]
            + num_3 * w_in[i - 2]
        ) / denom_1
        w_out[i] = w_tmp[3]  # Put the higher precision buffer into the desired output
        # shuffle the buffers
        w_tmp[0] = w_tmp[1]
        w_tmp[1] = w_tmp[2]
        w_tmp[2] = w_tmp[3]

    # Check the output
    if np.isnan(w_out).any():
        raise DSPFatal("RC-CR^2 filter produced nans in output.")