from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32[:])",
        "void(float64[:], float64, float64[:])",
    ],
    "(n),()->()",
    **nb_kwargs,
)
def get_wf_centroid(w_in: np.ndarray, shift: int, centroid: int) -> None:
    """Calculate waveform centroid.

    Note
    ----
    This processor calculate the centroid position when provided the convolution product with a step function.

    Parameters
    ----------
    w_in
        the input waveform.
    shift
        shift.
    centroid
        centroid position.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "centroid": {
          "function": "get_wf_centroid",
          "module": "dspeed.processors",
          "args": ["waveform", "shift", "centroid"],
          "unit": "ADC"
        }
    """

    centroid[0] = np.nan

    if np.isnan(w_in).any():
        return

    if np.isnan(shift):
        raise DSPFatal("shift is nan")
    if shift < 0:
        raise DSPFatal("shift must be positive")
    if shift > len(w_in) - 1:
        raise DSPFatal("shift must be shorter than input waveform size")

    c_a = (
        np.where(w_in[w_in.argmin() : w_in.argmax()] > 0)[0][0] + w_in.argmin() + shift
    )
    c_b = (
        np.where(w_in[w_in.argmin() : w_in.argmax()] < 0)[0][-1] + w_in.argmin() + shift
    )

    centroid[0] = round((c_a + c_b) / 2)