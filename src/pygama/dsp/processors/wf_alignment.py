from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),(),(m)",
    **nb_kwargs,
)
def wf_alignment(
    w_in: np.ndarray, centroid: int, shift: int, size: int, w_out: np.ndarray
) -> None:
    """Align waveform.

    Note
    ----
    This processor align the input waveform by setting the centroid position at the center of the output waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    centroid
        centroid.
    shift
        shift.
    size
        size of output waveform.
    w_out
        aligned waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_align": {
          "function": "wf_alignment",
          "module": "dspeed.processors",
          "args": ["waveform", "centroid", "shift", "size", "wf_align"],
          "unit": "ADC"
        }
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.isnan(centroid):
        raise DSPFatal("centroid is nan")

    if np.isnan(shift):
        raise DSPFatal("shift is nan")
    if shift < 0:
        raise DSPFatal("shift must be positive")
    if shift > len(w_in):
        raise DSPFatal("shift must be shorter than input waveform size")

    if np.isnan(size):
        raise DSPFatal("size is nan")
    if size <= 0:
        raise DSPFatal("size must be positive")
    if size > len(w_in):
        raise DSPFatal("size must be shorter than input waveform size")

    if (centroid >= size / 2) and (centroid < len(w_in) - size / 2):
        w_out[:] = w_in[int(centroid - size / 2) : int(centroid + size / 2)]
    elif (centroid > size / 2 - shift) and (centroid < size / 2):
        ss = int((size + 1) / 2 - centroid)
        w_out[:ss] = w_in[0]
        w_out[ss:] = w_in[: int(centroid + size / 2)]
    else:
        w_out[:] = w_in[:size]