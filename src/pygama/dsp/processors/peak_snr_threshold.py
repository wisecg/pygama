from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32, float32[:], uint32[:])",
        "void(float64[:], float64[:], float64, float64, float64[:], uint32[:])",
    ],
    "(n),(m),(),()->(m),()",
    **nb_kwargs,
)
def peak_snr_threshold(
    w_in: np.ndarray,
    idx_in: np.ndarray,
    ratio_in: float,
    width_in: int,
    idx_out: np.ndarray,
    n_idx_out: int,
) -> None:
    """Search for local minima in a window around the provided indices.

    If a minimum is found it is checked whether amplitude of the minimum
    divided by the amplitude of the waveform at the index is smaller then the
    given ratio. If this is the case the index is copied to the output.

    Parameters
    ----------
    w_in
        the input waveform.
    idx_in
        the array of indices of possible signal candidates.
    ratio_in
        ratio threshold value.
    width_in
        width of the local search window.
    idx_out
        indices of local minima.
    n_idx_out
        number of non-:any:`numpy.nan` indices in `idx_out`.
    """

    # prepare output

    idx_out[:] = np.nan
    n_idx_out[0] = 0

    # fill output
    k = 0
    for i in range(len(idx_in)):
        if not np.isnan(idx_in[i]):
            a = int(idx_in[i]) - int(width_in)
            b = int(idx_in[i]) + int(width_in)
            if a < 0:
                a = 0
            if b >= len(w_in):
                b = len(w_in) - 1
            min_index = a
            for j in range(a, b, 1):
                if w_in[j] < w_in[min_index]:
                    min_index = j
            if np.absolute(w_in[min_index] / w_in[int(idx_in[i])]) < ratio_in:
                idx_out[k] = idx_in[i]
                k = k + 1
    n_idx_out[0] = k