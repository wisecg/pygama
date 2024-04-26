from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32[:], float32[:])",
        "void(float64[:], float64, float64[:], float64[:])",
    ],
    "(n), (), (), (m)",
    **nb_kwargs,
)
def presum(w_in: np.ndarray, do_norm: int, ps_fact: int, w_out: np.ndarray) -> None:
    """Presum the waveform.

    Combine bins in chunks of ``len(w_in) / len(w_out)``, which is hopefully an
    integer. If it isn't, then some samples at the end will be omitted.

    Parameters
    ----------
    w_in
        the input waveform.
    do_norm
        a flag for setting returned chunks are normalized

        * ``0``: chunks are not normalized
        * ``1``: chunks are normalized

    ps_fact
        the presum factor/rate, determined by `len(w_in) // len(w_out)`
    w_out
        the output waveform.
    """
    w_out[:] = np.nan
    ps_fact[0] = np.nan

    if np.isnan(w_in).any():
        return
    if do_norm not in [int(0), int(1)]:
        raise DSPFatal("do_norm type not found.")

    ps_fact[0] = int(len(w_in) // len(w_out))
    for i in range(0, len(w_out), 1):
        j0 = i * int(ps_fact[0])
        if do_norm == 1:
            w_out[i] = w_in[j0] / ps_fact[0]
        else:
            w_out[i] = w_in[j0]

        for j in range(j0 + 1, j0 + ps_fact[0], 1):
            if do_norm == 1:
                w_out[i] += w_in[j] / ps_fact[0]
            else:
                w_out[i] += w_in[j]
