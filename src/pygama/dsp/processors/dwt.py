from __future__ import annotations

import numpy as np
from numba import guvectorize
from pywt import downcoef

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs

@guvectorize(
    [
        "void(float32[:], int32, char, char, float32[:])",
        "void(float64[:], int64, char, char, float64[:])",
    ],
    "(n),(),(),(),(m)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def discrete_wavelet_transform(
    w_in: np.ndarray, level: int, wave_type: int, coeff: int, w_out: np.ndarray
) -> None:
    """
    Apply a discrete wavelet transform to the waveform and return only
    the detailed or approximate coefficients.


    Parameters
    ----------

    w_in
       The input waveform
    level
       The level of decompositions to be performed ``(1, 2, ...)``
    wave_type
       The wavelet type for discrete convolution ``('h' = 'haar', 'd' = 'db1')``.
    coeff
       The coefficients to be saved ``('a', 'd')``
    w_out
       The approximate coefficients. The dimension of this array can be calculated
       by ``out_dim = len(w_in)/(filter_length^level)``, where ``filter_length``
       can be obtained via ``pywt.Wavelet(wave_type).dec_len``


    JSON Configuration Example
    --------------------------
    .. code-block :: json

        "dwt_haar":{
            "function": "discrete_wavelet_transform",
            "module": "dspeed.processors",
            "args": ["wf_blsub", 5, "'h'", "'a'", "dwt_haar(256, 'f')"],
            "unit": "ADC",
            "prereqs": ["wf_blsub"],
        }
    """

    w_out[:] = np.nan

    if level <= 0:
        raise DSPFatal("The level must be a positive integer")

    if np.isnan(w_in).any():
        return

    coeff = chr(coeff)

    if chr(wave_type) == "h":
        wave_type = "haar"

    elif chr(wave_type) == "d":
        wave_type = "db1"

    w_out[:] = downcoef(coeff, w_in, wave_type, level=level)