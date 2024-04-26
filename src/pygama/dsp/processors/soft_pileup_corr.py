from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64[:])",
    ],
    "(n),(),()->(n)",
    **nb_kwargs,
)
def soft_pileup_corr(
    w_in: np.ndarray, n_in: int, tau_in: float, w_out: np.ndarray
) -> None:
    """Fit the baseline to an exponential with the provided time constant and
    then subtract the best-fit function from the entire waveform.

    Parameters
    ----------
    w_in
        The input waveform.
    n_in
        The number of samples at the beginning of the waveform
        to fit to an exponential.
    tau_in
        The fixed, exponential time constant.
    w_out
        The output waveform with the exponential subtracted.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_bl": {
            "function": "soft_pileup_corr",
            "module": "dspeed.processors",
            "args": ["waveform", "1000", "500*us", "wf_bl"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(n_in) or np.isnan(tau_in):
        return

    if not np.floor(n_in) == n_in:
        raise DSPFatal("The number of samples is not an integer")

    if n_in < 2:
        raise DSPFatal("The number of samples is not enough for a fit")

    if n_in > len(w_in):
        raise DSPFatal("The number of samples is more than the waveform length")

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    for i in range(0, n_in, 1):
        s1 += 1.0
        s2 += np.exp(-1.0 * i / tau_in)
        s3 += np.exp(-2.0 * i / tau_in)
        s4 += np.exp(-1.0 * i / tau_in) * w_in[i]
        s5 += w_in[i]
    b = (s5 - s2 * (s4 * s1 - s2 * s5) / (s3 * s1 - s2 * s2)) / s1
    a = (s4 - b * s2) / s3
    for i in range(0, len(w_in), 1):
        w_out[i] = w_in[i] - (a * np.exp(-1.0 * i / tau_in) + b)


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),()->(n)",
    **nb_kwargs,
)
def soft_pileup_corr_bl(
    w_in: np.ndarray, n_in: int, tau_in: float, b_in: float, w_out: np.ndarray
) -> None:
    """Fit the baseline to an exponential with the provided time
    constant and then subtract the best-fit function from the
    entire waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    n_in
        the number of samples at the beginning of the waveform
        to fit to an exponential.
    tau_in
        the fixed, exponential time constant.
    b_in
        the fixed, exponential constant.
    w_out
        the output waveform with the exponential subtracted.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_bl": {
            "function": "soft_pileup_corr_bl",
            "module": "dspeed.processors",
            "args": ["waveform", "1000", "500*us", "baseline", "wf_bl"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(n_in) or np.isnan(tau_in) or np.isnan(b_in):
        return

    if not np.floor(n_in) == n_in:
        raise DSPFatal("The number of samples is not an integer")

    if n_in < 1:
        raise DSPFatal("The number of samples is not enough for a fit")

    if n_in > len(w_in):
        raise DSPFatal("The number of samples is more than the waveform length")

    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(0, n_in, 1):
        s2 += np.exp(-1.0 * i / tau_in)
        s3 += np.exp(-2.0 * i / tau_in)
        s4 += np.exp(-1.0 * i / tau_in) * w_in[i]
    a = (s4 - b_in * s2) / s3
    for i in range(0, len(w_in), 1):
        w_out[i] = w_in[i] - (a * np.exp(-1.0 * i / tau_in) + b_in)