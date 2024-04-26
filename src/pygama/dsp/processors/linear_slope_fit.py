from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n)->(),(),(),()",
    **nb_kwargs,
)
def linear_slope_fit(
    w_in: np.ndarray, mean: float, stdev: float, slope: float, intercept: float
) -> None:
    """
    Calculate the mean and standard deviation of the waveform using
    Welford's method as well as the slope an intercept of the waveform
    using linear regression.

    Parameters
    ----------
    w_in
        the input waveform.
    mean
        the mean of the waveform.
    stdev
        the standard deviation of the waveform.
    slope
        the slope of the linear fit.
    intercept
        the intercept of the linear fit.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "bl_mean, bl_std, bl_slope, bl_intercept": {
            "function": "linear_slope_fit",
            "module": "dspeed.processors",
            "args": ["wf_blsub[0:1650]", "bl_mean", "bl_std", "bl_slope", "bl_intercept"],
            "unit": ["ADC", "ADC", "ADC", "ADC"],
        }
    """
    mean[0] = np.nan
    stdev[0] = np.nan
    slope[0] = np.nan
    intercept[0] = np.nan

    if np.isnan(w_in).any():
        return

    sum_x = sum_x2 = sum_xy = sum_y = mean[0] = stdev[0] = 0
    isum = len(w_in)

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = w_in[i] - mean
        mean += temp / (i + 1)
        stdev += temp * (w_in[i] - mean)

        # linear regression
        sum_x += i
        sum_x2 += i * i
        sum_xy += w_in[i] * i
        sum_y += w_in[i]

    stdev /= isum - 1
    np.sqrt(stdev, stdev)

    slope[0] = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept[0] = (sum_y - sum_x * slope[0]) / isum


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(),()->(),()",
    **nb_kwargs,
)
def linear_slope_diff(
    w_in: np.ndarray, slope: float, intercept: float, mean: float, rms: float
) -> None:
    """
    Calculate the mean, standard deviation, slope and y-intercept of waveform.

    Uses Welford's method and linear regression.

    Parameters
    ----------
    w_in
        the input waveform.
    mean
        the mean of the waveform.
    stdev
        the standard deviation of the waveform.
    slope
        the slope of the linear fit.
    intercept
        the intercept of the linear fit.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "bl_mean, bl_std, bl_slope, bl_intercept": {
            "function": "linear_slope_fit",
            "module": "dspeed.processors",
            "args": ["wf_blsub[0:1650]", "bl_mean", "bl_std", "bl_slope", "bl_intercept"],
            "unit": ["ADC", "ADC", "ADC", "ADC"],
        }
    """
    mean[0] = np.nan
    rms[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(slope) or np.isnan(intercept):
        return

    mean[0] = rms[0] = 0
    isum = len(w_in)

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = w_in[i] - (slope * i + intercept)
        mean += temp / (i + 1)
        rms += temp * temp

    rms /= isum - 1
    np.sqrt(rms, rms)