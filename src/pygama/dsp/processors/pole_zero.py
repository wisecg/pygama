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
def pole_zero(w_in: np.ndarray, t_tau: float, w_out: np.ndarray) -> None:
    """Apply a pole-zero cancellation using the provided time
    constant to the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    t_tau
        the time constant of the exponential to be deconvolved.
    w_out
        the pole-zero cancelled waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_pz": {
            "function": "pole_zero",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "400*us", "wf_pz"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau):
        return

    const = np.exp(-1 / t_tau)
    w_out[0] = w_in[0]
    for i in range(1, len(w_in), 1):
        w_out[i] = w_out[i - 1] + w_in[i] - w_in[i - 1] * const

@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs,
)
def inverse_pole_zero(w_in: np.ndarray, t_tau: float, w_out: np.ndarray) -> None:
    """Apply an inverse pole-zero cancellation using the provided time
    constant to the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    t_tau
        the time constant of the exponential to be deconvolved.
    w_out
        the pole-zero cancelled waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_ipz": {
            "function": "inverse_pole_zero",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "400*us", "wf_ipz"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau):
        return

    const = np.exp(-1 / t_tau)
    w_out[0] = w_in[0]
    for i in range(1, len(w_in), 1):
        w_out[i] = const*w_out[i - 1] + w_in[i] - w_in[i - 1] 


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),()->(n)",
    **nb_kwargs,
)
def double_pole_zero(
    w_in: np.ndarray, t_tau1: float, t_tau2: float, frac: float, w_out: np.ndarray
) -> np.ndarray:
    r"""
    Apply a double pole-zero cancellation using the provided time
    constants to the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    t_tau1
        the time constant of the first exponential to be deconvolved.
    t_tau2
        the time constant of the second exponential to be deconvolved.
    frac
        the fraction which the second exponential contributes.
    w_out
        the pole-zero cancelled waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_pz": {
            "function": "double_pole_zero",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "400*us", "20*us", "0.02", "wf_pz"],
            "unit": "ADC"
        }

    Notes
    -----
    This algorithm is an IIR filter to deconvolve the function

    .. math::
        s(t) = A \left[ f \cdot \exp\left(-\frac{t}{\tau_2} \right)
               + (1-f) \cdot \exp\left(-\frac{t}{\tau_1}\right) \right]

    (:math:`f` = `frac`) into a single step function of amplitude :math:`A`.
    This filter is derived by :math:`z`-transforming the input (:math:`s(t)`)
    and output (step function) signals, and then determining the transfer
    function. For shorthand, define :math:`a=\exp(-1/\tau_1)` and
    :math:`b=\exp(-1/\tau_2)`, the transfer function is then:

    .. math::
        H(z) = \frac{1 - (a+b)z^{-1} + abz^{-2}}
                    {1 + (fb - fa - b - 1)z^{-1}-(fb - fa - b)z^{-2}}

    By equating the transfer function to the ratio of output to input waveforms
    :math:`H(z) = w_\text{out}(z) / w_\text{in}(z)` and then taking the
    inverse :math:`z`-transform yields the filter as run in the function, where
    :math:`f` is the `frac` parameter:

    .. math::
        w_\text{out}[n] =& w_\text{in}[n] - (a+b)w_\text{in}[n-1]
                           + abw_\text{in}[n-2] \\
                         & -(fb - fa - b - 1)w_\text{out}[n-1]
                           + (fb - fa - b)w_\text{out}[n-2]
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau1) or np.isnan(t_tau2) or np.isnan(frac):
        return
    if len(w_in) <= 3:
        raise DSPFatal(
            "The length of the waveform must be larger than 3 for the filter to work safely"
        )

    a = np.exp(-1 / t_tau1)
    b = np.exp(-1 / t_tau2)

    transfer_denom_1 = frac * b - frac * a - b - 1
    transfer_denom_2 = -1 * (frac * b - frac * a - b)
    transfer_num_1 = -1 * (a + b)
    transfer_num_2 = a * b

    w_out[0] = w_in[0]
    w_out[1] = w_in[1]
    w_out[2] = w_in[2]

    for i in range(2, len(w_in), 1):
        w_out[i] = (
            w_in[i]
            + transfer_num_1 * w_in[i - 1]
            + transfer_num_2 * w_in[i - 2]
            - transfer_denom_1 * w_out[i - 1]
            - transfer_denom_2 * w_out[i - 2]
        )
