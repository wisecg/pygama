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

@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64, float64, float64[:])",
    ],
    "(n),(),(),(),(),()->(n)",
    **nb_kwargs,
)
def triple_pole_zero(
    w_in: np.ndarray, t_tau1: float, t_tau2: float, t_tau3: float, frac1: float, frac2: float, w_out: np.ndarray
) -> np.ndarray:
    r"""
    Apply a triple pole-zero cancellation using the provided time
    constants to the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    t_tau1
        the time constant of the first exponential to be deconvolved.
    t_tau2
        the time constant of the second exponential to be deconvolved.
    t_tau3
        the time constant of the second exponential to be deconvolved.
    frac1
        the fraction which the second exponential contributes.
    frac2
        the fraction which the third exponential contributes.
    w_out
        the triple pole-zero cancelled waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_pz": {
            "function": "triple_pole_zero",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "400*us", "20*us", "100*ns", "0.02", "0.98", "wf_pz"],
            "unit": "ADC"
        }

    Notes
    -----
    This algorithm is an IIR filter to deconvolve the function

    .. math::
        s(t) = A \left[ f1 \cdot \exp\left(-\frac{t}{\tau_1} \right)
               + f2 \cdot \exp\left(-\frac{t}{\tau_2} \right) +  (1-f1-f2) \cdot \exp\left(-\frac{t}{\tau_3}\right) \right]

    (:math:`f` = `frac`) into a single step function of amplitude :math:`A`.
    This filter is derived by :math:`z`-transforming the input (:math:`s(t)`)
    and output (step function) signals, and then determining the transfer
    function. 
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau1) or np.isnan(t_tau2) or np.isnan(t_tau3) or np.isnan(frac1) or np.isnan(frac2):
        return
    if len(w_in) <= 3:
        raise DSPFatal(
            "The length of the waveform must be larger than 3 for the filter to work safely"
        )

    a = np.exp(-1 / t_tau1)
    b = np.exp(-1 / t_tau2)
    c = np.exp(-1 / t_tau3)

    w_out[0] = w_in[0]
    w_out[1] = w_in[1]
    w_out[2] = w_in[2]

    num_1 = 1
    num_2 = - a - b - c
    num_3 = a*b + a*c + b*c
    num_4 = -a*b*c
    
    denom_1 = 1 
    denom_2 = -1 - a - b + a*frac1 - c*frac1 + b*frac2 - c*frac2
    denom_3 = a + b + a*b - a*frac1 -a*b*frac1 + c*frac1 + b*c*frac1 - b*frac2 - a*b*frac2 + c*frac2 + a*c*frac2
    denom_4 = -a*b + a*b*frac1 - b*c*frac1 + a*b*frac2 - a*c*frac2
    
    for i in range(3, len(w_in)):
        w_out[i] = -denom_2*w_out[i-1] - denom_3*w_out[i-2] - denom_4*w_out[i-3] + num_1*w_in[i] + num_2*w_in[i-1] + num_3*w_in[i-2] + num_4*w_in[i-3]


@guvectorize(
    [
        "void(float32[:], float64, float64, float64, float64, float32[:])",
        "void(float64[:], float64, float64, float64, float64, float64[:])",
    ],
    "(n),(),(),(),()->(n)",
    fastmath=False,
)
def double_pole_zero_two_fracs(
    w_in: np.ndarray, t_tau1: float, t_tau2: float, A: float, B: float, w_out: np.ndarray
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

    if np.isnan(w_in).any() or np.isnan(t_tau1) or np.isnan(t_tau2) or np.isnan(A) or np.isnan(B):
        return
    if len(w_in) <= 3:
        raise DSPFatal(
            "The length of the waveform must be larger than 3 for the filter to work safely"
        )

    a = np.exp(-1 / t_tau1)
    b = np.exp(-1 / t_tau2)

    transfer_denom_1 = A+B
    transfer_denom_2 = -1*(A+ A*b + B + a*B)
    transfer_denom_3 = A*b + a*B
    transfer_num_1 = 1
    transfer_num_2 = -1*(a+b)
    transfer_num_3 = a*b
    

    w_out[0] = w_in[0]
    w_out[1] = w_in[1]
    w_out[2] = w_in[2]

    for i in range(2, len(w_in), 1):
        w_out[i] = (
            transfer_num_1 * w_in[i]
            + transfer_num_2 * w_in[i - 1]
            + transfer_num_3 * w_in[i - 2]
            - transfer_denom_2 * w_out[i - 1]
            - transfer_denom_3 * w_out[i - 2]
        )/transfer_denom_1
