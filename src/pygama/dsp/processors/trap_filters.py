from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], int32, int32, float32[:])",
        "void(float64[:], int32, int32, float64[:])",
    ],
    "(n),(),()->(n)",
    **nb_kwargs,
)
def trap_filter(w_in: np.ndarray, rise: int, flat: int, w_out: np.ndarray) -> None:
    """Apply a symmetric trapezoidal filter to the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    rise
        the number of samples averaged in the rise and fall sections.
    flat
        the delay between the rise and fall sections.
    w_out
        the filtered waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_tf": {
            "function": "trap_filter",
            "module": "dspeed.processors",
            "args": ["wf_pz", "10*us", "3*us", "wf_tf"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat):
        return

    if int(rise) < 0:
        raise DSPFatal("The number of samples in the rise section must be positive")

    if int(flat) < 0:
        raise DSPFatal("The number of samples in the flat section must be positive")

    if 2 * int(rise) + int(flat) > len(w_in):
        raise DSPFatal("The trapezoid width is wider than the waveform")

    w_out[0] = w_in[0]
    for i in range(1, rise, 1):
        w_out[i] = w_out[i - 1] + w_in[i]
    for i in range(rise, rise + flat, 1):
        w_out[i] = w_out[i - 1] + w_in[i] - w_in[i - rise]
    for i in range(rise + flat, 2 * rise + flat, 1):
        w_out[i] = w_out[i - 1] + w_in[i] - w_in[i - rise] - w_in[i - rise - flat]
    for i in range(2 * rise + flat, len(w_in), 1):
        w_out[i] = (
            w_out[i - 1]
            + w_in[i]
            - w_in[i - rise]
            - w_in[i - rise - flat]
            + w_in[i - 2 * rise - flat]
        )


@guvectorize(
    [
        "void(float32[:], int32, int32, float32[:])",
        "void(float64[:], int32, int32, float64[:])",
    ],
    "(n),(),()->(n)",
    **nb_kwargs,
)
def trap_norm(w_in: np.ndarray, rise: int, flat: int, w_out: np.ndarray) -> None:
    """Apply a symmetric trapezoidal filter to the waveform, normalized by the
    number of samples averaged in the rise and fall sections.

    Parameters
    ----------
    w_in
        the input waveform.
    rise
        the number of samples averaged in the rise and fall sections.
    flat
        the delay between the rise and fall sections.
    w_out
        the normalized, filtered waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_tf": {
            "function": "trap_norm",
            "module": "dspeed.processors",
            "args": ["wf_pz", "10*us", "3*us", "wf_tf"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat):
        return

    if int(rise) < 0:
        raise DSPFatal("The number of samples in the rise section must be positive")

    if int(flat) < 0:
        raise DSPFatal("The number of samples in the flat section must be positive")

    if 2 * int(rise) + int(flat) > len(w_in):
        raise DSPFatal("The trapezoid width is wider than the waveform")

    w_out[0] = w_in[0] / rise
    for i in range(1, rise, 1):
        w_out[i] = w_out[i - 1] + w_in[i] / rise
    for i in range(rise, rise + flat, 1):
        w_out[i] = w_out[i - 1] + (w_in[i] - w_in[i - rise]) / rise
    for i in range(rise + flat, 2 * rise + flat, 1):
        w_out[i] = (
            w_out[i - 1] + (w_in[i] - w_in[i - rise] - w_in[i - rise - flat]) / rise
        )
    for i in range(2 * rise + flat, len(w_in), 1):
        w_out[i] = (
            w_out[i - 1]
            + (
                w_in[i]
                - w_in[i - rise]
                - w_in[i - rise - flat]
                + w_in[i - 2 * rise - flat]
            )
            / rise
        )


@guvectorize(
    [
        "void(float32[:], int32, int32, int32, float32[:])",
        "void(float64[:], int32, int32, int32, float64[:])",
    ],
    "(n),(),(),()->(n)",
    **nb_kwargs,
)
def asym_trap_filter(
    w_in: np.ndarray, rise: int, flat: int, fall: int, w_out: np.ndarray
) -> None:
    """Apply an asymmetric trapezoidal filter to the waveform, normalized by
    the number of samples averaged in the rise and fall sections.

    Parameters
    ----------
    w_in
        the input waveform.
    rise
        the number of samples averaged in the rise section.
    flat
        the delay between the rise and fall sections.
    fall
        the number of samples averaged in the fall section.
    w_out
        the normalized, filtered waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_af": {
            "function": "asym_trap_filter",
            "module": "dspeed.processors",
            "args": ["wf_pz", "128*ns", "64*ns", "2*us", "wf_af"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat) or np.isnan(fall):
        return

    if int(rise) < 0:
        raise DSPFatal("The number of samples in the rise section must be positive")

    if int(flat) < 0:
        raise DSPFatal("The number of samples in the flat section must be positive")

    if int(fall) < 0:
        raise DSPFatal("The number of samples in the fall section must be positive")

    if int(rise) + int(flat) + int(fall) > len(w_in):
        raise DSPFatal("The trapezoid width is wider than the waveform")

    w_out[0] = w_in[0] / rise
    for i in range(1, rise, 1):
        w_out[i] = w_out[i - 1] + w_in[i] / rise
    for i in range(rise, rise + flat, 1):
        w_out[i] = w_out[i - 1] + (w_in[i] - w_in[i - rise]) / rise
    for i in range(rise + flat, rise + flat + fall, 1):
        w_out[i] = (
            w_out[i - 1]
            + (w_in[i] - w_in[i - rise]) / rise
            - w_in[i - rise - flat] / fall
        )
    for i in range(rise + flat + fall, len(w_in), 1):
        w_out[i] = (
            w_out[i - 1]
            + (w_in[i] - w_in[i - rise]) / rise
            - (w_in[i - rise - flat] - w_in[i - rise - flat - fall]) / fall
        )


@guvectorize(
    [
        "void(float32[:], int32, int32, float32, float32[:])",
        "void(float64[:], int32, int32, float64, float64[:])",
    ],
    "(n),(),(),()->()",
    **nb_kwargs,
)
def trap_pickoff(
    w_in: np.ndarray, rise: int, flat: int, t_pickoff: float, a_out: float
) -> None:
    """Pick off the value at the provided index of a symmetric trapezoidal
    filter to the input waveform, normalized by the number of samples averaged
    in the rise and fall sections.

    Parameters
    ----------
    w_in
        the input waveform.
    rise
        the number of samples averaged in the rise and fall sections.
    flat
        the delay between the rise and fall sections.
    t_pickoff
        the waveform index to pick off.
    a_out
        the output pick-off value of the filtered waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "ct_corr": {
            "function": "trap_pickoff",
            "module": "dspeed.processors",
            "args": ["wf_pz", "1.5*us", 0, "tp_0", "ct_corr"],
            "unit": "ADC"
        }
    """
    a_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat) or np.isnan(t_pickoff):
        return

    if np.floor(t_pickoff) != t_pickoff:
        raise DSPFatal("The pick-off index must be an integer")

    if int(rise) < 0:
        raise DSPFatal("The number of samples in the rise section must be positive")

    if int(flat) < 0:
        raise DSPFatal("The number of samples in the flat section must be positive")

    if 2 * int(rise) + int(flat) > len(w_in):
        raise DSPFatal("The trapezoid width is wider than the waveform")

    i_1 = 0.0
    i_2 = 0.0
    start_time = int(t_pickoff + 1)

    if not len(w_in) >= start_time >= 2 * rise + flat:
        return

    for i in range(start_time - rise, start_time, 1):
        i_1 += w_in[i]
    for i in range(start_time - 2 * rise - flat, start_time - rise - flat, 1):
        i_2 += w_in[i]
    a_out[0] = (i_1 - i_2) / rise