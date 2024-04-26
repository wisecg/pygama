from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[::1], float32[:], float32[:], float32[::1])",
        "void(float64[::1], float64[:], float64[:], float64[::1])",
    ],
    "(n),(m),(l)->(n)",
    **nb_kwargs,
)
def transfer_function_convolver(w_in, a_in, b_in, w_out):
    r"""
    Compute the difference equation of the form
    $a_0*w_out[i] = a_1*w_out[i-1] + \ldots b_0*w_in[i] \ldots$
    which is equivalent of convolving a signal with a transfer function.
    a_in always needs at least one element
    The numba signature specifies these arrays as contiguous, so that way the dot product is as fast as possible

    Parameters
    ----------
    w_in
        An array of the values that this lti filter will be performed on
    a_in
        An array [a_0, ..., a_n] of filter coefficients to apply recursively to the output
    b_in
        An array [b_0, ..., b_m] of filter coefficients to apply to the input
    w_out
        The output array to store all the filtered values

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_pz": {
            "function": "transfer_function_convolver",
            "module": "dspeed.processors",
            "args": ["waveform", "np.array([-1,1])", "np.array([1,2])", "wf_pz"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    # perform some checks
    if np.isnan(w_in).any() or np.isnan(a_in).any() or np.isnan(b_in).any():
        return
    if len(w_in) != len(w_out):
        return
    if len(a_in) < 1:
        raise DSPFatal("The parameter a_in must be an array of length at least 1")

    b = b_in / a_in[0]
    a = a_in / a_in[0]

    # copy and flip the arrays, needed so that way they are contiguous
    a_flip = np.copy(a[::-1])
    b_flip = np.copy(b[::-1])

    # Figure out if it's a FIR
    if a_flip.size == 1:
        # Perform the convolution
        for n in range(0, w_in.size):
            w_out[n] = np.dot(b_flip[-(n + 1) :], w_in[n + 1 - len(b_flip) : n + 1])

    # Otherwise we have an IIR
    else:
        a_flip = a_flip[:-1]
        # Initialize the output with the first input signals
        for i in range(a_flip.size):
            w_out[i] = w_in[i]

        # Perform the convolution
        for n in range(a_flip.size, w_in.size):
            w_out[n] = np.dot(a_flip, w_out[n - len(a_flip) : n])
            w_out[n] += np.dot(b_flip[-(n + 1) :], w_in[n + 1 - len(b_flip) : n + 1])