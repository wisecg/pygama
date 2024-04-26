import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import pole_zero, transfer_function_convolver


def test_transfer_function_convolver(compare_numba_vs_python):
    """
    Testing function for the transfer_function_convolver processor.
    Test that this function reproduces a single pole zero filter

    """

    # Create a single exponential pulse to pole-zero correct
    tau = 10
    ts = np.arange(0, 100)
    amplitude = 10
    pulse_in = np.insert(amplitude * np.exp(-ts / tau), 0, np.zeros(20))

    a_in = np.array([1, 1], dtype=np.float64)
    b_in = np.array([1, -np.exp(-1 / tau)], dtype=np.float64)
    pulse_out = np.zeros(pulse_in.size, dtype=np.float64)

    # ensure the DSPFatal is raised if the parameter a is a scalar
    with pytest.raises(DSPFatal):
        transfer_function_convolver(pulse_in, np.array([]), b_in, pulse_out)

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(pulse_in.size)
    w_in[4] = np.nan

    assert np.all(
        np.isnan(compare_numba_vs_python(transfer_function_convolver, w_in, a_in, b_in))
    )

    # ensure that if there is a nan in a_in, all nans are outputted
    a_wrong_in = np.ones(3)
    a_wrong_in[2] = np.nan

    assert np.all(
        np.isnan(
            compare_numba_vs_python(
                transfer_function_convolver, pulse_in, a_wrong_in, b_in
            )
        )
    )

    # ensure that if there is a nan in b_in, all nans are outputted
    b_wrong_in = np.ones(3)
    b_wrong_in[2] = np.nan

    assert np.all(
        np.isnan(
            compare_numba_vs_python(
                transfer_function_convolver, pulse_in, a_in, b_wrong_in
            )
        )
    )

    # ensure that a valid input gives the expected output when comparing the pole-zero correction with the pole-zero processor
    w_out_expected = compare_numba_vs_python(pole_zero, pulse_in, tau)

    assert np.allclose(
        compare_numba_vs_python(transfer_function_convolver, pulse_in, a_in, b_in),
        w_out_expected,
    )