import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import double_pole_zero, pole_zero


def test_pole_zero(compare_numba_vs_python):
    """
    Test that the pole-zero filter can correct an RC decay pulse into a step function
    """

    # Create a single exponential pulse to pole-zero correct
    tau = 30000
    ts = np.arange(0, 8192)
    amplitude = 17500
    pulse_in = np.insert(amplitude * np.exp(-ts / tau), 0, np.zeros(20))
    pulse_in = np.array(pulse_in, dtype=np.float32)

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(pulse_in.size)
    w_in[4] = np.nan

    assert np.all(np.isnan(compare_numba_vs_python(pole_zero, w_in, tau)))

    # ensure that a valid input gives the expected output when comparing the pole-zero correction with the pole-zero processor
    step = np.full(len(ts), amplitude)
    w_out_expected = np.insert(step, 0, np.zeros(20))

    # Check that it works at float32 precision
    pulse_in.astype(np.float32)
    tau = np.array([tau], dtype=np.float32)[0]
    assert np.allclose(
        compare_numba_vs_python(pole_zero, pulse_in, tau), w_out_expected, rtol=1e-07
    )

    # Check that it works at float64 precision
    pulse_in.astype(np.float64)
    tau = np.array([tau], dtype=np.float64)[0]
    assert np.allclose(
        compare_numba_vs_python(pole_zero, pulse_in, tau), w_out_expected, rtol=1e-07
    )


def test_double_pole_zero(compare_numba_vs_python):
    """
    Test that the double pole-zero filter can correct the sum of two RC decays into a step function
    """

    # Create a double exponential pulse to double-pole-zero correct
    wf_len = 8192
    tp0 = 2
    amplitude = 17500
    tau1 = 1000
    tau2 = 30000
    frac = 0.98
    ts = np.arange(0, wf_len - tp0)
    ys = amplitude * (1 - frac) * np.exp(-ts / tau1) + amplitude * frac * np.exp(
        -ts / tau2
    )
    pulse_in = np.insert(ys, 0, np.zeros(tp0))

    # ensure the DSPFatal is raised if the waveform is too short
    pulse_out = np.zeros(2, dtype=np.float64)
    with pytest.raises(DSPFatal):
        double_pole_zero(np.ones(2), tau1, tau2, frac, pulse_out)

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(pulse_in.size)
    w_in[4] = np.nan

    assert np.all(
        np.isnan(compare_numba_vs_python(double_pole_zero, w_in, tau1, tau2, frac))
    )

    # ensure that a valid input gives the expected output when comparing the pole-zero correction with the pole-zero processor
    step = np.full(len(ts), amplitude)
    w_out_expected = np.insert(step, 0, np.zeros(tp0))

    assert np.allclose(
        compare_numba_vs_python(double_pole_zero, pulse_in, tau1, tau2, frac),
        w_out_expected,
        rtol=1e-7,
    )

    # Make sure that the processor also works for float32 precision
    pulse_in.astype(np.float32)
    tau1 = np.array([tau1], dtype=np.float32)[0]
    tau2 = np.array([tau2], dtype=np.float32)[0]
    frac = np.array([frac], dtype=np.float32)[0]
    assert np.allclose(
        compare_numba_vs_python(double_pole_zero, pulse_in, tau1, tau2, frac),
        w_out_expected,
        rtol=1e-7,
    )