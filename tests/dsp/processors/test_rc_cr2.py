import numpy as np
import pytest


from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import rc_cr2


def test_rc_cr2(compare_numba_vs_python):
    # Create a single exponential pulse to RC-CR^2 filter
    zeta = 30000
    ts = np.arange(0, 8192 // 2)
    amplitude = 17500
    pulse_in = np.insert(amplitude * np.exp(-ts / zeta), 0, np.zeros(8192 // 2))
    pulse_in = np.array(pulse_in, dtype=np.float32)
    tau = 500

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(pulse_in.size)
    w_in[4] = np.nan

    assert np.all(np.isnan(compare_numba_vs_python(rc_cr2, w_in, tau)))

    # Ensure that if the input waveform is less than 3 samples, an error is raised
    with pytest.raises(DSPFatal):
        rc_cr2(np.zeros(3), tau, np.zeros(3))

    # ensure that a valid input gives the expected output
    out_pulse = np.zeros_like(pulse_in)
    rc_cr2(pulse_in, tau, out_pulse)

    t = (
        np.arange(8192 // 2, len(out_pulse) + 1) - 8192 // 2
    )  # shift by 1 because of the 1 sample rise time
    # This is the exact form of an RC-CR^2 filter applied to an exponential pulse...
    w_out_expected = (
        -zeta * t**2 * np.exp(-t / tau) / (2 * tau * (zeta - tau))
        + t * (zeta**2 - 2 * zeta * tau) * np.exp(-t / tau) / (zeta - tau) ** 2
        + zeta * tau**3 * np.exp(-t / zeta) / (zeta - tau) ** 3
        - zeta * tau**3 * np.exp(-t / tau) / (zeta - tau) ** 3
    )
    w_out_expected *= amplitude
    w_out_expected *= np.amax(out_pulse) / np.amax(
        w_out_expected
    )  # rescale because we abandoned gain scaling in the filter
    w_out_expected = np.insert(
        w_out_expected, 0, np.zeros(8192 // 2 - 1)
    )  # subtract off the 1 we shifted

    # Check that it works at float32 precision
    pulse_in.astype(np.float32)
    tau = np.array([tau], dtype=np.float32)[0]
    assert np.allclose(
        compare_numba_vs_python(rc_cr2, pulse_in, tau), w_out_expected, rtol=1e-01
    )

    pulse_in.astype(np.float64)
    tau = np.array([tau], dtype=np.float64)[0]
    assert np.allclose(
        compare_numba_vs_python(rc_cr2, pulse_in, tau), w_out_expected, rtol=1e-01
    )