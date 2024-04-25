import inspect

import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import fixed_time_pickoff, bi_level_zero_crossing_time_points, rc_cr2


def test_fixed_time_pickoff(compare_numba_vs_python):
    """Testing function for the fixed_time_pickoff processor."""

    len_wf = 20

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    assert np.isnan(compare_numba_vs_python(fixed_time_pickoff, w_in, 1, ord("i")))

    # test for nan if nan is passed to t_in
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(fixed_time_pickoff, w_in, np.nan, ord("i")))

    # test for nan if t_in is negative
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(fixed_time_pickoff, w_in, -1, ord("i")))

    # test for nan if t_in is too large
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(fixed_time_pickoff, w_in, len_wf, ord("i")))

    # test for DSPFatal errors being raised
    # noninteger t_in with integer interpolation
    with pytest.raises(DSPFatal):
        w_in = np.ones(len_wf)
        fixed_time_pickoff(w_in, 1.5, ord("i"))
    with pytest.raises(DSPFatal):
        a_out = np.empty(len_wf)
        inspect.unwrap(fixed_time_pickoff)(w_in, 1.5, ord("i"), a_out)

    # unsupported mode_in character
    with pytest.raises(DSPFatal):
        w_in = np.ones(len_wf)
        fixed_time_pickoff(w_in, 1.5, ord(" "))
    with pytest.raises(DSPFatal):
        a_out = np.empty(len_wf)
        inspect.unwrap(fixed_time_pickoff)(w_in, 1.5, ord(" "), a_out)

    # linear tests
    w_in = np.arange(len_wf, dtype=float)
    assert compare_numba_vs_python(fixed_time_pickoff, w_in, 3, ord("i")) == 3

    chars = ["n", "f", "c", "l", "h", "s"]
    sols = [4, 3, 4, 3.5, 3.5, 3.5]

    for char, sol in zip(chars, sols):
        assert compare_numba_vs_python(fixed_time_pickoff, w_in, 3.5, ord(char)) == sol

    # sine wave tests
    w_in = np.sin(np.arange(len_wf))

    chars = ["n", "f", "c", "l", "h", "s"]
    sols = [
        0.1411200080598672,
        0.1411200080598672,
        -0.7568024953079282,
        -0.08336061778208165,
        -0.09054574599004982,
        -0.10707938709427486,
    ]

    for char, sol in zip(chars, sols):
        assert np.isclose(
            compare_numba_vs_python(fixed_time_pickoff, w_in, 3.25, ord(char)), sol
        )

    # last few corner cases of 'h'
    w_in = np.sin(np.arange(len_wf))
    ftps = [0.2, len_wf - 1.8]
    sols = [
        0.1806725096462211,
        -0.6150034250096629,
    ]

    for ftp, sol in zip(ftps, sols):
        assert np.isclose(
            compare_numba_vs_python(fixed_time_pickoff, w_in, ftp, ord("h")), sol
        )



def test_bi_level_zero_crossing_time_points(compare_numba_vs_python):
    # Test exceptions and initial checks
    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(100)
    w_in[4] = np.nan
    t_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(w_in, 100, 100, 100, 0, pol_out, t_out)
    assert np.isnan(t_out).all()

    # ensure the ValueError is raised if the polarity output array is different length than the time point output array
    t_start_in = 1.02
    with pytest.raises(ValueError):
        bi_level_zero_crossing_time_points(
            np.ones(9), 100, 100, 100, t_start_in, pol_out, np.zeros(1)
        )

    # ensure the DSPFatal is raised if initial timepoint is not an integer
    t_start_in = 1.02
    with pytest.raises(DSPFatal):
        bi_level_zero_crossing_time_points(
            np.ones(9), 100, 100, 100, t_start_in, pol_out, t_out
        )

    # ensure the DSPFatal is raised if initial timepoint is not negative
    t_start_in = -2
    with pytest.raises(DSPFatal):
        bi_level_zero_crossing_time_points(
            np.ones(9), 100, 100, 100, t_start_in, pol_out, t_out
        )

    # ensure the DSPFatal is raised if initial timepoint is outside length of waveform
    t_start_in = 100
    with pytest.raises(DSPFatal):
        bi_level_zero_crossing_time_points(
            np.ones(9), 100, 100, 100, t_start_in, pol_out, t_out
        )

    early_trig = 500  # start pulse1 500 samples from the start of the wf
    late_trig = 200  # start pulse2 200 samples after the midpoint of 8192 length wf
    zeta = 30000  # the decay time constant of a pulse, in samples
    amplitude = 1750
    tau = 100  # the RC-CR^2 filter time constant

    # Make the first pulse and RC-CR^2 filter it
    ts = np.arange(0, 8192 // 2 - early_trig)
    pulse = amplitude * np.exp(-1 * ts / zeta)
    pulse = np.insert(pulse, 0, np.zeros(5))  # pad with 0s to avoid edge effects
    out_pulse = np.zeros_like(pulse)

    rc_cr2(pulse, tau, out_pulse)

    pulse = np.insert(
        out_pulse[5:], 0, np.zeros(early_trig)
    )  # delay the start of the wf by early_trig

    # Make the second pulse
    ts = np.arange(0, 8192 // 2 - late_trig)
    pulse2 = amplitude * np.exp(-1 * ts / zeta)
    pulse2 = np.insert(pulse2, 0, np.zeros(5))  # avoid edge effects
    out_pulse = np.zeros_like(pulse2)
    rc_cr2(pulse2, tau, out_pulse)
    pulse = np.insert(pulse, -1, np.zeros(late_trig))
    pulse = np.insert(pulse, -1, out_pulse[5:])

    gate_time = 1000
    # Test that the filter reproduces 0 crossings at the expected points
    # Test on positive polarity
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        pulse, 2000, -2000, gate_time, 0, pol_out, t_trig_times_out
    )

    cross_1 = early_trig + 2 * tau - 1  # minus 1 from delay?
    cross_2 = (
        8192 // 2 + late_trig + 2 * tau - 2
    )  # minus 1 from 1st pulse delay and again
    assert np.allclose(int(t_trig_times_out[0]), cross_1, rtol=1)
    assert np.allclose(int(t_trig_times_out[1]), cross_2, rtol=1)
    assert int(pol_out[0]) == 1
    assert int(pol_out[1]) == 1

    # Check on negative polarity pulses
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        -1 * pulse, 2000, -2000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.allclose(int(t_trig_times_out[0]), cross_1, rtol=1)
    assert np.allclose(int(t_trig_times_out[1]), cross_2, rtol=1)
    assert int(pol_out[0]) == 0
    assert int(pol_out[1]) == 0

    # Check positive polarity pulses that cross 0 and never reach negative threshold return all nan
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        pulse, 2000, -300000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.isnan(t_trig_times_out).all()
    assert np.isnan(pol_out).all()

    # Check negative polarity pulses that cross 0 and never reach positive threshold return all nan
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        -1 * pulse, 300000, -2000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.isnan(t_trig_times_out).all()
    assert np.isnan(pol_out).all()

    # Check that pulses that never reach either threshold return all nan
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        pulse, 300000, 300000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.isnan(t_trig_times_out).all()
    assert np.isnan(pol_out).all()

    # Check that pulses that go up and never cross zero again return all nan
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        np.linspace(-1, 100, 101), 4, -4, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.isnan(t_trig_times_out).all()
    assert np.isnan(pol_out).all()

    # Check that pulses that go down and never cross zero again return all nan
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        -1 * np.linspace(-1, 100, 101), 4, -4, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.isnan(t_trig_times_out).all()
    assert np.isnan(pol_out).all()

    # Check positive polarity pulses where only 2nd peak crosses the threshold
    scale_arr = np.full(8192 // 2, 1)
    scale_arr = np.insert(scale_arr, -1, np.full(8192 // 2, 5))
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        pulse * scale_arr, 2000, -20000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.allclose(
        int(t_trig_times_out[0]), cross_2, rtol=1
    )  # only the 2nd time point should have been crossed
    assert int(pol_out[0]) == 1

    # Check positive polarity pulses where only 1st peak crosses the threshold
    scale_arr = np.full(8192 // 2, 5)
    scale_arr = np.insert(scale_arr, -1, np.full(8192 // 2, 1))
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        pulse * scale_arr, 2000, -20000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.allclose(
        int(t_trig_times_out[0]), cross_1, rtol=1
    )  # only the 1st time point should have been crossed
    assert int(pol_out[0]) == 1

    # Check positive polarity pulses where only 2nd peak crosses both thresholds, but 1st peak passes negative but not within gate
    scale_arr = np.full(8192 // 2, 1)
    scale_arr = np.insert(scale_arr, -1, np.full(8192 // 2, 5))
    t_trig_times_out = np.zeros(5)
    pol_out = np.zeros(5)
    bi_level_zero_crossing_time_points(
        pulse * scale_arr, 50000, -2000, gate_time, 0, pol_out, t_trig_times_out
    )
    assert np.allclose(
        int(t_trig_times_out[0]), cross_2, rtol=1
    )  # only the 2nd time point should have been crossed
    assert int(pol_out[0]) == 1