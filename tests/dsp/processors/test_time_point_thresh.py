import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import (
    bi_level_zero_crossing_time_points,
    interpolated_time_point_thresh,
    rc_cr2,
    time_point_thresh,
)


def test_time_point_thresh(compare_numba_vs_python):
    """Testing function for the time_point_thresh processor."""

    # test for nan if w_in has a nan
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    w_in[4] = np.nan
    assert np.isnan(
        compare_numba_vs_python(
            time_point_thresh,
            w_in,
            1,
            11,
            0,
        )
    )

    # test for nan if nan is passed to a_threshold
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert np.isnan(
        compare_numba_vs_python(
            time_point_thresh,
            w_in,
            np.nan,
            11,
            0,
        )
    )

    # test for nan if nan is passed to t_start
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert np.isnan(
        compare_numba_vs_python(
            time_point_thresh,
            w_in,
            1,
            np.nan,
            0,
        )
    )

    # test for nan if nan is passed to walk_forward
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert np.isnan(
        compare_numba_vs_python(
            time_point_thresh,
            w_in,
            1,
            11,
            np.nan,
        )
    )

    # test for error if t_start non integer
    with pytest.raises(DSPFatal):
        w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
        time_point_thresh(w_in, 1, 10.5, 0, np.array([0.0]))

    # test for error if walk_forward non integer
    with pytest.raises(DSPFatal):
        w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
        time_point_thresh(w_in, 1, 11, 0.5, np.array([0.0]))

    # test for error if t_start out of range
    with pytest.raises(DSPFatal):
        w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
        time_point_thresh(w_in, 1, 12, 0, np.array([0.0]))

    # test walk backward
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert compare_numba_vs_python(time_point_thresh, w_in, 1, 11, 0) == 8.0

    # test walk forward
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert compare_numba_vs_python(time_point_thresh, w_in, 3, 0, 1) == 4.0


def test_interpolated_time_point_thresh(compare_numba_vs_python):
    """Testing function for the interpolated_time_point_thresh processor."""

    # test for nan if w_in has a nan
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    w_in[4] = np.nan
    assert np.isnan(
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1.0, 11.0, 0, 105)
    )

    # test for nan if nan is passed to a_threshold
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert np.isnan(
        compare_numba_vs_python(
            interpolated_time_point_thresh, w_in, np.nan, 11.0, 0, 105
        )
    )

    # test for nan if nan is passed to t_start
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert np.isnan(
        compare_numba_vs_python(
            interpolated_time_point_thresh, w_in, 1.0, np.nan, 0, 105
        )
    )

    # test for nan if t_start out of range
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert np.isnan(
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1.0, 12, 0, 105)
    )

    # test walk backward mode 'i'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 105)
        == 7.0
    )

    # test walk forward mode 'i'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 105)
        == 4.0
    )

    # test walk backward mode 'f'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 102)
        == 8.0
    )

    # test walk forward mode 'f'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 102)
        == 5.0
    )

    # test walk backward mode 'f'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 99)
        == 7.0
    )

    # test walk forward mode 'f'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 99)
        == 4.0
    )

    # test walk backward mode 'n'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 110)
        == 7.5
    )

    # test walk forward mode 'n'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 110)
        == 4.5
    )

    # test walk backward mode 'l'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1.5, 11, 0, 108)
        == 8.5
    )

    # test walk forward mode 'l'
    w_in = np.concatenate([np.arange(-1, 5, 1), np.arange(-1, 5, 1)], dtype="float")
    assert (
        compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3.5, 0, 1, 108)
        == 4.5
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