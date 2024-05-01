from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),()->()",
    **nb_kwargs,
)
def time_point_thresh(
    w_in: np.ndarray, a_threshold: float, t_start: int, walk_forward: int, t_out: float
) -> None:
    """Find the index where the waveform value crosses the threshold, walking
    either forward or backward from the starting index.

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold
        the threshold value.
    t_start
        the starting index.
    walk_forward
        the backward (``0``) or forward (``1``) search direction.
    t_out
        the index where the waveform value crosses the threshold.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "tp_0": {
            "function": "time_point_thresh",
            "module": "dspeed.processors",
            "args": ["wf_atrap", "bl_std", "tp_start", 0, "tp_0"],
            "unit": "ns"
        }
    """
    t_out[0] = np.nan

    if (
        np.isnan(w_in).any()
        or np.isnan(a_threshold)
        or np.isnan(t_start)
        or np.isnan(walk_forward)
    ):
        return

    if np.floor(t_start) != t_start:
        raise DSPFatal("The starting index must be an integer")

    if np.floor(walk_forward) != walk_forward:
        raise DSPFatal("The search direction must be an integer")

    if int(t_start) < 0 or int(t_start) >= len(w_in):
        raise DSPFatal("The starting index is out of range")

    if int(walk_forward) == 1:
        for i in range(int(t_start), len(w_in) - 1, 1):
            if w_in[i] <= a_threshold < w_in[i + 1]:
                t_out[0] = i
                return
    else:
        for i in range(int(t_start), 0, -1):
            if w_in[i - 1] < a_threshold <= w_in[i]:
                t_out[0] = i
                return


@guvectorize(
    [
        "void(float32[:], float32, float32, int64, char, float32[:])",
        "void(float64[:], float64, float64, int64, char, float64[:])",
    ],
    "(n),(),(),(),()->()",
    **nb_kwargs,
)
def interpolated_time_point_thresh(
    w_in: np.ndarray,
    a_threshold: float,
    t_start: int,
    walk_forward: int,
    mode_in: np.int8,
    t_out: float,
) -> None:
    """Find the time where the waveform value crosses the threshold

    Search performed walking either forward or backward from the starting
    index. Use interpolation to estimate a time between samples. Interpolation
    mode selected with `mode_in`.

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold
        the threshold value.
    t_start
        the starting index.
    walk_forward
        the backward (``0``) or forward (``1``) search direction.
    mode_in
        Character selecting which interpolation method to use. Note this
        must be passed as a ``int8``, e.g. ``ord('i')``. Options:

        * ``i`` -- integer `t_in`; equivalent to
          :func:`~.dsp.processors.fixed_sample_pickoff`
        * ``b`` -- before; closest integer sample before threshold crossing
        * ``a`` -- after; closest integer sample after threshold crossing
        * ``r`` -- round; round to nearest integer sample to threshold crossing
        * ``l`` -- linear interpolation

        The following modes are meant to mirror the options
        dspeed.upsampler

        * ``f`` -- floor; interpolated values are at previous neighbor.
          Equivalent to ``a``
        * ``c`` -- ceiling, interpolated values are at next neighbor.
          Equivalent to ``b``
        * ``n`` -- nearest-neighbor interpolation; threshold crossing is
          half-way between samples
        * ``h`` -- Hermite cubic spline interpolation (*not implemented*)
        * ``s`` -- natural cubic spline interpolation (*not implemented*)
    t_out
        the index where the waveform value crosses the threshold.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "tp_0": {
            "function": "time_point_thresh",
            "module": "dspeed.processors",
            "args": ["wf_atrap", "bl_std", "tp_start", 0, "'l'", "tp_0"],
            "unit": "ns"
        }
    """
    t_out[0] = np.nan

    if (
        np.isnan(w_in).any()
        or np.isnan(a_threshold)
        or np.isnan(t_start)
        or np.isnan(walk_forward)
    ):
        return

    if t_start < 0 or t_start >= len(w_in):
        return

    i_cross = -1
    if walk_forward > 0:
        for i in range(int(t_start), len(w_in) - 1, 1):
            if w_in[i] <= a_threshold < w_in[i + 1]:
                i_cross = i
                break
    else:
        for i in range(int(t_start), 1, -1):
            if w_in[i - 1] < a_threshold <= w_in[i]:
                i_cross = i - 1
                break

    if i_cross == -1:
        return

    if mode_in == ord("i"):  # return index before crossing
        t_out[0] = i_cross
    elif mode_in in (ord("a"), ord("f")):  # return index after crossing
        t_out[0] = i_cross + 1
    elif mode_in in (ord("b"), ord("c")):  # return index before crossing
        t_out[0] = i_cross
    elif mode_in == ord("r"):  # return closest index to crossing
        if abs(a_threshold - w_in[i_cross]) < abs(a_threshold - w_in[i_cross + 1]):
            t_out[0] = i_cross
        else:
            t_out[0] = i_cross + 1
    elif mode_in == ord("n"):  # nearest-neighbor; return half-way between samps
        t_out[0] = i_cross + 0.5
    elif mode_in == ord("l"):  # linear
        t_out[0] = i_cross + (a_threshold - w_in[i_cross]) / (
            w_in[i_cross + 1] - w_in[i_cross]
        )
    else:
        raise DSPFatal("Unrecognized interpolation mode")


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32, char, float32[:])",
        "void(float64[:], float64[:], float64, float64, char, float64[:])",
    ],
    "(n),(m),(),(),()->(m)",
    **nb_kwargs,
)
def multi_time_point_thresh(
    w_in: np.ndarray,
    a_threshold: np.ndarray,
    t_start: int,
    polarity: int,
    mode_in: np.int8,
    t_out: np.ndarray,
) -> None:
    """Find the time where the waveform value crosses the threshold

    Search performed walking either forward or backward from the starting
    index. Use interpolation to estimate a time between samples. Interpolation
    mode selected with `mode_in`.

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold
        list of threshold values.
    t_start
        the starting index.
    polarity
        is the average slope of the waveform up (>0) or down (<0) in the
        search region; only sign matters, not value. Raise Exception if 0.
    mode_in
        Character selecting which interpolation method to use. Note this
        must be passed as a ``int8``, e.g. ``ord('i')``. Options:

        * ``i`` -- integer `t_in`; equivalent to
          :func:`~.dsp.processors.fixed_sample_pickoff`
        * ``b`` -- before; closest integer sample before threshold crossing
        * ``a`` -- after; closest integer sample after threshold crossing
        * ``r`` -- round; round to nearest integer sample to threshold crossing
        * ``l`` -- linear interpolation

        The following modes are meant to mirror the options
        dspeed.upsampler

        * ``f`` -- floor; interpolated values are at previous neighbor.
          Equivalent to ``a``
        * ``c`` -- ceiling, interpolated values are at next neighbor.
          Equivalent to ``b``
        * ``n`` -- nearest-neighbor interpolation; threshold crossing is
          half-way between samples
        * ``h`` -- Hermite cubic spline interpolation (*not implemented*)
        * ``s`` -- natural cubic spline interpolation (*not implemented*)
    t_out
        the index where the waveform value crosses the threshold.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "tp_0": {
            "function": "time_point_thresh",
            "module": "dspeed.processors",
            "args": ["wf_atrap", "bl_std", "tp_start", 0, "'l'", "tp_0"],
            "unit": "ns"
        }
    """
    t_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(a_threshold).any() or np.isnan(t_start):
        return

    if t_start < 0 or t_start >= len(w_in):
        return

    # make polarity +/- 1
    if polarity > 0:
        polarity = 1
    elif polarity < 0:
        polarity = -1
    else:
        raise DSPFatal("polarity cannot be 0")

    sorted_idx = np.argsort(a_threshold)

    # Get initial values for search
    t_start = int(t_start)
    a_start = w_in[t_start]
    i_start = len(sorted_idx)
    for i in range(len(sorted_idx)):
        if a_threshold[sorted_idx[i]] >= a_start:
            i_start = i
            break

    # Search for timepoints at larger values
    i_tp = i_start
    if i_tp < len(sorted_idx):
        idx = sorted_idx[i_tp]
        for i_wf in range(t_start, len(w_in) - 1 if polarity > 0 else -1, polarity):
            if i_tp >= len(sorted_idx):
                break
            while w_in[i_wf] <= a_threshold[idx] < w_in[i_wf + polarity]:
                if mode_in == ord("i"):  # return index closest to start of search
                    t_out[idx] = i_wf
                elif mode_in in (ord("a"), ord("f")):  # return index after crossing
                    t_out[idx] = i_wf if polarity < 0 else i_wf + 1
                elif mode_in in (ord("b"), ord("c")):  # return index before crossing
                    t_out[idx] = i_wf if polarity > 0 else i_wf - 1
                elif mode_in == ord("r"):  # round; return closest index
                    if (
                        a_threshold[idx] - w_in[i_wf]
                        < w_in[i_wf + polarity] - a_threshold[sorted_idx[i_tp]]
                    ):
                        t_out[idx] = i_wf
                    else:
                        t_out[idx] = i_wf + polarity
                elif mode_in == ord(
                    "n"
                ):  # nearest-neighbor; return half-way between samps
                    t_out[idx] = i_wf + 0.5 * polarity
                elif mode_in == ord("l"):  # linear
                    t_out[idx] = i_wf + (a_threshold[idx] - w_in[i_wf]) / (
                        w_in[i_wf + polarity] - w_in[i_wf]
                    )
                else:
                    raise DSPFatal("Unrecognized interpolation mode")
                i_tp += 1
                if i_tp >= len(sorted_idx):
                    break
                idx = sorted_idx[i_tp]

    # Search for timepoints at smaller values
    i_tp = i_start - 1
    if i_tp >= 0:
        idx = sorted_idx[i_tp]
        for i_wf in range(
            t_start - 1, len(w_in) - 1 if polarity < 0 else -1, -polarity
        ):
            if i_tp < 0:
                break
            while w_in[i_wf] <= a_threshold[idx] < w_in[i_wf + polarity]:
                if mode_in == ord("i"):  # return index closest to start of search
                    t_out[idx] = i_wf
                elif mode_in in (ord("a"), ord("f")):  # return index after crossing
                    t_out[idx] = i_wf if polarity < 0 else i_wf + 1
                elif mode_in in (ord("b"), ord("c")):  # return index before crossing
                    t_out[idx] = i_wf if polarity > 0 else i_wf - 1
                elif mode_in == ord("r"):  # round; return closest index
                    if (
                        a_threshold[idx] - w_in[i_wf]
                        < w_in[i_wf + polarity] - a_threshold[idx]
                    ):
                        t_out[idx] = i_wf
                    else:
                        t_out[idx] = i_wf + polarity
                elif mode_in == ord(
                    "n"
                ):  # nearest-neighbor; return half-way between samps
                    t_out[idx] = i_wf + 0.5 * polarity
                elif mode_in == ord("l"):  # linear
                    t_out[idx] = i_wf + (a_threshold[idx] - w_in[i_wf]) / (
                        w_in[i_wf + polarity] - w_in[i_wf]
                    )
                else:
                    raise DSPFatal("Unrecognized interpolation mode")
                i_tp -= 1
                if i_tp < 0:
                    break
                idx = sorted_idx[i_tp]


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32, uint32[:], float32[:], float32[:])",
        "void(float64[:], float64, float64, float64, float64, uint32[:], float64[:], float64[:])",
    ],
    "(n),(),(),(),(),(),(m),(m)",
    **nb_kwargs,
)
def bi_level_zero_crossing_time_points(
    w_in: np.ndarray,
    a_pos_threshold_in: float,
    a_neg_threshold_in: float,
    gate_time_in: int,
    t_start_in: int,
    n_crossings_out: int,
    polarity_out: np.array,
    t_trig_times_out: np.array,
) -> None:
    """
    Find the indices where a waveform value crosses 0 after crossing the threshold and reaching the next threshold within some gate time.
    Works on positive and negative polarity waveforms.
    Useful for finding pileup events with the RC-CR^2 filter.

    Parameters
    ----------
    w_in
        the input waveform.
    a_pos_threshold_in
        the positive threshold value.
    a_neg_threshold_in
        the negative threshold value.
    gate_time_in
        The number of samples that the next threshold crossing has to be within in order to count a 0 crossing
    t_start_in
        the starting index.
    n_crossings_out
        the number of zero-crossings found. Note: if there are more zeros than elements in output arrays, this will continue to increment but the polarity and trigger time will not be added to the output buffers
    polarity_out
        An array holding the polarity of identified pulses. 0 for negative and 1 for positive
    t_trig_times_out
        the indices where the waveform value has crossed the threshold and returned to 0.
        Arrays of fixed length (padded with :any:`numpy.nan`) that hold the
        indices of the identified trigger times.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "trig_times_out": {
            "function": "multi_trigger_time",
            "module": "dspeed.processors",
            "args": ["wf_rc_cr2", "5", "-10", 0, "n_crossings", "polarity_out(shape = 20)", "trig_times_out(shape = 20)"],
            "unit": "ns"
        }
    """
    # prepare output
    t_trig_times_out[:] = np.nan
    polarity_out[:] = np.nan

    # Check everything is ok
    if (
        np.isnan(w_in).any()
        or np.isnan(a_pos_threshold_in)
        or np.isnan(a_neg_threshold_in)
        or np.isnan(t_start_in)
    ):
        return

    if np.floor(t_start_in) != t_start_in:
        raise DSPFatal("The starting index must be an integer")

    if int(t_start_in) < 0 or int(t_start_in) >= len(w_in):
        raise DSPFatal("The starting index is out of range")

    if len(polarity_out) != len(t_trig_times_out):
        raise DSPFatal("The output arrays are of different lengths.")

    gate_time_in = int(gate_time_in)  # make sure this is an integer!
    # Perform the processing
    is_above_thresh = False
    is_below_thresh = False
    crossed_zero = False
    n_crossings_out[0] = 0
    for i in range(int(t_start_in), len(w_in) - 1, 1):
        if is_below_thresh and (w_in[i] <= 0 < w_in[i + 1]):
            crossed_zero = True
            neg_trig_time_candidate = i

        # Either we go above threshold
        if w_in[i] <= a_pos_threshold_in < w_in[i + 1]:
            if crossed_zero and is_below_thresh:
                if i - is_below_thresh < gate_time_in:
                    if n_crossings_out[0] < len(polarity_out):
                        t_trig_times_out[n_crossings_out[0]] = neg_trig_time_candidate
                        polarity_out[n_crossings_out[0]] = 0
                    n_crossings_out[0] += 1
                else:
                    is_above_thresh = i

                is_below_thresh = False
                crossed_zero = False
            else:
                is_above_thresh = i

        if is_above_thresh and (w_in[i] >= 0 > w_in[i + 1]):
            crossed_zero = True
            pos_trig_time_candidate = i

        # Or we go below threshold
        if w_in[i] >= a_neg_threshold_in > w_in[i + 1]:
            if crossed_zero and is_above_thresh:
                if i - is_above_thresh < gate_time_in:
                    if n_crossings_out[0] < len(polarity_out):
                        t_trig_times_out[n_crossings_out[0]] = pos_trig_time_candidate
                        polarity_out[n_crossings_out[0]] = 1
                    n_crossings_out[0] += 1
                else:
                    is_below_thresh = i
                is_above_thresh = False
                crossed_zero = False
            else:
                is_below_thresh = i