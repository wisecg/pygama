from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32, float32, float32[:], float32[:], uint32[:], uint32[:])",
        "void(float64[:], float64, float64, float64, float64, float64, float64[:], float64[:], uint32[:], uint32[:])",
    ],
    "(n),(),(),(),(),(),(m),(m),(),()",
    **nb_kwargs,
)
def get_multi_local_extrema(
    w_in: np.ndarray,
    a_delta_max_in: float,
    a_delta_min_in: float,
    search_direction: int,
    a_abs_max_in: float,
    a_abs_min_in: float,
    vt_max_out: np.ndarray,
    vt_min_out: np.ndarray,
    n_max_out: int,
    n_min_out: int,
) -> None:
    """Get lists of indices of the local maxima and minima of data.

    Converted from a `MATLAB script <http://billauer.co.il/peakdet.html>`_ by
    E. Billauer. See the parameters description for details about search modes
    and further settings.

    Parameters
    ----------
    w_in
        the array of data within which extrema will be found.
    a_delta_min_in
        the relative level by which data must vary (in the search direction)
        about a candidate minimum in order for it to be tagged.
    a_delta_max_in
        the relative level by which data must vary (in the search direction)
        about a candidate maximum in order for it to be tagged.
    search_direction
        the direction in which the input waveform is processed.

        * ``0``: one sweep, first to last sample
        * ``1``: one sweep, last to first sample
        * ``2``: two sweeps, in both directions. Largest common set of found
          extrema is returned (logical AND)
        * ``3``: two sweeps, in both directions. Union of found extrema is
          returned (logical OR)

    a_abs_min_in
        a candidate minimum is tagged only if its value is smaller than this.
    a_abs_max_in
        a candidate maximum is tagged only if its value is larger than this.
    vt_min_out
        arrays of fixed length (padded with :any:`numpy.nan`) that hold the
        indices of the identified minima.
    vt_max_out
        arrays of fixed length (padded with :any:`numpy.nan`) that hold the
        indices of the identified maxima.
    n_min_out
        the number of minima found in a waveform.
    n_max_out
        the number of maxima found in a waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "vt_max_out, vt_min_out, n_max_out, n_min_out": {
            "function": "get_multi_local_extrema",
            "module": "dspeed.processors",
            "args": [
                "waveform",
                5, 0.1,
                1,
                20, 0,
                "vt_max_out(20)",
                "vt_min_out(20)",
                "n_max_out",
                "n_min_out"
            ],
            "unit": ["ns", "ns", "none", "none"]
        }
    """
    # prepare output
    vt_max_out[:] = np.nan
    vt_min_out[:] = np.nan
    n_max_out[0] = 0
    n_min_out[0] = 0

    # initialize internal counters
    n_max_left_counter = 0
    n_max_right_counter = 0
    n_min_left_counter = 0
    n_min_right_counter = 0

    # initialize temporary arrays
    left_vt_max = np.zeros(len(vt_max_out), dtype=np.float_)
    left_vt_min = np.zeros(len(vt_min_out), dtype=np.float_)
    right_vt_max = np.zeros(len(vt_max_out), dtype=np.float_)
    right_vt_min = np.zeros(len(vt_min_out), dtype=np.float_)

    left_vt_max[:] = np.nan
    left_vt_min[:] = np.nan
    right_vt_max[:] = np.nan
    right_vt_min[:] = np.nan

    # Checks

    if np.isnan(w_in).any() or np.isnan(a_delta_max_in) or np.isnan(a_delta_min_in):
        return

    if (not len(vt_max_out) < len(w_in)) or (not len(vt_min_out) < len(w_in)):
        raise DSPFatal(
            "The length of your return array must be smaller than the length of your waveform"
        )
    if (not a_delta_max_in >= 0) or (not a_delta_min_in >= 0):
        raise DSPFatal("Delta must be positive")

    # now loop over data
    # left to right search
    if (search_direction == 0) or (search_direction > 1):
        find_max = True
        imax, imin = 0, 0
        for i in range(len(w_in)):
            if w_in[i] > w_in[imax]:
                imax = i
            if w_in[i] < w_in[imin]:
                imin = i
            if find_max:
                # if the sample is less than the current max by more than a_delta_in,
                # declare the previous one a maximum, then set this as the new "min"
                if (
                    (w_in[i] < w_in[imax] - a_delta_max_in)
                    and (int(n_max_left_counter) < int(len(left_vt_max)))
                    and (w_in[imax] > a_abs_max_in)
                ):
                    left_vt_max[int(n_max_left_counter)] = imax
                    n_max_left_counter += 1
                    imin = i
                    find_max = False
            else:
                # if the sample is more than the current min by more than a_delta_in,
                # declare the previous one a minimum, then set this as the new "max"
                if (
                    (w_in[i] > w_in[imin] + a_delta_min_in)
                    and (int(n_min_left_counter) < int(len(left_vt_min)))
                    and (w_in[imin] < a_abs_min_in)
                ):
                    left_vt_min[int(n_min_left_counter)] = imin
                    n_min_left_counter += 1
                    imax = i
                    find_max = True

    # right to left search
    if search_direction > 0:
        find_max = True
        imax, imin = len(w_in) - 1, len(w_in) - 1
        for i in range(len(w_in) - 1, -1, -1):
            if w_in[i] > w_in[imax]:
                imax = i
            if w_in[i] < w_in[imin]:
                imin = i
            if find_max:
                # if the sample is less than the current max by more than a_delta_in,
                # declare the previous one a maximum, then set this as the new "min"
                if (
                    w_in[i] < w_in[imax] - a_delta_max_in
                    and int(n_max_right_counter) < int(len(right_vt_max))
                    and w_in[imax] > a_abs_max_in
                ):
                    right_vt_max[int(n_max_right_counter)] = imax
                    n_max_right_counter += 1
                    imin = i
                    find_max = False
            else:
                # if the sample is more than the current min by more than a_delta_in,
                # declare the previous one a minimum, then set this as the new "max"
                if (
                    w_in[i] > w_in[imin] + a_delta_min_in
                    and int(n_min_right_counter) < int(len(right_vt_min))
                    and w_in[imin] < a_abs_min_in
                ):
                    right_vt_min[int(n_min_right_counter)] = imin
                    n_min_right_counter += 1
                    imax = i
                    find_max = True

    # set output
    # left search
    if search_direction == 0:
        n_max_out[0] = n_max_left_counter
        n_min_out[0] = n_min_left_counter
        vt_max_out[:] = left_vt_max
        vt_min_out[:] = left_vt_min

    # right search
    elif search_direction == 1:
        n_max_out[0] = n_max_right_counter
        n_min_out[0] = n_min_right_counter
        vt_max_out[:] = right_vt_max
        vt_min_out[:] = right_vt_min

    # conservative search (only extrema found in both directions)
    elif search_direction == 2:
        # sort the right search result. Left search should be already sorted
        right_vt_max = np.sort(right_vt_max)
        right_vt_min = np.sort(right_vt_min)

        rge_right = (right_vt_max[~np.isnan(right_vt_max)]).astype(np.int_)
        rge_left = (left_vt_max[~np.isnan(left_vt_max)]).astype(np.int_)

        # only continue if both arrays have something in them
        if len(rge_right) > 0 and len(rge_left) > 0:
            r_max = int(rge_right[-1])
            r_min = int(rge_right[0])
            rge = r_max - r_min

            # coincidence mask
            coin_mask = np.zeros(len(rge_left), dtype=np.bool_)

            # helper array: 1 if integer exists in rge_right
            helper_ar = np.zeros(rge + 1, dtype=np.bool_)
            helper_ar[rge_right - r_min] = 1

            # masking
            # only integers in [r_min,r_max] could be in coincidence
            basic_mask = (rge_left <= r_max) & (rge_left >= r_min)

            # take candidates of rge_left surviving the basic mask,
            # shift by rmin to get the indices for the integer mask.
            # Apply basic mask to coincidenc mask and indexed integer mask.
            coin_mask[basic_mask] = helper_ar[rge_left[basic_mask] - r_min]

            # Apply coincidence mask to rge_left & fill into output
            rge_left = rge_left[coin_mask]
            vt_max_out[: len(rge_left)] = rge_left
            n_max_out[0] = len(rge_left)
        else:
            n_max_out[0] = 0
        # Do the same for the minima
        rge_right = (right_vt_max[~np.isnan(right_vt_min)]).astype(np.int_)
        rge_left = (left_vt_max[~np.isnan(left_vt_min)]).astype(np.int_)

        # only continue if both arrays have something in them
        if len(rge_right) > 0 and len(rge_left) > 0:
            r_max = int(rge_right[-1])
            r_min = int(rge_right[0])
            rge = r_max - r_min

            # coincidence mask
            coin_mask = np.zeros(len(rge_left), dtype=np.bool_)

            # helper array: 1 if integer exists in rge_right
            helper_ar = np.zeros(rge + 1, dtype=np.bool_)
            helper_ar[rge_right - r_min] = 1

            # masking
            # only integers in [r_min,r_max] could be in coincidence
            basic_mask = (rge_left <= r_max) & (rge_left >= r_min)

            # take candidates of rge_left surviving the basic mask,
            # shift by rmin to get the indices for the integer mask.
            # Apply basic mask to coincidenc mask and indexed integer mask.
            coin_mask[basic_mask] = helper_ar[rge_left[basic_mask] - r_min]

            # Apply coincidence mask to rge_left & fill into output
            rge_left = rge_left[coin_mask]
            vt_min_out[: len(rge_left)] = rge_left
            n_min_out[0] = len(rge_left)
        else:
            n_min_out[0] = 0

    # aggressive search (extrema found in either directions)
    elif search_direction == 3:
        both = np.unique(np.append(left_vt_max, right_vt_max))
        if len(vt_max_out) <= len(both):
            vt_max_out[:] = both[: len(vt_max_out)]
            n_max_out[0] = len(vt_max_out[~np.isnan(vt_max_out)])
        else:
            vt_max_out[: len(both)] = both
            n_max_out[0] = len(both[~np.isnan(both)])

        both = np.unique(np.append(left_vt_min, right_vt_min))
        if len(vt_min_out) <= len(both):
            vt_min_out[:] = both[: len(vt_min_out)]
            n_min_out[0] = len(vt_min_out[~np.isnan(vt_min_out)])
        else:
            vt_min_out[: len(both)] = both
            n_min_out[0] = len(both[~np.isnan(both)])

    else:
        raise DSPFatal("search direction type not found.")