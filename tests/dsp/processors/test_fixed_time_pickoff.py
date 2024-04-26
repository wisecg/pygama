import inspect

import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import fixed_time_pickoff


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