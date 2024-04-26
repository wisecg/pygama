import inspect

import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import get_multi_local_extrema

#                .
#               . .
#              .   .
#             .     .
#            .       .
#       .   .         .   .
#      . . .           . . .
#     .   .             .   .
#    .                       .
#   .                         .
# ..                           ..
# 0123456789012345678901234567890
#           1         2         3
wf = np.array([0, 0, 1, 2, 3, 4, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10,
               9, 8, 7, 6, 5, 4, 3, 4, 5, 4, 3, 2, 1, 0, 0])  # fmt: skip


# delta_min=delta_max tests (all searches should behave identical)


# L->R
def test_get_multi_local_extrema_ltor(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        0,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert n_min_out[0] == 0


# L<-R
def test_get_multi_local_extrema_rtol(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        1,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert n_min_out[0] == 0


# L<-c->R
def test_get_multi_local_extrema_both_cons(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        2,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1


# L<-a->R
def test_get_multi_local_extrema_both_agro(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        3,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1


# delta_min != delta_max (result should be search direction dependent)


# L->R
def test_get_multi_local_extrema_ltor_asym(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        0,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, 24, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([22, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 2
    assert n_min_out[0] == 1


# L<-R
def test_get_multi_local_extrema_rtol_asym(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        1,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, 6, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([8, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 2
    assert n_min_out[0] == 1


# L<-c->R
def test_get_multi_local_extrema_both_cons_asym(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        2,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert n_min_out[0] == 0


# L<-a->R
def test_get_multi_local_extrema_both_agro_asym(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        3,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([6, 15, 24]), equal_nan=True)
    assert np.array_equal(min_out, np.array([8, 22, np.nan]), equal_nan=True)
    assert n_max_out[0] == 3
    assert n_min_out[0] == 2


# Absolute threshold test with aggressive both direction search
def test_get_multi_local_extrema_both_agro_asym_abs(compare_numba_vs_python):
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        3,
        8,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([8, 22, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert n_min_out[0] == 2


# And now test if everything works well when vectorized


# return nan tests
def test_get_multi_local_extrema_return_on_nan(compare_numba_vs_python):
    wf = np.ones(20)
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    wf[4] = np.nan
    assert (
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            3,
            1,
            3,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
        )
        == 0
    )

    wf[4] = 3
    assert (
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            np.nan,
            3,
            3,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
        )
        == 0
    )
    assert (
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            np.nan,
            3,
            np.nan,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
        )
        == 0
    )
    assert (
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            np.nan,
            np.nan,
            3,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
        )
        == 0
    )


# DSP Fatal test
def test_get_multi_local_extrema_dsp_fatal():
    wf = np.ones(20)
    max_out = np.zeros(len(wf) + 1)
    max_out[:] = np.nan
    min_out = np.zeros(len(wf) + 1)
    min_out[:] = np.nan
    n_min_out = np.zeros(1, dtype="uint32")
    n_max_out = np.zeros(1, dtype="uint32")

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, 3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )
    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, 3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )

    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, -3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )

    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, -3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, 3, -1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )

    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, 3, -1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, 3, 1, -3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )

    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, 3, 1, -3, 8, 20, max_out, min_out, n_max_out, n_min_out
        )