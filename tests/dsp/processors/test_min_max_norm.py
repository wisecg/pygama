import numpy as np

from pygama.dsp.processors import min_max_norm


def test_min_max_norm(compare_numba_vs_python):
    """Testing function for the max_min_norm processor."""

    # set up values to use for each test case
    len_wf = 10

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    a = np.array([1])
    assert np.all(np.isnan(compare_numba_vs_python(min_max_norm, w_in, a, a)))

    # test for division by 0
    w_in = np.ones(len_wf)
    a = np.array([0])
    w_out_expected = np.ones(len_wf)
    assert np.allclose(
        compare_numba_vs_python(min_max_norm, w_in, a, a), w_out_expected
    )

    # test for abs(a_max) > abs(a_min)
    a_max = np.array([2])
    a_min = np.array([-1])
    w_out_expected = np.ones(len_wf) / abs(a_max[0])
    assert np.allclose(
        compare_numba_vs_python(min_max_norm, w_in, a_min, a_max), w_out_expected
    )

    # test for abs(a_max) < abs(a_min)
    a_max = np.array([1])
    a_min = np.array([-2])
    w_out_expected = np.ones(len_wf) / abs(a_min[0])
    assert np.allclose(
        compare_numba_vs_python(min_max_norm, w_in, a_min, a_max), w_out_expected
    )