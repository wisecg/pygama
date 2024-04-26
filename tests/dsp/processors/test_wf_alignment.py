import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import wf_alignment


def test_wf_alignment(compare_numba_vs_python):
    len_wf = 20
    size = 10

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    w_out = np.empty(size)
    assert np.all(compare_numba_vs_python(wf_alignment, w_in, 1, 1, size, w_out))

    # ensure to have a valid output
    w_in = np.ones(len_wf)
    assert np.all(compare_numba_vs_python(wf_alignment, w_in, 1, 1, size, w_out))

    # tests on centroid
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, np.nan, 1, size, w_out)

    # tests on shift
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, np.nan, size, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, -1, size, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, len_wf + 1, size, w_out)

    # tests on size
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, 1, np.nan, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, 1, -1, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, 1, 0, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_alignment, w_in, 1, 1, len_wf + 1, w_out)