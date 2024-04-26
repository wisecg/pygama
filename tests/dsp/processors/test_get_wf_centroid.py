import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import get_wf_centroid


def test_get_wf_centroid(compare_numba_vs_python):
    len_wf = 20

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    assert np.isnan(compare_numba_vs_python(get_wf_centroid, w_in, 1))

    # test for nan if nan is passed to shift
    w_in = np.ones(len_wf)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get_wf_centroid, w_in, np.nan)

    # test for nan if shift is negative
    w_in = np.ones(len_wf)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get_wf_centroid, w_in, -1)

    # test for nan if shift is too large
    w_in = np.ones(len_wf)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get_wf_centroid, w_in, len_wf)