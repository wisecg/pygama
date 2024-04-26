import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import discrete_wavelet_transform


def test_discrete_wavelet_transform(compare_numba_vs_python):
    """Testing function for the discrete_wavelet_transform processor."""

    # set up values to use for each test case
    len_wf_in = 16
    wave_type = ord("h")
    level = 2
    coeff = ord("a")
    len_wf_out = 4

    # ensure the DSPFatal is raised for a negative level
    w_in = np.ones(len_wf_in)
    w_out = np.empty(len_wf_out)
    with pytest.raises(DSPFatal):
        discrete_wavelet_transform(w_in, -1, wave_type, coeff, w_out)

    # ensure that a valid input gives the expected output
    w_out_expected = np.ones(len_wf_out) * 2 ** (level / 2)
    assert np.allclose(
        compare_numba_vs_python(
            discrete_wavelet_transform, w_in, level, wave_type, coeff, w_out
        ),
        w_out_expected,
    )

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(len_wf_in)
    w_in[4] = np.nan
    w_out = np.empty(len_wf_out)
    assert np.all(
        np.isnan(
            compare_numba_vs_python(
                discrete_wavelet_transform, w_in, level, wave_type, coeff, w_out
            )
        )
    )