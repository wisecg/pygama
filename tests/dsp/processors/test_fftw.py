import numpy as np
import pytest

from pygama.dsp.processors import dft, inv_dft, psd


def test_dft(compare_numba_vs_python):
    """Testing function for the discrete fourier transform."""

    # set up arrays to use
    w_in = np.zeros(shape=(16, 100), dtype="float32")
    w_dft = np.zeros(shape=(16, 51), dtype="complex64")

    # ensure the DSPFatal is raised for incorrect shaps/types
    with pytest.raises(ValueError):
        dft(w_in, np.zeros_like(w_in))

    # ensure that a valid input gives the expected output
    w_expected = np.zeros_like(w_dft)
    w_expected[:, 0] = 100.0

    dft_func = dft(w_in, w_dft)
    w_in[:] = 1.0

    assert np.allclose(
        compare_numba_vs_python(dft_func, w_in, w_dft),
        w_expected,
    )

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in[:, 10] = np.nan
    assert np.all(np.isnan(compare_numba_vs_python(dft_func, w_in, w_dft)))


def test_inv_dft(compare_numba_vs_python):
    """Testing function for the inverse discrete fourier transform."""

    # set up arrays to use
    w_in = np.zeros(shape=(16, 51), dtype="complex64")
    w_inv_dft = np.zeros(shape=(16, 100), dtype="float32")

    # ensure the DSPFatal is raised for incorrect shaps/types
    with pytest.raises(ValueError):
        inv_dft(w_in, np.zeros(shape=(16, 51), dtype="float32"))

    # ensure that a valid input gives the expected output
    w_expected = np.zeros_like(w_inv_dft)
    w_expected[:, 0] = 1.0

    inv_dft_func = inv_dft(w_in, w_inv_dft)
    w_in[:] = 1.0

    assert np.allclose(
        compare_numba_vs_python(inv_dft_func, w_in, w_inv_dft),
        w_expected,
    )

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in[:, 10] = np.nan
    assert np.all(np.isnan(compare_numba_vs_python(inv_dft_func, w_in, w_inv_dft)))


def test_psd(compare_numba_vs_python):
    """Testing function for the power spectral density."""

    # set up arrays to use
    w_in = np.zeros(shape=(16, 100), dtype="float32")
    w_psd = np.zeros(shape=(16, 51), dtype="float32")

    # ensure the DSPFatal is raised for incorrect shaps/types
    with pytest.raises(ValueError):
        psd(w_in, np.zeros_like(w_in))

    # ensure that a valid input gives the expected output
    w_expected = np.zeros_like(w_psd)
    w_expected[:, 0] = 100.0

    psd_func = psd(w_in, w_psd)
    w_in[:] = 1.0

    assert np.allclose(
        compare_numba_vs_python(psd_func, w_in, w_psd),
        w_expected,
    )

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in[:, 10] = np.nan
    assert np.all(np.isnan(compare_numba_vs_python(psd_func, w_in, w_psd)))