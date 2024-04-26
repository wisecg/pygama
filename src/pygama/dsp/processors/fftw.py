from __future__ import annotations

from typing import Callable

import numpy as np
from numba import guvectorize
from pyfftw import FFTW

from pygama.dsp.utils import ProcChainVarBase
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


def dft(
    w_in: np.ndarray | ProcChainVarBase, w_out: np.ndarray | ProcChainVarBase
) -> Callable:
    """Perform discrete Fourier transforms using the FFTW library.

    Parameters
    ----------
    w_in
        the input waveform.
    w_out
        the output fourier transform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_dft": {
            "function": "dft",
            "module": "dspeed.processors",
            "args": ["wf", "wf_dft"],
            "init_args": ["wf", "wf_dft"]
        }

    Note
    ----
    FFTW optimizes the FFT algorithm based on the size of the arrays, with SIMD
    parallelized commands.  This optimization requires initialization, so this
    is a factory function that returns a Numba gufunc that performs the FFT.
    FFTW works on fixed memory buffers, so you must tell it what memory to use
    ahead of time.  When using this with
    :class:`~.dsp.processing_chain.ProcessingChain`, the output waveform's size,
    dtype and coordinate grid units can be set automatically.  The
    possible `dtypes` for the input/outputs are:

    =============================== ========= =============================== =============
    :class:`numpy.dtype`            Size      :class:`numpy.dtype`            Size
    ------------------------------- --------- ------------------------------- -------------
    ``float32``/``float``           :math:`n` ``complex64``                   :math:`n/2+1`
    ``float64``/``double``          :math:`n` ``complex128``                  :math:`n/2+1`
    ``float128``/``longdouble``     :math:`n` ``complex256``/``clongdouble``  :math:`n/2+1`
    ``complex64``                   :math:`n` ``complex64``                   :math:`n`
    ``complex128``                  :math:`n` ``complex128``                  :math:`n`
    ``complex256``/``clongdouble``  :math:`n` ``complex256``/``clongdouble``  :math:`n`
    =============================== ========= =============================== =============
    """
    # if we have a ProcChainVar, set up the output and get numpy arrays
    if isinstance(w_in, ProcChainVarBase) and isinstance(w_out, ProcChainVarBase):
        c = w_in.dtype.kind
        s = w_in.dtype.itemsize
        if c == "f":
            w_out.update_auto(
                shape=w_in.shape[:-1] + (w_in.shape[-1] // 2 + 1,),
                dtype=np.dtype(f"c{2*s}"),
                period=1.0 / w_in.period / w_in.shape[-1],
            )
        elif c == "c":
            w_out.update_auto(
                shape=w_in.shape,
                dtype=np.dtype(f"c{s}"),
                period=1.0 / w_in.period / w_in.shape[-1],
            )
        w_in = w_in.buffer
        w_out = w_out.buffer

    try:
        dft_fun = FFTW(w_in, w_out, axes=(-1,), direction="FFTW_FORWARD")
    except ValueError:
        raise ValueError(
            "incompatible array types/shapes. See function documentation for allowed values"
        )

    typesig = "void(" + str(w_in.dtype) + "[:, :], " + str(w_out.dtype) + "[:, :])"
    sizesig = "(m, n)->(m, n)" if w_in.shape == w_out.shape else "(m, n),(m, l)"

    @guvectorize(
        [typesig],
        sizesig,
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def dft(wf_in: np.ndarray, dft_out: np.ndarray) -> None:
        dft_fun(wf_in, dft_out)

    return dft


def inv_dft(w_in: np.ndarray, w_out: np.ndarray) -> Callable:
    """Perform inverse discrete Fourier transforms using the FFTW library.

    Parameters
    ----------
    w_in
        the input fourier transformed waveform.
    w_out
        the output time-domain waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_invdft": {
            "function": "inv_dft",
            "module": "dspeed.processors",
            "args": ["wf_dft", "wf_invdft"],
            "init_args": ["wf_dft", "wf_invdft"]
        }

    Note
    ----
    FFTW optimizes the FFT algorithm based on the size of the arrays, with SIMD
    parallelized commands.  This optimization requires initialization, so this
    is a factory function that returns a Numba gufunc that performs the FFT.
    FFTW works on fixed memory buffers, so you must tell it what memory to use
    ahead of time.  When using this with
    :class:`~.dsp.processing_chain.ProcessingChain`, the output waveform's size,
    dtype and coordinate grid units can be set automatically.  The automated
    behavior will produce a real output by default, unless you specify a complex
    output. Possible `dtypes` for the input/outputs are:

    =============================== ============= =============================== =========
    :class:`numpy.dtype`            Size          :class:`numpy.dtype`            Size
    ------------------------------- ------------- ------------------------------- ---------
    ``complex64``                   :math:`n/2+1` ``float32``/``float``           :math:`n`
    ``complex128``                  :math:`n/2+1` ``float64``/``double``          :math:`n`
    ``complex256``/``clongdouble``  :math:`n/2+1` ``float128``/``longdouble``     :math:`n`
    ``complex64``                   :math:`n`     ``complex64``                   :math:`n`
    ``complex128``                  :math:`n`     ``complex128``                  :math:`n`
    ``complex256``/``clongdouble``  :math:`n`     ``complex256``/``clongdouble``  :math:`n`
    =============================== ============= =============================== =========
    """
    # if we have a ProcChainVar, set up the output and get numpy arrays
    if isinstance(w_in, ProcChainVarBase) and isinstance(w_out, ProcChainVarBase):
        s = w_in.dtype.itemsize
        if w_out.dtype == "auto":
            w_out.update_auto(
                shape=w_in.shape[:-1] + (2 * (w_in.shape[-1] - 1),),
                dtype=np.dtype(f"f{s//2}"),
                period=1.0 / w_in.period / w_in.shape[-1],
            )
        else:
            w_out.update_auto(
                shape=(
                    w_in.shape
                    if w_out.dtype.kind == "c"
                    else w_in.shape[:-1] + (2 * (w_in.shape[-1] - 1),)
                ),
                period=1.0 / w_in.period / w_in.shape[-1],
            )
        w_in = w_in.buffer
        w_out = w_out.buffer

    try:
        idft_fun = FFTW(w_in, w_out, axes=(-1,), direction="FFTW_BACKWARD")
    except ValueError:
        raise ValueError(
            "incompatible array types/shapes. See function documentation for allowed values"
        )

    typesig = "void(" + str(w_in.dtype) + "[:, :], " + str(w_out.dtype) + "[:, :])"
    sizesig = "(m, n)->(m, n)" if w_in.shape == w_out.shape else "(m, n),(m, l)"

    @guvectorize(
        [typesig],
        sizesig,
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def inv_dft(wf_in: np.ndarray, dft_out: np.ndarray) -> None:
        idft_fun(wf_in, dft_out)

    return inv_dft


def psd(w_in: np.ndarray, w_out: np.ndarray) -> Callable:
    """Perform discrete Fourier transforms using the FFTW library, and use it to get
    the power spectral density.

    Parameters
    ----------
    w_in
        the input waveform.
    w_out
        the output fourier transform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_psd": {
            "function": "psd",
            "module": "dspeed.processors",
            "args": ["wf", "wf_psd"],
            "init_args": ["wf", "wf_psd"]
        }

    Note
    ----
    FFTW optimizes the FFT algorithm based on the size of the arrays, with SIMD
    parallelized commands.  This optimization requires initialization, so this
    is a factory function that returns a Numba gufunc that performs the FFT.
    FFTW works on fixed memory buffers, so you must tell it what memory to use
    ahead of time.  When using this with
    :class:`~.dsp.processing_chain.ProcessingChain`, the output waveform's size,
    dtype and coordinate grid units can be set automatically.  The
    possible `dtypes` for the input/outputs are:

    =============================== ========= ============================ =============
    :class:`numpy.dtype`            Size      :class:`numpy.dtype`         Size
    ------------------------------- --------- ---------------------------- -------------
    ``complex64``                   :math:`n` ``float32``/``float``        :math:`n`
    ``complex128``                  :math:`n` ``float64``/``double``       :math:`n`
    ``complex256``/``clongdouble``  :math:`n` ``float128``/``longdouble``  :math:`n`
    ``float32``/``float``           :math:`n` ``float32``/``float``        :math:`n/2+1`
    ``float64``/``double``          :math:`n` ``float64``/``double``       :math:`n/2+1`
    ``float128``/``longdouble``     :math:`n` ``float128``/``longdouble``  :math:`n/2+1`
    =============================== ========= ============================ =============
    """
    # if we have a ProcChainVar, set up the output and get numpy arrays
    if isinstance(w_in, ProcChainVarBase) and isinstance(w_out, ProcChainVarBase):
        c = w_in.dtype.kind
        s = w_in.dtype.itemsize
        if c == "f":
            w_out.update_auto(
                shape=w_in.shape[:-1] + (w_in.shape[-1] // 2 + 1,),
                dtype=np.dtype(f"f{s}"),
                period=1.0 / w_in.period / w_in.shape[-1],
            )
        elif c == "c":
            w_out.update_auto(
                shape=w_in.shape,
                dtype=np.dtype(f"f{s//2}"),
                period=1.0 / w_in.period / w_in.shape[-1],
            )
        w_in = w_in.buffer
        w_out = w_out.buffer

    # build intermediate array for the dft, which will be abs'd to get the PSD
    w_dft = np.ndarray(w_out.shape, np.dtype(f"c{w_in.dtype.itemsize*2}"))
    try:
        dft_fun = FFTW(w_in, w_dft, axes=(-1,), direction="FFTW_FORWARD")
    except ValueError:
        raise ValueError(
            "incompatible array types/shapes. See function documentation for allowed values"
        )

    typesig = "void(" + str(w_in.dtype) + "[:, :], " + str(w_out.dtype) + "[:, :])"
    sizesig = "(m, n)->(m, n)" if w_in.shape == w_out.shape else "(m, n),(m, l)"

    @guvectorize(
        [typesig],
        sizesig,
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def psd(wf_in: np.ndarray, psd_out: np.ndarray) -> None:
        dft_fun(wf_in, w_dft)
        np.abs(w_dft, psd_out)

    return psd