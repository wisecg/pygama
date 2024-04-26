from __future__ import annotations

import pickle
from typing import Callable

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


def svm_predict(svm_file: str) -> Callable:
    """
    Apply a Support Vector Machine (SVM) to an input waveform to
    predict a data cleaning label.

    Note
    ----
    This processor is composed of a factory function that is called
    using the ``init_args`` argument. The input waveform and output
    label are passed using ``args``.


    Parameters
    ----------
    svm_file
       The name of the file with the trained SVM ``svm_p*_r*_T***Z.sav``


    JSON Configuration Example
    --------------------------
    .. code-block :: json

        "svm_label":{
            "function": "svm_predict",
            "module": "dspeed.processors",
            "args": ["dwt_norm", "svm_label"],
            "unit": "",
            "prereqs": ["dwt_norm"],
            "init_args": ["'svm_p*_r*_T***Z.sav'"]
        }
    """

    with open(svm_file, "rb") as f:
        svm = pickle.load(f)

    @guvectorize(
        [
            "void(float32[:], float32[:])",
            "void(float64[:], float64[:])",
        ],
        "(n),()",
        **nb_kwargs(
            forceobj=True,
        ),
    )
    def svm_out(w_in: np.ndarray, label_out: float) -> None:
        """
        Parameters
        ----------
        w_in
           The input waveform (has to be a max_min normalized discrete wavelet transform)
        label_out
           The predicted label by the trained SVM for the input waveform.
        """
        label_out[0] = np.nan

        if np.isnan(w_in).any():
            return

        if w_in.ndim == 1:
            label_out[0] = svm.predict(w_in.reshape(1, -1))
        else:
            label_out[0] = svm.predict(w_in)

    return svm_out