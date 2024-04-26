from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32, float32, float32, float32[:])",
        "void(float64, float64, float64, float64[:])",
    ],
    "(),(),(),(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def cusp_filter(sigma: float, flat: int, decay: int, kernel: np.array) -> None:
    """Calculates CUSP kernel.

    Parameters
    ----------
    sigma
        the curvature of the rising and falling part of the kernel.
    flat
        the length of the flat section.
    decay
        the decay constant of the exponential to be convolved.
    kernel
        the calculated kernel

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "kern_cusp": {
            "function": "cusp_filter",
            "module": "dspeed.processors",
            "args": ["10*us", "3*us", "400*us", "kern_cusp"],
            "unit": "ADC"
        }
    """

    if sigma < 0:
        raise DSPFatal("The curvature parameter must be positive")

    if flat < 0:
        raise DSPFatal("The length of the flat section must be positive")

    if np.floor(flat) != flat:
        raise DSPFatal("The length of the flat section must be an integer")

    if decay < 0:
        raise DSPFatal("The decay constant must be positive")

    lt = int((len(kernel) - flat) / 2)
    flat_int = int(flat)
    for ind in range(0, lt, 1):
        kernel[ind] = float(np.sinh(ind / sigma) / np.sinh(lt / sigma))
    for ind in range(lt, lt + flat_int + 1, 1):
        kernel[ind] = 1
    for ind in range(lt + flat_int + 1, len(kernel), 1):
        kernel[ind] = float(np.sinh((len(kernel) - ind) / sigma) / np.sinh(lt / sigma))

    den = [1, -np.exp(-1 / decay)]
    kernel[:] = np.convolve(kernel, den, "same")


@guvectorize(
    [
        "void(float32, float32, float32, float32[:])",
        "void(float64, float64, float64, float64[:])",
    ],
    "(),(),(),(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def zac_filter(sigma: float, flat: int, decay: int, kernel: np.array) -> None:
    """Calculates ZAC (Zero Area CUSP) kernel.

    Parameters
    ----------
    sigma
        the curvature of the rising and falling part of the kernel.
    flat
        the length of the flat section.
    decay
        the decay constant of the exponential to be convolved.
    kernel
        the calculated kernel

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "kern_zac": {
            "function": "zac_filter",
            "module": "dspeed.processors",
            "args": ["10*us", "3*us", "400*us", "kern_zac"],
            "unit": "ADC"
        }
    """

    if sigma < 0:
        raise DSPFatal("The curvature parameter must be positive")

    if flat < 0:
        raise DSPFatal("The length of the flat section must be positive")

    if np.floor(flat) != flat:
        raise DSPFatal("The length of the flat section must be an integer")

    if decay < 0:
        raise DSPFatal("The decay constant must be positive")

    lt = int((len(kernel) - flat) / 2)
    flat_int = int(flat)
    length = len(kernel)

    # calculate cusp filter and negative parables
    cusp = np.zeros(length)
    par = np.zeros(length)
    for ind in range(0, lt, 1):
        cusp[ind] = float(np.sinh(ind / sigma) / np.sinh(lt / sigma))
        par[ind] = np.power(ind - lt / 2, 2) - np.power(lt / 2, 2)
    for ind in range(lt, lt + flat_int + 1, 1):
        cusp[ind] = 1
    for ind in range(lt + flat_int + 1, length, 1):
        cusp[ind] = float(np.sinh((length - ind) / sigma) / np.sinh(lt / sigma))
        par[ind] = np.power(length - ind - lt / 2, 2) - np.power(lt / 2, 2)

    # calculate area of cusp and parables
    areapar, areacusp = 0, 0
    for i in range(0, length, 1):
        areapar += par[i]
        areacusp += cusp[i]

    # normalize parables area
    par = -par / areapar * areacusp

    # create zac filter
    zac = cusp + par

    # deconvolve zac filter
    den = [1, -np.exp(-1 / decay)]
    kernel[:] = np.convolve(zac, den, "same")


@guvectorize(
    [
        "void(float32[:,:], float32[:], float32, float32, float32, float32, float32[:])",
        "void(float64[:,:], float64[:], float64, float64, float64, float64, float64[:])",
    ],
    "(n,n),(m),(),(),(),(),(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def dplms(
    noise_mat: list,
    reference: list,
    a1: float,
    a2: float,
    a3: float,
    ff: int,
    kernel: np.array,
) -> None:
    """Calculate and apply an optimum DPLMS filter to the waveform.

    The processor takes the noise matrix and the reference signal as input and
    calculates the optimum filter according to the provided length and
    penalized coefficients [DPLMS]_.

    .. [DPLMS] V. D'Andrea et al. “Optimum Filter Synthesis with DPLMS
        Method for Energy Reconstruction” Eur. Phys. J. C 83, 149 (2023).
        https://doi.org/10.1140/epjc/s10052-023-11299-z


    Parameters
    ----------
    noise_mat
        noise matrix
    reference
        reference signal
    a1
        penalized coefficient for the noise matrix
    a2
        penalized coefficient for the reference matrix
    a3
        penalized coefficient for the zero area matrix
    ff
        flat top length for the reference signal
    kernel
        output kernel


    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "kern_dplms": {
            "function": "dplms",
            "module": "dspeed.processors",
            "args": ["db.dplms.noise_matrix",
                "db.dplms.reference",
                "50", "0.1", "1", "1"
                "kern_dplms"],
            "unit": "ADC",
        }
    """

    noise_mat = np.array(noise_mat)
    reference = np.array(reference)

    if len(kernel) != noise_mat.shape[0]:
        raise DSPFatal(
            "The length of the filter is not consistent with the noise matrix"
        )

    if len(reference) <= 0:
        raise DSPFatal("The length of the reference signal must be positive")

    if a1 <= 0:
        raise DSPFatal("The penalized coefficient for the noise must be positive")

    if a2 <= 0:
        raise DSPFatal("The penalized coefficient for the reference must be positive")

    if a3 <= 0:
        raise DSPFatal("The penalized coefficient for the zero area must be positive")

    if ff <= 0:
        raise DSPFatal("The penalized coefficient for the ref matrix must be positive")

    # reference matrix
    length = len(kernel)
    ssize = len(reference)
    flo = int(ssize / 2 - length / 2)
    fhi = int(ssize / 2 + length / 2)
    ref_mat = np.zeros([length, length])
    ref_sig = np.zeros([length])
    if ff == 0:
        ff = [0]
    elif ff == 1:
        ff = [-1, 0, 1]
    else:
        raise DSPFatal("The penalized coefficient for the ref matrix must be 0 or 1")
    for i in ff:
        ref_mat += np.outer(reference[flo + i : fhi + i], reference[flo + i : fhi + i])
        ref_sig += reference[flo + i : fhi + i]
    ref_mat /= len(ff)

    # filter calculation
    mat = a1 * noise_mat + a2 * ref_mat + a3 * np.ones([length, length])
    kernel[:] = np.flip(np.linalg.solve(mat, ref_sig))
    y = np.convolve(reference, kernel, mode="valid")
    maxy = np.amax(y)
    kernel[:] /= maxy