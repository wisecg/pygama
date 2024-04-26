""":class:`~.lgdo.typing.vectorofvectors.VectorOfVectors` utilities."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import awkward as ak
import numba
import numpy as np
from numpy.typing import NDArray

from pygama.lgdo.utils import numba_defaults_kwargs as nb_kwargs
from pygama.lgdo.array import Array

log = logging.getLogger(__name__)


def build_cl(
    sorted_array_in: NDArray, cumulative_length_out: NDArray | None = None
) -> NDArray:
    """Build a cumulative length array from an array of sorted data.

    Examples
    --------
    >>> build_cl(np.array([3, 3, 3, 4])
    array([3., 4.])

    For a `sorted_array_in` of indices, this is the inverse of
    :func:`.explode_cl`, in the sense that doing
    ``build_cl(explode_cl(cumulative_length))`` would recover the original
    `cumulative_length`.

    Parameters
    ----------
    sorted_array_in
        array of data already sorted; each N matching contiguous entries will
        be converted into a new row of `cumulative_length_out`.
    cumulative_length_out
        a pre-allocated array for the output `cumulative_length`. It will
        always have length <= `sorted_array_in`, so giving them the same length
        is safe if there is not a better guess.

    Returns
    -------
    cumulative_length_out
        the output cumulative length array. If the user provides a
        `cumulative_length_out` that is too long, this return value is sliced
        to contain only the used portion of the allocated memory.
    """
    if len(sorted_array_in) == 0:
        return None
    sorted_array_in = np.asarray(sorted_array_in)
    if cumulative_length_out is None:
        cumulative_length_out = np.zeros(len(sorted_array_in), dtype=np.uint64)
    else:
        cumulative_length_out.fill(0)
    if len(cumulative_length_out) == 0 and len(sorted_array_in) > 0:
        msg = "cumulative_length_out too short ({len(cumulative_length_out)})"
        raise ValueError(msg)
    return _nb_build_cl(sorted_array_in, cumulative_length_out)


@numba.njit(**nb_kwargs)
def _nb_build_cl(sorted_array_in: NDArray, cumulative_length_out: NDArray) -> NDArray:
    """numbified inner loop for build_cl"""
    ii = 0
    last_val = sorted_array_in[0]
    for val in sorted_array_in:
        if val != last_val:
            ii += 1
            cumulative_length_out[ii] = cumulative_length_out[ii - 1]
            if ii >= len(cumulative_length_out):
                msg = "cumulative_length_out too short"
                raise RuntimeError(msg)
            last_val = val
        cumulative_length_out[ii] += 1
    ii += 1
    return cumulative_length_out[:ii]


@numba.guvectorize(
    [
        f"{data_type}[:,:],{size_type}[:],{data_type}[:]"
        for data_type in [
            "b1",
            "i1",
            "i2",
            "i4",
            "i8",
            "u1",
            "u2",
            "u4",
            "u8",
            "f4",
            "f8",
            "c8",
            "c16",
        ]
        for size_type in ["i4", "i8", "u4", "u8"]
    ],
    "(l,m),(l),(n)",
    **nb_kwargs,
)
def _nb_fill(aoa_in: NDArray, len_in: NDArray, flattened_array_out: NDArray):
    """Vectorized function to fill flattened array from array of arrays and
    lengths. Values in aoa_in past lengths will not be copied.

    Parameters
    ----------
    aoa_in
        array of arrays containing values to be copied
    len_in
        array of vector lengths for each row of aoa_in
    flattened_array_out
        flattened array to copy values into. Must be longer than sum of
        lengths in len_in
    """

    if len(flattened_array_out) < len_in.sum():
        msg = "flattened array not large enough to hold values"
        raise ValueError(msg)

    start = 0
    for i, ll in enumerate(len_in):
        stop = start + ll
        flattened_array_out[start:stop] = aoa_in[i, :ll]
        start = stop


def explode_cl(cumulative_length: NDArray, array_out: NDArray | None = None) -> NDArray:
    """Explode a `cumulative_length` array.

    Examples
    --------
    >>> explode_cl(np.array([2, 3]))
    array([0., 0., 1.])

    This is the inverse of :func:`.build_cl`, in the sense that doing
    ``build_cl(explode_cl(cumulative_length))`` would recover the original
    `cumulative_length`.

    Parameters
    ----------
    cumulative_length
        the cumulative length array to be exploded.
    array_out
        a pre-allocated array to hold the exploded cumulative length array.
        The length should be equal to ``cumulative_length[-1]``.

    Returns
    -------
    array_out
        the exploded cumulative length array.
    """
    cumulative_length = np.asarray(cumulative_length)
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if array_out is None:
        array_out = np.empty(int(out_len), dtype=np.uint64)
    if len(array_out) != out_len:
        msg = f"bad lengths: cl[-1] ({cumulative_length[-1]}) != out ({len(array_out)})"
        raise ValueError(msg)
    return _nb_explode_cl(cumulative_length, array_out)


@numba.njit(**nb_kwargs)
def _nb_explode_cl(cumulative_length: NDArray, array_out: NDArray) -> NDArray:
    """numbified inner loop for explode_cl"""
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if len(array_out) != out_len:
        msg = "bad lengths"
        raise ValueError(msg)
    start = 0
    for ii in range(len(cumulative_length)):
        nn = int(cumulative_length[ii] - start)
        for jj in range(nn):
            array_out[int(start + jj)] = ii
        start = cumulative_length[ii]
    return array_out


def explode(
    cumulative_length: NDArray, array_in: NDArray, array_out: NDArray | None = None
) -> NDArray:
    """Explode a data array using a `cumulative_length` array.

    This is identical to :func:`.explode_cl`, except `array_in` gets exploded
    instead of `cumulative_length`.

    Examples
    --------
    >>> explode(np.array([2, 3]), np.array([3, 4]))
    array([3., 3., 4.])

    Parameters
    ----------
    cumulative_length
        the cumulative length array to use for exploding.
    array_in
        the data to be exploded. Must have same length as `cumulative_length`.
    array_out
        a pre-allocated array to hold the exploded data. The length should be
        equal to ``cumulative_length[-1]``.

    Returns
    -------
    array_out
        the exploded cumulative length array.
    """
    cumulative_length = np.asarray(cumulative_length)
    array_in = np.asarray(array_in)
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if array_out is None:
        array_out = np.empty(out_len, dtype=array_in.dtype)
    if len(cumulative_length) != len(array_in) or len(array_out) != out_len:
        msg = (
            f"bad lengths: cl ({len(cumulative_length)}) != in ({len(array_in)}) "
            f"and cl[-1] ({cumulative_length[-1]}) != out ({len(array_out)})"
        )
        raise ValueError(msg)
    return _nb_explode(cumulative_length, array_in, array_out)


@numba.njit(**nb_kwargs)
def _nb_explode(
    cumulative_length: NDArray, array_in: NDArray, array_out: NDArray
) -> NDArray:
    """Numbified inner loop for :func:`.explode`."""
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if len(cumulative_length) != len(array_in) or len(array_out) != out_len:
        msg = "bad lengths"
        raise ValueError(msg)
    ii = 0
    for jj in range(len(array_out)):
        while ii < len(cumulative_length) and jj >= cumulative_length[ii]:
            ii += 1
        array_out[jj] = array_in[ii]
    return array_out


def explode_arrays(
    cumulative_length: Array,
    arrays: Sequence[NDArray],
    arrays_out: Sequence[NDArray] | None = None,
) -> list:
    """Explode a set of arrays using a `cumulative_length` array.

    Parameters
    ----------
    cumulative_length
        the cumulative length array to use for exploding.
    arrays
        the data arrays to be exploded. Each array must have same length as
        `cumulative_length`.
    arrays_out
        a list of pre-allocated arrays to hold the exploded data. The length of
        the list should be equal to the length of `arrays`, and each entry in
        arrays_out should have length ``cumulative_length[-1]``. If not
        provided, output arrays are allocated for the user.

    Returns
    -------
    arrays_out
        the list of exploded cumulative length arrays.
    """
    cumulative_length = np.asarray(cumulative_length)
    for ii in range(len(arrays)):
        arrays[ii] = np.asarray(arrays[ii])
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if arrays_out is None:
        arrays_out = []
        for array in arrays:
            arrays_out.append(np.empty(out_len, dtype=array.dtype))
    for ii in range(len(arrays)):
        explode(cumulative_length, arrays[ii], arrays_out[ii])
    return arrays_out


def _ak_is_jagged(type_: ak.types.Type) -> bool:
    """Returns ``True`` if :class:`ak.Array` is jagged at all axes.

    This assures that :func:`ak.to_buffers` returns the expected data
    structures.
    """
    if isinstance(type_, ak.Array):
        return _ak_is_jagged(type_.type)

    if isinstance(type_, (ak.types.ArrayType, ak.types.ListType)):
        return _ak_is_jagged(type_.content)

    if isinstance(type_, ak.types.ScalarType):
        msg = "Expected ArrayType or its content"
        raise TypeError(msg)

    return not isinstance(type_, ak.types.RegularType)


# https://github.com/scikit-hep/awkward/discussions/3049
def _ak_is_valid(type_: ak.types.Type) -> bool:
    """Returns ``True`` if :class:`ak.Array` contains only elements we can serialize to LH5."""
    if isinstance(type_, ak.Array):
        return _ak_is_valid(type_.type)

    if isinstance(type_, (ak.types.ArrayType, ak.types.ListType)):
        return _ak_is_valid(type_.content)

    if isinstance(type_, ak.types.ScalarType):
        msg = "Expected ArrayType or its content"
        raise TypeError(msg)

    return not isinstance(
        type_,
        (
            ak.types.OptionType,
            ak.types.UnionType,
            ak.types.RecordType,
        ),
    )

    return isinstance(type_, ak.types.NumpyType)