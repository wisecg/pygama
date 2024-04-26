"""
Implements a LEGEND Data Object representing an array of equal-sized arrays and
corresponding utilities.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd

import pygama.lgdo.lgdo_utils as utils
import pygama.lgdo.vectorofvectors as vov
from pygama.lgdo.array import Array


class ArrayOfEqualSizedArrays(Array):
    """An array of equal-sized arrays.

    Arrays of equal size within a file but could be different from application
    to application. Canonical example: array of same-length waveforms.
    """

    def __init__(
        self,
        dims: tuple[int, ...] | None = None,
        nda: np.ndarray = None,
        shape: tuple[int, ...] = (),
        dtype: np.dtype = None,
        fill_val: int | float | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dims
            specifies the dimensions required for building the
            :class:`ArrayOfEqualSizedArrays`' `datatype` attribute.
        nda
            An :class:`numpy.ndarray` to be used for this object's internal
            array. Note: the array is used directly, not copied. If not
            supplied, internal memory is newly allocated based on the `shape`
            and `dtype` arguments.
        shape
            A NumPy-format shape specification for shape of the internal
            array. Required if `nda` is ``None``, otherwise unused.
        dtype
            Specifies the type of the data in the array. Required if `nda` is
            ``None``, otherwise unused.
        fill_val
            If ``None``, memory is allocated without initialization. Otherwise,
            the array is allocated with all elements set to the corresponding
            fill value. If `nda` is not ``None``, this parameter is ignored.
        attrs
            A set of user attributes to be carried along with this LGDO.

        Notes
        -----
        If shape is not "1D array of arrays of shape given by axes 1-N" (of
        `nda`) then specify the dimensionality split in the constructor.

        See Also
        --------
        :class:`.Array`
        """
        if dims is None:
            # If no dims are provided, assume that it's a 1D Array of (N-1)-D Arrays
            if nda is None:
                s = shape
            else:
                if not isinstance(nda, np.ndarray):
                    nda = np.array(nda)
                s = nda.shape
            self.dims = (1, len(s) - 1)
        else:
            self.dims = dims
        super().__init__(
            nda=nda, shape=shape, dtype=dtype, fill_val=fill_val, attrs=attrs
        )

    def datatype_name(self) -> str:
        return "array_of_equalsized_arrays"

    def form_datatype(self) -> str:
        dt = self.datatype_name()
        nd = str(len(self.nda.shape))
        if self.dims is not None:
            nd = ",".join([str(i) for i in self.dims])
        et = utils.get_element_type(self)
        return dt + "<" + nd + ">{" + et + "}"

    def __len__(self) -> int:
        return len(self.nda)

    def __iter__(self) -> Iterator[np.array]:
        return self.nda.__iter__()

    def __next__(self) -> np.ndarray:
        return self.nda.__next__()

    def to_vov(self, cumulative_length: np.ndarray = None) -> vov.VectorOfVectors:
        """Convert (and eventually resize) to :class:`.vectorofvectors.VectorOfVectors`.

        Parameters
        ----------
        cumulative_length
            cumulative length array of the output vector of vectors. Each
            vector in the output is filled with values found in the
            :class:`ArrayOfEqualSizedArrays`, starting from the first index. if
            ``None``, use all of the original 2D array and make vectors of
            equal size.
        """
        attrs = self.getattrs()

        if cumulative_length is None:
            return vov.VectorOfVectors(
                flattened_data=self.nda.flatten(),
                cumulative_length=(np.arange(self.nda.shape[0], dtype="uint32") + 1)
                * self.nda.shape[1],
                attrs=attrs,
            )

        if not isinstance(cumulative_length, np.ndarray):
            cumulative_length = np.array(cumulative_length)

        flattened_data = self.nda[
            np.arange(self.nda.shape[1])
            < np.diff(cumulative_length, prepend=0)[:, None]
        ]

        return vov.VectorOfVectors(
            flattened_data=flattened_data,
            cumulative_length=cumulative_length,
            attrs=attrs,
        )

    def view_as(
        self, library: str, with_units: bool = False
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        """View the array as a third-party format data structure.

        See Also
        --------
        .LGDO.view_as
        """
        return super().view_as(library, with_units=with_units)