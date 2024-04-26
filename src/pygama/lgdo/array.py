"""
Implements a LEGEND Data Object representing an n-dimensional array and
corresponding utilities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
import pint_pandas  # noqa: F401

import pygama.lgdo.lgdo_utils as utils
from pygama.lgdo.units import default_units_registry as u
from pygama.lgdo.lgdo import LGDO

log = logging.getLogger(__name__)


class Array(LGDO):
    r"""Holds an :class:`numpy.ndarray` and attributes.

    :class:`Array` (and the other various array types) holds an `nda` instead
    of deriving from :class:`numpy.ndarray` for the following reasons:

    - It keeps management of the `nda` totally under the control of the user. The
      user can point it to another object's buffer, grab the `nda` and toss the
      :class:`Array`, etc.
    - It allows the management code to send just the `nda`'s the central routines
      for data manpulation. Keeping LGDO's out of that code allows for more
      standard, reusable, and (we expect) performant Python.
    - It allows the first axis of the `nda` to be treated as "special" for storage
      in :class:`.Table`\ s.
    """

    def __init__(
        self,
        nda: np.ndarray = None,
        shape: tuple[int, ...] = (),
        dtype: np.dtype = None,
        fill_val: float | int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        nda
            An :class:`numpy.ndarray` to be used for this object's internal
            array. Note: the array is used directly, not copied. If not
            supplied, internal memory is newly allocated based on the shape and
            dtype arguments.
        shape
            A numpy-format shape specification for shape of the internal
            ndarray. Required if `nda` is ``None``, otherwise unused.
        dtype
            Specifies the type of the data in the array. Required if `nda` is
            ``None``, otherwise unused.
        fill_val
            If ``None``, memory is allocated without initialization. Otherwise,
            the array is allocated with all elements set to the corresponding
            fill value. If `nda` is not ``None``, this parameter is ignored.
        attrs
            A set of user attributes to be carried along with this LGDO.
        """
        if nda is None:
            if fill_val is None:
                nda = np.empty(shape, dtype=dtype)
            elif fill_val == 0:
                nda = np.zeros(shape, dtype=dtype)
            else:
                nda = np.full(shape, fill_val, dtype=dtype)

        elif isinstance(nda, Array):
            nda = nda.nda

        elif not isinstance(nda, np.ndarray):
            nda = np.array(nda)

        self.nda = nda
        self.dtype = self.nda.dtype

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        dt = self.datatype_name()
        nd = str(len(self.nda.shape))
        et = utils.get_element_type(self)
        return dt + "<" + nd + ">{" + et + "}"

    def __len__(self) -> int:
        return len(self.nda)

    def resize(self, new_size: int) -> None:
        new_shape = (new_size,) + self.nda.shape[1:]
        return self.nda.resize(new_shape, refcheck=True)

    def append(self, value: np.ndarray) -> None:
        self.resize(len(self) + 1)
        self.nda[-1] = value

    def insert(self, i: int, value: int | float) -> None:
        self.nda = np.insert(self.nda, i, value)

    def __getitem__(self, key):
        return self.nda[key]

    def __setitem__(self, key, value):
        return self.nda.__setitem__(key, value)

    def __eq__(self, other: Array) -> bool:
        if isinstance(other, Array):
            return self.attrs == other.attrs and np.array_equal(self.nda, other.nda)

        return False

    def __iter__(self) -> Iterator:
        yield from self.nda

    def __str__(self) -> str:
        attrs = self.getattrs()
        string = str(self.nda)
        if attrs:
            string += f" with attrs={attrs}"
        return string

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + np.array2string(
                self.nda,
                prefix=self.__class__.__name__ + " ",
                formatter={
                    "int": lambda x: f"0x{x:02x}" if self.dtype == np.ubyte else str(x)
                },
            )
            + f", attrs={self.attrs!r})"
        )

    def view_as(
        self, library: str, with_units: bool = False
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        """View the Array data as a third-party format data structure.

        This is a zero-copy operation. Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.Series`
        - ``np``: returns the internal `nda` attribute (:class:`numpy.ndarray`)
        - ``ak``: returns an :class:`ak.Array` initialized with `self.nda`

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.

        See Also
        --------
        .LGDO.view_as
        """
        # TODO: does attaching units imply a copy?
        attach_units = with_units and "units" in self.attrs

        if library == "pd":
            if attach_units:
                if self.nda.ndim == 1:
                    return pd.Series(
                        self.nda, dtype=f"pint[{self.attrs['units']}]", copy=False
                    )

                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            if self.nda.ndim == 1:
                return pd.Series(self.nda, copy=False)

            # if array is multi-dim, use awkward
            return akpd.from_awkward(self.view_as("ak"))

        if library == "np":
            if attach_units:
                return self.nda * u(self.attrs["units"])

            return self.nda

        if library == "ak":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            # NOTE: this is zero-copy!
            return ak.Array(self.nda)

        msg = f"{library} is not a supported third-party format."
        raise ValueError(msg)