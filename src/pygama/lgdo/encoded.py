from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import pygama.lgdo.lgdo_utils as utils
from pygama.lgdo.array import Array
from pygama.lgdo.lgdo import LGDO
from pygama.lgdo.scalar import Scalar
from pygama.lgdo.vectorofvectors import VectorOfVectors


class VectorOfEncodedVectors(LGDO):
    """An array of variable-length encoded arrays.

    Used to represent an encoded :class:`.VectorOfVectors`. In addition to an
    internal :class:`.VectorOfVectors` `self.encoded_data` storing the encoded
    data, a 1D :class:`.Array` in `self.encoded_size` holds the original sizes
    of the encoded vectors.

    See Also
    --------
    .VectorOfVectors
    """

    def __init__(
        self,
        encoded_data: VectorOfVectors = None,
        decoded_size: Array = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        encoded_data
            the vector of encoded vectors.
        decoded_size
            an array holding the original length of each encoded vector in
            `encoded_data`.
        attrs
            A set of user attributes to be carried along with this LGDO. Should
            include information about the codec used to encode the data.
        """
        if isinstance(encoded_data, VectorOfVectors):
            self.encoded_data = encoded_data
        elif encoded_data is None:
            self.encoded_data = VectorOfVectors(dtype="ubyte")
        else:
            msg = "encoded_data must be a valid VectorOfVectors"
            raise ValueError(msg)

        if isinstance(decoded_size, Array):
            self.decoded_size = decoded_size
        elif decoded_size is not None:
            self.decoded_size = Array(decoded_size)
        elif encoded_data is not None:
            self.decoded_size = Array(
                shape=len(encoded_data), dtype="uint32", fill_val=0
            )
        elif decoded_size is None:
            self.decoded_size = Array()

        if len(self.encoded_data) != len(self.decoded_size):
            msg = "encoded_data vs. decoded_size shape mismatch"
            raise RuntimeError(msg)

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        et = utils.get_element_type(self.encoded_data)
        return "array<1>{encoded_array<1>{" + et + "}}"

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __eq__(self, other: VectorOfEncodedVectors) -> bool:
        if isinstance(other, VectorOfEncodedVectors):
            return (
                self.encoded_data == other.encoded_data
                and self.decoded_size == other.decoded_size
                and self.attrs == other.attrs
            )

        return False

    def resize(self, new_size: int) -> None:
        """Resize vector along the first axis.

        See Also
        --------
        .VectorOfVectors.resize
        """
        self.encoded_data.resize(new_size)
        self.decoded_size.resize(new_size)

    def append(self, value: tuple[NDArray, int]) -> None:
        """Append a 1D encoded vector at the end.

        Parameters
        ----------
        value
            a tuple holding the encoded array and its decoded size.

        See Also
        --------
        .VectorOfVectors.append
        """
        self.encoded_data.append(value[0])
        self.decoded_size.append(value[1])

    def insert(self, i: int, value: tuple[NDArray, int]) -> None:
        """Insert an encoded vector at index `i`.

        Parameters
        ----------
        i
            the new vector will be inserted before this index.
        value
            a tuple holding the encoded array and its decoded size.

        See Also
        --------
        .VectorOfVectors.insert
        """
        self.encoded_data.insert(i, value[0])
        self.decoded_size.insert(i, value[1])

    def replace(self, i: int, value: tuple[NDArray, int]) -> None:
        """Replace the encoded vector (and decoded size) at index `i` with a new one.

        Parameters
        ----------
        i
            index of the vector to be replaced.
        value
            a tuple holding the encoded array and its decoded size.

        See Also
        --------
        .VectorOfVectors.replace
        """
        self.encoded_data.replace(i, value[0])
        self.decoded_size[i] = value[1]

    def __setitem__(self, i: int, value: tuple[NDArray, int]) -> None:
        """Set an encoded vector at index `i`.

        Parameters
        ----------
        i
            the new vector will be set at this index.
        value
            a tuple holding the encoded array and its decoded size.
        """
        self.encoded_data[i] = value[0]
        self.decoded_size[i] = value[1]

    def __getitem__(self, i: int) -> tuple[NDArray, int]:
        """Return vector at index `i`.

        Returns
        -------
        (encoded_data, decoded_size)
            the encoded array and its decoded length.
        """
        return (self.encoded_data[i], self.decoded_size[i])

    def __iter__(self) -> Iterator[tuple[NDArray, int]]:
        yield from zip(self.encoded_data, self.decoded_size)

    def __str__(self) -> str:
        string = ""
        for pos, res in enumerate(self):
            vec, size = res[0], res[1]
            if pos != 0:
                string += " "

            string += (
                np.array2string(
                    vec,
                    prefix=" ",
                    formatter={
                        "int": lambda x, vec=vec: f"0x{x:02x}"
                        if vec.dtype == np.ubyte
                        else str(x)
                    },
                )
                + f" decoded_size = {size}"
            )

            if pos < len(self.encoded_data.cumulative_length):
                string += ",\n"

        string = f"[{string}]"

        attrs = self.getattrs()
        if len(attrs) > 0:
            string += f" with attrs={attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(
            threshold=5,
            edgeitems=2,
            linewidth=100,
        )
        out = (
            "VectorOfEncodedVectors(encoded_data="
            + repr(self.encoded_data)
            + ", decoded_size="
            + repr(self.decoded_size)
            + ", attrs="
            + repr(self.attrs)
            + ")"
        )
        np.set_printoptions(**npopt)
        return out

    def view_as(
        self, library: str, with_units: bool = False
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        """View the encoded data as a third-party format data structure.

        This is a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.DataFrame`
        - ``ak``: returns an :class:`ak.Array` (record type)

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
        attach_units = with_units and "units" in self.attrs

        if library == "ak":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            records_list = {
                "encoded_data": self.encoded_data.view_as("ak"),
                "decoded_size": np.array(self.decoded_size),
            }
            return ak.Array(records_list)

        if library == "np":
            msg = f"Format {library} is not supported for VectorOfEncodedVectors."
            raise TypeError(msg)
        if library == "pd":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            return pd.DataFrame(
                {
                    "encoded_data": akpd.from_awkward(self.encoded_data.view_as("ak")),
                    "decoded_size": self.decoded_size,
                }
            )

        msg = f"{library} is not a supported third-party format."
        raise ValueError(msg)


class ArrayOfEncodedEqualSizedArrays(LGDO):
    """An array of encoded arrays with equal decoded size.

    Used to represent an encoded :class:`.ArrayOfEqualSizedArrays`. In addition
    to an internal :class:`.VectorOfVectors` `self.encoded_data` storing the
    encoded data, the size of the decoded arrays is stored in a
    :class:`.Scalar` `self.encoded_size`.

    See Also
    --------
    .ArrayOfEqualSizedArrays
    """

    def __init__(
        self,
        encoded_data: VectorOfVectors = None,
        decoded_size: Scalar | int = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        encoded_data
            the vector of vectors holding the encoded data.
        decoded_size
            the length of the decoded arrays.
        attrs
            A set of user attributes to be carried along with this LGDO. Should
            include information about the codec used to encode the data.
        """
        if isinstance(encoded_data, VectorOfVectors):
            self.encoded_data = encoded_data
        elif encoded_data is None:
            self.encoded_data = VectorOfVectors(dtype="ubyte")
        else:
            msg = "encoded_data must be a valid VectorOfVectors"
            raise ValueError(msg)

        if isinstance(decoded_size, Scalar):
            self.decoded_size = decoded_size
        elif decoded_size is not None:
            self.decoded_size = Scalar(int(decoded_size))
        else:
            self.decoded_size = Scalar(0)

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        et = utils.get_element_type(self.encoded_data)
        return "array_of_encoded_equalsized_arrays<1,1>{" + et + "}"

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __eq__(self, other: ArrayOfEncodedEqualSizedArrays) -> bool:
        if isinstance(other, ArrayOfEncodedEqualSizedArrays):
            return (
                self.encoded_data == other.encoded_data
                and self.decoded_size == other.decoded_size
                and self.attrs == other.attrs
            )

        return False

    def resize(self, new_size: int) -> None:
        """Resize array along the first axis.

        See Also
        --------
        .VectorOfVectors.resize
        """
        self.encoded_data.resize(new_size)

    def append(self, value: NDArray) -> None:
        """Append a 1D encoded array at the end.

        See Also
        --------
        .VectorOfVectors.append
        """
        self.encoded_data.append(value)

    def insert(self, i: int, value: NDArray) -> None:
        """Insert an encoded array at index `i`.

        See Also
        --------
        .VectorOfVectors.insert
        """
        self.encoded_data.insert(i, value)

    def replace(self, i: int, value: NDArray) -> None:
        """Replace the encoded array at index `i` with a new one.

        See Also
        --------
        .VectorOfVectors.replace
        """
        self.encoded_data.replace(i, value)

    def __setitem__(self, i: int, value: NDArray) -> None:
        """Set an encoded array at index `i`."""
        self.encoded_data[i] = value

    def __getitem__(self, i: int) -> NDArray:
        """Return encoded array at index `i`."""
        return self.encoded_data[i]

    def __iter__(self) -> Iterator[NDArray]:
        yield from self.encoded_data

    def __str__(self) -> str:
        string = ""
        for pos, vec in enumerate(self):
            if pos != 0:
                string += " "

            string += np.array2string(
                vec,
                prefix=" ",
                formatter={
                    "int": lambda x, vec=vec: f"0x{x:02x}"
                    if vec.dtype == np.ubyte
                    else str(x)
                },
            )

            if pos < len(self.encoded_data.cumulative_length):
                string += ",\n"

        string = f"[{string}] decoded_size={self.decoded_size}"

        attrs = self.getattrs()
        if len(attrs) > 0:
            string += f" with attrs={attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(
            threshold=5,
            edgeitems=2,
            linewidth=100,
        )
        out = (
            "ArrayOfEncodedEqualSizedArrays(encoded_data="
            + repr(self.encoded_data)
            + ", decoded_size="
            + repr(self.decoded_size)
            + ", attrs="
            + repr(self.attrs)
            + ")"
        )
        np.set_printoptions(**npopt)
        return out

    def view_as(
        self, library: str, with_units: bool = False
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        """View the encoded data as a third-party format data structure.

        This is nearly a zero-copy operation.

        Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.DataFrame`
        - ``ak``: returns an :class:`ak.Array` (record type)

        Note
        ----
        In the view, `decoded_size` is expanded into an array.

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
        attach_units = with_units and "units" in self.attrs

        if library == "ak":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            records_list = {
                "encoded_data": self.encoded_data.view_as("ak"),
                "decoded_size": np.full(
                    len(self.encoded_data.cumulative_length), self.decoded_size.value
                ),
            }
            return ak.Array(records_list)

        if library == "np":
            msg = (
                f"Format {library} is not supported for ArrayOfEncodedEqualSizedArrays."
            )
            raise TypeError(msg)

        if library == "pd":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            return pd.DataFrame(
                {
                    "encoded_data": akpd.from_awkward(self.encoded_data.view_as("ak")),
                    "decoded_size": np.full(
                        len(self.encoded_data.cumulative_length),
                        self.decoded_size.value,
                    ),
                }
            )

        msg = f"{library} is not a supported third-party format."
        raise ValueError(msg)