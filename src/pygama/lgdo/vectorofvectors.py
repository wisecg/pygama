"""
Implements a LEGEND Data Object representing a variable-length array of
variable-length arrays and corresponding utilities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike, NDArray

import pygama.lgdo.lgdo_utils as utils
import pygama.lgdo.arrayofequalsizedarrays as aoesa
import pygama.lgdo.vovutils as vovutils
from pygama.lgdo.array import Array
from pygama.lgdo.lgdo import LGDO

log = logging.getLogger(__name__)


class VectorOfVectors(LGDO):
    """A n-dimensional variable-length 1D array of variable-length 1D arrays.

    If the vector is 2-dimensional, the internal representation is as two NumPy
    arrays, one to store the flattened data contiguosly
    (:attr:`flattened_data`) and one to store the cumulative sum of lengths of
    each vector (:attr:`cumulative_length`). When the dimension is more than 2,
    :attr:`flattened_data` is a :class:`VectorOfVectors` itself.

    Examples
    --------
    >>> from lgdo import VectorOfVectors
    >>> data = VectorOfVectors(
    ...   [[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]]],
    ...   attrs={"units": "m"}
    ... )
    >>> print(data)
    [[[1, 2], [3, 4, 5]],
     [[2], [4, 8, 9, 7]],
     [[5, 3, 1]]
    ] with attrs={'units': 'm'}
    >>> data.view_as("ak")
    <Array [[[1, 2], [3, 4, 5]], ..., [[5, ..., 1]]] type='3 * var * var * int64'>

    Note
    ----
    Many class methods are currently implemented only for 2D vectors and will
    raise an exception on higher dimensional data.
    """

    def __init__(
        self,
        data: ArrayLike | None = None,
        flattened_data: ArrayLike | None = None,
        cumulative_length: ArrayLike | VectorOfVectors | None = None,
        shape_guess: Sequence[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        fill_val: int | float | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            Any array-like structure accepted by the :class:`ak.Array`
            constructor, with the exception that elements cannot be of type
            ``OptionType``, ``UnionType`` or ``RecordType``.  Takes priority
            over `flattened_data` and `cumulative_length`. The serialization of
            the :class:`ak.Array` is performed through :func:`ak.to_buffers`.
            Since the latter returns non-data-owning NumPy arrays, which would
            prevent later modifications like resizing, a copy is performed.
        flattened_data
            if not ``None``, used as the internal array for
            `self.flattened_data`.  Otherwise, an internal `flattened_data` is
            allocated based on `cumulative_length` (or `shape_guess`) and `dtype`.
        cumulative_length
            if not ``None``, used as the internal array for
            `self.cumulative_length`. Should be `dtype` :any:`numpy.uint32`. If
            `cumulative_length` is ``None``, an internal `cumulative_length` is
            allocated based on the first element of `shape_guess`.
        shape_guess
            a NumPy-format shape specification, required if either of
            `flattened_data` or `cumulative_length` are not supplied.  The
            first element should not be a guess and sets the number of vectors
            to be stored. The second element is a guess or approximation of the
            typical length of a stored vector, used to set the initial length
            of `flattened_data` if it was not supplied.
        dtype
            sets the type of data stored in `flattened_data`. Required if
            `flattened_data` and `array` are ``None``.
        fill_val
            fill all of `self.flattened_data` with this value.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        # sanitize
        if cumulative_length is not None and not isinstance(cumulative_length, Array):
            cumulative_length = Array(cumulative_length)
        if flattened_data is not None and not isinstance(
            flattened_data, (Array, VectorOfVectors)
        ):
            flattened_data = Array(flattened_data)

        if data is not None:
            if not isinstance(data, ak.Array):
                data = ak.Array(data)

            if data.ndim < 2:
                msg = (
                    "cannot initialize a VectorOfVectors with "
                    f"{data.ndim}-dimensional data"
                )
                raise ValueError(msg)

            # make sure it's not a record array
            if not vovutils._ak_is_valid(data):
                msg = "input array type is not supported!"
                raise ValueError(msg)

            # array might be non-jagged! ('container' will hold a ndim NumPy array)
            if not vovutils._ak_is_jagged(data):
                data = ak.from_regular(data, axis=None)

            # ak.to_buffer helps in de-serialization
            # NOTE: ak.to_packed() needed?
            form, length, container = ak.to_buffers(ak.to_packed(data))

            # NOTE: node#-data is not even in the dict if the awkward array is empty
            # NOTE: if the data arg was a numpy array, to_buffers() preserves
            # the original dtype
            # FIXME: have to copy the buffers, otherwise self will not own the
            # data and self.resize() will fail. Is it possible to avoid this?
            flattened_data = np.copy(
                container.pop(f"node{data.ndim-1}-data", np.empty(0, dtype=dtype))
            )

            # if user-provided dtype is different than dtype from Awkward, cast
            # NOTE: makes a copy only if needed
            flattened_data = np.asarray(flattened_data, dtype=dtype)

            # start from innermost VoV and build nested structure
            for i in range(data.ndim - 2, -1, -1):
                # NOTE: remember, omit the leading 0 from ak.Array offsets
                cumulative_length = np.copy(container[f"node{i}-offsets"][1:])

                if i != 0:
                    # at the beginning of the loop: initialize innermost
                    # flattened_data and replace current flattened_data
                    # reference. in the following iterations flattened_data is
                    # a VectorOfVectors
                    flattened_data = VectorOfVectors(
                        flattened_data=flattened_data,
                        cumulative_length=cumulative_length,
                    )

                else:
                    # at end we need to initialize self with the latest flattened_data
                    self.__init__(
                        flattened_data=flattened_data,
                        cumulative_length=cumulative_length,
                    )

        else:
            self.flattened_data = None
            self.cumulative_length = None

            # let's first setup cumulative_length...
            if cumulative_length is None:
                # initialize based on shape_guess
                if shape_guess is None:
                    # just make an empty 2D vector
                    shape_guess = (0, 0)

                # sanity check
                if len(shape_guess) < 2:
                    msg = "shape_guess must be a sequence of 2 integers or more"
                    raise ValueError(msg)

                # let's Awkward do the job here, we're lazy
                if fill_val is not None:
                    self.__init__(
                        np.full(shape=shape_guess, fill_value=fill_val, dtype=dtype)
                    )
                else:
                    self.__init__(np.empty(shape=shape_guess, dtype=dtype))
            else:
                # if it's user provided just use it
                self.cumulative_length = cumulative_length

            # ...then flattened_data
            # NOTE: self.flattened_data might have already been initialized
            # above
            if flattened_data is None and self.flattened_data is None:
                # this happens when the cumulative_length arg is not None
                if dtype is None:
                    msg = "flattened_data and dtype cannot both be None!"
                    raise ValueError(msg)

                # now ready to initialize the object!
                self.flattened_data = Array(
                    shape=(self.cumulative_length[-1],), dtype=dtype, fill_val=fill_val
                )
            elif self.flattened_data is None:
                self.flattened_data = flattened_data

            # finally set dtype
            self.dtype = self.flattened_data.dtype

        # set ndim
        self.ndim = 2
        pointer = self.flattened_data
        while True:
            if isinstance(pointer, Array):
                break

            self.ndim += 1
            pointer = pointer.flattened_data

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        eltype = (
            "array<1>{" + utils.get_element_type(self) + "}"
            if self.ndim == 2
            else self.flattened_data.form_datatype()
        )
        return "array<1>{" + eltype + "}"

    def __len__(self) -> int:
        """Return the number of stored vectors along the first axis (0)."""
        return len(self.cumulative_length)

    def __eq__(self, other: VectorOfVectors) -> bool:
        if isinstance(other, VectorOfVectors):
            if self.ndim == 2 and len(self.cumulative_length) != 0:
                fldata_eq = np.array_equal(
                    self.flattened_data[: self.cumulative_length[-1]],
                    other.flattened_data[: other.cumulative_length[-1]],
                )
            else:
                fldata_eq = self.flattened_data == other.flattened_data

            return (
                self.cumulative_length == other.cumulative_length
                and fldata_eq
                and self.dtype == other.dtype
                and self.attrs == other.attrs
            )

        return False

    def __getitem__(self, i: int) -> NDArray:
        """Return a view of the vector at index `i` along the first axis."""
        if self.ndim == 2:
            stop = self.cumulative_length[i]
            if i in (0, -len(self)):
                return self.flattened_data[0:stop]

            return self.flattened_data[self.cumulative_length[i - 1] : stop]

        raise NotImplementedError

    def __setitem__(self, i: int, new: NDArray) -> None:
        if self.ndim == 2:
            self.__getitem__(i)[:] = new
        else:
            raise NotImplementedError

    def resize(self, new_size: int) -> None:
        """Resize vector along the first axis.

        `self.flattened_data` is resized only if `new_size` is smaller than the
        current vector length.

        If `new_size` is larger than the current vector length,
        `self.cumulative_length` is padded with its last element.  This
        corresponds to appending empty vectors.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.resize(3)
        >>> print(vov)
        [[1 2 3],
         [4 5],
         [],
        ]

        >>> vov = VectorOfVectors([[1, 2], [3], [4, 5]])
        >>> vov.resize(2)
        >>> print(vov)
        [[1 2],
         [3],
        ]
        """
        if self.ndim == 2:
            vidx = self.cumulative_length
            old_s = len(self)
            dlen = new_size - old_s
            csum = vidx[-1] if len(self) > 0 else 0

            # first resize the cumulative length
            self.cumulative_length.resize(new_size)

            # if new_size > size, new elements are filled with zeros, let's fix
            # that
            if dlen > 0:
                self.cumulative_length[old_s:] = csum

            # then resize the data array
            # if dlen > 0 this has no effect
            if len(self.cumulative_length) > 0:
                self.flattened_data.resize(self.cumulative_length[-1])
        else:
            raise NotImplementedError

    def append(self, new: NDArray) -> None:
        """Append a 1D vector `new` at the end.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.append([8, 9])
        >>> print(vov)
        [[1 2 3],
         [4 5],
         [8 9],
        ]
        """
        if self.ndim == 2:
            # first extend cumulative_length by +1
            self.cumulative_length.resize(len(self) + 1)
            # set it at the right value
            newlen = (
                self.cumulative_length[-2] + len(new) if len(self) > 1 else len(new)
            )
            self.cumulative_length[-1] = newlen
            # then resize flattened_data to accommodate the new vector
            self.flattened_data.resize(len(self.flattened_data) + len(new))
            # finally set it
            self[-1] = new
        else:
            raise NotImplementedError

    def insert(self, i: int, new: NDArray) -> None:
        """Insert a vector at index `i`.

        `self.flattened_data` (and therefore `self.cumulative_length`) is
        resized in order to accommodate the new element.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.insert(1, [8, 9])
        >>> print(vov)
        [[1 2 3],
         [8 9],
         [4 5],
        ]

        Warning
        -------
        This method involves a significant amount of memory re-allocation and
        is expected to perform poorly on large vectors.
        """
        if self.ndim == 2:
            if i >= len(self):
                msg = f"index {i} is out of bounds for vector owith size {len(self)}"
                raise IndexError(msg)

            self.flattened_data = Array(
                np.insert(self.flattened_data, self.cumulative_length[i - 1], new)
            )
            self.cumulative_length = Array(
                np.insert(self.cumulative_length, i, self.cumulative_length[i - 1])
            )
            self.cumulative_length[i:] += np.uint32(len(new))
        else:
            raise NotImplementedError

    def replace(self, i: int, new: NDArray) -> None:
        """Replace the vector at index `i` with `new`.

        `self.flattened_data` (and therefore `self.cumulative_length`) is
        resized, if the length of `new` is different from the vector currently
        at index `i`.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.replace(0, [8, 9])
        >>> print(vov)
        [[8 9],
         [4 5],
        ]

        Warning
        -------
        This method involves a significant amount of memory re-allocation and
        is expected to perform poorly on large vectors.
        """
        if self.ndim == 2:
            if i >= len(self):
                msg = f"index {i} is out of bounds for vector with size {len(self)}"
                raise IndexError(msg)

            vidx = self.cumulative_length
            dlen = len(new) - len(self[i])

            if dlen == 0:
                # don't waste resources
                self[i] = new
            elif dlen < 0:
                start = vidx[i - 1]
                stop = start + len(new)
                # set the already allocated indices
                self.flattened_data[start:stop] = new
                # then delete the extra indices
                self.flattened_data = Array(
                    np.delete(self.flattened_data, np.s_[stop : vidx[i]])
                )
            else:
                # set the already allocated indices
                self.flattened_data[vidx[i - 1] : vidx[i]] = new[: len(self[i])]
                # then insert the remaining
                self.flattened_data = Array(
                    np.insert(self.flattened_data, vidx[i], new[len(self[i]) :])
                )

            vidx[i:] = vidx[i:] + dlen
        else:
            raise NotImplementedError

    def _set_vector_unsafe(
        self, i: int, vec: NDArray, lens: ArrayLike | None = None
    ) -> None:
        r"""Insert vector `vec` at position `i`.

        Assumes that ``j = self.cumulative_length[i-1]`` is the index (in
        `self.flattened_data`) of the end of the `(i-1)`\ th vector and copies
        `vec` in ``self.flattened_data[j:sum(lens)]``. Finally updates
        ``self.cumulative_length[i]`` with the new flattened data array length.

        Vectors stored after index `i` can be overridden, producing unintended
        behavior. This method is typically used for fast sequential fill of a
        pre-allocated vector of vectors.

        If i`vec` is 1D array and `lens` is ``None``, set using full array. If
        `vec` is 2D, require `lens` to be included, and fill each array only up
        to lengths in `lens`.

        Danger
        ------
        This method can lead to undefined behavior or vector invalidation if
        used improperly. Use it only if you know what you are doing.

        See Also
        --------
        append, replace, insert
        """
        if self.ndim == 2:
            # check if current vector is empty and get the start index in
            # flattened_data
            start = 0 if i == 0 else self.cumulative_length[i - 1]

            # if the new element is 1D, convert to dummy 2D
            if len(vec.shape) == 1:
                vec = np.expand_dims(vec, axis=0)
                if lens is None:
                    lens = np.array([vec.shape[1]], dtype="u4")

            # this in case lens is 02, convert to 1D
            if not isinstance(lens, np.ndarray):
                lens = np.array([lens], dtype="u4")

            # calculate stop index in flattened_data
            cum_lens = start + lens.cumsum()

            # fill with fast vectorized routine
            vovutils._nb_fill(vec, lens, self.flattened_data.nda[start : cum_lens[-1]])

            # add new vector(s) length to cumulative_length
            self.cumulative_length[i : i + len(lens)] = cum_lens
        else:
            raise NotImplementedError

    def __iter__(self) -> Iterator[NDArray]:
        if self.ndim == 2:
            for j, stop in enumerate(self.cumulative_length):
                if j == 0:
                    yield self.flattened_data[0:stop]
                else:
                    yield self.flattened_data[self.cumulative_length[j - 1] : stop]
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        string = self.view_as("ak").show(stream=None)

        string = string.strip().removesuffix("]")
        string += "\n]"

        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        if len(tmp_attrs) > 0:
            string += f" with attrs={tmp_attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = (
            "VectorOfVectors(flattened_data="
            + repr(self.flattened_data)
            + ", cumulative_length="
            + repr(self.cumulative_length)
            + ", attrs="
            + repr(self.attrs)
            + ")"
        )
        np.set_printoptions(**npopt)
        return out

    def to_aoesa(
        self,
        max_len: int | None = None,
        fill_val: bool | int | float = np.nan,
        preserve_dtype: bool = False,
    ) -> aoesa.ArrayOfEqualSizedArrays:
        """Convert to :class:`ArrayOfEqualSizedArrays`.

        Note
        ----
        The dtype of the original vector is typically not strictly preserved.
        The output dtype will be either :class:`np.float64` or :class:`np.int64`.
        If you want to use the same exact dtype, set `preserve_dtype` to
        ``True``.

        Parameters
        ----------
        max_len
            the length of the returned array along its second dimension. Longer
            vectors will be truncated, shorter will be padded with `fill_val`.
            If ``None``, the length will be equal to the length of the longest
            vector.
        fill_val
            value used to pad shorter vectors up to `max_len`. The dtype of the
            output array will be such that both `fill_val` and the vector
            values can be represented in the same data structure.
        preserve_dtype
            whether the output array should have exactly the same dtype as the
            original vector of vectors. The type `fill_val` must be a
            compatible one.
        """
        if self.ndim == 2:
            ak_arr = self.view_as("ak")

            if max_len is None:
                max_len = int(ak.max(ak.count(ak_arr, axis=-1)))

            nda = ak.fill_none(
                ak.pad_none(ak_arr, max_len, clip=True), fill_val
            ).to_numpy(allow_missing=False)

            if preserve_dtype:
                nda = nda.astype(self.flattened_data.dtype, copy=False)

            return aoesa.ArrayOfEqualSizedArrays(nda=nda, attrs=self.getattrs())

        raise NotImplementedError

    def view_as(
        self,
        library: str,
        with_units: bool = False,
        fill_val: bool | int | float = np.nan,
        preserve_dtype: bool = False,
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        r"""View the vector data as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.Series` (supported through the
          ``awkward-pandas`` package)
        - ``np``: returns a :class:`numpy.ndarray`, padded with zeros to make
          it rectangular. This implies memory re-allocation.
        - ``ak``: returns an :class:`ak.Array`. ``self.cumulative_length`` is
          currently re-allocated for technical reasons.

        Notes
        -----
        Awkward array views partially involve memory re-allocation (the
        `cumulative_length`\ s), while NumPy "exploded" views clearly imply a
        full copy.

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.
        fill_val
            forwarded to :meth:`.to_aoesa`, if `library` is ``np``.
        preserve_dtype
            forwarded to :meth:`.to_aoesa`, if `library` is ``np``.

        See Also
        --------
        .LGDO.view_as
        """
        attach_units = with_units and "units" in self.attrs

        if library == "ak":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            # see https://github.com/scikit-hep/awkward/discussions/2848

            # cannot avoid making a copy here. we should add the leading 0 to
            # cumulative_length inside VectorOfVectors at some point in the
            # future
            offsets = np.empty(
                len(self.cumulative_length) + 1, dtype=self.cumulative_length.dtype
            )
            offsets[1:] = self.cumulative_length
            offsets[0] = 0

            content = (
                ak.contents.NumpyArray(self.flattened_data.nda)
                if self.ndim == 2
                else self.flattened_data.view_as(library, with_units=with_units).layout
            )

            layout = ak.contents.ListOffsetArray(
                offsets=ak.index.Index(offsets),
                content=content,
            )
            return ak.Array(layout)

        if library == "np":
            if preserve_dtype:
                return self.to_aoesa(fill_val=fill_val, preserve_dtype=True).view_as(
                    "np", with_units=with_units
                )

            return self.to_aoesa().view_as("np", with_units=with_units)

        if library == "pd":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            return akpd.from_awkward(self.view_as("ak"))

        msg = f"{library} is not a supported third-party format."
        raise ValueError(msg)