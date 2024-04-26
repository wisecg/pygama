"""
Implements a LEGEND Data Object representing a special struct of arrays of
equal length and corresponding utilities.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from warnings import warn

import awkward as ak
import numexpr as ne
import numpy as np
import pandas as pd
from pandas.io.formats import format as fmt

from pygama.lgdo.array import Array
from pygama.lgdo.arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from pygama.lgdo.lgdo import LGDO
from pygama.lgdo.scalar import Scalar
from pygama.lgdo.struct import Struct
from pygama.lgdo.vectorofvectors import VectorOfVectors

log = logging.getLogger(__name__)


class Table(Struct):
    """A special struct of arrays or subtable columns of equal length.

    Holds onto an internal read/write location ``loc`` that is useful in
    managing table I/O using functions like :meth:`push_row`, :meth:`is_full`,
    and :meth:`clear`.

    Note
    ----
    If you write to a table and don't fill it up to its total size, be sure to
    resize it before passing to data processing functions, as they will call
    :meth:`__len__` to access valid data, which returns the ``size`` attribute.
    """

    def __init__(
        self,
        size: int | None = None,
        col_dict: dict[str, LGDO] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Parameters
        ----------
        size
            sets the number of rows in the table. :class:`.Array`\ s in
            `col_dict will be resized to match size if both are not ``None``.
            If `size` is left as ``None``, the number of table rows is
            determined from the length of the first array in `col_dict`. If
            neither is provided, a default length of 1024 is used.
        col_dict
            instantiate this table using the supplied named array-like LGDO's.
            Note 1: no copy is performed, the objects are used directly.
            Note 2: if `size` is not ``None``, all arrays will be resized to
            match it.  Note 3: if the arrays have different lengths, all will
            be resized to match the length of the first array.
        attrs
            A set of user attributes to be carried along with this LGDO.

        Notes
        -----
        the :attr:`loc` attribute is initialized to 0.
        """
        super().__init__(obj_dict=col_dict, attrs=attrs)

        # if col_dict is not empty, set size according to it
        # if size is also supplied, resize all fields to match it
        # otherwise, warn if the supplied fields have varying size
        if col_dict is not None and len(col_dict) > 0:
            do_warn = size is None
            self.resize(new_size=size, do_warn=do_warn)

        # if no col_dict, just set the size (default to 1024)
        else:
            self.size = size if size is not None else None

        # always start at loc=0
        self.loc = 0

    def datatype_name(self) -> str:
        return "table"

    def __len__(self) -> int:
        """Provides ``__len__`` for this array-like class."""
        return self.size

    def resize(self, new_size: int | None = None, do_warn: bool = False) -> None:
        # if new_size = None, use the size from the first field
        for field, obj in self.items():
            if new_size is None:
                new_size = len(obj)
            elif len(obj) != new_size:
                if do_warn:
                    log.warning(
                        f"warning: resizing field {field}"
                        f"with size {len(obj)} != {new_size}"
                    )
                if isinstance(obj, Table):
                    obj.resize(new_size)
                else:
                    obj.resize(new_size)
        self.size = new_size

    def push_row(self) -> None:
        self.loc += 1

    def is_full(self) -> bool:
        return self.loc >= self.size

    def clear(self) -> None:
        self.loc = 0

    def add_field(self, name: str, obj: LGDO, use_obj_size: bool = False) -> None:
        """Add a field (column) to the table.

        Use the name "field" here to match the terminology used in
        :class:`.Struct`.

        Parameters
        ----------
        name
            the name for the field in the table.
        obj
            the object to be added to the table.
        use_obj_size
            if ``True``, resize the table to match the length of `obj`.
        """
        if not hasattr(obj, "__len__"):
            msg = "cannot add field of type"
            raise TypeError(msg, type(obj).__name__)

        super().add_field(name, obj)

        if self.size is None:
            self.size = len(obj)

        # check / update sizes
        if self.size != len(obj):
            warn(
                f"warning: you are trying to add {name} with length {len(obj)} to a table with size {self.size} and data might be lost. \n"
                f"With 'use_obj_size' set to:\n"
                f"  - True, the table will be resized to length {len(obj)} by padding/clipping its columns.\n"
                f"  - False (default), object {name} will be padded/clipped to length {self.size}.",
                UserWarning,
                stacklevel=2,
            )
            new_size = len(obj) if use_obj_size else self.size
            self.resize(new_size=new_size)

    def add_column(self, name: str, obj: LGDO, use_obj_size: bool = False) -> None:
        """Alias for :meth:`.add_field` using table terminology 'column'."""
        self.add_field(name, obj, use_obj_size=use_obj_size)

    def remove_column(self, name: str, delete: bool = False) -> None:
        """Alias for :meth:`.remove_field` using table terminology 'column'."""
        super().remove_field(name, delete)

    def join(
        self, other_table: Table, cols: list[str] | None = None, do_warn: bool = True
    ) -> None:
        """Add the columns of another table to this table.

        Notes
        -----
        Following the join, both tables have access to `other_table`'s fields
        (but `other_table` doesn't have access to this table's fields). No
        memory is allocated in this process. `other_table` can go out of scope
        and this table will retain access to the joined data.

        Parameters
        ----------
        other_table
            the table whose columns are to be joined into this table.
        cols
            a list of names of columns from `other_table` to be joined into
            this table.
        do_warn
            set to ``False`` to turn off warnings associated with mismatched
            `loc` parameter or :meth:`add_column` warnings.
        """
        if other_table.loc != self.loc and do_warn:
            log.warning(f"other_table.loc ({other_table.loc}) != self.loc({self.loc})")
        if cols is None:
            cols = other_table.keys()
        for name in cols:
            self.add_column(name, other_table[name])

    def get_dataframe(
        self,
        cols: list[str] | None = None,
        copy: bool = False,  # noqa: ARG002
        prefix: str = "",
    ) -> pd.DataFrame:
        """Get a :class:`pandas.DataFrame` from the data in the table.

        Warning
        -------
        This method is deprecated. Use :meth:`.view_as` to view the table as a
        Pandas dataframe.

        Notes
        -----
        The requested data must be array-like, with the ``nda`` attribute.

        Parameters
        ----------
        cols
            a list of column names specifying the subset of the table's columns
            to be added to the dataframe.
        copy
            When ``True``, the dataframe allocates new memory and copies data
            into it. Otherwise, the raw ``nda``'s from the table are used directly.
        prefix
            The prefix to be added to the column names. Used when recursively getting the
            dataframe of a Table inside this Table
        """
        warn(
            "`get_dataframe` is deprecated and will be removed in a future release. "
            "Instead use `view_as` to get the Table data as a pandas dataframe "
            "or awkward Array. ",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.view_as(library="pd", cols=cols, prefix=prefix)

    def flatten(self, _prefix="") -> Table:
        """Flatten the table, if nested.

        Returns a new :class:`Table` (that references, not copies, the existing
        columns) with columns in nested tables being moved to the first level
        (and renamed appropriately).

        Examples
        --------
        >>> repr(tbl)
        "Table(dict={'a': Array([1 2 3], attrs={'datatype': 'array<1>{real}'}), 'tbl': Table(dict={'b': Array([4 5 6], attrs={'datatype': 'array<1>{real}'}), 'tbl1': Table(dict={'z': Array([9 9 9], attrs={'datatype': 'array<1>{real}'})}, attrs={'datatype': 'table{z}'})}, attrs={'datatype': 'table{b,tbl1}'})}, attrs={'datatype': 'table{a,tbl}'})"
        >>> tbl.flatten().keys()
        dict_keys(['a', 'tbl__b', 'tbl__tbl1__z'])
        """
        flat_table = Table(size=self.size)
        for key, obj in self.items():
            if isinstance(obj, Table):
                flat_table.join(obj.flatten(_prefix=f"{_prefix}{key}__"))
            else:
                flat_table.add_column(_prefix + key, obj)

        return flat_table

    def eval(
        self,
        expr: str,
        parameters: Mapping[str, str] | None = None,
    ) -> LGDO:
        """Apply column operations to the table and return a new LGDO.

        Internally uses :func:`numexpr.evaluate` if dealing with columns
        representable as NumPy arrays or :func:`eval` if
        :class:`.VectorOfVectors` are involved. In the latter case, the VoV
        columns are viewed as :class:`ak.Array` and the respective routines are
        therefore available.

        To columns nested in subtables can be accessed by scoping with two
        underscores (``__``). For example: ::

          tbl.eval("a + tbl2__b")

        computes the sum of column `a` and column `b` in the subtable `tbl2`.

        Parameters
        ----------
        expr
            if the expression only involves non-:class:`.VectorOfVectors`
            columns, the syntax is the one supported by
            :func:`numexpr.evaluate` (see `here
            <https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/index.html>`_
            for documentation). Note: because of internal limitations,
            reduction operations must appear the last in the stack. If at least
            one considered column is a :class:`.VectorOfVectors`, plain
            :func:`eval` is used and :class:`ak.Array` transforms can be used
            through the ``ak.`` prefix. (NumPy functions are analogously
            accessible through ``np.``). See also examples below.
        parameters
            a dictionary of function parameters. Passed to
            :func:`numexpr.evaluate`` as `local_dict` argument or to
            :func:`eval` as `locals` argument.

        Examples
        --------
        >>> import lgdo
        >>> tbl = lgdo.Table(
        ...   col_dict={
        ...     "a": lgdo.Array([1, 2, 3]),
        ...     "b": lgdo.VectorOfVectors([[5], [6, 7], [8, 9, 0]]),
        ...   }
        ... )
        >>> print(tbl.eval("a + b"))
        [[6],
         [8 9],
         [11 12  3],
        ]
        >>> print(tbl.eval("np.sum(a) + ak.sum(b)"))
        41
        """
        if parameters is None:
            parameters = {}

        # get the valid python variable names in the expression
        c = compile(expr, "0vbb is real!", "eval")

        # make a dictionary of low-level objects (numpy or awkward)
        # for later computation
        flat_self = self.flatten()
        self_unwrap = {}
        has_ak = False
        for obj in c.co_names:
            if obj in flat_self:
                if isinstance(flat_self[obj], VectorOfVectors):
                    self_unwrap[obj] = flat_self[obj].view_as("ak", with_units=False)
                    has_ak = True
                else:
                    self_unwrap[obj] = flat_self[obj].view_as("np", with_units=False)

        msg = f"evaluating {expr!r} with locals={(self_unwrap | parameters)} and {has_ak=}"
        log.debug(msg)

        # use numexpr if we are only dealing with numpy data types
        if not has_ak:
            out_data = ne.evaluate(
                expr,
                local_dict=(self_unwrap | parameters),
            )

            msg = f"...the result is {out_data!r}"
            log.debug(msg)

            # need to convert back to LGDO
            # np.evaluate should always return a numpy thing?
            if out_data.ndim == 0:
                return Scalar(out_data.item())
            if out_data.ndim == 1:
                return Array(out_data)
            if out_data.ndim == 2:
                return ArrayOfEqualSizedArrays(nda=out_data)

            msg = (
                f"evaluation resulted in {out_data.ndim}-dimensional data, "
                "I don't know which LGDO this corresponds to"
            )
            raise RuntimeError(msg)

        # resort to good ol' eval()
        globs = {"ak": ak, "np": np}
        out_data = eval(expr, globs, (self_unwrap | parameters))

        msg = f"...the result is {out_data!r}"
        log.debug(msg)

        # need to convert back to LGDO
        if isinstance(out_data, ak.Array):
            if out_data.ndim == 1:
                return Array(out_data.to_numpy())
            return VectorOfVectors(out_data)

        if np.isscalar(out_data):
            return Scalar(out_data)

        msg = (
            f"evaluation resulted in a {type(out_data)} object, "
            "I don't know which LGDO this corresponds to"
        )
        raise RuntimeError(msg)

    def __str__(self):
        opts = fmt.get_dataframe_repr_params()
        opts["show_dimensions"] = False
        opts["index"] = False

        try:
            string = self.view_as("pd").to_string(**opts)
        except ValueError:
            string = "Cannot print Table with VectorOfVectors yet!"

        string += "\n"
        for k, v in self.items():
            attrs = v.getattrs()
            if attrs:
                string += f"\nwith attrs['{k}']={attrs}"

        attrs = self.getattrs()
        if attrs:
            string += f"\nwith attrs={attrs}"

        return string

    def view_as(
        self,
        library: str,
        with_units: bool = False,
        cols: list[str] | None = None,
        prefix: str = "",
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        r"""View the Table data as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.DataFrame`
        - ``ak``: returns an :class:`ak.Array` (record type)

        Notes
        -----
        Conversion to Awkward array only works when the key is a string.

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.
        cols
            a list of column names specifying the subset of the table's columns
            to be added to the data view structure.
        prefix
            The prefix to be added to the column names. Used when recursively
            getting the dataframe of a :class:`Table` inside this
            :class:`Table`.

        See Also
        --------
        .LGDO.view_as
        """
        if cols is None:
            cols = self.keys()

        if library == "pd":
            df = pd.DataFrame()

            for col in cols:
                data = self[col]

                if isinstance(data, Table):
                    log.debug(f"viewing Table {col=!r} recursively")

                    tmp_df = data.view_as(
                        "pd", with_units=with_units, prefix=f"{prefix}{col}_"
                    )
                    for k, v in tmp_df.items():
                        df[k] = v

                else:
                    log.debug(
                        f"viewing {type(data).__name__} column {col!r} as Pandas Series"
                    )
                    df[f"{prefix}{col}"] = data.view_as("pd", with_units=with_units)

            return df

        if library == "np":
            msg = f"Format {library!r} is not supported for Tables."
            raise TypeError(msg)

        if library == "ak":
            if with_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            # NOTE: passing the Table directly (which inherits from a dict)
            # makes it somehow really slow. Not sure why, but this could be due
            # to extra LGDO fields (like "attrs")
            return ak.Array({col: self[col].view_as("ak") for col in cols})

        msg = f"{library!r} is not a supported third-party format."
        raise TypeError(msg)