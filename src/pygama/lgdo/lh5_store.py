"""
This module implements routines from reading and writing LEGEND Data Objects in
HDF5 files.
"""
from __future__ import annotations

import fnmatch
import glob
import logging
import os
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any, Union

import h5py
import numba as nb
import numpy as np
import pandas as pd

from pygama.lgdo.array import Array
from pygama.lgdo.arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from pygama.lgdo.fixedsizearray import FixedSizeArray
from pygama.lgdo.lgdo_utils import expand_path, parse_datatype
from pygama.lgdo.scalar import Scalar
from pygama.lgdo.struct import Struct
from pygama.lgdo.table import Table
from pygama.lgdo.vectorofvectors import VectorOfVectors
from pygama.lgdo.waveform_table import WaveformTable

LGDO = Union[Array, Scalar, Struct, VectorOfVectors]

log = logging.getLogger(__name__)


class LH5Store:
    """
    Class to represent a store of LEGEND HDF5 files. The two main methods
    implemented by the class are :meth:`read_object` and :meth:`write_object`.

    Examples
    --------
    >>> from pygama.lgdo import LH5Store
    >>> store = LH5Store()
    >>> obj, _ = store.read_object("/geds/waveform", "file.lh5")
    >>> type(obj)
    pygama.lgdo.waveform_table.WaveformTable
    """

    def __init__(self, base_path: str = "", keep_open: bool = False) -> None:
        """
        Parameters
        ----------
        base_path
            directory path to prepend to LH5 files.
        keep_open
            whether to keep files open by storing the :mod:`h5py` objects as
            class attributes.
        """
        self.base_path = "" if base_path == "" else expand_path(base_path)
        self.keep_open = keep_open
        self.files = {}

    def gimme_file(self, lh5_file: str | h5py.File, mode: str = "r") -> h5py.File:
        """Returns a :mod:`h5py` file object from the store or creates a new one.

        Parameters
        ----------
        lh5_file
            LH5 file name.
        mode
            mode in which to open file. See :class:`h5py.File` documentation.
        """
        if isinstance(lh5_file, h5py.File):
            return lh5_file
        if mode == "r":
            lh5_file = expand_path(lh5_file)
        if lh5_file in self.files.keys():
            return self.files[lh5_file]
        if self.base_path != "":
            full_path = os.path.join(self.base_path, lh5_file)
        else:
            full_path = lh5_file
        if mode != "r":
            directory = os.path.dirname(full_path)
            if directory != "" and not os.path.exists(directory):
                log.debug(f"making path {directory}")
                os.makedirs(directory)
        if mode == "r" and not os.path.exists(full_path):
            raise FileNotFoundError(f"file {full_path} not found")
        if mode != "r" and os.path.exists(full_path):
            log.debug(f"opening existing file {full_path} in mode '{mode}'")
        h5f = h5py.File(full_path, mode)
        if self.keep_open:
            self.files[lh5_file] = h5f
        return h5f

    def gimme_group(
        self,
        group: str,
        base_group: h5py.Group,
        grp_attrs: dict[str, Any] = None,
        overwrite: bool = False,
    ) -> h5py.Group:
        """
        Returns an existing :class:`h5py` group from a base group or creates a
        new one. Can also set (or replace) group attributes.

        Parameters
        ----------
        group
            name of the HDF5 group.
        base_group
            HDF5 group to be used as a base.
        grp_attrs
            HDF5 group attributes.
        overwrite
            whether overwrite group attributes, ignored is `grp_attrs` is
            ``None``.
        """
        if not isinstance(group, h5py.Group):
            if group in base_group:
                group = base_group[group]
            else:
                group = base_group.create_group(group)
                if grp_attrs is not None:
                    group.attrs.update(grp_attrs)
                return group
        if (
            grp_attrs is not None
            and len(set(grp_attrs.items()) ^ set(group.attrs.items())) > 0
        ):
            if not overwrite:
                raise RuntimeError("grp_attrs != group.attrs but overwrite not set")
            else:
                log.debug(f"overwriting {group}.attrs...")
                for key in group.attrs.keys():
                    group.attrs.pop(key)
                group.attrs.update(grp_attrs)
        return group

    def get_buffer(
        self,
        name: str,
        lh5_file: str | h5py.File | list[str | h5py.File],
        size: int = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] = None,
    ) -> LGDO:
        """Returns an LH5 object appropriate for use as a pre-allocated buffer
        in a read loop. Sets size to `size` if object has a size.
        """
        obj, n_rows = self.read_object(name, lh5_file, n_rows=0, field_mask=field_mask)
        if hasattr(obj, "resize") and size is not None:
            obj.resize(new_size=size)
        return obj

    def read_object(
        self,
        name: str,
        lh5_file: str | h5py.File | list[str | h5py.File],
        start_row: int = 0,
        n_rows: int = sys.maxsize,
        idx: np.ndarray | list | tuple | list[np.ndarray | list | tuple] = None,
        use_h5idx: bool = False,
        field_mask: dict[str, bool] | list[str] | tuple[str] = None,
        obj_buf: LGDO = None,
        obj_buf_start: int = 0,
    ) -> tuple[LGDO, int]:
        """Read LH5 object data from a file.

        Use the ``idx`` parameter to read out particular rows of the data. The ``use_h5idx`` flag
        controls whether *only* those rows are read from disk or if the rows are indexed after reading
        the entire object. Reading individual rows can be orders of magnitude slower than reading
        the whole object and then indexing the desired rows. The default behavior (``use_h5idx=False``)
        is to use slightly more memory for a much faster read. See
        `legend-pydataobj #29 <https://github.com/legend-exp/legend-pydataobj/issues/29>`_
        for additional information.

        Parameters
        ----------
        name
            Name of the LH5 object to be read (including its group path).
        lh5_file
            The file(s) containing the object to be read out. If a list of
            files, array-like object data will be concatenated into the output
            object.
        start_row
            Starting entry for the object read (for array-like objects). For a
            list of files, only applies to the first file.
        n_rows
            The maximum number of rows to read (for array-like objects). The
            actual number of rows read will be returned as one of the return
            values (see below).
        idx
            For NumPy-style "fancying indexing" for the read to select only some
            rows, e.g. after applying some cuts to particular columns.
            Only selection along the first axis is supported, so tuple arguments
            must be one-tuples.  If `n_rows` is not false, `idx` will be truncated to
            `n_rows` before reading. To use with a list of files, can pass in a list of
            `idx`'s (one for each file) or use a long contiguous list (e.g. built from a previous
            identical read). If used in conjunction with `start_row` and `n_rows`,
            will be sliced to obey those constraints, where `n_rows` is
            interpreted as the (max) number of *selected* values (in `idx`) to be
            read out. Note that the ``use_h5idx`` parameter controls some behaviour of the
            read and that the default behavior (``use_h5idx=False``) prioritizes speed over
            a small memory penalty.
        use_h5idx
            ``True`` will directly pass the ``idx`` parameter to the underlying
            ``h5py`` call such that only the selected rows are read directly into memory,
            which conserves memory at the cost of speed. There can be a significant penalty
            to speed for larger files (1 - 2 orders of magnitude longer time).
            ``False`` (default) will read the entire object into memory before
            performing the indexing. The default is much faster but requires additional memory,
            though a relatively small amount in the typical use case. It is recommended to
            leave this parameter as its default.
        field_mask
            For tables and structs, determines which fields get written out.
            Only applies to immediate fields of the requested objects. If a dict
            is used, a default dict will be made with the default set to the
            opposite of the first element in the dict. This way if one specifies
            a few fields at ``False``, all but those fields will be read out,
            while if one specifies just a few fields as ``True``, only those
            fields will be read out. If a list is provided, the listed fields
            will be set to ``True``, while the rest will default to ``False``.
        obj_buf
            Read directly into memory provided in `obj_buf`. Note: the buffer
            will be expanded to accommodate the data requested. To maintain the
            buffer length, send in ``n_rows = len(obj_buf)``.
        obj_buf_start
            Start location in ``obj_buf`` for read. For concatenating data to
            array-like objects.

        Returns
        -------
        (object, n_rows_read)
            `object` is the read-out object `n_rows_read` is the number of rows
            successfully read out. Essential for arrays when the amount of data
            is smaller than the object buffer.  For scalars and structs
            `n_rows_read` will be``1``. For tables it is redundant with
            ``table.loc``.
        """
        # Handle list-of-files recursively
        if not isinstance(lh5_file, (str, h5py._hl.files.File)):
            lh5_file = list(lh5_file)
            n_rows_read = 0

            # to know whether we are reading in a list of files.
            # this is part of the fix for reading data by idx
            # (see https://github.com/legend-exp/legend-pydataobj/issues/29)
            # so that we only make a copy of the data if absolutely necessary
            # or if we can read the data from file without having to make a copy
            self.in_file_loop = True

            for i, h5f in enumerate(lh5_file):
                if isinstance(idx, list) and len(idx) > 0 and not np.isscalar(idx[0]):
                    # a list of lists: must be one per file
                    idx_i = idx[i]
                elif idx is not None:
                    # make idx a proper tuple if it's not one already
                    if not (isinstance(idx, tuple) and len(idx) == 1):
                        idx = (idx,)
                    # idx is a long continuous array
                    n_rows_i = self.read_n_rows(name, h5f)
                    # find the length of the subset of idx that contains indices
                    # that are less than n_rows_i
                    n_rows_to_read_i = bisect_left(idx[0], n_rows_i)
                    # now split idx into idx_i and the remainder
                    idx_i = (idx[0][:n_rows_to_read_i],)
                    idx = (idx[0][n_rows_to_read_i:] - n_rows_i,)
                else:
                    idx_i = None
                n_rows_i = n_rows - n_rows_read

                # maybe someone passed in a list of len==1?
                if i == (len(lh5_file) - 1):
                    self.in_file_loop = False

                obj_buf, n_rows_read_i = self.read_object(
                    name,
                    lh5_file[i],
                    start_row=start_row,
                    n_rows=n_rows_i,
                    idx=idx_i,
                    use_h5idx=use_h5idx,
                    field_mask=field_mask,
                    obj_buf=obj_buf,
                    obj_buf_start=obj_buf_start,
                )

                n_rows_read += n_rows_read_i
                if n_rows_read >= n_rows or obj_buf is None:
                    return obj_buf, n_rows_read
                start_row = 0
                obj_buf_start += n_rows_read_i

                self.in_file_loop = False

            return obj_buf, n_rows_read

        # start read from single file. fail if the object is not found
        log.debug(
            f"reading '{name}' from {lh5_file}"
            + (f" with field mask {field_mask}" if field_mask else "")
        )

        # get the file from the store
        h5f = self.gimme_file(lh5_file, "r")
        if not h5f or name not in h5f:
            raise KeyError(f"'{name}' not in {lh5_file}")

        # make idx a proper tuple if it's not one already
        if not (isinstance(idx, tuple) and len(idx) == 1):
            if idx is not None:
                idx = (idx,)

        # get the object's datatype
        if "datatype" not in h5f[name].attrs:
            raise RuntimeError(
                f"'{name}' in file {lh5_file} is missing the datatype attribute"
            )

        datatype = h5f[name].attrs["datatype"]
        datatype, shape, elements = parse_datatype(datatype)

        # check field_mask and make it a default dict
        if datatype == "struct" or datatype == "table":
            if field_mask is None:
                field_mask = defaultdict(lambda: True)
            elif isinstance(field_mask, dict):
                default = True
                if len(field_mask) > 0:
                    default = not field_mask[list(field_mask.keys())[0]]
                field_mask = defaultdict(lambda: default, field_mask)
            elif isinstance(field_mask, (list, tuple)):
                field_mask = defaultdict(
                    lambda: False, {field: True for field in field_mask}
                )
            elif not isinstance(field_mask, defaultdict):
                raise RuntimeError("bad field_mask of type", type(field_mask).__name__)
        elif field_mask is not None:
            raise RuntimeError(f"datatype {datatype} does not accept a field_mask")

        # Scalar
        # scalars are dim-0 datasets
        if datatype == "scalar":
            value = h5f[name][()]
            if elements == "bool":
                value = np.bool(value)
            if obj_buf is not None:
                obj_buf.value = value
                obj_buf.attrs.update(h5f[name].attrs)
                return obj_buf, 1
            else:
                return Scalar(value=value, attrs=h5f[name].attrs), 1

        # Struct
        # recursively build a struct, return as a dictionary
        if datatype == "struct":

            # ignore obj_buf.
            # TODO: could append new fields or overwrite/concat to existing
            # fields. If implemented, get_buffer() above should probably also
            # (optionally?) prep buffers for each field
            if obj_buf is not None:
                raise NotImplementedError("obj_buf not implemented for LGOD Structs")

            # loop over fields and read
            obj_dict = {}
            for field in elements:
                if not field_mask[field]:
                    continue
                # TODO: it's strange to pass start_row, n_rows, idx to struct
                # fields. If they all had shared indexing, they should be in a
                # table... Maybe should emit a warning? Or allow them to be
                # dicts keyed by field name?
                obj_dict[field], _ = self.read_object(
                    name + "/" + field, h5f, start_row=start_row, n_rows=n_rows, idx=idx
                )
            # modify datatype in attrs if a field_mask was used
            attrs = dict(h5f[name].attrs)
            if field_mask is not None:
                selected_fields = []
                for field in elements:
                    if field_mask[field]:
                        selected_fields.append(field)
                attrs["datatype"] = "struct" + "{" + ",".join(selected_fields) + "}"
            return Struct(obj_dict=obj_dict, attrs=attrs), 1

        # Below here is all array-like types. So trim idx if needed
        if idx is not None:
            # chop off indices < start_row
            i_first_valid = bisect_left(idx[0], start_row)
            idxa = idx[0][i_first_valid:]
            # don't readout more than n_rows indices
            idx = (idxa[:n_rows],)  # works even if n_rows > len(idxa)

        # Table or WaveformTable
        if datatype == "table":
            col_dict = {}

            # read out each of the fields
            rows_read = []
            for field in elements:
                if not field_mask[field]:
                    continue
                fld_buf = None
                if obj_buf is not None:
                    if not isinstance(obj_buf, Table) or field not in obj_buf:
                        raise ValueError(
                            f"obj_buf for LGDO Table '{name}' not formatted correctly"
                        )

                    else:
                        fld_buf = obj_buf[field]
                col_dict[field], n_rows_read = self.read_object(
                    name + "/" + field,
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx,
                    use_h5idx=use_h5idx,
                    obj_buf=fld_buf,
                    obj_buf_start=obj_buf_start,
                )
                if obj_buf is not None and obj_buf_start + n_rows_read > len(obj_buf):
                    obj_buf.resize(obj_buf_start + n_rows_read)
                rows_read.append(n_rows_read)
            # warn if all columns don't read in the same number of rows
            if len(rows_read) > 0:
                n_rows_read = rows_read[0]
            else:
                n_rows_read = 0
                log.warning(f"Table '{name}' has no subgroups accepted by field mask")

            for n in rows_read[1:]:
                if n != n_rows_read:
                    log.warning(
                        f"Table '{name}' got strange n_rows_read = {n}, {n_rows_read} was expected"
                    )

            # modify datatype in attrs if a field_mask was used
            attrs = dict(h5f[name].attrs)
            if field_mask is not None:
                selected_fields = []
                for field in elements:
                    if field_mask[field]:
                        selected_fields.append(field)
                attrs["datatype"] = "table" + "{" + ",".join(selected_fields) + "}"

            # fields have been read out, now return a table
            if obj_buf is None:
                # if col_dict contains just 3 objects called t0, dt, and values,
                # return a WaveformTable
                if (
                    len(col_dict) == 3
                    and "t0" in col_dict
                    and "dt" in col_dict
                    and "values" in col_dict
                ):
                    table = WaveformTable(
                        t0=col_dict["t0"], dt=col_dict["dt"], values=col_dict["values"]
                    )
                else:
                    table = Table(col_dict=col_dict, attrs=attrs)
                # set (write) loc to end of tree
                table.loc = n_rows_read
                return table, n_rows_read
            else:
                # We have read all fields into the object buffer. Run
                # checks: All columns should be the same size. So update
                # table's size as necessary, warn if any mismatches are found
                obj_buf.resize(do_warn=True)
                # set (write) loc to end of tree
                obj_buf.loc = obj_buf_start + n_rows_read
                # check attributes
                if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                    raise RuntimeError(
                        f"attrs mismatch. obj_buf.attrs: "
                        f"{obj_buf.attrs}, h5f[{name}].attrs: {attrs}"
                    )
                return obj_buf, n_rows_read

        # VectorOfVectors
        # read out vector of vectors of different size
        if elements.startswith("array"):
            if obj_buf is not None and not isinstance(obj_buf, VectorOfVectors):
                raise ValueError(f"obj_buf for '{name}' not a LGDO VectorOfVectors")

            # read out cumulative_length
            cumulen_buf = None if obj_buf is None else obj_buf.cumulative_length
            cumulative_length, n_rows_read = self.read_object(
                f"{name}/cumulative_length",
                h5f,
                start_row=start_row,
                n_rows=n_rows,
                idx=idx,
                use_h5idx=use_h5idx,
                obj_buf=cumulen_buf,
                obj_buf_start=obj_buf_start,
            )
            # get a view of just what was read out for cleaner code below
            this_cumulen_nda = cumulative_length.nda[
                obj_buf_start : obj_buf_start + n_rows_read
            ]

            if idx is not None and n_rows_read > 0:
                # get the starting indices for each array in flattended data:
                # the starting index for array[i] is cumulative_length[i-1]
                idx2 = (np.asarray(idx[0]).copy() - 1,)
                # re-read cumulative_length with these indices
                # note this will allocate memory for fd_starts!
                fd_start = None
                if idx2[0][0] == -1:
                    idx2 = (idx2[0][1:],)
                    fd_start = 0  # this variable avoids an ndarray append
                fd_starts, fds_n_rows_read = self.read_object(
                    f"{name}/cumulative_length",
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx2,
                    use_h5idx=use_h5idx,
                )
                fd_starts = fd_starts.nda  # we just need the nda
                if fd_start is None:
                    fd_start = fd_starts[0]

                # compute the length that flattened_data will have after the
                # fancy-indexed read
                fd_n_rows = np.sum(this_cumulen_nda[-len(fd_starts) :] - fd_starts)
                if fd_start == 0:
                    fd_n_rows += this_cumulen_nda[0]

                # now make fd_idx
                fd_idx = np.empty(fd_n_rows, dtype="uint32")
                fd_idx = _make_fd_idx(fd_starts, this_cumulen_nda, fd_idx)

                # Now clean up this_cumulen_nda, to be ready
                # to match the in-memory version of flattened_data. Note: these
                # operations on the view change the original array because they are
                # numpy arrays, not lists.
                this_cumulen_nda[-len(fd_starts) :] -= fd_starts
                np.cumsum(this_cumulen_nda, out=this_cumulen_nda)

            else:
                fd_idx = None

                # determine the start_row and n_rows for the flattened_data readout
                fd_start = 0
                if start_row > 0 and n_rows_read > 0:
                    # need to read out the cumulen sample -before- the first sample
                    # read above in order to get the starting row of the first
                    # vector to read out in flattened_data
                    fd_start = h5f[f"{name}/cumulative_length"][start_row - 1]

                    # check limits for values that will be used subsequently
                    if this_cumulen_nda[-1] < fd_start:
                        log.debug(
                            f"this_cumulen_nda[-1] = {this_cumulen_nda[-1]}, "
                            f"fd_start = {fd_start}, "
                            f"start_row = {start_row}, "
                            f"n_rows_read = {n_rows_read}"
                        )
                        raise RuntimeError(
                            f"cumulative_length non-increasing between entries "
                            f"{start_row} and {start_row+n_rows_read} ??"
                        )

                # determine the number of rows for the flattened_data readout
                fd_n_rows = this_cumulen_nda[-1] if n_rows_read > 0 else 0

                # Now done with this_cumulen_nda, so we can clean it up to be ready
                # to match the in-memory version of flattened_data. Note: these
                # operations on the view change the original array because they are
                # numpy arrays, not lists.
                #
                # First we need to subtract off the in-file offset for the start of
                # read for flattened_data
                this_cumulen_nda -= fd_start

            # If we started with a partially-filled buffer, add the
            # appropriate offset for the start of the in-memory flattened
            # data for this read.
            fd_buf_start = np.uint32(0)
            if obj_buf_start > 0:
                fd_buf_start = cumulative_length.nda[obj_buf_start - 1]
                this_cumulen_nda += fd_buf_start

            # Now prepare the object buffer if necessary
            fd_buf = None
            if obj_buf is not None:
                fd_buf = obj_buf.flattened_data
                # grow fd_buf if necessary to hold the data
                fdb_size = fd_buf_start + fd_n_rows
                if len(fd_buf) < fdb_size:
                    fd_buf.resize(fdb_size)

            # now read
            flattened_data, dummy_rows_read = self.read_object(
                f"{name}/flattened_data",
                h5f,
                start_row=fd_start,
                n_rows=fd_n_rows,
                idx=fd_idx,
                use_h5idx=use_h5idx,
                obj_buf=fd_buf,
                obj_buf_start=fd_buf_start,
            )
            if obj_buf is not None:
                return obj_buf, n_rows_read
            return (
                VectorOfVectors(
                    flattened_data=flattened_data,
                    cumulative_length=cumulative_length,
                    attrs=h5f[name].attrs,
                ),
                n_rows_read,
            )

        # Array
        # FixedSizeArray
        # ArrayOfEqualSizedArrays
        # read out all arrays by slicing
        if "array" in datatype:
            if obj_buf is not None:
                if not isinstance(obj_buf, Array):
                    raise ValueError(f"obj_buf for '{name}' not an LGDO Array")
                    obj_buf = None

            # compute the number of rows to read
            # we culled idx above for start_row and n_rows, now we have to apply
            # the constraint of the length of the dataset
            ds_n_rows = h5f[name].shape[0]
            if idx is not None:
                if len(idx[0]) > 0 and idx[0][-1] >= ds_n_rows:
                    log.warning(
                        "idx indexed past the end of the array in the file. Culling..."
                    )
                    n_rows_to_read = bisect_left(idx[0], ds_n_rows)
                    idx = (idx[0][:n_rows_to_read],)
                    if len(idx[0]) == 0:
                        log.warning("idx empty after culling.")
                n_rows_to_read = len(idx[0])
            else:
                n_rows_to_read = ds_n_rows - start_row
            if n_rows_to_read > n_rows:
                n_rows_to_read = n_rows

            # if idx is passed, check if we can make it a slice instead (faster)
            change_idx_to_slice = False

            # prepare the selection for the read. Use idx if available
            if idx is not None:
                # check if idx is empty and convert to slice instead
                if len(idx[0]) == 0:
                    source_sel = np.s_[0:0]
                    change_idx_to_slice = True
                # check if idx is contiguous and increasing
                # if so, convert it to a slice instead (faster)
                elif np.all(np.diff(idx[0]) == 1):
                    source_sel = np.s_[idx[0][0] : idx[0][-1] + 1]
                    change_idx_to_slice = True
                else:
                    source_sel = idx
            else:
                source_sel = np.s_[start_row : start_row + n_rows_to_read]

            # Now read the array
            if obj_buf is not None and n_rows_to_read > 0:
                buf_size = obj_buf_start + n_rows_to_read
                if len(obj_buf) < buf_size:
                    obj_buf.resize(buf_size)
                dest_sel = np.s_[obj_buf_start:buf_size]

                # this is required to make the read of multiple files faster
                # until a better solution found.

                if change_idx_to_slice or idx is None or use_h5idx:
                    # NOTE: if your script fails on this line, it may be because you
                    # have to apply this patch to h5py (or update h5py, if it's
                    # fixed): https://github.com/h5py/h5py/issues/1792
                    h5f[name].read_direct(obj_buf.nda, source_sel, dest_sel)
                else:
                    # it is faster to read the whole object and then do fancy indexing
                    obj_buf.nda[dest_sel] = h5f[name][...][source_sel]

                nda = obj_buf.nda
            else:
                if n_rows == 0:
                    tmp_shape = (0,) + h5f[name].shape[1:]
                    nda = np.empty(tmp_shape, h5f[name].dtype)
                else:
                    if change_idx_to_slice or idx is None or use_h5idx:
                        nda = h5f[name][source_sel]
                    else:
                        # it is faster to read the whole object and then do fancy indexing
                        nda = h5f[name][...][source_sel]

                        # if reading a list of files recursively, this is given to obj_buf on                        
                        # the first file read. obj_buf needs to be resized and therefore
                        # it needs to hold the data itself (not a view of the data).
                        # a view is returned by the source_sel indexing, which cannot be resized
                        # by ndarray.resize().
                        if hasattr(self, "in_file_loop") and self.in_file_loop:
                            nda = np.copy(nda)
                            
            # special handling for bools
            # (c and Julia store as uint8 so cast to bool)
            if elements == "bool":
                nda = nda.astype(np.bool)

            # Finally, set attributes and return objects
            attrs = h5f[name].attrs
            if obj_buf is None:
                if datatype == "array":
                    return Array(nda=nda, attrs=attrs), n_rows_to_read
                if datatype == "fixedsize_array":
                    return FixedSizeArray(nda=nda, attrs=attrs), n_rows_to_read
                if datatype == "array_of_equalsized_arrays":
                    return (
                        ArrayOfEqualSizedArrays(nda=nda, dims=shape, attrs=attrs),
                        n_rows_to_read,
                    )
            else:
                if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                    raise RuntimeError(
                        f"attrs mismatch. "
                        f"obj_buf.attrs: {obj_buf.attrs}, "
                        f"h5f[{name}].attrs: {attrs}"
                    )
                return obj_buf, n_rows_to_read

        raise RuntimeError("don't know how to read datatype {datatype}")

    def write_object(
        self,
        obj: LGDO,
        name: str,
        lh5_file: str | h5py.File,
        group: str | h5py.Group = "/",
        start_row: int = 0,
        n_rows: int = None,
        wo_mode: str = "append",
        write_start: int = 0,
    ) -> None:
        """Write an LGDO into an LH5 file.

        Parameters
        ----------
        obj
            LH5 object. if object is array-like, writes `n_rows` starting from
            `start_row` in `obj`.
        name
            name of the object in the output HDF5 file.
        lh5_file
            HDF5 file name or :class:`h5py.File` object.
        group
            HDF5 group name or :class:`h5py.Group` object in which `obj` should
            be written.
        start_row
            first row in `obj` to be written.
        n_rows
            number of rows in `obj` to be written.
        wo_mode
            - ``write_safe`` or ``w``: only proceed with writing if the
              object does not already exist in the file.
            - ``append`` or ``a``: append along axis 0 (the first dimension)
              of array-like objects and array-like subfields of structs.
              :class:`~.lgdo.scalar.Scalar` objects get overwritten.
            - ``overwrite`` or ``o``: replace data in the file if present,
              starting from `write_start`. Note: overwriting with `write_start` =
              end of array is the same as ``append``.
            - ``overwrite_file`` or ``of``: delete file if present prior to
              writing to it. `write_start` should be 0 (its ignored).
        write_start
            row in the output file (if already existing) to start overwriting
            from.
        """
        if wo_mode == "write_safe":
            wo_mode = "w"
        if wo_mode == "append":
            wo_mode = "a"
        if wo_mode == "overwrite":
            wo_mode = "o"
        if wo_mode == "overwrite_file":
            wo_mode = "of"
            write_start = 0
        if wo_mode != "w" and wo_mode != "a" and wo_mode != "o" and wo_mode != "of":
            raise ValueError(f"unknown wo_mode '{wo_mode}'")

        # "mode" is for the h5df.File and wo_mode is for this function
        # In hdf5, 'a' is really "modify" -- in addition to appending, you can
        # change any object in the file. So we use file:append for
        # write_object:overwrite.
        mode = "w" if wo_mode == "of" else "a"
        lh5_file = self.gimme_file(lh5_file, mode=mode)
        group = self.gimme_group(group, lh5_file)
        if wo_mode == "w" and name in group:
            raise RuntimeError(f"can't overwrite '{name}' in wo_mode 'write_safe'")

        # struct or table or waveform table
        if isinstance(obj, Struct):
            group = self.gimme_group(
                name, group, grp_attrs=obj.attrs, overwrite=(wo_mode == "o")
            )
            for field in obj.keys():
                self.write_object(
                    obj[field],
                    field,
                    lh5_file,
                    group=group,
                    start_row=start_row,
                    n_rows=n_rows,
                    wo_mode=wo_mode,
                    write_start=write_start,
                )
            return

        # scalars
        elif isinstance(obj, Scalar):
            if name in group:
                if wo_mode in ["o", "a"]:
                    log.debug(f"overwriting {name} in {group}")
                    del group[name]
                else:
                    raise RuntimeError(
                        f"tried to overwrite {name} in {group} for wo_mode {wo_mode}"
                    )
            ds = group.create_dataset(name, shape=(), data=obj.value)
            ds.attrs.update(obj.attrs)
            return

        # vector of vectors
        elif isinstance(obj, VectorOfVectors):
            group = self.gimme_group(
                name, group, grp_attrs=obj.attrs, overwrite=(wo_mode == "o")
            )
            if (
                n_rows is None
                or n_rows > obj.cumulative_length.nda.shape[0] - start_row
            ):
                n_rows = obj.cumulative_length.nda.shape[0] - start_row

            # if appending we need to add an appropriate offset to the
            # cumulative lengths as appropriate for the in-file object
            offset = 0  # declare here because we have to subtract it off at the end
            if (wo_mode == "a" or wo_mode == "o") and "cumulative_length" in group:
                len_cl = len(group["cumulative_length"])
                if wo_mode == "a":
                    write_start = len_cl
                if len_cl > 0:
                    offset = group["cumulative_length"][write_start - 1]

            # First write flattened_data array. Only write rows with data.
            fd_start = 0 if start_row == 0 else obj.cumulative_length.nda[start_row - 1]
            fd_n_rows = obj.cumulative_length.nda[start_row + n_rows - 1] - fd_start
            self.write_object(
                obj.flattened_data,
                "flattened_data",
                lh5_file,
                group=group,
                start_row=fd_start,
                n_rows=fd_n_rows,
                wo_mode=wo_mode,
                write_start=offset,
            )

            # now offset is used to give appropriate in-file values for
            # cumulative_length. Need to adjust it for start_row
            if start_row > 0:
                offset -= obj.cumulative_length.nda[start_row - 1]

            # Add offset to obj.cumulative_length itself to avoid memory allocation.
            # Then subtract it off after writing! (otherwise it will be changed
            # upon return)
            obj.cumulative_length.nda += offset

            self.write_object(
                obj.cumulative_length,
                "cumulative_length",
                lh5_file,
                group=group,
                start_row=start_row,
                n_rows=n_rows,
                wo_mode=wo_mode,
                write_start=write_start,
            )
            obj.cumulative_length.nda -= offset

            return

        # if we get this far, must be one of the Array types
        elif isinstance(obj, Array):
            if n_rows is None or n_rows > obj.nda.shape[0] - start_row:
                n_rows = obj.nda.shape[0] - start_row
            nda = obj.nda[start_row : start_row + n_rows]
            # hack to store bools as uint8 for c / Julia compliance
            if nda.dtype.name == "bool":
                nda = nda.astype(np.uint8)
            # need to create dataset from ndarray the first time for speed
            # creating an empty dataset and appending to that is super slow!
            if (wo_mode != "a" and write_start == 0) or name not in group:
                maxshape = (None,) + nda.shape[1:]
                if wo_mode == "o" and name in group:
                    log.debug(f"overwriting {name} in {group}")
                    del group[name]
                ds = group.create_dataset(name, data=nda, maxshape=maxshape)
                ds.attrs.update(obj.attrs)
                return

            # Now append or overwrite
            ds = group[name]
            old_len = ds.shape[0]
            if wo_mode == "a":
                write_start = old_len
            add_len = write_start + nda.shape[0] - old_len
            ds.resize(old_len + add_len, axis=0)
            ds[write_start:] = nda
            return

        else:
            raise RuntimeError(
                f"do not know how to write '{name}' of type '{type(obj).__name__}'"
            )

    def read_n_rows(self, name: str, lh5_file: str | h5py.File) -> int | None:
        """Look up the number of rows in an Array-like object called `name` in
        `lh5_file`.

        Return ``None`` if it is a :class:`.Scalar` or a :class:`.Struct`."""
        # this is basically a stripped down version of read_object
        h5f = self.gimme_file(lh5_file, "r")
        if not h5f or name not in h5f:
            raise KeyError(f"'{name}' not in {lh5_file}")

        # get the datatype
        if "datatype" not in h5f[name].attrs:
            raise RuntimeError(
                f"'{name}' in file {lh5_file} is missing the datatype attribute"
            )

        datatype = h5f[name].attrs["datatype"]
        datatype, shape, elements = parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == "scalar":
            return None

        # structs don't have rows
        if datatype == "struct":
            return None

        # tables should have elements with all the same length
        if datatype == "table":
            # read out each of the fields
            rows_read = None
            for field in elements:
                n_rows_read = self.read_n_rows(name + "/" + field, h5f)
                if not rows_read:
                    rows_read = n_rows_read
                elif rows_read != n_rows_read:
                    log.warning(
                        f"table '{name}' got strange n_rows_read = {rows_read}, "
                        f"{n_rows_read} was expected"
                    )
            return rows_read

        # length of vector of vectors is the length of its cumulative_length
        if elements.startswith("array"):
            return self.read_n_rows(f"{name}/cumulative_length", h5f)

        # return array length (without reading the array!)
        if "array" in datatype:
            # compute the number of rows to read
            return h5f[name].shape[0]

        raise RuntimeError(f"don't know how to read datatype '{datatype}'")


def ls(lh5_file: str | h5py.Group, lh5_group: str = "") -> list[str]:
    """Return a list of LH5 groups in the input file and group, similar
    to ``ls`` or ``h5ls``. Supports wildcards in group names.


    Parameters
    ----------
    lh5_file
        name of file.
    lh5_group
        group to search. add a ``/`` to the end of the group name if you want to
        list all objects inside that group.
    """

    log.debug(
        f"Listing objects in '{lh5_file}'"
        + ("" if lh5_group == "" else f" (and group {lh5_group})")
    )

    lh5_st = LH5Store()
    # To use recursively, make lh5_file a h5group instead of a string
    if isinstance(lh5_file, str):
        lh5_file = lh5_st.gimme_file(lh5_file, "r")
        if lh5_group.startswith("/"):
            lh5_group = lh5_group[1:]

    if lh5_group == "":
        lh5_group = "*"

    splitpath = lh5_group.split("/", 1)
    matchingkeys = fnmatch.filter(lh5_file.keys(), splitpath[0])

    if len(splitpath) == 1:
        return matchingkeys
    else:
        ret = []
        for key in matchingkeys:
            ret.extend([f"{key}/{path}" for path in ls(lh5_file[key], splitpath[1])])
        return ret


def show(
    lh5_file: str | h5py.Group,
    lh5_group: str = "/",
    indent: str = "",
    header: bool = True,
) -> None:
    """Print a tree of LH5 file contents with LGDO datatype.

    Parameters
    ----------
    lh5_file
        the LH5 file.
    lh5_group
        print only contents of this HDF5 group.
    indent
        indent the diagram with this string.
    header
        print `lh5_group` at the top of the diagram.

    Examples
    --------
    >>> from pygama.lgdo import show
    >>> show("file.lh5", "/geds/raw")
    /geds/raw
    ├── channel · array<1>{real}
    ├── energy · array<1>{real}
    ├── timestamp · array<1>{real}
    ├── waveform · table{t0,dt,values}
    │   ├── dt · array<1>{real}
    │   ├── t0 · array<1>{real}
    │   └── values · array_of_equalsized_arrays<1,1>{real}
    └── wf_std · array<1>{real}
    """
    # open file
    if isinstance(lh5_file, str):
        lh5_file = h5py.File(expand_path(lh5_file), "r")

    # go to group
    if lh5_group != "/":
        lh5_file = lh5_file[lh5_group]

    if header:
        print(f"\033[1m{lh5_group}\033[0m")  # noqa: T201

    # get an iterator over the keys in the group
    it = iter(lh5_file)
    key = None

    # make sure there is actually something in this file/group
    try:
        key = next(it)  # get first key
    except StopIteration:
        print(f"{indent}└──  empty")  # noqa: T201
        return

    # loop over keys
    while True:
        val = lh5_file[key]
        # we want to print the LGDO datatype
        attr = val.attrs.get("datatype", default="no datatype")
        if attr == "no datatype" and isinstance(val, h5py.Group):
            attr = "HDF5 group"

        # is this the last key?
        killme = False
        try:
            k_new = next(it)  # get next key
        except StopIteration:
            char = "└──"
            killme = True  # we'll have to kill this loop later
        else:
            char = "├──"

        print(f"{indent}{char} \033[1m{key}\033[0m · {attr}")  # noqa: T201

        # if it's a group, call this function recursively
        if isinstance(val, h5py.Group):
            show(val, indent=indent + ("    " if killme else "│   "), header=False)

        # break or move to next key
        if killme:
            break
        else:
            key = k_new


def load_nda(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] = None,
) -> dict[str, np.ndarray]:
    r"""Build a dictionary of :class:`numpy.ndarray`\ s from LH5 data.

    Given a list of files, a list of LH5 table parameters, and an optional
    group path, return a NumPy array with all values for each parameter.

    Parameters
    ----------
    f_list
        A list of files. Can contain wildcards.
    par_list
        A list of parameters to read from each file.
    lh5_group
        group path within which to find the specified parameters.
    idx_list
        for fancy-indexed reads. Must be one index array for each file in
        `f_list`.

    Returns
    -------
    par_data
        A dictionary of the parameter data keyed by the elements of `par_list`.
        Each entry contains the data for the specified parameter concatenated
        over all files in `f_list`.
    """
    if isinstance(f_list, str):
        f_list = [f_list]
        if idx_list is not None:
            idx_list = [idx_list]
    if idx_list is not None and len(f_list) != len(idx_list):
        raise ValueError(
            f"f_list length ({len(f_list)}) != idx_list length ({len(idx_list)})!"
        )

    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    sto = LH5Store()
    par_data = {par: [] for par in par_list}
    for ii, f in enumerate(f_list):
        f = sto.gimme_file(f, "r")
        for par in par_list:
            if f"{lh5_group}/{par}" not in f:
                raise RuntimeError(f"'{lh5_group}/{par}' not in file {f_list[ii]}")

            if idx_list is None:
                data, _ = sto.read_object(f"{lh5_group}/{par}", f)
            else:
                data, _ = sto.read_object(f"{lh5_group}/{par}", f, idx=idx_list[ii])
            if not data:
                continue
            par_data[par].append(data.nda)
    par_data = {par: np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] = None,
) -> pd.DataFrame:
    """Build a :class:`pandas.DataFrame` from LH5 data.

    Given a list of files (can use wildcards), a list of LH5 columns, and
    optionally the group path, return a :class:`pandas.DataFrame` with all
    values for each parameter.

    See Also
    --------
    :func:`load_nda`

    Returns
    -------
    dataframe
        contains columns for each parameter in `par_list`, and rows containing
        all data for the associated parameters concatenated over all files in
        `f_list`.
    """
    return pd.DataFrame(
        load_nda(f_list, par_list, lh5_group=lh5_group, idx_list=idx_list)
    )


class LH5Iterator:
    """
    A class for iterating through one or more LH5 files, one block of entries
    at a time. This also accepts an entry list/mask to enable event selection,
    and a field mask.

    This class can be used either for random access:

    >>> lh5_obj, n_rows = lh5_it.read(entry)

    to read the block of entries starting at entry. In case of multiple files
    or the use of an event selection, entry refers to a global event index
    across files and does not count events that are excluded by the selection.

    This can also be used as an iterator:

    >>> for lh5_obj, entry, n_rows in LH5Iterator(...):
    >>>    # do the thing!

    This is intended for if you are reading a large quantity of data but
    want to limit your memory usage (particularly when reading in waveforms!).
    The ``lh5_obj`` that is read by this class is reused in order to avoid
    reallocation of memory; this means that if you want to hold on to data
    between reads, you will have to copy it somewhere!
    """

    def __init__(
        self,
        lh5_files: str | list[str],
        group: str,
        base_path: str = "",
        entry_list: list[int] | list[list[int]] = None,
        entry_mask: list[bool] | list[list[bool]] = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] = None,
        buffer_len: int = 3200,
    ) -> None:
        """
        Parameters
        ----------
        lh5_files
            file or files to read from. May include wildcards and environment
            variables.
        group
            HDF5 group to read.
        base_path
            HDF5 path to prepend.
        entry_list
            list of entry numbers to read. If a nested list is provided,
            expect one top-level list for each file, containing a list of
            local entries. If a list of ints is provided, use global entries.
        entry_mask
            mask of entries to read. If a list of arrays is provided, expect
            one for each file. Ignore if a selection list is provided.
        field_mask
            mask of which fields to read. See :meth:`LH5Store.read_object` for
            more details.
        buffer_len
            number of entries to read at a time while iterating through files.
        """
        self.lh5_st = LH5Store(base_path=base_path, keep_open=True)

        # List of files, with wildcards and env vars expanded
        if isinstance(lh5_files, str):
            lh5_files = expand_path(lh5_files, True)
        elif not isinstance(lh5_files, list):
            raise ValueError("lh5_files must be a string or list of strings")

        if entry_list is not None and entry_mask is not None:
            raise ValueError(
                "entry_list and entry_mask arguments are mutually exclusive"
            )

        self.lh5_files = [f for f_wc in lh5_files for f in expand_path(f_wc, True)]

        # Map to last row in each file
        self.file_map = np.array(
            [self.lh5_st.read_n_rows(group, f) for f in self.lh5_files], "int64"
        ).cumsum()
        self.group = group
        self.buffer_len = buffer_len

        if len(self.lh5_files) > 0:
            self.lh5_buffer = self.lh5_st.get_buffer(
                self.group,
                self.lh5_files[0],
                size=self.buffer_len,
                field_mask=field_mask,
            )
        else:
            raise RuntimeError(f"can't open any files from {lh5_files}")

        self.n_rows = 0
        self.current_entry = 0

        self.field_mask = field_mask

        # List of entry indices from each file
        self.entry_list = None
        if entry_list is not None:
            entry_list = list(entry_list)
            if isinstance(entry_list[0], int):
                entry_list.sort()
                i_start = 0
                self.entry_list = []
                for f_end in self.file_map:
                    i_stop = bisect_right(entry_list, f_end, lo=i_start)
                    self.entry_list.append(entry_list[i_start:i_stop])
                    i_start = i_stop

            else:
                self.entry_list = [[]] * len(self.file_map)
                for i_file, local_list in enumerate(entry_list):
                    self.entry_list[i_file] = list(local_list)

        elif entry_mask is not None:
            # Convert entry mask into an entry list
            if isinstance(entry_mask, pd.Series):
                entry_mask = entry_mask.values
            if isinstance(entry_mask, np.ndarray):
                self.entry_list = []
                f_start = 0
                for f_end in self.file_map:
                    self.entry_list.append(
                        list(np.nonzero(entry_mask[f_start:f_end])[0])
                    )
                    f_start = f_end
            else:
                self.entry_list = [[]] * len(self.file_map)
                for i_file, local_mask in enumerate(entry_mask):
                    self.entry_list[i_file] = list(np.nonzero(local_mask)[0])

        # Map to last entry of each file
        self.entry_map = (
            self.file_map
            if self.entry_list is None
            else np.array([len(elist) for elist in self.entry_list]).cumsum()
        )

    def read(self, entry: int) -> tuple[LGDO, int]:
        """Read the next chunk of events, starting at entry. Return the
        LH5 buffer and number of rows read."""
        i_file = np.searchsorted(self.entry_map, entry, "right")
        local_entry = entry
        if i_file > 0:
            local_entry -= self.entry_map[i_file - 1]
        self.n_rows = 0

        while self.n_rows < self.buffer_len and i_file < len(self.file_map):
            # Loop through files
            local_idx = self.entry_list[i_file] if self.entry_list is not None else None
            i_local = local_idx[local_entry] if local_idx is not None else local_entry
            self.lh5_buffer, n_rows = self.lh5_st.read_object(
                self.group,
                self.lh5_files[i_file],
                start_row=i_local,
                n_rows=self.buffer_len - self.n_rows,
                idx=local_idx,
                field_mask=self.field_mask,
                obj_buf=self.lh5_buffer,
                obj_buf_start=self.n_rows,
            )

            self.n_rows += n_rows
            i_file += 1
            local_entry = 0

        self.current_entry = entry
        return (self.lh5_buffer, self.n_rows)

    def __len__(self) -> int:
        """Return the total number of entries."""
        return self.entry_map[-1] if len(self.entry_map) > 0 else 0

    def __iter__(self) -> tuple[LGDO, int, int]:
        """Loop through entries in blocks of size buffer_len."""
        entry = 0
        while entry < len(self):
            buf, n_rows = self.read(entry)
            yield (buf, entry, n_rows)
            entry += n_rows


@nb.njit(parallel=False, fastmath=True)
def _make_fd_idx(starts, stops, idx):
    k = 0
    if len(starts) < len(stops):
        for i in range(stops[0]):
            idx[k] = i
            k += 1
        stops = stops[1:]
    for j in range(len(starts)):
        for i in range(starts[j], stops[j]):
            idx[k] = i
            k += 1
    return (idx,)
