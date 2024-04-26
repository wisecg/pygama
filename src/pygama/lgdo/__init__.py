"""
Pygama works with "LEGEND Data Objects" (LGDO) defined in the `LEGEND data
format specification <https://github.com/legend-exp/legend-data-format-specs>`_.
This subpackage serves as the Python implementation of that specification. The
general strategy for the implementation is to dress standard Python and NumPy
objects with an ``attr`` dictionary holding LGDO metadata, plus some convenience
functions. The basic data object classes are:

* :class:`.Scalar`: typed Python scalar. Access data via the :attr:`value`
  attribute
* :class:`.Array`: basic :class:`numpy.ndarray`. Access data via the
  :attr:`nda` attribute.
* :class:`.FixedSizeArray`: basic :class:`numpy.ndarray`. Access data via the
  :attr:`nda` attribute.
* :class:`.ArrayOfEqualSizedArrays`: multi-dimensional :class:`numpy.ndarray`.
  Access data via the :attr:`nda` attribute.
* :class:`.VectorOfVectors`: a variable length array of variable length arrays.
  Implemented as a pair of :class:`.Array`: :attr:`flattened_data` holding the
  raw data, and :attr:`cumulative_length` whose ith element is the sum of the
  lengths of the vectors with ``index <= i``
* :class:`.Struct`: a dictionary containing LGDO objects. Derives from
  :class:`dict`
* :class:`.Table`: a :class:`.Struct` whose elements ("columns") are all array
  types with the same length (number of rows)

Currently the primary on-disk format for LGDO object is LEGEND HDF5 (LH5) files. IO
is done via the class :class:`.lh5_store.LH5Store`. LH5 files can also be
browsed easily in python like any `HDF5 <https://www.hdfgroup.org>`_ file using
`h5py <https://www.h5py.org>`_.
"""

from pygama.lgdo.array import Array
from pygama.lgdo.arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from pygama.lgdo.fixedsizearray import FixedSizeArray
from pygama.lgdo.encoded import ArrayOfEncodedEqualSizedArrays, VectorOfEncodedVectors
from pygama.lgdo.lh5_store import LH5Iterator, LH5Store, load_dfs, load_nda, ls, show
from pygama.lgdo.scalar import Scalar
from pygama.lgdo.struct import Struct
from pygama.lgdo.table import Table
from pygama.lgdo.vectorofvectors import (
    VectorOfVectors
)
from pygama.lgdo.waveform_table import WaveformTable
from pygama.lgdo.vovutils import build_cl, explode, explode_arrays, explode_cl

__all__ = [
    "Array",
    "ArrayOfEqualSizedArrays",
    "FixedSizeArray",
    "Scalar",
    "Struct",
    "Table",
    "VectorOfVectors",
    "WaveformTable",
    "LH5Iterator",
    "LH5Store",
    "load_dfs",
    "load_nda",
    "ls",
    "show",
    "build_cl",
    "explode",
    "explode_arrays",
    "explode_cl",
    "ArrayOfEncodedEqualSizedArrays", 
    "VectorOfEncodedVectors"
]

import numpy as np

np.set_printoptions(threshold=10)
