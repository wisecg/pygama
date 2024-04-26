from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd


class LGDO(ABC):
    """Abstract base class representing a LEGEND Data Object (LGDO)."""

    @abstractmethod
    def __init__(self, attrs: dict[str, Any] | None = None) -> None:
        self.attrs = {} if attrs is None else dict(attrs)

        if "datatype" in self.attrs:
            if self.attrs["datatype"] != self.form_datatype():
                msg = (
                    f"datatype attribute ({self.attrs['datatype']}) does "
                    f"not match class datatype ({self.form_datatype()})!"
                )
                raise ValueError(msg)
        else:
            self.attrs["datatype"] = self.form_datatype()

    @abstractmethod
    def datatype_name(self) -> str:
        """The name for this LGDO's datatype attribute."""

    @abstractmethod
    def form_datatype(self) -> str:
        """Return this LGDO's datatype attribute string."""

    @abstractmethod
    def view_as(
        self, library: str, with_units: bool = False
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        r"""View the LGDO data object as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation unless
        explicitly stated in the concrete LGDO documentation. The view can be
        turned into a copy explicitly by the user with the appropriate methods.
        If requested by the user, the output format supports it and the LGDO
        carries a ``units`` attribute, physical units are attached to the view
        through the :mod:`pint` package.

        Typical supported third-party libraries are:

        - ``pd``: :mod:`pandas`
        - ``np``: :mod:`numpy`
        - ``ak``: :mod:`awkward`

        Note
        ----
        Awkward does not support attaching units through Pint, at the moment.

        but the actual supported formats may vary depending on the concrete
        LGDO class.

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.
        """

    def getattrs(self, datatype: bool = False) -> dict:
        """Return a copy of the LGDO attributes dictionary.

        Parameters
        ----------
        datatype
            if ``False``, remove ``datatype`` attribute from the output
            dictionary.
        """
        d = dict(self.attrs)
        if not datatype:
            d.pop("datatype", None)
        return d

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(attrs={self.attrs!r})"