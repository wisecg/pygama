"""Implements a LEGEND Data Object representing a scalar and corresponding utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import pygama.lgdo.lgdo_utils as utils
from pygama.lgdo.units import default_units_registry as u
from pygama.lgdo.lgdo import LGDO

log = logging.getLogger(__name__)


class Scalar(LGDO):
    """Holds just a scalar value and some attributes (datatype, units, ...)."""

    # TODO: do scalars need proper numpy dtypes?

    def __init__(
        self, value: int | float | str, attrs: dict[str, Any] | None = None
    ) -> None:
        """
        Parameters
        ----------
        value
            the value for this scalar.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if not np.isscalar(value):
            msg = "cannot instantiate a Scalar with a non-scalar value"
            raise ValueError(msg)

        self.value = value
        super().__init__(attrs)

    def datatype_name(self) -> str:
        if hasattr(self.value, "datatype_name"):
            return self.value.datatype_name

        return utils.get_element_type(self.value)

    def form_datatype(self) -> str:
        return self.datatype_name()

    def view_as(self, with_units: bool = False):
        r"""Dummy function, returns the scalar value itself.

        See Also
        --------
        .LGDO.view_as
        """
        if with_units:
            return self.value * u[self.attrs["units"]]
        return self.value

    def __eq__(self, other: Scalar) -> bool:
        if isinstance(other, Scalar):
            return self.value == other.value and self.attrs == self.attrs

        return False

    def __str__(self) -> str:
        attrs = self.getattrs()
        return f"{self.value!s} with attrs={attrs!r}"

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(value={self.value!r}, attrs={self.attrs!r})"