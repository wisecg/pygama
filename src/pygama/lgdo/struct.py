"""
Implements a LEGEND Data Object representing a struct and corresponding
utilities.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pygama.lgdo.lgdo import LGDO

log = logging.getLogger(__name__)


class Struct(LGDO, dict):
    """A dictionary of LGDO's with an optional set of attributes.

    After instantiation, add fields using :meth:`add_field` to keep the
    datatype updated, or call :meth:`update_datatype` after adding.
    """

    def __init__(
        self,
        obj_dict: dict[str, LGDO] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        obj_dict
            instantiate this Struct using the supplied named LGDO's.  Note: no
            copy is performed, the objects are used directly.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if obj_dict is not None:
            self.update(obj_dict)

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "struct"

    def form_datatype(self) -> str:
        return (
            self.datatype_name() + "{" + ",".join([str(k) for k in self.keys()]) + "}"
        )

    def update_datatype(self) -> None:
        self.attrs["datatype"] = self.form_datatype()

    def add_field(self, name: str | int, obj: LGDO) -> None:
        """Add a field to the table."""
        super().__setitem__(name, obj)
        self.update_datatype()

    def __setitem__(self, name: str, obj: LGDO) -> None:
        return self.add_field(name, obj)

    def __getattr__(self, name: str) -> LGDO:
        if hasattr(super(), name):
            return super().__getattr__(name)

        if name in self.keys():
            return super().__getitem__(name)

        raise AttributeError(name)

    def remove_field(self, name: str | int, delete: bool = False) -> None:
        """Remove a field from the table.

        Parameters
        ----------
        name
            name of the field to be removed.
        delete
            if ``True``, delete the field object by calling :any:`del`.
        """
        if delete:
            del self[name]
        else:
            self.pop(name)
        self.update_datatype()

    def __str__(self) -> str:
        """Convert to string (e.g. for printing)."""

        thr_orig = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=8)

        string = "{\n"
        for k, v in self.items():
            if "\n" in str(v):
                rv = str(v).replace("\n", "\n    ")
                string += f" '{k}':\n    {rv},\n"
            else:
                string += f" '{k}': {v},\n"
        string += "}"

        attrs = self.getattrs()
        if attrs:
            string += f" with attrs={attrs}"

        np.set_printoptions(threshold=thr_orig)

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = (
            self.__class__.__name__
            + "(dict="
            + dict.__repr__(self)
            + f", attrs={self.attrs!r})"
        )
        np.set_printoptions(**npopt)
        return " ".join(out.replace("\n", " ").split())

    def view_as(self) -> None:
        r"""View the Struct data as a third-party format data structure.

        Error
        -----
        Not implemented. Since Struct's fields can have different lengths,
        converting to a NumPy, Pandas or Awkward is generally not possible.
        Call :meth:`.LGDO.view_as` on the fields instead.

        See Also
        --------
        .LGDO.view_as
        """
        msg = (
            "Since Struct's fields can have different lengths, "
            "converting to a NumPy, Pandas or Awkward is generally "
            "not possible. Call view_as() on the fields instead."
        )
        raise NotImplementedError(msg)