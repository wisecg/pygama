from __future__ import annotations

import pint

default_units_registry = pint.get_application_registry()
default_units_registry.default_format = "~P"