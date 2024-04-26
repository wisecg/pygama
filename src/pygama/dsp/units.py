from pint import UnitRegistry, set_application_registry
import pint

unit_registry = UnitRegistry(auto_reduce_dimensions=True)
set_application_registry(unit_registry)

default_units_registry = pint.get_application_registry()
default_units_registry.default_format = "~P"