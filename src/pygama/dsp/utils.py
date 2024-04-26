import os
from abc import ABCMeta
from collections.abc import MutableMapping
from typing import Any, Iterator


def getenv_bool(name: str, default: bool = False) -> bool:
    """Get environment value as a boolean, returning True for 1, t and true
    (caps-insensitive), and False for any other value and default if undefined.
    """
    val = os.getenv(name)
    if not val:
        return default
    elif val.lower() in ("1", "t", "true"):
        return True
    else:
        return False


class NumbaDefaults(MutableMapping):
    """Bare-bones class to store some Numba default options. Defaults values
    are set from environment variables

    Examples
    --------
    Set all default option values for a processor at once by expanding the
    provided dictionary:

    >>> from numba import guvectorize
    >>> from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs
    >>> @guvectorize([], "", **nb_kwargs, nopython=True) # def proc(...): ...

    Customize one argument but still set defaults for the others:

    >>> from pygama.dsp.utils import numba_defaults as nb_defaults
    >>> @guvectorize([], "", **nb_defaults(cache=False) # def proc(...): ...

    Override global options at runtime:

    >>> from pygama.dsp.utils import numba_defaults
    >>> from pygama.dsp import build_dsp
    >>> # must set options before explicitly importing pygama.dsp.processors!
    >>> numba_defaults.cache = False
    >>> numba_defaults.boundscheck = True
    >>> build_dsp(...) # if not explicit, processors imports happen here
    """

    def __init__(self) -> None:
        self.cache: bool = getenv_bool("PYGAMA_CACHE")
        self.boundscheck: bool = getenv_bool("PYGAMA_BOUNDSCHECK")

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def __setitem__(self, item: str, val: Any) -> None:
        self.__dict__[item] = val

    def __delitem__(self, item: str) -> None:
        del self.__dict__[item]

    def __iter__(self) -> Iterator:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __call__(self, **kwargs) -> dict:
        mapping = self.__dict__.copy()
        mapping.update(**kwargs)
        return mapping

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)


numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults

class ProcChainVarBase(metaclass=ABCMeta):
    r"""Base class.

    :class:`ProcChainVar` implements this class. This base class is used
    by processors that use ProcChainVar in their constructors.
    """

    pass