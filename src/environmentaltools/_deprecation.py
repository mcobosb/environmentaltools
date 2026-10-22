"""Aliases that keep old import paths working after functions were moved.

A module that lost functions defines::

    __getattr__ = moved(__name__, {
        "old_name": ("environmentaltools.models.cshore", "read_output", {"file_": "file_type"}),
    })

Accessing ``module.old_name`` returns a wrapper of the new function that emits a
``DeprecationWarning`` on every call and renames old keyword arguments.
"""

import functools
import importlib
import warnings

REMOVAL = "4.0"


def deprecated_alias(func, old, new, renames=None):
    """Wrap ``func`` so that calling it as ``old`` warns and points to ``new``.

    Args:
        func (callable): Function that now holds the implementation.
        old (str): Old dotted name (e.g. ``"environmentaltools.processes.read_swan"``).
        new (str): New dotted name.
        renames (dict | None): Old keyword names mapped to the new ones.

    Returns:
        callable: Wrapper with the metadata of ``func``.
    """
    renames = renames or {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{old} is deprecated and will be removed in version {REMOVAL}; use {new} instead.",
                      DeprecationWarning, stacklevel=2)
        for old_key, new_key in renames.items():
            if old_key in kwargs:
                kwargs[new_key] = kwargs.pop(old_key)
        return func(*args, **kwargs)

    return wrapper


def moved(module_name, mapping):
    """Build a module ``__getattr__`` (PEP 562) that resolves moved names lazily.

    Targets are imported on first access, which avoids import cycles between
    ``common``/``processes``/``graphics`` and ``models``.

    Args:
        module_name (str): ``__name__`` of the module that lost the functions.
        mapping (dict): ``old_name -> (new_module, new_name)`` or
            ``old_name -> (new_module, new_name, renames)``.

    Returns:
        callable: Function to assign to the module ``__getattr__``.
    """
    cache = {}

    def __getattr__(name):
        if name not in mapping:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        if name not in cache:
            target_module, target_name, *rest = mapping[name]
            func = getattr(importlib.import_module(target_module), target_name)
            cache[name] = deprecated_alias(func, f"{module_name}.{name}", f"{target_module}.{target_name}",
                                           rest[0] if rest else None)
        return cache[name]

    return __getattr__
