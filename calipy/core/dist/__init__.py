# file: calipy/core/dist/__init__.py

# file: calipy/dist/__init__.py

import sys
import inspect
import pyro.distributions as pyro_dists
from calipy.core.dist.distributions import CalipyDistribution
from calipy.core.dist.distributions import generate_init_for_distribution

__all__ = []

def _make_calipy_distribution(dist_name, dist_cls):
    """
    Factory that dynamically creates a subclass of CalipyDistribution
    for a given Pyro distribution, preserving the original signature.
    """

    # Build a dict of attributes for our new class
    attrs = {}

    # Generate a custom __init__ that respects Pyro's signature
    new_init = generate_init_for_distribution(dist_cls, base_cls=CalipyDistribution)
    attrs['__init__'] = new_init

    # Create a new type
    new_class = type(dist_name, (CalipyDistribution,), attrs)

    # Optionally set a nicer __qualname__
    new_class.__qualname__ = dist_name

    return new_class

# Dynamically scan pyro.distributions
for _name in dir(pyro_dists):
    _obj = getattr(pyro_dists, _name)
    if not inspect.isclass(_obj):
        continue
    # Simple check: must have a 'sample' method
    if not hasattr(_obj, 'sample'):
        continue

    # Create a dynamic subclass
    wrapper_class = _make_calipy_distribution(_name, _obj)

    # Install it into our module
    setattr(sys.modules[__name__], _name, wrapper_class)
    __all__.append(_name)
