# file: calipy/core/dist/__init__.py

# file: calipy/dist/__init__.py

import sys
import inspect
import pyro.distributions as pyro_dists
from calipy.core.dist.distributions import CalipyDistribution
from calipy.core.dist.distributions import generate_init_for_distribution, build_default_nodestructure

__all__ = []

def _make_calipy_distribution(dist_name, pyro_dist_cls):
    """
    Factory that dynamically creates a subclass of CalipyDistribution
    for a given Pyro distribution, preserving the original signature.
    """

    # Build a dict of attributes for our new class
    attrs = {}

    # Generate a custom __init__ that respects Pyro's signature
    new_init = generate_init_for_distribution(pyro_dist_cls, base_cls=CalipyDistribution)
    attrs['__init__'] = new_init


    # # Create a new type
    # new_class = type(dist_name, (CalipyDistribution,), attrs)

    # # Optionally set a nicer __qualname__
    # new_class.__qualname__ = dist_name
    # new_class.name = 'CalipyDistribution ' + dist_name

    # return new_class
    
    
    # # Create a new subclass of CalipyDistribution
    # new_class = type(
    #     dist_name,
    #     (CalipyDistribution,),
    #     attrs,
    # )
    
    new_class = CalipyDistribution.create_distribution_class(pyro_dist_cls)

    # Set the qualname to nest under CalipyDistribution
    new_class.__qualname__ = f"CalipyDistribution.{dist_name}"
    new_class.__name__ = dist_name
    new_class.name = new_class.__qualname__
    new_class.default_nodestructure = build_default_nodestructure(new_class.name)

    # Attach the subclass to CalipyDistribution as a nested class
    # setattr(CalipyDistribution, dist_name, new_class)

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
