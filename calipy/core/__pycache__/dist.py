# file: calipy/dist/distribution_base.py

import inspect

class CalipyDistribution:
    """
    Base class for all auto-generated calipy distributions.
    Wraps a pyro distribution class and associates a NodeStructure.
    """

    def __init__(self, pyro_dist_cls, node_structure=None, **dist_params):
        """
        :param pyro_dist_cls: The Pyro distribution class being wrapped.
        :param node_structure: A NodeStructure instance for dimension management.
        :param dist_params: All other distribution parameters (e.g., loc, scale for Normal).
        """
        self.pyro_dist_cls = pyro_dist_cls
        self.node_structure = node_structure
        self.dist_params = dist_params  # store for later usage (e.g., sampling)

    def create_pyro_dist(self):
        """
        Create the actual Pyro distribution instance from the stored parameters.
        (Sampling and dimension logic can be implemented later.)
        """
        return self.pyro_dist_cls(**self.dist_params)

    def __repr__(self):
        return (f"<CalipyDistribution({self.pyro_dist_cls.__name__}) "
                f"with node_structure={self.node_structure} and "
                f"params={self.dist_params}>")
