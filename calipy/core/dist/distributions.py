# file: calipy/core/dist/distributions.py


class CalipyDistribution:
    """
    Base class for all auto-generated Calipy distributions.
    It wraps a Pyro distribution class and associates a NodeStructure
    (or any additional dimension-aware info).
    """

    def __init__(self, pyro_dist_cls, node_structure=None, **dist_params):
        """
        :param pyro_dist_cls: The Pyro distribution class being wrapped.
        :param node_structure: A NodeStructure for dimension logic.
        :param dist_params: All other distribution parameters (e.g., loc, scale for Normal).
        """
        self.pyro_dist_cls = pyro_dist_cls
        self.node_structure = node_structure
        self.dist_params = dist_params

    def create_pyro_dist(self):
        """
        Instantiate the underlying Pyro distribution with stored parameters.
        (Sampling or dimension logic can be added later.)
        """
        return self.pyro_dist_cls(**self.dist_params)

    def __repr__(self):
        return (f"<CalipyDistribution({self.pyro_dist_cls.__name__}) "
                f"node_structure={self.node_structure} "
                f"params={self.dist_params}>")

import inspect

def generate_init_for_distribution(dist_cls, base_cls):
    original_sig = inspect.signature(dist_cls.__init__)
    original_params = list(original_sig.parameters.values())

    # Inject node_structure as keyword-only
    node_param = inspect.Parameter(
        "node_structure",
        inspect.Parameter.KEYWORD_ONLY,
        default=None
    )

    # Find insertion point after positional parameters
    insert_pos = len(original_params)
    for i, param in enumerate(original_params):
        if param.kind in (param.KEYWORD_ONLY, param.VAR_KEYWORD):
            insert_pos = i
            break

    original_params.insert(insert_pos, node_param)
    new_sig = original_sig.replace(parameters=original_params)

    def __init__(self, *args, **kwargs):
        bound_args = new_sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        node_structure = bound_args.arguments.pop("node_structure", None)
        bound_args.arguments.pop("self", None)  # Remove self from params

        # EXPLICITLY CALL BASE CLASS INIT (NO AMBIGUOUS super())
        base_cls.__init__(
            self,
            pyro_dist_cls=dist_cls,
            node_structure=node_structure,
            **bound_args.arguments
        )

    __init__.__signature__ = new_sig
    __init__.__doc__ = f"{dist_cls.__doc__}\n\nnode_structure: Optional[NodeStructure] = None"

    return __init__
