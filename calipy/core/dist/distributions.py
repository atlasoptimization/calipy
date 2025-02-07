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





# import inspect

# def generate_init_for_distribution(dist_cls, base_cls):
#     """
#     Dynamically create an __init__ method that:
#       1) Matches the signature of dist_cls.__init__.
#       2) Adds a 'node_structure=None' parameter (keyword-only).
#       3) Calls the base class's __init__ with the resolved parameters.
#     """

#     # Original signature of the Pyro dist constructor
#     original_sig = inspect.signature(dist_cls.__init__)
#     original_params = list(original_sig.parameters.values())

#     # Remove 'self' if present
#     if original_params and original_params[0].name == 'self':
#         original_params = original_params[1:]

#     # Insert our extra param for node_structure (keyword-only with default=None)
#     node_param = inspect.Parameter(
#         name='node_structure',
#         kind=inspect.Parameter.KEYWORD_ONLY,
#         default=None
#     )
    
#     # Insert node_param before a **kwargs param if it exists:
#     for i, param in enumerate(original_params):
#         if param.kind == inspect.Parameter.VAR_KEYWORD:
#             original_params.insert(i, node_param)
#             break
#     else:
#         original_params.append(node_param)

#     # Build the new signature
#     new_sig = inspect.Signature(parameters=original_params)

#     def __init__(self, *args, **kwargs):
#         # Bind arguments to the new signature
#         bound_args = new_sig.bind(self, *args, **kwargs)
#         bound_args.apply_defaults()
        
#         # Extract node_structure, if present
#         node_structure = bound_args.arguments.pop('node_structure', None)

#         # Remove 'self'
#         bound_args.arguments.pop('self', None)

#         # Everything else is part of dist_params
#         dist_params = dict(bound_args.arguments)

#         # Call the base class __init__
#         super(self.__class__, self).__init__(
#             pyro_dist_cls=dist_cls,
#             node_structure=node_structure,
#             **dist_params
#         )

#     # Attach the newly created signature and docstring for IDE/tooltips
#     __init__.__signature__ = new_sig
#     # Optionally copy the docstring from the original Pyro distribution
#     doc = (dist_cls.__doc__ or "") + "\n\nOriginal __init__ doc:\n"
#     doc += (dist_cls.__init__.__doc__ or "")
#     __init__.__doc__ = doc

#     return __init__

import inspect

def generate_init_for_distribution(dist_cls, base_cls):
    original_sig = inspect.signature(dist_cls.__init__)
    original_params = list(original_sig.parameters.values())

    # Keep 'self' in the parameter list (don't remove it)
    # Inject node_structure as keyword-only after positional params
    node_param = inspect.Parameter(
        "node_structure", 
        inspect.Parameter.KEYWORD_ONLY, 
        default=None
    )

    # Find the first keyword-only or var-kwarg to insert after
    insert_pos = len(original_params)
    for i, param in enumerate(original_params):
        if param.kind in (param.KEYWORD_ONLY, param.VAR_KEYWORD):
            insert_pos = i
            break

    original_params.insert(insert_pos, node_param)
    new_sig = original_sig.replace(parameters=original_params)

    def __init__(self, *args, **kwargs):
        bound_args = new_sig.bind(*args, **kwargs)  # No explicit 'self' here
        bound_args.apply_defaults()
        node_structure = bound_args.arguments.pop("node_structure", None)
        dist_params = bound_args.arguments
        
        super(base_cls, self).__init__(
            pyro_dist_cls=dist_cls,
            node_structure=node_structure,
            **dist_params
        )

    __init__.__signature__ = new_sig
    __init__.__doc__ = f"{dist_cls.__doc__}\n\nnode_structure: Optional[NodeStructure] = None"

    return __init__
