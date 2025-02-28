# file: calipy/core/dist/distributions.py

import inspect

from calipy.core.base import NodeStructure, CalipyNode
from calipy.core.utils import dim_assignment
from calipy.core.primitives import sample


def build_default_nodestructure(class_name):

    # Initialize CalipyDistribution wide default NodeStructure
    batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_sizes = [10])
    event_dims = dim_assignment(dim_names = ['event_dim'], dim_sizes = [2])
    batch_dims_description = 'The batch dimension, in which realizations are independent'
    event_dims_description = 'The event dimension, in which a realization is dependent'
    
    default_nodestructure = NodeStructure()
    default_nodestructure.set_dims(batch_dims = batch_dims, event_dims = event_dims)
    default_nodestructure.set_dim_descriptions(batch_dims = batch_dims_description,
                                               event_dims = event_dims_description)
    
    default_nodestructure.set_name(class_name)
    return default_nodestructure


# class CalipyDistribution(CalipyNode):
#     """
#     Base class for all auto-generated Calipy distributions.
#     It wraps a Pyro distribution class and associates a NodeStructure
#     (or any additional dimension-aware info).
#     """
    
#     # _registry = {}  # Track subclasses like Normal, Poisson
#     default_nodestructure = default_nodestructure

#     def __init__(self, pyro_dist_cls, node_structure=None, **dist_params):
#         """
#         :param pyro_dist_cls: The Pyro distribution class being wrapped.
#         :param node_structure: A NodeStructure for dimension logic.
#         :param dist_params: All other distribution parameters (e.g., loc, scale for Normal).
#         """
#         self.pyro_dist_cls = pyro_dist_cls
#         self.node_structure = node_structure
#         self.dist_params = dist_params
#         self.dist_pyro = self.create_pyro_dist()
        
#     @classmethod
#     def register_subclass(cls, name, subclass):
#         # cls._registry[name] = subclass
#         setattr(cls, name, subclass)  # Class-level attribute

#     # Forward pass is generating samples
#     def forward(self, input_vars, observations = None, subsample_index = None):
#         # thoughts: interaction forward, sampling?
#         # Different flow needed
#         #   CalipyNormal = calipy.core.dist.Normal should be of type abc.CalipyDistribution
#         #   calipy_normal = CalipyNormal(node_structure) should be of type abc.CalipyDistribution.Normal
#         #   calipy_normal.forward() should produce some numbers and should effectively be the sampling.
#         # Does it make sense to separate Distributions and DistributionNodes?
#         #   thoughs: 
            
#         # forward should build the distribution (i.e. the pyro distribution by integrating
#         # loc and scale and then produce some numbers by calling calipy_sample
#         pass

#     def create_pyro_dist(self):
#         """
#         Instantiate the underlying Pyro distribution with stored parameters.
#         (Sampling or dimension logic can be added later.)
#         """
#         return self.pyro_dist_cls(**self.dist_params)

#     # def __repr__(self):
#     #     return (f"<CalipyDistribution({self.pyro_dist_cls.__name__}) "
#     #             f"node_structure={self.node_structure} "
#     #             f"params={self.dist_params}>")
    
#     def __repr__(self):
#         return f"<{self.__class__.__qualname__}(node_structure={self.node_structure}, params={self.dist_params})>"

# class CalipyDistribution(CalipyNode):
#     """Base distribution class. Subclasses will be dynamically generated."""

#     def __init__(self, node_structure=None):
#         super().__init__()
#         self.node_structure = node_structure

#     @classmethod
#     def create_distribution_class(cls, pyro_dist_cls):
#         """Dynamically creates a subclass for a Pyro distribution."""
#         dist_name = pyro_dist_cls.__name__
        
#         # Create subclass with proper inheritance
#         class Subclass(cls):
#             _pyro_dist_cls = pyro_dist_cls
#             input_vars = inspect.signature(pyro_dist_cls.__init__).parameters

#             def __init__(self, node_structure=None):
#                 super().__init__(node_structure=node_structure)

#             def forward(self, **input_vars):
#                 # Validate input vars match the Pyro distribution's __init__ signature
#                 bound_args = inspect.signature(self._pyro_dist_cls.__init__).bind(
#                     None,  # Skip 'self'
#                     **input_vars
#                 )
#                 bound_args.apply_defaults()
                
#                 # Create and sample from Pyro distribution
#                 pyro_dist = self._pyro_dist_cls(**bound_args.arguments)
#                 return pyro_dist.sample()

#         # Set class metadata
#         Subclass.__name__ = dist_name
#         Subclass.__qualname__ = f"CalipyDistribution.{dist_name}"
#         Subclass.__module__ = __name__
        
#         return Subclass




class CalipyDistribution(CalipyNode):
    """
    Base class for all auto-generated Calipy distributions.
    It wraps a Pyro distribution class and associates a NodeStructure
    (or any additional dimension-aware info).
    """
    
    
    dists = {}  # Maps distribution names to subclasses   
    

    def __init__(self, node_structure=None):
        super().__init__()
        self.node_structure = node_structure
        
    @classmethod
    def create_distribution_class(cls, pyro_dist_cls):
        """Dynamically creates a subclass for a Pyro distribution."""
        dist_name = pyro_dist_cls.__name__
        
        # Create subclass with proper inheritance
        class Subclass(cls):
            _pyro_dist_cls = pyro_dist_cls
            cls.input_vars = inspect.signature(pyro_dist_cls.__init__).parameters

            def __init__(self, node_structure=None):
                super().__init__(node_structure=node_structure)
                
            def create_pyro_dist(self, input_vars):
                """
                Instantiate the underlying Pyro distribution with input vars as
                parameters. Sampling as dimension handled by sample function.
                """
                
                return pyro_dist_cls(**input_vars.as_dict())


            def forward(self, input_vars, observations = None, subsample_index = None, **kwargs):
                # input vars should be
                
                # Formatting arguments
                vec = kwargs.get('vectorizable', True)
                ssi = subsample_index
                obs = observations
                name = self.id + '_sample'
                dims = self.node_structure.dims
                
                # Building pyro distribution
                n_event_dims = len(self.node_structure.dims['event_dims'].sizes)
                pyro_dist = self.create_pyro_dist(input_vars).to_event(n_event_dims)
 
                # Sampling and compiling
                calipy_sample = sample(name, pyro_dist, dims, observations = obs,
                                       subsample_index = ssi, vectorizable = vec)
                return calipy_sample

        # Set class metadata
        Subclass.__name__ = dist_name
        Subclass.__qualname__ = f"CalipyDistribution.{dist_name}"
        Subclass.__module__ = __name__
        
        # Register the subclass without adding it as an attribute
        cls.dists[dist_name] = Subclass
        return Subclass


    def create_pyro_dist(self):
        """
        Instantiate the underlying Pyro distribution with stored parameters.
        (Sampling or dimension logic can be added later.)
        """
        return self.pyro_dist_cls(**self.dist_params)

    
    def __repr__(self):
        return f"<{self.__class__.__qualname__}(node_structure={self.node_structure})>"



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
