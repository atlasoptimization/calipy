# file: calipy/core/dist/distributions.py

import inspect
from inspect import Parameter, signature
from typing import Optional, Dict, Any, List

from calipy.core.tensor import CalipyTensor
from calipy.core.base import NodeStructure, CalipyNode
from calipy.core.utils import dim_assignment, InputSchema
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



class CalipyDistribution(CalipyNode):
    """
    Base class for all auto-generated Calipy distributions.
    It wraps a Pyro distribution class and associates a NodeStructure
    (or any additional dimension-aware info).
    """
    
    # Default empty schemas (will be overridden)
    input_vars_schema = None
    observation_schema = None
    dists = {}  # Maps distribution names to subclasses   
    
    

    def __init__(self, node_structure=None):
        super().__init__()
        self.node_structure = node_structure
        
    @classmethod
    def create_distribution_class(cls, pyro_dist_cls):
        """Dynamically creates a subclass for a Pyro distribution."""
        dist_name = pyro_dist_cls.__name__
        input_schema, obs_schema = generate_schemas_from_pyro(pyro_dist_cls)
        
        # Create subclass with proper inheritance
        class Subclass(cls):
            input_vars_schema = input_schema
            observation_schema = obs_schema
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


def generate_schemas_from_pyro(pyro_dist_cls: type) -> tuple:
    """Generates input_vars and observation schemas from a Pyro distribution class."""
    # Get __init__ parameters (skip 'self')
    init_params = list(signature(pyro_dist_cls.__init__).parameters.values())[1:]  

    # Extract parameters for input_vars_schema
    required_keys = []
    optional_keys = []
    defaults = {}
    key_types = {}
    
    for param in init_params:
        if param.name == "validate_args":
            continue  # Skip common Pyro parameter
        
        if param.default == Parameter.empty:
            required_keys.append(param.name)
        else:
            optional_keys.append(param.name)
            defaults[param.name] = param.default
        
        # Get type annotation or fallback to CalipyTensor
        key_types[param.name] = param.annotation if param.annotation != Parameter.empty else CalipyTensor

    # Input vars schema (distribution parameters)
    input_vars_schema = InputSchema(
        required_keys=required_keys,
        optional_keys=optional_keys,
        defaults=defaults,
        key_types=key_types
    )

    # Observation schema (standardized to value/key)
    observation_schema = InputSchema(
        required_keys=["value"],
        key_types={"value": Any}  # Can specialize based on distribution
    )

    return input_vars_schema, observation_schema