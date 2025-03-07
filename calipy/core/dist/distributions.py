# file: calipy/core/dist/distributions.py

import inspect
import textwrap
from inspect import Parameter, signature
from typing import Optional, Dict, Any, List

from calipy.core.tensor import CalipyTensor
from calipy.core.base import NodeStructure, CalipyNode
from calipy.core.utils import dim_assignment, InputSchema
from calipy.core.primitives import sample
from calipy.core.data import CalipyDict


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
    Base class for all auto-generated Calipy distributions. It wraps a Pyro 
    distribution class and associates a NodeStructure (or any additional 
    dimension-aware info). CalipyDistributions are Subclasses of CalipyNode and
    therefore come together with a method forward(input_vars, observations, 
    subsample_index, **kwargs) method. The forward() method of distributions can
    be seen as a way of passing parameters and sampling; the format of inputs 
    and observations is documented within the methods create_input_vars or 
    create_observations that help turn data into the DataTuples needed as input
    for .forward(). Al CalipyDistributions come with a default_nodestructure
    consisting of just batch_dims and event_dims being a single dimension with
    either size 10 or 2.
    
    CalipyDistribution is not usually called by the user, it is called by an
    script __init__.py in calipy.core.dist automatically executed during package
    import. The subclasses of CalipyDistribution are e.g. CalipyDistribution.Normal
    or CalipyDistribution.Gamma which can be accessed via calipy.core.dist.Normal
    or calipy.core.dist.Gamma
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # CalipyDistribution objects are CalipyNodes --------------------------
        #
        # i) Imports and definitions
        import torch
        import calipy
        from calipy.core.base import NodeStructure
        from calipy.core.tensor import CalipyTensor
        from calipy.core.data import DataTuple
        #
        # ii) Invoke and investigate CalipyDistribution
        CalipyNormal = calipy.core.dist.Normal
        CalipyNormal.dists
        CalipyNormal.input_vars
        CalipyNormal.input_vars_schema
        
        # iii) Build a concrete Node
        normal_ns = NodeStructure(CalipyNormal)
        print(normal_ns)
        calipy_normal = CalipyNormal(node_structure = normal_ns, node_name = 'Normal')
        
        calipy_normal.id
        calipy_normal.node_structure
        CalipyNormal.default_nodestructure
        
        # Calling the forward method
        normal_dims = normal_ns.dims['batch_dims'] + normal_ns.dims['event_dims']
        normal_ns_sizes = normal_dims.sizes
        mean = CalipyTensor(torch.zeros(normal_ns_sizes), normal_dims)
        standard_deviation = CalipyTensor(torch.ones(normal_ns_sizes), normal_dims)
        input_vars_normal = DataTuple(['loc', 'scale'], [mean, standard_deviation])
        samples_normal = calipy_normal.forward(input_vars_normal)
        samples_normal
        samples_normal.dims
        
        # A more convenient way of creating the input_vars and observations data or
        # at least getting the info on the input signatures
        create_input_vars = CalipyNormal.create_input_vars
        help(create_input_vars)
        input_vars_normal_alt = create_input_vars(loc = mean, scale = standard_deviation)
        samples_normal_alt = calipy_normal.forward(input_vars_normal_alt)
        
        # Since distributions are nodes, we can illustrate them
        calipy_normal.dtype_chain
        calipy_normal.id
        render_1 = calipy_normal.render(input_vars_normal)
        render_1
        render_2 = calipy_normal.render_comp_graph(input_vars_normal)
        render_2
    """
    
    # Default empty schemas (will be overridden)
    input_vars_schema = None
    observation_schema = None
    dists = {}  # Maps distribution names to subclasses   
    
    

    def __init__(self, node_structure=None, node_name = None):
        super().__init__(node_name = node_name)
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
            input_vars = inspect.signature(pyro_dist_cls.__init__).parameters

            def __init__(self, node_structure=None, node_name = dist_name):
                super().__init__(node_structure=node_structure, node_name = node_name)
                
            def create_pyro_dist(self, input_vars):
                """
                Instantiate the underlying Pyro distribution with input vars as
                parameters. Sampling as dimension handled by sample function.
                """
                
                input_vars_tensors = input_vars.as_datatuple().get_tensors()
                return pyro_dist_cls(**input_vars_tensors.as_dict())


            def forward(self, input_vars, observations = None, subsample_index = None, **kwargs):
                # wrap input_vars and observations to CalipyDict
                input_vars_cp = CalipyDict(input_vars)
                observations_cp = CalipyDict(observations)
                
                # Formatting arguments
                vec = kwargs.get('vectorizable', True)
                ssi = subsample_index
                obs = observations_cp
                name = '{}__sample__{}'.format(self.id_short, self.name)
                dims = self.node_structure.dims
                
                # Building pyro distribution
                n_event_dims = len(self.node_structure.dims['event_dims'].sizes)
                pyro_dist = self.create_pyro_dist(input_vars_cp).to_event(n_event_dims)
 
                # Sampling and compiling
                # obs_or_None = obs['sample'].tensor if obs is not None else None
                calipy_sample = sample(name, pyro_dist, dims, observations = obs.value,
                                       subsample_index = ssi, vectorizable = vec)
                return CalipyDict(calipy_sample)

        # Set class metadata
        Subclass.__name__ = dist_name
        Subclass.__qualname__ = f"CalipyDistribution.{dist_name}"
        Subclass.__module__ = __name__
        Subclass.__doc__ = textwrap.dedent(f"""\
            {dist_name} Distribution Subclass of CalipyDistribution 
            inherited from pyro.distributions.{dist_name}.
            
            Input variables to `forward()` are:
            {input_schema}
            
            Observations are:
            {obs_schema}.
        """)
        
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
        required_keys=["sample"],
        key_types={"sample": Any}  # Can specialize based on distribution
    )

    return input_vars_schema, observation_schema