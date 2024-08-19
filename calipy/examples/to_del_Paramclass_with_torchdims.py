import pyro
import torch
import math
from calipy.core.base import CalipyNode
from calipy.core.utils import multi_unsqueeze, context_plate_stack, dim_assignment, generate_trivial_dims, DimTuple
from calipy.core.base import NodeStructure
from pyro.distributions import constraints
from abc import ABC, abstractmethod
from functorch.dim import dims




class CalipyQuantity(CalipyNode):
    """
    The CalipyQuantity class provides a comprehensive representation of a specific 
    quantity used in the construction of a CalipyEffect object. This could be a
    known parameter, an unknown parameter, or a random variable. This quantity
    is named, explained, and referenced in the quantity description. Quantities
    are incorporated into the differentiable function that define the CalipyEffect
    forward pass. Each quantity is subservient to an effect and gets a unique id
    that reflects this, quantities are local and cannot be shared between effects.
    """
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        



class UnknownParameter(CalipyQuantity):
    """ UnknownParameter is a subclass of CalipyQuantity that produces an object whose
    forward() method produces a parameter that is subject to inference.

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :param constraint: Pyro constraint that constrains the parameter of a distribution
        to lie in a pre-defined subspace of R^n like e.g. simplex, positive, ...
    :type constraint: pyro.distributions.constraints.Constraint
    :return: Instance of the UnknownParameter class built on the basis of node_structure
    :rtype: UnknownParameter (subclass of CalipyQuantity subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D bias tensor -------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.effects import UnknownParameter
        node_structure = UnknownParameter.example_node_structure
        bias_object = UnknownParameter(node_structure, name = 'tutorial')
        #
        # ii) Produce bias value
        bias = bias_object.forward()
        #
        # iii) Investigate object
        bias_object.dtype_chain
        bias_object.id
        bias_object.node_structure.description
        render_1 = bias_object.render()
        render_1
        render_2 = bias_object.render_comp_graph()
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
    example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure, constraint = constraints.real, **kwargs):  
        # The whole setup is entirely possible if we instead of shapes use dims
        # and e.g. set_dims(dims = 'batch_dims', dim_names = ('batch_dim_1', 'batch_dim_2'), dim_sizes = (10, 7))
        # maybe it is also possible, to set a 'required' flag for some of these
        # quantities and have this info pop up as class attribute.
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_shape = self.node_structure.shapes['batch_shape']
        self.event_shape = self.node_structure.shapes['event_shape']
        
        self.constraint = constraint
        
        trivial_dims_batch = generate_trivial_dims(len(self.batch_shape))
        trivial_dims_event = generate_trivial_dims(len(self.event_shape))
        batch_dims = dim_assignment(dim_names =  ['batch_dim'], dim_shapes = self.batch_shape)
        event_dims = dim_assignment(dim_names =  ['event_dim'], dim_shapes = self.event_shape)
        
        extension_tensor_dims = trivial_dims_batch + event_dims
        init_tensor_dims = batch_dims + trivial_dims_event
        
        self.extension_tensor = torch.ones(extension_tensor_dims.get_sizes())   #ones[ 1, 1, 2,3] 
        self.init_tensor = torch.ones(init_tensor_dims.get_sizes())             # ones [10,5, 1, 1]
    
    # Forward pass is initializing and passing parameter
    def forward(self, input_vars = None, observations = None):
        self.param = pyro.param('{}__param_{}'.format(self.id_short, self.name), init_tensor = self.init_tensor, constraint = self.constraint)
        self.extended_param = self.extension_tensor * self.param
        return self.extended_param
    

    
batch_dims = dim_assignment(dim_names = ['batch_dim_1', 'batch_dim_2'], dim_shapes = [10,5])
event_dims = dim_assignment(dim_names = ['event_dim_1', 'event_dim_2'], dim_shapes = [2,3])


trivial_dims_batch = generate_trivial_dims(len(batch_dims))
trivial_dims_event = generate_trivial_dims(len(event_dims))

extension_tensor_dims = trivial_dims_batch + event_dims
init_tensor_dims = batch_dims + trivial_dims_event
extension_tensor = torch.ones(extension_tensor_dims.get_sizes())
init_tensor = torch.ones(init_tensor_dims.get_sizes())

# node_structure_up = NodeStructure()
# node_structure_up.set_shape('batch_shape', batch_dims, 'description batch dims')
# node_structure_up.set_shape('event_shape', event_dims, 'description event dims')
  
node_structure_up = NodeStructure()
node_structure_up.set_shape('batch_shape', (10,5), 'description batch dims')
node_structure_up.set_shape('event_shape', (2,3), 'description event dims')  

unknown_param = UnknownParameter(node_structure_up)
unknown_param.node_structure.print_shapes_and_plates()
unknown_param.forward().shape
    
    
    
    
bd_1, bd_2 = dims(2)
ed_1 = dims(1)
batch_dims = DimTuple((bd_1,bd_2))
event_dims = DimTuple((ed_1,))    
full_dims = batch_dims + event_dims
    
    
    
def model():
    with pyro.plate('batch_plate', size = 10, subsample_size = 3, dim =-1) as ind:
        print(ind)
    with pyro.plate('batch_plate', size = 10, subsample_size = 3, dim =-1) as ind:
        print(ind)
        
    