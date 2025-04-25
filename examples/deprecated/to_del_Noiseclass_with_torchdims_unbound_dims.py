import pyro
import torch
import math
import varname
from calipy.core.base import CalipyNode
from calipy.core.utils import multi_unsqueeze, context_plate_stack, dim_assignment, generate_trivial_dims, DimTuple
from calipy.core.base import NodeStructure
from pyro.distributions import constraints
from abc import ABC, abstractmethod
from functorch.dim import dims


class CalipyEffect(CalipyNode):
    """
    The CalipyEffect class provides a comprehensive representation of a specific 
    effect. It is named, explained, and referenced in the effect description. The
    effect is incorporated as a differentiable function based on torch. This function
    can depend on known parameters, unknown parameters, and random variables. Known 
    parameters have to be provided during invocation of the effect. During training,
    unknown parameters and the posterior density of the random variables is inferred.
    This requires providing a unique name, a prior distribution, and a variational
    distribution for the random variables.
    """
    
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        
        
        self._effect_model = None
        self._effect_guide = None



class NoiseAddition(CalipyEffect):
    """ NoiseAddition is a subclass of CalipyEffect that produces an object whose
    forward() method emulates uncorrelated noise being added to an input. 

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :return: Instance of the NoiseAddition class built on the basis of node_structure
    :rtype: NoiseAddition (subclass of CalipyEffect subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D noise ------------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.effects import NoiseAddition
        node_structure = NoiseAddition.example_node_structure
        noisy_meas_object = NoiseAddition(node_structure, name = 'tutorial')
        #
        # ii) Sample noise
        mean = torch.zeros([10,5])
        std = torch.ones([10,5])
        noisy_meas = noisy_meas_object.forward(input_vars = (mean, std))
        #
        # iii) Investigate object
        noisy_meas_object.dtype_chain
        noisy_meas_object.id
        noisy_meas_object.noise_dist
        noisy_meas_object.node_structure.description
        noisy_meas_object.plate_stack
        render_1 = noisy_meas_object.render((mean, std))
        render_1
        render_2 = noisy_meas_object.render_comp_graph((mean, std))
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_shape('batch_shape', (10, 5), 'Batch shape description')
    example_node_structure.set_shape('event_shape', (2, 3), 'Event shape description')
    # example_node_structure.set_shape('event_shape', (2, ), 'Event shape description')
    # example_node_structure.set_plate_stack('noise_stack', [('batch_plate_1', 5, -2, 'plate denoting independence in row dim'),
    #                                                           ('batch_plate_2', 10, -1, 'plate denoting independence in col dim')],
    #                                        'Plate stack for noise ')

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        
        self.batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_shapes = self.node_structure.shapes['batch_shape'])
        self.event_dims = dim_assignment(dim_names = ['event_dim'], dim_shapes = self.node_structure.shapes['event_shape'])
        self.full_dims = self.batch_dims + self.event_dims
    
    def plate_stack_from_shape(self, plate_stack_name, dim_tuple, stack_description = None):
        # This function could be part of the node_structure. It is supposed to
        # create a plate_stack with a certain name based on some dimensions.
        # It takes the batch dimensions, finds their location w.r.t. all dimensions
        # and sets the appropriate plate stack
        dim_name_list = dim_tuple.names
        dim_size_list = dim_tuple.sizes
        dim_loc_list = self.full_dims.find_indices(dim_name_list)
        dim_doc_list = dim_tuple.descriptions
        
        plate_data_list = [(name, size, loc, doc) for name, size, loc, doc in
                           zip(dim_name_list, dim_size_list, dim_loc_list, dim_doc_list)]
        self.node_structure.set_plate_stack(plate_stack_name, plate_data_list, stack_description)
        
    
    # Forward pass is passing input_vars and sampling from noise_dist
    def forward(self, input_vars, observations = None):
        """
        Create noisy samples using input_vars = (mean, standard_deviation) with
        shapes as indicated in the node_structures' plate_stack 'noise_stack' used
        for noisy_meas_object = NoiseAddition(node_structure).
        
        :param input vars: 2-tuple (mean, standard_deviation) of tensors with 
            equal (or at least broadcastable) shapes. 
        :type input_vars: 2-tuple of instances of torch.Tensor
        :return: Tensor representing simulation of a noisy measurement of the mean.
        :rtype: torch.Tensor
        """
        
        batch_dims = self.batch_dims.get_local_copy()
        event_dims = self.event_dims.get_local_copy()
        full_dims = batch_dims + event_dims
        
        mean_fd = input_vars[0][full_dims]
        # mean_ordered = mean_fd.order(*full_dims)
        
        self.plate_stack_from_shape('noise_stack', batch_dims, 'Plate stack for noise')
        self.noise_stack = self.node_structure.plate_stacks['noise_stack']
        
        self.noise_dist = pyro.distributions.Normal(loc = input_vars[0], scale = input_vars[1])
        
        # Sample within independence context
        with context_plate_stack(self.noise_stack):
            output = pyro.sample('{}__noise_{}'.format(self.id_short, self.name), self.noise_dist, obs = observations)
        return output
    


  
node_structure_na = NodeStructure()
node_structure_na.set_shape('batch_shape', (None,), ('description batch dim_1',))
node_structure_na.set_shape('event_shape', (3,), ('description event dim_1', ))
# node_structure_na.set_shape('event_shape', (2,3), 'description event dims')  

# batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_shapes = node_structure_na.shapes['batch_shape'], 
#                             dim_descriptions = ['row direction', ' column direction'])
# event_dims = dim_assignment(dim_names = ['event_dim'], dim_shapes = node_structure_na.shapes['event_shape'])
# full_dims = batch_dims + event_dims

# dim_name_list = batch_dims.names
# dim_size_list = batch_dims.sizes
# dim_loc_list = batch_dims.find_indices(dim_name_list)
# dim_doc_list = batch_dims.descriptions

# plate_data_list = [(name, size, loc, doc) for name, size, loc, doc in
#                    zip(dim_name_list, dim_size_list, dim_loc_list, dim_doc_list)]
        
noise_addition = NoiseAddition(node_structure_na)
noise_addition.node_structure.print_shapes_and_plates()
mu = torch.ones([13,3])
sigma = torch.ones([1])
noise_addition.forward(input_vars = (mu, sigma))
    
    
    
    
# bd_1, bd_2 = dims(2)
# ed_1 = dims(1)
# batch_dims = DimTuple((bd_1,bd_2))
# event_dims = DimTuple((ed_1,))    
# full_dims = batch_dims + event_dims
    
    
    
    
    
    