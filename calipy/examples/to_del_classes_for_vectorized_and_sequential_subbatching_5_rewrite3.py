# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. We investigate 
# here, how this can be used to perform subbatching without a model-rewrite
# We make our life a bit easier by only considering one batch dim for now.
# We feature nontrivial event_shape and functorch.dim based indexing.



import pyro
import pyro.distributions as dist
import torch
import contextlib
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functorch.dim import dims
from calipy.core.utils import dim_assignment, generate_trivial_dims, context_plate_stack, DimTuple

import numpy as np
import random
import varname
import copy
import matplotlib.pyplot as plt

torch.manual_seed(42)
pyro.set_rng_seed(42)


# i) Generate synthetic data A and B

# Generate data_A corresponding e.g. to two temperature vals on a grid on a plate
n_data_1_A = 6
n_data_2_A = 4
batch_shape_A = [n_data_1_A, n_data_2_A]
n_event_A = 2
n_event_dims_A = 1
n_data_total_A = n_data_1_A * n_data_2_A


# # Plate names and sizes
plate_names_A = ['batch_plate_1_A', 'batch_plate_2_A']
plate_sizes_A = [n_data_1_A, n_data_2_A]

mu_true_A = torch.zeros([1,1,n_event_A])
sigma_true_A = torch.tensor([[1,0],[0,0.01]])

extension_tensor_A = torch.ones([n_data_1_A,n_data_2_A,1])
data_dist_A = pyro.distributions.MultivariateNormal(loc = mu_true_A * extension_tensor_A, covariance_matrix = sigma_true_A)
data_A = data_dist_A.sample()

# Generate data_B corresponding e.g. to three temperature vals on each location on a rod
n_data_1_B = 5
batch_shape_B = [n_data_1_B]
n_event_B = 3
n_event_dims_B = 1
n_data_total_B = n_data_1_B


# # Plate names and sizes
plate_names_B = ['batch_plate_1_B']
plate_sizes_B = [n_data_1_B]

mu_true_B = torch.zeros([1,n_event_B])
sigma_true_B = torch.tensor([[1,0,0],[0,0.1,0],[0,0,0.01]])

extension_tensor_B = torch.ones([n_data_1_B,1])
data_dist_B = pyro.distributions.MultivariateNormal(loc = mu_true_B * extension_tensor_B, covariance_matrix = sigma_true_B)
data_B = data_dist_B.sample()


# functorch dims
batch_dims_A = dim_assignment(dim_names = ['bd_1_A', 'bd_2_A'])
event_dims_A = dim_assignment(dim_names = ['ed_1_A'])
data_dims_A = batch_dims_A + event_dims_A

batch_dims_B = dim_assignment(dim_names = ['bd_1_B'])
event_dims_B = dim_assignment(dim_names = ['ed_1_B'])
data_dims_B = batch_dims_B + event_dims_B


# i) DataTuple Class
class DataTuple:
    """
    Custom class for holding tuples of various objects with explicit names.
    Provides methods for easily distributing functions over the entries in the
    tuple and thereby makes modifying collections of objects easier. This is
    routinely used to perform actions on grouped observation tensors, batch_dims,
    or event_dims.
    
    :param names: A list of names serving as keys for the DataTuple.
    :type names: list of string
    :param values: A list of objects serving as values for the DataTuple.
    :type values: list of obj
    
    :return: An instance of DataTuple containing the key, value pairs and additional
        attributes and methods.
    :rtype: DataTuple

    Example usage:

    .. code-block:: python

        # Create DataTuple of tensors
        names = ['tensor_A', 'tensor_B']
        values = [torch.ones(2, 3), torch.ones(4, 5)]
        data_tuple = DataTuple(names, values)
        data_tuple['tensor_A']
        
        # Apply functions
        fun = lambda x: x +1
        result_tuple_1 = data_tuple.apply_elementwise(fun)
        print("Result of applying function:", result_tuple_1, result_tuple_1['tensor_A'], result_tuple_1['tensor_B'])
        fun_dict = {'tensor_A': lambda x: x + 1, 'tensor_B': lambda x: x - 1}
        result_tuple_2 = data_tuple.apply_from_dict(fun_dict)
        print("Result of applying function dictionary:", result_tuple_2, result_tuple_2['tensor_A'], result_tuple_2['tensor_B'])
        

        
        # Create DataTuple of dimensions
        batch_dims_A = dim_assignment(dim_names = ['bd_A',])
        event_dims_A = dim_assignment(dim_names = ['ed_A'])       
        batch_dims_B = dim_assignment(dim_names = ['bd_B'])
        event_dims_B = dim_assignment(dim_names = ['ed_B'])
        
        batch_dims_tuple = DataTuple(names, [batch_dims_A, batch_dims_B])
        event_dims_tuple = DataTuple(names, [event_dims_A, event_dims_B])
        
        # Add them 
        added_tensor_tuple = data_tuple + data_tuple
        full_dims_datatuple = batch_dims_tuple + event_dims_tuple
        
        # Construct indexer
        data_tuple.indexer_construct(full_dims_datatuple)
        augmented_tensor = data_tuple['tensor_A']
        augmented_tensor.calipy.indexer.local_index
        
        # Access subattributes
        shapes_tuple = data_tuple.get_subattributes('shape')
        print("Shapes of each tensor in DataTuple:", shapes_tuple)
        batch_dims_datatuple.get_subattributes('sizes')
        batch_dims_datatuple.get_subattributes('build_torchdims')
        
        # Set new item
        data_tuple['tensor_C'] = torch.ones([6,6])
        print(data_tuple)
        
        # Apply class over each element
        class DifferentClass:
            def __init__(self, tensor):
                self.tensor = tensor
        
            def __repr__(self):
                return f"DifferentClass(tensor={self.tensor})"
        
        different_tuple = data_tuple.apply_class(DifferentClass)
        print("Result of applying DifferentClass to DataTuple:", different_tuple)
        
    """
    def __init__(self, names, values):
        if len(names) != len(values):
            raise ValueError("Length of names must match length of values.")
        self._data_dict = {name: value for name, value in zip(names, values)}

    def __getitem__(self, key):
        return self._data_dict[key]

    def __setitem__(self, key, value):
        """
        Allows assignment of new key-value pairs or updating existing ones.
        """
        self._data_dict[key] = value

    def keys(self):
        return self._data_dict.keys()

    def values(self):
        return self._data_dict.values()

    def items(self):
        return self._data_dict.items()
    
    
    def apply_from_dict(self, fun_dict):
        """
        Applies functions from a dictionary to corresponding entries in the DataTuple.
        If a key in fun_dict matches a key in DataTuple, the function is applied.
    
        :param fun_dict: Dictionary with keys corresponding to DataTuple keys and values as functions.
        :return: New DataTuple with functions applied where specified.
        """
        new_dict = {}
        for key, value in self._data_dict.items():
            if key in fun_dict:
                new_value = fun_dict[key](value)
            else:
                new_value = value
            new_dict[key] = new_value
        key_list, value_list = zip(*new_dict.items())
        return DataTuple(key_list, value_list)

    def apply_elementwise(self, function):
        """ 
        Returns a new DataTuple with keys = self._data_dict.keys() and associated
        values = function(self._data_dict.values())
        """
        
        fun_dict = {key: function for key in self._data_dict.keys()}
        return_tuple = self.apply_from_dict(fun_dict)        
            
        return return_tuple

    def get_subattributes(self, attr):
        """
        Allows direct access to attributes or methods of elements inside the DataTuple.
        For example, calling `data_tuple.get_subattributes('shape')` will return a DataTuple
        containing the shapes of each tensor in `data_tuple`.

        :param attr: The attribute or method name to be accessed.
        :return: DataTuple containing the attribute values or method results for each element.
        """
        new_dict = {}
        for key, value in self._data_dict.items():
            if hasattr(value, attr):
                attribute = getattr(value, attr)
                # If it's a method, call it
                new_dict[key] = attribute() if callable(attribute) else attribute
            else:
                raise AttributeError(f"Object '{key}' of type '{type(value)}' has no attribute '{attr}'")
        key_list, value_list = zip(*new_dict.items())
        return DataTuple(key_list, value_list)
    
    def apply_class(self, class_type):
        """
        Allows applying a class constructor to all elements in the DataTuple.
        For example, DifferentClass(data_tuple) will apply DifferentClass to each element in data_tuple.

        :param class_type: The class constructor to be applied to each element.
        :return: New DataTuple with the class constructor applied to each element.
        """
        if not callable(class_type):
            raise ValueError("The provided class_type must be callable.")

        new_dict = {}
        for key, value in self._data_dict.items():
            new_dict[key] = class_type(value)
        
        key_list, value_list = zip(*new_dict.items())
        return DataTuple(key_list, value_list)
 
    def indexer_construct(self, dims_datatuple):
        """
        Applies construction of the CalipyIndexer to build for each tensor in self
        the self.calipy.indexer instance used for indexing. Requires all elements
        of self to be tensors and requires dims_datatuple to be a DataTuple containing
        DimTuples.

        :param self: A DataTuple containing indexable tensors to be indexed
        :param dims_datatuple: A DataTuple containing the DimTuples used for indexing
        :type dim_datatuple: DataTuple
        :return: Nothing returned, calipy.indexer integrated into tensors in self
        :rtype: None, CalipyIndexer
        :raises ValueError: If both DataTuples do not have matching keys.
        """
        # Check consistency
        if self.keys() != dims_datatuple.keys():
            raise ValueError("Both DataTuples self and dims_datatuple must have the same keys.")
        # Apply indexer_construction
        fun_dict = {}
        for key, value in self.items():
            fun_dict[key] = lambda tensor, key = key: tensor.calipy.indexer_construct(dims_datatuple[key], name = key, silent = False)
        self.apply_from_dict(fun_dict)
        return
    
    # def bind_dims(self, datatuple_dims):
    #     """ 
    #     Returns a new DataTuple of tensors with dimensions bound to the dims
    #     recorded in the DataTuple datatuple_dims.
    #     """
        
    #     for key, value in self._data_dict.items():
    #         if isinstance(value, torch.Tensor) or value is None:
    #             pass
    #         else:
    #             raise Exception('bind dims only available for tensors or None '\
    #                             'but tuple element is {}'.format(value.__class__))
    #     new_dict = {}
    #     key_list = []
    #     value_list = []
        
    #     for key, value in self._data_dict.items():
    #         new_key = key
    #         new_value =  value[datatuple_dims[key]]
            
    #         new_dict[new_key] = new_value
    #         key_list.append(new_key)
    #         value_list.append(new_value)
            
    #     return DataTuple(key_list, value_list)
                    

    # def get_local_copy(self):
    #     """
    #     Returns a new DataTuple with local copies of all DimTuple instances.
    #     Non-DimTuple items remain unchanged.
    #     """
        
    #     return_tuple = self.get_subattributes('get_local_copy')
    #     return return_tuple
        # local_copy_data = {}
        # key_list = []
        # value_list = []
        
        # for key, value in self._data_dict.items():
        #     if isinstance(value, DimTuple):
        #         local_copy_data[key] = value.get_local_copy()
        #     else:
        #         local_copy_data[key] = value
        #     key_list.append(key)
        #     value_list.append(value)
            
        # return DataTuple(key_list, value_list)

    def __add__(self, other):
        """ 
        Overloads the + operator to return a new DataTuple when adding two DataTuple objects.
        
        :param other: The DataTuple to add.
        :type other: DataTuple
        :return: A new DataTuple with elements from each tuple added elementwise.
        :rtype: DataTuple
        :raises ValueError: If both DataTuples do not have matching keys.
        """
        if not isinstance(other, DataTuple):
            return NotImplemented

        if self.keys() != other.keys():
            raise ValueError("Both DataTuples must have the same keys for elementwise addition.")

        combined_data = {}
        for key in self.keys():
            combined_data[key] = self[key] + other[key]

        return DataTuple(list(combined_data.keys()), list(combined_data.values()))

    def __repr__(self):
        repr_items = []
        for k, v in self._data_dict.items():
            if isinstance(v, torch.Tensor):
                repr_items.append(f"{k}: shape={v.shape}")
            else:
                repr_items.append(f"{k}: {v.__repr__()}")
        return f"DataTuple({', '.join(repr_items)})"

# Test the DataTuple definitions
# Define names and values for DataTuple
names = ['tensor_a', 'tensor_b']
values = [torch.ones(2, 3), torch.zeros(4, 5)]

# Create DataTuple instance
data_tuple = DataTuple(names, values)

# 1. Apply a function dictionary to DataTuple
fun_dict = {'tensor_a': lambda x: x + 1, 'tensor_b': lambda x: x - 1}
result_tuple = data_tuple.apply_from_dict(fun_dict)
print("Result of applying function dictionary:", result_tuple)

# 2. Get the shapes of each tensor in DataTuple
shapes_tuple = data_tuple.get_subattributes('shape')
print("Shapes of each tensor in DataTuple:", shapes_tuple)

# 3. Apply a custom class DifferentClass to each element in DataTuple
class DifferentClass:
    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        return f"DifferentClass(tensor={self.tensor})"

different_tuple = data_tuple.apply_class(DifferentClass)
print("Result of applying DifferentClass to DataTuple:", different_tuple)

# 4. Adding two DataTuples
another_values = [torch.ones(2, 3) * 2, torch.ones(4, 5) * 3]
another_data_tuple = DataTuple(names, another_values)
added_tuple = data_tuple + another_data_tuple
print("Result of adding two DataTuples:", added_tuple)   
  

# New CalipyIndexer Class
class CalipyIndex:
    """ 
    Class acting as a collection of infos on a specific index tensor collecting
    basic index_tensor, index_tensor_tuple, and index_tensor_named. This class
    represents a specific index tensor.
        index_tensor.tensor is the original index tensor
        index_tensor.tuple can be used for indexing via data[tuple]
        index_tensor.named can be used for dimension specific operations
    """
    def __init__(self, index_tensor, index_tensor_dims, name = None):
        self.name = name
        self.tensor = index_tensor
        self.tuple = index_tensor.unbind(-1)
        # self.named = index_tensor[index_tensor_dims]
        self.dims = index_tensor_dims
        self.index_name_dict = self.generate_index_name_dict()
        
    def generate_index_name_dict(self):
        """
        Generate a dictionary that maps indices to unique names.

        :param global_index: CalipyIndex containing global index tensor.
        :return: Dict mapping each elment of index tensor to unique name.
        """

        index_to_name_dict = {}
        indextensor_flat = self.tensor.flatten(0,-2)
        for k in range(indextensor_flat.shape[0]):
            idx = indextensor_flat[k, :]
            dim_name_list = [dim.name for dim in self.dims[0:-1]]
            idx_name_list = [str(i.long().item()) for i in idx]
            idx_str = f"{self.name}__sample__{'_'.join(dim_name_list)}__{'_'.join(idx_name_list)}"
            index_to_name_dict[tuple(idx.tolist())] = idx_str

        return index_to_name_dict

    def __repr__(self):
        sizes = [size for size in self.tensor.shape]
        repr_string = 'CalipyIndex for tensor with dims {} and sizes {}'.format(self.dims.names, sizes)
        return repr_string





class CalipyIndexer:
    """
    Class to handle indexing operations for observations, including creating local and global indices,
    managing subsampling, and generating named dictionaries for indexing purposes. Takes as input
    a tensor and a DimTuple object and creates a CalipyIndexer object that can be used to produce
    indices, bind dimensions, order the tensor and similar other support functionality.
    
    :param tensor: The tensor for which the indexer is to be constructed
    :type tensor: torch.Tensor
    :param dims: A DimTuple containing the dimensions of the tensor
    :type dims: DimTuple
    :param name: A name for the indexer, useful for keeping track of subservient indexers.
        Default is None.
    :type name: string

    
    :return: An instance of CalipyIndexer containing functionality for indexing the
        input tensor including subbatching, naming, index tensors.
    :rtype: CalipyIndexer

    Example usage:

    .. code-block:: python
    
        # Create DimTuples and tensors
        data_A = torch.normal(0,1,[6,4,2])
        batch_dims_A = dim_assignment(dim_names = ['bd_1_A', 'bd_2_A'])
        event_dims_A = dim_assignment(dim_names = ['ed_1_A'])
        data_dims_A = batch_dims_A + event_dims_A
        

        # Evoke indexer
        data_A.calipy.indexer_construct(data_dims_A, 'data_A')
        indexer = data_A.calipy.indexer
        print(indexer)
        
        # Indexer contains the tensor, its dims, and bound tensor
        indexer.tensor
        indexer.tensor_dims
        indexer.tensor_dims.__class__
        indexer.tensor_dims.sizes
        indexer.tensor_torchdims
        indexer.tensor_torchdims.__class__
        indexer.tensor_torchdims.sizes
        indexer.tensor_named
        indexer.index_dim
        indexer.index_tensor_dims
        
        # Functionality indexer
        attr_list = [attr for attr in dir(indexer) if '__' not in attr]
        print(attr_list)
        
        # Functionality index
        local_index = data_A.calipy.indexer.local_index
        local_index
        local_index.dims
        local_index.tensor.shape
        local_index.index_name_dict
        assert (data_A[local_index.tuple] == data_A).all()
        
        # Subbatching along one or multiple dims
        subsamples, subsample_indices = data_A.calipy.indexer.simple_subsample(batch_dims_A[0], 5)
        print('Shape subsamples = {}'.format([subsample.shape for subsample in subsamples]))
        block_batch_dims_A = batch_dims_A
        block_subsample_sizes_A = [5,3]
        block_subsamples, block_subsample_indices = data_A.calipy.indexer.block_subsample(block_batch_dims_A, block_subsample_sizes_A)
        print('Shape block subsamples = {}'.format([subsample.shape for subsample in block_subsamples]))
        
        # Inheritance - by construction
        # Suppose we got data_C as a subset of data_B with derived ssi CalipyIndex and
        # now want to index data_C with proper names and references
        #   1. generate data_B
        batch_dims_B = dim_assignment(['bd_1_B', 'bd_2_B'])
        event_dims_B = dim_assignment(['ed_1_B'])
        data_dims_B = batch_dims_B + event_dims_B
        data_B = torch.normal(0,1,[7,5,2])
        data_B.calipy.indexer_construct(data_dims_B, 'data_B')
        
        #   2. subsample data_C from data_B
        block_data_C, block_indices_C = data_B.calipy.indexer.block_subsample(batch_dims_B, [5,3])
        block_nr = 3
        data_C = block_data_C[block_nr]
        block_index_C = block_indices_C[block_nr]
        
        #   3. subsampling has created an indexer for data_C
        data_C.calipy.indexer
        data_C.calipy.indexer.local_index
        data_C.calipy.indexer.global_index
        data_C.calipy.indexer.local_index.tensor
        data_C.calipy.indexer.global_index.tensor
        data_C.calipy.indexer.global_index.index_name_dict
        data_C.calipy.indexer.data_source_name
        
        data_C_local_index = data_C.calipy.indexer.local_index
        data_C_global_index = data_C.calipy.indexer.global_index
        assert (data_C[data_C_local_index.tuple] == data_B[data_C_global_index.tuple]).all()
        
        # Inheritance - by declaration
        # If data comes out of some external subsampling and only the corresponding indextensors
        # are known, the calipy_indexer can be evoked manually.
        data_D = copy.copy(data_C)
        data_D.calipy.indexer = None
        index_tensor_D = block_index_C.tensor
        
        data_D.calipy.indexer_construct(data_dims_B, 'data_D')
        data_D.calipy.indexer.create_global_index(index_tensor_D, 'from_data_D')
        data_D_global_index = data_D.calipy.indexer.global_index
        
        assert (data_D == data_B[data_D_global_index.tuple]).all()
        
        # Alternative way of calling via DataTuples
        data_E = torch.normal(0,1,[5,3])
        batch_dims_E = dim_assignment(dim_names = ['bd_1_E'])
        event_dims_E = dim_assignment(dim_names = ['ed_1_E'])
        data_dims_E = batch_dims_E + event_dims_E
        
        data_names_list = ['data_A', 'data_E']
        data_list = [data_A, data_E]
        data_datatuple = DataTuple(data_names_list, data_list)
        
        batch_dims_datatuple = DataTuple(data_names_list, [batch_dims_A, batch_dims_E])
        event_dims_datatuple = DataTuple(data_names_list, [event_dims_A, event_dims_E])
        data_dims_datatuple = batch_dims_datatuple + event_dims_datatuple
        
        data_datatuple.indexer_construct(data_dims_datatuple)
        data_datatuple['data_A'].calipy.indexer
    """
    
    
    def __init__(self, tensor, dims, name = None):
        # Integrate initial data
        self.name = name
        self.tensor = tensor
        self.tensor_dims = dims
        self.tensor_torchdims = dims.build_torchdims()
        self.tensor_named = tensor[self.tensor_torchdims]
        
        # Create index tensors
        self.local_index = self.create_local_index()
        # self.global_index = self.create_global_index()


    def create_local_index(self):
        """
        Create a local index tensor enumerating all possible indices for all the dims.
        The indices local_index_tensor are chosen such that they can be used for 
        indexing the tensor via value = tensor[i,j,k] = tensor[local_index_tensor[i,j,k,:]],
        i.e. the index at [i,j,k] is [i,j,k]. A more compact form of indexing
        is given by directly accessing the index tuples via tensor = tensor[local_index_tensor_tuple]

        :return: Writes torch tensors with indices representing all possible positions within the tensor
            local_index_tensor: index_tensor containing an index at each location of value in tensor
            local_index_tensor_tuple: index_tensor split into tuple for straightforward indexing
            local_index_tensor_named: index_tensor with dimensions named and bound
        """
        # Set up dims
        self.index_dim = dim_assignment(['index_dim'])
        self.index_tensor_dims = self.tensor_dims + self.index_dim
        
        # Iterate through ranges
        index_ranges = [torch.arange(dim_size) for dim_size in self.tensor_torchdims.sizes]
        meshgrid = torch.meshgrid(*index_ranges, indexing='ij')
        index_tensor = torch.stack(meshgrid, dim=-1)
        
        # Write out results
        local_index = CalipyIndex(index_tensor, self.index_tensor_dims, name = self.name)

        return local_index

    def create_global_index(self, subsample_indextensor = None, data_source_name = None):
        """
        Create a global CalipyIndex object enumerating all possible indices for all the dims. The
        indices global_index_tensor are chosen such that they can be used to access the data
        in data_source with name data_source_name via self.tensor  = data_source[global_index_tensor_tuple] 
        
        :param subsample_index: An index tensor that enumerates for all the entries of
            self.tensor which index needs to be used to access it in some global dataset.
        :param data_source_name: A string serving as info to record which object the global indices are indexing.
        
        :return: A CalipyIndex object global index containing indexing data that
            describes how the tensor is related to the superpopulation it has been
            sampled from.
        """
        # Record source
        self.data_source_name = data_source_name
        
        # If no subsample_indextensor given, global = local
        if subsample_indextensor is None:
            global_index = self.local_index

        # Else derive global indices from subsample_indextensor
        else:
            global_index = CalipyIndex(subsample_indextensor, self.index_tensor_dims, name = self.name)
        self.global_index = global_index
        
        return global_index


    def block_subsample(self, batch_dims, subsample_sizes):
        """
        Generate indices for block subbatching across multiple batch dimensions
        and extract the subbatches.

        :param batch_dims: DimTuple with dims along which subbatching happens
        :param subsample_sizes: Tuple with sizes of the blocks to create.
        :return: List of tensors and CalipyIndex representing the block subatches.
        """
        
        # Extract shapes
        batch_tdims = self.tensor_torchdims[batch_dims]
        tensor_reordered = self.tensor_named.order(*batch_tdims)
        self.block_batch_shape = tensor_reordered.shape
        
        # Compute number of blocks in dims 
        self.num_blocks = [(self.block_batch_shape[i] + subsample_sizes[i] - 1) // subsample_sizes[i]
            for i in range(len(subsample_sizes))]
        self.block_identifiers = list(itertools.product(*[range(n) for n in self.num_blocks]))
        random.shuffle(self.block_identifiers)

        block_data = []
        block_indices = []
        for block_idx in self.block_identifiers:
            block_tensor, block_index = self._get_indextensor_from_block(block_idx, batch_tdims, subsample_sizes)
            block_data.append(block_tensor)
            block_indices.append(block_index)
        self.block_indices = block_indices
        self.block_data = block_data
        return block_data, block_indices
    
    def _get_indextensor_from_block(self, block_index,  batch_tdims, subsample_sizes):
        # Manage dimensions
        # batch_dims = batch_dims.get_local_copy()
        nonbatch_tdims = self.tensor_torchdims.delete_dims(batch_tdims.names)
        block_index_dims = self.tensor_dims + self.index_dim
        
        # Block index is n-dimensional with n = number of dims in batch_dims
        indices_ranges_batch = {}
        for i, (b, s, d) in enumerate(zip(block_index, subsample_sizes, batch_tdims)):
            start = b * s
            end = min(start + s, self.block_batch_shape[i])
            indices_ranges_batch[d] = torch.arange(start, end)
        
        # Include event dimensions in the slices
        nonbatch_shape = self.tensor_named.order(nonbatch_tdims).shape
        indices_ranges_nonbatch = {d: torch.arange(d.size) for d in nonbatch_tdims}
        
        indices_ordered = {}
        indices_ordered_list = []
        for i, (d, dn) in enumerate(zip(self.tensor_torchdims, self.tensor_torchdims.names)):
            if dn in batch_tdims.names:
                indices_ordered[d] = indices_ranges_batch[d]
            elif dn in nonbatch_tdims.names:
                indices_ordered[d] = indices_ranges_nonbatch[d]
            indices_ordered_list.append(indices_ordered[d])
        
        # Compile to indices and tensors
        meshgrid = torch.meshgrid(*indices_ordered_list, indexing='ij')
        block_index_tensor = torch.stack(meshgrid, dim=-1)
        block_index = CalipyIndex(block_index_tensor, block_index_dims, name = 'from_' + self.name)
        block_tensor = self.tensor[block_index.tuple]
        
        # Clarify global index
        block_tensor.calipy.indexer_construct(self.tensor_dims, self.name)
        block_tensor.calipy.indexer.create_global_index(block_index.tensor, 'from_' + self.name)
        
        return block_tensor, block_index   
    
    def simple_subsample(self, batch_dim, subsample_size):
        """
        Generate indices for subbatching across a single batch dimension and 
        extract the subbatches.

        :param batch_dim: Element of DimTuple (typically CalipyDim) along which
            subbatching happens.
        :param subsample_size: Single size determining length of batches to create.
        :return: List of tensors and CalipyIndex representing the subbatches.
        """
        
        subbatch_data, subbatch_indices = self.block_subsample(DimTuple((batch_dim,)), [subsample_size])
        
        return subbatch_data, subbatch_indices
    
    # def indexfun(tuple_of_indices, vectorizable = True):
    #     """ Function to create a multiindex that can handle an arbitrary number of indices 
    #     (both integers and vectors), while preserving the shape of the original tensor.
    #     # Example usage 1 
    #     A = torch.randn([4,5])  # Tensor of shape [4, 5]
    #     i = torch.tensor([1])
    #     j = torch.tensor([3, 4])
    #     # Call indexfun to generate indices
    #     indices_A, symbol = indexfun((i, j))
    #     # Index the tensor
    #     result_A = A[indices_A]     # has shape [1,2]
    #     # Example usage 2
    #     B = torch.randn([4,5,6,7])  # Tensor of shape [4, 5, 6, 7]
    #     k = torch.tensor([1])
    #     l = torch.tensor([3, 4])
    #     m = torch.tensor([3])
    #     n = torch.tensor([0,1,2,3])
    #     # Call indexfun to generate indices
    #     indices_B, symbol = indexfun((k,l,m,n))
    #     # Index the tensor
    #     result_B = B[indices_B]     # has shape [1,2,1,4]
    #     """
    #     # Calculate the shape needed for each index to broadcast correctly
    #     if vectorizable == True:        
    #         idx = tuple_of_indices
    #         shape = [1] * len(idx)  # Start with all singleton dimensions
    #         broadcasted_indices = []
        
    #         for i, x in enumerate(idx):
    #             target_shape = list(shape)
    #             target_shape[i] = len(x)
    #             # Reshape the index to target shape
    #             x_reshaped = x.view(target_shape)
    #             # Expand to match the full broadcast size
    #             x_broadcast = x_reshaped.expand(*[len(idx[j]) if j != i else x_reshaped.shape[i] for j in range(len(idx))])
    #             broadcasted_indices.append(x_broadcast)
    #             indexsymbol = 'vectorized'
    #     else:
    #         broadcasted_indices = tuple_of_indices
    #         indexsymbol = broadcasted_indices
        
    #     # Convert list of broadcasted indices into a tuple for direct indexing
    #     return tuple(broadcasted_indices), indexsymbol  
    
        
    
    def __repr__(self):
        dim_name_list = self.tensor_torchdims.names
        dim_sizes_list = self.tensor_torchdims.sizes
        repr_string = 'CalipyIndexer for tensor with dims {} of sizes {}'.format(dim_name_list, dim_sizes_list)
        return repr_string
        
    
# # Monkey Patching
        

# Set up new propety of tensors with centralized internal name
INTERNAL_NAME_CALIPY = '_calipy_namespace'
class CalipyNamespace:
    def __init__(self, tensor):
        self.tensor = tensor
    
    # Define the function that will be monkey-patched onto torch.Tensor
    def indexer_construct(self, tensor_dims, name, silent = True):
        """Constructs a CalipyIndexer for the tensor."""
        self.indexer = CalipyIndexer(self.tensor, tensor_dims, name)
        return self if silent == False else None
        
    def __repr__(self):
        repr_string = 'CalipyNamespace for methods and attributes in torch.Tensor'
        return repr_string
        
def get_calipy_namespace(self):
    if not hasattr(self, INTERNAL_NAME_CALIPY):
        setattr(self, INTERNAL_NAME_CALIPY, CalipyNamespace(self))
    return getattr(self, INTERNAL_NAME_CALIPY)

torch.Tensor.calipy = property(get_calipy_namespace)

# # Conflict mitigation approaches
# if hasattr(torch.Tensor, 'calipy'):
#     print("Warning: 'calipy' attribute already exists on torch.Tensor. Potential conflict.")
# else:
#     torch.Tensor.calipy = property(get_calipy_namespace)
    
# if hasattr(torch.Tensor, 'calipy'):
#     # Handle the conflict: choose a different name or raise an error
#     # For example, use an alternative name
#     setattr(torch.Tensor, 'calipy_custom', property(get_calipy_namespace))
#     calipy_attribute_name = 'calipy_custom'
# else:
#     # Safe to use 'calipy'
#     setattr(torch.Tensor, 'calipy', property(get_calipy_namespace))
#     calipy_attribute_name = 'calipy'
    
# Attach the method to torch.Tensor
# # Define the function that will be monkey-patched onto torch.Tensor
# def indexer_construct(self, tensor_dims, name, silent = False):
#     """Constructs a CalipyIndexer for the tensor."""
#     self.calipy_indexer = CalipyIndexer(self.tensor, tensor_dims, name)
#     return self if silent == False else None


# def indexer_construct_silent(self, tensor_dims, name):
#     """Constructs a CalipyIndexer for the tensor."""
#     self.indexer_construct(tensor_dims, name, silent = True)
#     return
# torch.Tensor.calipy_indexer_construct = indexer_construct_silent

# # End Monkey Patching


# Check the functionality of this class with monkeypatch
data_A.calipy.indexer_construct(data_dims_A, 'data_A')
local_index = data_A.calipy.indexer.local_index
local_index.tensor.shape
local_index.dims
assert (data_A[local_index.tuple] == data_A).all()

simple_subsamples, simple_subsample_indices = data_A.calipy.indexer.simple_subsample(batch_dims_A[0], 5)
block_batch_dims_A = batch_dims_A
block_subsample_sizes_A = [5,3]
block_subsamples, block_subsample_indices = data_A.calipy.indexer.block_subsample(block_batch_dims_A, block_subsample_sizes_A)

# Check subsample indexing 
# Suppose we got data_D as a subset of data_C with derived ssi CalipyIndex and
# now want to index data_D with proper names and references
# First, generate data_C
batch_dims_C = dim_assignment(['bd_1_C', 'bd_2_C'])
event_dims_C = dim_assignment(['ed_1_C'])
data_dims_C = batch_dims_C + event_dims_C
data_C = torch.normal(0,1,[7,5,2])
data_C.calipy.indexer_construct(data_dims_C, 'data_C')

# Then, subsample data_D from data_C
block_data_D, block_indices_D = data_C.calipy.indexer.block_subsample(batch_dims_C, [5,3])
block_nr = 3
data_D = block_data_D[block_nr]
block_index_D = block_indices_D[block_nr]

# Now look at global indices and names; the indexer has been inherited during subsampling
data_D.calipy.indexer
data_D.calipy.indexer.local_index
data_D.calipy.indexer.global_index
data_D.calipy.indexer.local_index.tensor
data_D.calipy.indexer.global_index.tensor
data_D.calipy.indexer.data_source_name

# If data comes out of some external subsampling and only the corresponding indextensors
# are known, the calipy_indexer can be evoked manually.
data_E = copy.copy(data_D)
data_E.calipy.indexer = None
index_tensor_E = block_index_D.tensor

data_E.calipy.indexer_construct(data_dims_C, 'data_E')
data_E.calipy.indexer.create_global_index(index_tensor_E, 'from_data_E')
data_E_index_tuple = data_E.calipy.indexer.global_index.tuple

assert (data_E == data_C[data_E_index_tuple]).all()


# Check interaction with DataTuple class
data_names_list = ['data_A', 'data_B']
data_list = [data_A, data_B]
data_datatuple = DataTuple(data_names_list, data_list)

batch_dims_datatuple = DataTuple(data_names_list, [batch_dims_A, batch_dims_B])
event_dims_datatuple = DataTuple(data_names_list, [event_dims_A, event_dims_B])
data_dims_datatuple = batch_dims_datatuple + event_dims_datatuple

# Build the calipy_indexer; it does not exist before this line for e.g data_datatuple['data_B']
data_datatuple.indexer_construct(data_dims_datatuple)



# ii) Build dataloader
class CalipySample:
    """
    Stores samples along with batch and event dimensions in a new object that 
    provides indexing and naming functionality.
    
    :param samples_datatuple: A DataTuple containing tensors considered as samples
    :type obs: DataTuple
    :param batch_dims_datatuple: A DataTuple containing the batch_dims DimTuple for each
        tensor in the sample_datatuple
    :type batch_dims_datatuple: DataTuple
    :param event_dims_datatuple: A DataTuple containing the event_dims DimTuple for each
        tensor in the sample_datatuple
    :type event_dims_datatuple: DataTuple
    :param subsample_indices: ?
    :type subsample_indices: ?
    :param vectorizable: Flag indicating if observations are vectorizable
    :type vectorizable: bool
    :return: CalipySample object 
    :rtype: CalipySample
    
    
    """
    def __init__(self, samples_datatuple, batch_dims_datatuple, event_dims_datatuple,
                 subsample_indices=None, vectorizable=True):

        # Metadata: keep it directly tied with samples
        self.entry_names = [key for key in samples_datatuple.keys()]
        self.batch_dims = batch_dims_datatuple
        self.event_dims = event_dims_datatuple
        self.sample_dims = self.batch_dims + self.event_dims
        # self.index_dims = DataTuple(self.entry_names, [dim_assignment(['id_{}'.format(key)],
        #                             [len(self.sample_dims[key])]) for key in self.entry_names])
        # self.ssi_dims = self.sample_dims + self.index_dims
        self.vectorizable = vectorizable
        
        # Handle tensor tuples for samples and ssi
        self.samples = samples_datatuple
        # self.samples_bound = self.samples.bind_dims(self.obs_dims)
        self.subsample_indices = subsample_indices
        # self.subsample_indices_bound = subsample_indices.bind_dims(self.ssi_dims) if subsample_indices is not None else None

        # Initialize local and global indices for easy reference
        
        self.samples.indexer_construct(self.sample_dims)
        index_names, indexers = zip(*[(name, tensor.calipy.indexer) for name, tensor in self.samples.items()])
        self.indexers = DataTuple(list(index_names), list(indexers))


    def get_entry(self, **batch_dims_spec):
        # Retrieve an observation by specifying batch dimensions explicitly
        indices = [batch_dims_spec[dim] for dim in self.batch_dims]
        obs_values = {key: tensor[tuple(indices)] if len(tensor) > 0 else None for key, tensor in self.observations.items()}
        obs_name = {key: self.index_to_name_dict[(key, tuple(indices))] for key in obs_values.keys()}
        return obs_name, obs_values

    def get_local_index(self, key, idx):
        return self.obs_local_indices[key][idx]

    def get_global_index(self, key, idx):
        return self.obs_global_indices[key][idx]

    def __repr__(self):
        repr_str = 'CalipySample object with samples: {} \nand sample dims : {}'\
            .format(self.samples.__repr__(), self.sample_dims.__repr__())
        return repr_str

# Instantiate CalipyObservation
obs_name_list = ['T_grid', 'T_rod']
observations = DataTuple(obs_name_list, [data_A, data_B])
batch_dims = DataTuple(obs_name_list, [batch_dims_A, batch_dims_B])
event_dims = DataTuple(obs_name_list, [event_dims_A, event_dims_B])
obs_dims = batch_dims + event_dims

calipy_obs = CalipySample(observations, batch_dims, event_dims)
print(calipy_obs)



class SubbatchDataset(Dataset):
    def __init__(self, data, subsample_sizes, batch_shape=None, event_shape=None):
        self.data = data
        self.batch_shape = batch_shape
        self.subsample_sizes = subsample_sizes

        # Determine batch_shape and event_shape
        if batch_shape is None:
            batch_dims = len(subsample_sizes)
            self.batch_shape = data.shape[:batch_dims]
        else:
            self.batch_shape = batch_shape

        if event_shape is None:
            self.event_shape = data.shape[len(self.batch_shape):]
        else:
            self.event_shape = event_shape

        # Compute number of blocks using ceiling division to include all data
        self.num_blocks = [
            (self.batch_shape[i] + subsample_sizes[i] - 1) // subsample_sizes[i]
            for i in range(len(subsample_sizes))
        ]
        self.block_indices = list(itertools.product(*[range(n) for n in self.num_blocks]))
        random.shuffle(self.block_indices)

    def __len__(self):
        return len(self.block_indices)

    def __getitem__(self, idx):
        block_idx = self.block_indices[idx]
        slices = []
        indices_ranges = []
        for i, (b, s) in enumerate(zip(block_idx, self.subsample_sizes)):
            start = b * s
            end = min(start + s, self.batch_shape[i])
            slices.append(slice(start, end))
            indices_ranges.append(torch.arange(start, end))
        # Include event dimensions in the slices
        slices.extend([slice(None)] * len(self.event_shape))
        meshgrid = torch.meshgrid(*indices_ranges, indexing='ij')
        indices = torch.stack(meshgrid, dim=-1).reshape(-1, len(self.subsample_sizes))
        obs_block = CalipyObservation(self.data[tuple(slices)], self.batch_shape, self.event_shape, subsample_indices = indices)
        return obs_block, indices

# Usage example
subsample_sizes = [4, 2]
subbatch_dataset = SubbatchDataset(data, subsample_sizes, batch_shape=batch_shape, event_shape=[2])
subbatch_dataloader = DataLoader(subbatch_dataset, batch_size=None)


index_list = []
obs_block_list = []
mean_list = []
for obs_block, index_tensor in subbatch_dataloader:
    print("Obs block shape:", obs_block.observations.shape)
    print("Index tensor shape:", index_tensor.shape)
    print("obs block:", obs_block)
    print("Indices:", index_tensor)
    index_list.append(index_tensor)
    obs_block_list.append(obs_block)
    mean_list.append(torch.mean(obs_block.observations,0))
    print("----")
    
    
# iii) Support classes
    
    
def calipy_sample(name, dist, plate_names, plate_sizes, vectorizable=True, obs=None, subsample_indices=None):
    """
    Flexible sampling function handling multiple plates and four cases based on obs and subsample_indices.

    Parameters:
    -----------
    name : str
        Base name for the sample site.
    dist : pyro.distributions.Distribution
        The distribution to sample from.
    plate_names : list of str
        Names of the plates (batch dimensions).
    plate_sizes : list of int
        Sizes of the plates (batch dimensions).
    vectorizable : bool, optional
        If True, uses vectorized sampling. If False, uses sequential sampling. Default is True.
    obs : CalipyObservation or None, optional
        Observations wrapped in CalipyObservation. If provided, sampling is conditioned on these observations.
    subsample_indices : list of torch.Tensor or None, optional
        Subsample indices for each plate dimension. If provided, sampling is performed over these indices.

    Returns:
    --------
    CalipySample
        The sampled data, preserving batch and event dimensions.
    """
    
    event_shape = dist.event_shape
    batch_shape = dist.batch_shape
    n_plates = len(plate_sizes)
    n_default = 3
    batch_shape_default = [n_default]*len(batch_shape)
    ssi = subsample_indices

    # cases [1,x,x] vectorizable
    if vectorizable == True:
        # Vectorized sampling using pyro.plate
        with contextlib.ExitStack() as stack:
            # Determine dimensions for plates
            batch_dims_from_event = [-(n_plates + len(event_shape)) + i + 1 for i in range(n_plates)]
            batch_dims_from_right = [-(n_plates ) + i + 1 for i in range(n_plates)]
            current_obs = obs.observations if obs is not None else None
            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                pass
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                pass
            
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                pass
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                pass
            
            # Handle multiple plates
            for i, (plate_name, plate_size, dim) in enumerate(zip(plate_names, plate_sizes, batch_dims_from_event)):
                subsample = subsample_indices if subsample_indices is not None else None
                size = plate_size
                stack.enter_context(pyro.plate(plate_name, size=size, subsample=subsample, dim=dim))

                # # Not necessary anymore since obs are assumed to be subsampled already
                # # Index observations if subsampling
                # if current_obs is not None and subsample is not None:
                #     current_obs = torch.index_select(current_obs, dim=batch_dims_from_right[i], index=subsample.flatten())

            # Sample data
            data = pyro.sample(name, dist, obs=current_obs)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=True)
        
    # cases [0,x,x] nonvectorizable
    elif vectorizable == False:
            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                # Create new observations of shape batch_shape_default with ssi
                # a flattened list of product(range(n_default))
                batch_shape_obs = batch_shape_default
                ssi_lists = [list(range(bs)) for bs in batch_shape_obs]
                ssi_list = [torch.tensor(idx) for idx in itertools.product(*ssi_lists)]
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = len(ssi_list)*[None]
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                # Create len(ssi) new observations with given ssi's
                ssi_list = ssi
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = len(ssi_list)*[None]
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                #  Create obs with standard ssi derived from obs batch_shape
                batch_shape_obs = obs.shape[:len(batch_shape_default)]
                ssi_lists = [list(range(bs)) for bs in batch_shape_obs]
                ssi_list = [torch.tensor(idx) for idx in itertools.product(*ssi_lists)]
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = [obs[ssi,...] for ssi in ssi_list]
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                # Create obs associated to given ssi's
                batch_shape_obs = obs.shape[:len(batch_shape_default)]
                ssi_list = ssi
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = [obs[ssi,...] for ssi in ssi_list]
            
            # Iterate over the lists and sample
            n_samples = len(obs_name_list)
            samples = []
            plate_list = [pyro.plate(plate_names[plate], plate_sizes[plate], subsample = ssi_tensor[:,plate]) for plate in range(len(plate_names))]
            
            
    
            # Iterate over all combinations of indices
            samples = []
            for idx in itertools.product(*batch_index_lists):
                idx_dict = {plate_names[i]: idx[i] for i in range(n_plates)}
                # Use actual data indices for naming
                idx_str = '_'.join(f"{plate_names[i]}_{idx[i]}" for i in range(n_plates))
    
                # Get observation if available
                if obs is not None:
                    obs_name, obs_value = obs.get_observation(idx)
                    sample_name = f"{obs_name}"
                else:
                    obs_name, obs_value = None, None
                    sample_name = f"{name}_{idx_str}"
    
                # Sample with unique name based on actual data indices
                sample_value = pyro.sample(sample_name, dist, obs=obs_value)
                samples.append((idx, sample_value))
    
            # Construct tensor from samples
            # Need to sort samples based on indices to ensure correct tensor shape
            samples.sort(key=lambda x: x[0])
            sample_values = [s[1] for s in samples]
            batch_shapes = [len(r) for r in batch_index_lists]
            data = torch.stack(sample_values).reshape(*batch_shapes, *event_shape)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=False)
    
    
# iv) Test the classes

obs_object = CalipyObservation(data, plate_names, batch_shape=[n_data_1], event_shape=[2])

# Subsample indices (if any)
subsample_indices = index_list[0]

sample_results = []
# Sample using calipy_sample and check results
for vectorizable in [True, False]:
    for obs in [None, obs_block_list[0]]:
        for ssi in [None, subsample_indices]:
            
            sample_dist = pyro.distributions.MultivariateNormal(loc = mu_true, covariance_matrix = sigma_true)
            sample_result = calipy_sample('my_sample', sample_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs=obs, subsample_indices=ssi)
            
            vflag = 1 if vectorizable == True else 0
            oflag = 1 if obs is not None else 0
            sflag = 1 if ssi is not None else 0
            
            print('vect_{}_obs_{}_ssi_{}_batch_shape'.format(vflag, oflag, sflag), sample_result.batch_shape )
            print('vect_{}_obs_{}_ssi_{}_event_shape'.format(vflag, oflag, sflag), sample_result.event_shape )
            print('vect_{}_obs_{}_ssi_{}_data_shape'.format(vflag, oflag, sflag), sample_result.data.shape )
            # print('vect_{}_obs_{}_ssi_{}_data'.format(vflag, oflag, sflag), sample_result.data )
            sample_results.append(sample_result)
            
            

def model(observations = None, vectorizable = True, subsample_indices = None):
    # observations is CalipyObservation object
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0,5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = 5*torch.eye(2), constraint = dist.constraints.positive_definite)
    
    plate_names = ['batch_dim_1']
    plate_sizes = [n_data_1]
    obs_dist = dist.MultivariateNormal(loc = mu, covariance_matrix = sigma)
    obs = calipy_sample('obs', obs_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs = observations, subsample_indices = subsample_indices)
    
    return obs


# Example usage:
# Run the model with vectorizable=True
output_vectorized = model(vectorizable=True)
print("Vectorized Output Shape:", output_vectorized.data.shape)
print("Vectorized Output:", output_vectorized)

# Run the model with vectorizable=False
output_non_vectorized = model(vectorizable=False)
print("Non-Vectorized Output Shape:", output_non_vectorized.data.shape)
print("Non-Vectorized Output:", output_non_vectorized)

# Plot the shapes
calipy_observation = CalipyObservation(data, plate_names, batch_shape = batch_shape, event_shape = [2])
model_trace = pyro.poutine.trace(model)
spec_trace = model_trace.get_trace(calipy_observation, vectorizable = True, subsample_indices = None)
spec_trace.nodes
print(spec_trace.format_shapes())
            
            
# v) Perform inference in different configurations

# Set up guide
def guide(observations = None, vectorizable= True, subsample_indices = None):
    pass


adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# Set up svi
signature = [1,1,1] # vect, obs, ssi = None or values
# signature = [1,1,0]
# signature = [1,0,1]
# signature = [1,0,0]
# signature = [0,1,1] 
# signature = [0,1,0]
# signature = [0,0,1]
# signature = [0,0,0]


# Handle DataLoader case
loss_sequence = []
for epoch in range(1000):
    epoch_loss = 0
    for batch_data, subsample_indices in subbatch_dataloader:
        
        # Set up kwargs for svi step
        obs_svi_dict = {}
        obs_svi_dict['vectorizable'] = True if signature[0] ==1 else False
        obs_svi_dict['observations'] = batch_data if signature[1] == 1 else None
        obs_svi_dict['subsample_indices'] = subsample_indices if signature[2] == 1 else None
        
        # loss = svi.step(batch_data, True, subsample_indices)
        loss = svi.step(**obs_svi_dict)
        # loss = svi.step(observations = obs_svi_dict['observations'],
        #                 vectorizable = obs_svi_dict['observations'],
        #                 subsample_indices = obs_svi_dict['subsample_indices'])
        # loss = svi.step(obs_svi_dict['observations'],
        #                 obs_svi_dict['observations'],
        #                 obs_svi_dict['subsample_indices'])
        epoch_loss += loss
    
    epoch_loss /= len(subbatch_dataloader)
    loss_sequence.append(epoch_loss)

    if epoch % 100 == 0:
        print(f'epoch: {epoch} ; loss : {epoch_loss}')


"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {}, Inferred sigma = {} \n mean = {} \n mean of batch means = {}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'],
              pyro.get_param_store()['sigma'],
              torch.mean(data,0),
              torch.mean(torch.vstack(mean_list),0)))
































   