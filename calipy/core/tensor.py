#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides basic dimension aware tensor functionality needed for 
subsampling, indexing, attaching dimensions to tensors and other quality of
life features that allow tensors to keep and communicate extra structure.

The classes and functions are
    CalipyIndex: Class of objects that can be used to index normal torch.tensors
        and CalipyTensors or CalipyDistributions. Contains indextensors, tuples,
        dims, names of the elements in the tensor and functionality for reducing
        and expanding indices.
    
    CalipyIndexer: Abstract class of objects that forms the basis for TensorIndexer
        and DistributionIndexer and bundles indexing methods and attributes used
        by both of these classes.
        
    TensorIndexer: Class responsible for indexing tensors. Is attached directly
        to CalipyTensors and takes over active duties like subsampling, keeping
        track of the origin of subsampled data, and creating CalipyIndex objects
        based on user demands.
  
The TensorIndexer and CalipyIndex classes provide basic functionality that is
used regularly in the context of defining basic effects and enabling primitives
that need to implement subsampling (like sampling or calling parameters.)
        

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


import torch
import pandas as pd
import warnings
import itertools
import random
import einops


from calipy.core.utils import DimTuple, CalipyDim, TorchdimTuple, dim_assignment, multi_unsqueeze, ensure_tuple, robust_meshgrid


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
        
    @property
    def is_empty(self):
        """ Indicates if no indexing is actually performed because e.g. indexed
        quantity is a scalar and does not have dims"""
        
        return len(self.tensor) == 0
        
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
    
    def is_reducible(self, dims_to_keep):
        """
        Determine if the index is reducible to the specified dimensions without loss.
    
        :param dims_to_keep: DimTuple of CalipyDims to keep.
        :return: True if reducible without loss, False otherwise.
        """
    
        # Extract indices for the dimensions to keep
        dim_positions_keep = self.dims.find_indices(dims_to_keep.names, from_right = False)
        dims_to_remove = self.dims.delete_dims(dims_to_keep.names + ['index_dim'])
        dim_positions_remove = self.dims.find_indices(dims_to_remove.names, from_right = False)
    
        # Flatten the index tensor
        num_elements = self.tensor.shape[:-1].numel()
        index_tensor_flat = self.tensor.view(num_elements, -1)  # Shape: [N, num_indices]
    
        # Extract indices for kept and removed dimensions
        indices_keep = index_tensor_flat[:, dim_positions_keep]  # Shape: [N, num_dims_keep]
        indices_remove = index_tensor_flat[:, dim_positions_remove]  # Shape: [N, num_dims_remove]
        
        # Convert indices to NumPy arrays
        indices_keep_np = indices_keep.numpy()
        indices_remove_np = indices_remove.numpy()
    
        # Convert indices to tuples for grouping
        keep_tuples = [tuple(row) for row in indices_keep_np]
        remove_tuples = [tuple(row) for row in indices_remove_np]
    
        # Create a DataFrame with tupled indices
        df = pd.DataFrame({'keep': keep_tuples, 'remove': remove_tuples})
        grouped = df.groupby('keep')['remove'].apply(set)
    
        # Convert sets to identify unique ones
        frozenset_remove_sets = grouped.apply(frozenset)
        unique_remove_sets = set(frozenset_remove_sets)
        # Check if all 'remove' sets are identical
        is_reducible = len(unique_remove_sets) == 1
    
        return is_reducible
    
    
    def reduce_to_dims(self, dims_to_keep):
        """
        Reduce the current index to cover some subset of dimensions.

        :param dims_to_keep: A DimTuple containing the target dimensions.
        :return: A new CalipyIndex instance with the reduced index tensor.
        
        """
        
        # i) Check reducibility
        
        if not self.is_reducible(dims_to_keep):
            warnings.warn("Index tensor cannot be reduced without loss of information.")
        
            
        # ii) Set up dimensions
        
        # Extract indices for the dimensions to keep
        index_dim = DimTuple(self.dims[-1:]).bind([len(dims_to_keep)])
        dim_positions_keep = self.dims.find_indices(dims_to_keep.names, from_right = False)
        dims_to_remove = self.dims.delete_dims(dims_to_keep.names + ['index_dim'])
        dim_positions_remove = self.dims.find_indices(dims_to_remove.names, from_right = False)


        # iii) Extract indices

        # Extract indices for dimensions to keep
        index_slices = [slice(None)] * len(self.dims[0:-1])  # Initialize slices for all dimensions
        for pos in dim_positions_remove:
            index_slices[pos] = 0  # Select index 0 along dimensions to remove
        reduced_tensor = self.tensor[tuple(index_slices + [slice(None)])]
        reduced_index_tensor = reduced_tensor[..., dim_positions_keep]

        # Create new CalipyIndex
        reduced_tensor_dims = dims_to_keep + index_dim
        reduced_index = CalipyIndex(reduced_index_tensor, reduced_tensor_dims, name=self.name + '_reduced')

        return reduced_index
    
    def expand_to_dims(self, dims, dim_sizes):
        """
        Expand the current index to include additional dimensions.

        :param dims: A DimTuple containing the target dimensions.
        :param dim_sizes: A list containing sizes for the target dimensions.
        :return: A new CalipyIndex instance with the expanded index tensor.
        
        """
        
        # Set up current and expanded dimensions
        index_dim = DimTuple(self.dims[-1:]).bind([len(dims)])
        current_tensor_dims = DimTuple(self.dims[0:-1]).bind(self.tensor.shape[0:-1])
        expanded_tensor_dims = dims.bind(dim_sizes)
        
        # current_indextensor_dims = self.dims
        expanded_indextensor_dims = expanded_tensor_dims + index_dim
        new_dims = expanded_tensor_dims.delete_dims(current_tensor_dims.names)
        default_order_dims = current_tensor_dims + new_dims + index_dim
        
        # Build index tensor with default order [current_dims, new_dims, index_dim]
        # i) Set up torchdims
        current_tdims = current_tensor_dims.build_torchdims(fix_size = True)
        new_tdims = new_dims.build_torchdims(fix_size = True)
        index_tdim = index_dim.build_torchdims(fix_size = True)
        default_order_tdim = current_tdims + new_tdims + index_tdim
        expanded_order_tdims = default_order_tdim[expanded_indextensor_dims]
        
        
        # ii) Build indextensor new_dims
        new_ranges = []
        for d in new_dims:
            new_ranges.append(torch.arange(d.size)) 
        new_meshgrid = robust_meshgrid(new_ranges, indexing='ij')
        new_dims_indextensor = torch.stack(new_meshgrid, dim=-1)
        current_dims_indextensor = self.tensor
        
        # iii) Combine current_dims_indextensor and new_dims_indextensor
        broadcast_sizes = default_order_tdim.sizes[:-1] + [1]
        broadcast_tensor = torch.ones(broadcast_sizes).long()
        current_expanded = multi_unsqueeze(current_dims_indextensor, [-2]*len(new_dims))
        new_expanded = multi_unsqueeze(new_dims_indextensor, [0]*len(current_tensor_dims))
        default_order_indextensor = torch.cat((current_expanded*broadcast_tensor, new_expanded*broadcast_tensor), dim = -1)
        
        # Order index_tensor to [expanded_tensor_dims, index_dim]
        default_order_indextensor_named = default_order_indextensor[default_order_tdim]  
        expanded_indextensor = default_order_indextensor_named.order(*expanded_order_tdims)
        # Also reorder in the [..., index_dim] so that indices and entries in index_dim align
        index_signature = default_order_dims.find_indices(expanded_indextensor_dims.names, from_right = False)
        expanded_indextensor = expanded_indextensor[..., index_signature[0:-1]]
 
        # Create new CalipyIndex
        expanded_index = CalipyIndex(expanded_indextensor, expanded_indextensor_dims, name = self.name + '_expanded')

        return expanded_index
        

    def __repr__(self):
        sizes = [size for size in self.tensor.shape]
        repr_string = 'CalipyIndex for tensor with dims {} and sizes {}'.format(self.dims.names, sizes)
        return repr_string



class CalipyIndexer:
    """
    Base class of an Indexer that implements methods for assigning dimensions
    to specific slices of e.g. tensors or distributions. The methods and attributes
    are rarely called directly; user interaction happens mostly with the subclasses
    TensorIndexer and DistIndexer. Within these, functionality for subsampling,
    batching, name generation etc are conretized. See those classes for examples
    and specific implementation details.
    
    """
    def __init__(self, dims, name=None):
        self.name = name
        self.dims = dims
        self.index_to_dim_dict = self._create_index_to_dim_dict(dims)
        self.dim_to_index_dict = self._create_dim_to_index_dict(dims)

    def _create_index_to_dim_dict(self, dim_tuple):
        """
        Creates a dict that contains as key: value pairs the integer indices
        (keys) and corresponding CalipyDim dimensions (values) of the self.tensor.
        """
        index_to_dim_dict = dict()
        for i, d in enumerate(dim_tuple):
            index_to_dim_dict[i] = d
        return index_to_dim_dict
    
    def _create_dim_to_index_dict(self, dim_tuple):
        """
        Creates a dict that contains as key: value pairs the CalipyDim dimensions
        (keys) and corresponding integer indices (values) of the self.tensor.
        """
        dim_to_index_dict = dict()
        for i, d in enumerate(dim_tuple):
            dim_to_index_dict[d] = i
        return dim_to_index_dict
    
    def _create_local_index(self, dim_sizes):
        """
        Create a local index tensor enumerating all possible indices for all the dims.
        The indices local_index_tensor are chosen such that they can be used for 
        indexing the tensor via value = tensor[i,j,k] = tensor[local_index_tensor[i,j,k,:]],
        i.e. the index at [i,j,k] is [i,j,k]. A more compact form of indexing
        is given by directly accessing the index tuples via tensor = tensor[local_index_tensor_tuple]

        :return: Writes torch tensors with indices representing all possible positions into the index
            local_index.tensor: index_tensor containing an index at each location of value in tensor
            local_index.tuple: index_tensor split into tuple for straightforward indexing
        """
        
        # Set up dims
        self.index_dim = dim_assignment(['index_dim'])
        self.index_tensor_dims = self.dims + self.index_dim
        
        # Iterate through ranges
        index_ranges = [torch.arange(dim_size) for dim_size in dim_sizes]
        meshgrid = robust_meshgrid(index_ranges, indexing='ij')
        index_tensor = torch.stack(meshgrid, dim=-1)
        
        # Write out results
        local_index = CalipyIndex(index_tensor, self.index_tensor_dims, name = self.name)

        return local_index
    
    def _create_global_index(self, subsample_indextensor = None, data_source_name = None):
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
    
    @classmethod
    def convert_slice_to_indextensor(cls, indexslice_list):
        """
        Converts an indexslice_list to an indextensor so that their actions for indexing
        are equivalent in the sense that sliced tensor has entries from tensor indexed
        by indextensor values: tensor[indexslice_list]  = tensor[indextensor.unbind(-1)].
        
        :param indexslice: An index slice that can be used to index tensors
        :return: An indextensor containing an index at each location of value
            in tensor
        """
        # Iterate through ranges
        index_ranges = [torch.arange(indexslice.stop)[indexslice] for indexslice in indexslice_list]
        meshgrid = robust_meshgrid(index_ranges, indexing='ij')
        indextensor = torch.stack(meshgrid, dim=-1)
        return indextensor
    
    @classmethod
    def convert_tuple_to_indextensor(cls, indextuple):
        """
        Converts an indexstuple to an indextensor so that their actions for indexing
        are equivalent in the sense that indexed tensor has entries from tensor indexed
        by indextensor values: tensor[indextuple]  = tensor[indextensor.unbind(-1)].
        
        :param indextuple: An index tuple that can be used to index tensors
        :return: An indextensor containing an index at each location of value
            in tensor
        """
        indextensor = torch.stack(indextuple.bind,-1)
        return indextensor
    
    @classmethod
    def convert_indextensor_to_tuple(cls, indextensor):
        """
        Converts an indextensor to an indextuple so that their actions for indexing
        are equivalent in the sense that indexed tensor has entries from tensor indexed
        by indextensor values: tensor[indextuple]  = tensor[indextensor.unbind(-1)].
        
        :param indextensor: An indextensor containing an index at each location of value
            in tensor
        :return: An index tuple that can be used to index tensors
        """
        indextuple = indextensor.unbind(-1)
        return indextuple


    def __repr__(self):
        dim_name_list = self.dims.names
        dim_sizes_list = self.dims.sizes
        repr_string = 'CalipyIndexer for tensor with dims {} of sizes {}'.format(dim_name_list, dim_sizes_list)
        return repr_string


class TensorIndexer(CalipyIndexer):
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

    :return: An instance of TensorIndexer containing functionality for indexing the
        input tensor including subbatching, naming, index tensors.
    :rtype: TensorIndexer

    Example usage:

    .. code-block:: python
    
        # Create DimTuples and tensors
        data_A_torch = torch.normal(0,1,[6,4,2])
        batch_dims_A = dim_assignment(dim_names = ['bd_1_A', 'bd_2_A'])
        event_dims_A = dim_assignment(dim_names = ['ed_1_A'])
        data_dims_A = batch_dims_A + event_dims_A
        

        # Evoke indexer
        data_A = CalipyTensor(data_A_torch, data_dims_A, 'data_A')
        indexer = data_A.indexer
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
        local_index = data_A.indexer.local_index
        local_index
        local_index.dims
        local_index.tensor.shape
        local_index.index_name_dict
        assert (data_A.tensor[local_index.tuple] == data_A.tensor).all()
        assert ((data_A[local_index] - data_A).tensor == 0).all()

        
        # Reordering and indexing by DimTuple
        reordered_dims = DimTuple((data_dims_A[1], data_dims_A[2], data_dims_A[0]))
        data_A_reordered = data_A.indexer.reorder(reordered_dims)
        data_tdims_A = data_dims_A.build_torchdims()
        data_tdims_A_reordered = data_tdims_A[reordered_dims]
        data_A_named_tensor = data_A.tensor[data_tdims_A]
        data_A_named_tensor_reordered = data_A_reordered.tensor[data_tdims_A_reordered]
        assert (data_A_named_tensor.order(*data_tdims_A) == data_A_named_tensor_reordered.order(*data_tdims_A)).all()
        
        # Subbatching along one or multiple dims
        subsamples, subsample_indices = data_A.indexer.simple_subsample(batch_dims_A[0], 5)
        print('Shape subsamples = {}'.format([subsample.shape for subsample in subsamples]))
        block_batch_dims_A = batch_dims_A
        block_subsample_sizes_A = [5,3]
        block_subsamples, block_subsample_indices = data_A.indexer.block_subsample(block_batch_dims_A, block_subsample_sizes_A)
        print('Shape block subsamples = {}'.format([subsample.shape for subsample in block_subsamples]))
        
        # Inheritance - by construction
        # Suppose we got data_C as a subset of data_B with derived ssi CalipyIndex and
        # now want to index data_C with proper names and references
        #   1. generate data_B
        batch_dims_B = dim_assignment(['bd_1_B', 'bd_2_B'])
        event_dims_B = dim_assignment(['ed_1_B'])
        data_dims_B = batch_dims_B + event_dims_B
        data_B_torch = torch.normal(0,1,[7,5,2])
        data_B = CalipyTensor(data_B_torch, data_dims_B, 'data_B')
        
        #   2. subsample data_C from data_B
        block_data_C, block_indices_C = data_B.indexer.block_subsample(batch_dims_B, [5,3])
        block_nr = 3
        data_C = block_data_C[block_nr]
        block_index_C = block_indices_C[block_nr]
        
        #   3. subsampling has created an indexer for data_C
        data_C.indexer
        data_C.indexer.local_index
        data_C.indexer.global_index
        data_C.indexer.local_index.tensor
        data_C.indexer.global_index.tensor
        data_C.indexer.global_index.index_name_dict
        data_C.indexer.data_source_name
        
        data_C_local_index = data_C.indexer.local_index
        data_C_global_index = data_C.indexer.global_index
        assert (data_C.tensor[data_C_local_index.tuple] == data_B.tensor[data_C_global_index.tuple]).all()
        assert ((data_C[data_C_local_index] - data_B[data_C_global_index]).tensor == 0).all()
        
        # Inheritance - by declaration
        # If data comes out of some external subsampling and only the corresponding indextensors
        # are known, the calipy_indexer can be evoked manually.
        data_D_torch = copy.copy(data_C.tensor)
        index_tensor_D = block_index_C.tensor
        
        data_D = CalipyTensor(data_D_torch, data_dims_B, 'data_D')
        data_D.indexer.create_global_index(index_tensor_D, 'from_data_D')
        data_D_global_index = data_D.indexer.global_index
        
        assert (data_D.tensor == data_B.tensor[data_D_global_index.tuple]).all()
        assert ((data_D - data_B[data_D_global_index]).tensor == 0).all()
        
        # Alternative way of calling via DataTuples
        data_E_torch = torch.normal(0,1,[5,3])
        batch_dims_E = dim_assignment(dim_names = ['bd_1_E'])
        event_dims_E = dim_assignment(dim_names = ['ed_1_E'])
        data_dims_E = batch_dims_E + event_dims_E
        
        data_names_list = ['data_A', 'data_E']
        data_list = [data_A_torch, data_E_torch]
        data_datatuple_torch = DataTuple(data_names_list, data_list)
        
        batch_dims_datatuple = DataTuple(data_names_list, [batch_dims_A, batch_dims_E])
        event_dims_datatuple = DataTuple(data_names_list, [event_dims_A, event_dims_E])
        data_dims_datatuple = batch_dims_datatuple + event_dims_datatuple
        
        data_datatuple = data_datatuple_torch.calipytensor_construct(data_dims_datatuple)
        data_datatuple['data_A'].indexer
        
        
        # Functionality for creating indices with TensorIndexer class methods
        # It is possible to create subsample_indices even when no tensor is given
        # simply by calling the class method TensorIndexer.create_block_subsample_indices
        # or TensorIndexer.create_simple_subsample_indices and providing the 
        # appropriate size specifications.         
        # i) Create the dims (with unspecified size so no conflict later when subbatching)
        batch_dims_FG = dim_assignment(['bd_1_FG', 'bd_2_FG'])
        event_dims_F = dim_assignment(['ed_1_F', 'ed_2_F'])
        event_dims_G = dim_assignment(['ed_1_G'])
        data_dims_F = batch_dims_FG + event_dims_F
        data_dims_G = batch_dims_FG + event_dims_G
        
        # ii) Sizes
        batch_dims_FG_sizes = [10,7]
        event_dims_F_sizes = [6,5]
        event_dims_G_sizes = [4]
        data_dims_F_sizes = batch_dims_FG_sizes + event_dims_F_sizes
        data_dims_G_sizes = batch_dims_FG_sizes + event_dims_G_sizes
        
        # iii) Then create the data
        data_F_torch = torch.normal(0,1, data_dims_F_sizes)
        data_F = CalipyTensor(data_F_torch, data_dims_F, 'data_F')
        data_G_torch = torch.normal(0,1, data_dims_G_sizes)
        data_G = CalipyTensor(data_G_torch, data_dims_G, 'data_G')
        
        # iv) Create and expand the reduced_index
        indices_reduced = TensorIndexer.create_block_subsample_indices(batch_dims_FG, batch_dims_FG_sizes, [9,5])
        index_reduced = indices_reduced[0]
        
        # Functionality for expanding, reducing, and reordering indices
        # Indices like the ones above can be used flexibly by expanding them to
        # fit tensors with various dimensions. They can also be changed w.r.t 
        # their order.
        
        # i) Expand index to fit data_F and data_G
        index_expanded_F = index_reduced.expand_to_dims(data_dims_F, [None]*len(batch_dims_FG) + event_dims_F_sizes)
        index_expanded_G = index_reduced.expand_to_dims(data_dims_G, [None]*len(batch_dims_FG) + event_dims_G_sizes)
        assert (data_F.tensor[index_expanded_F.tuple] == data_F.tensor[index_reduced.tensor[:,:,0], index_reduced.tensor[:,:,1], :,:]).all()
        assert ((data_F[index_expanded_F] - data_F[index_reduced.tensor[:,:,0], index_reduced.tensor[:,:,1], :,:]).tensor ==0).all()
        
        # ii) Reordering is done by passing in a differently ordered DimTuple
        data_dims_F_reordered = dim_assignment(['ed_2_F', 'bd_2_FG', 'ed_1_F', 'bd_1_FG'])
        data_dims_F_reordered_sizes = [5, None, 6, None]
        index_expanded_F_reordered = index_reduced.expand_to_dims(data_dims_F_reordered, data_dims_F_reordered_sizes)
        data_F_reordered = data_F.indexer.reorder(data_dims_F_reordered)
        data_F_subsample = data_F[index_expanded_F]
        data_F_reordered_subsample = data_F_reordered[index_expanded_F_reordered]
        assert (data_F_subsample.tensor == data_F_reordered_subsample.tensor.permute([3,1,2,0])).all()
        
        # iii) Index expansion can also be performed by the indexer of a tensor;
        # this is usually more convenient
        index_expanded_F_alt = data_F.indexer.expand_index(index_reduced)
        index_expanded_G_alt = data_G.indexer.expand_index(index_reduced)
        data_F_subsample_alt = data_F[index_expanded_F_alt.tuple]
        data_G_subsample_alt = data_G[index_expanded_G_alt.tuple]
        assert (data_F_subsample.tensor == data_F_subsample_alt.tensor).all()
        assert ((data_F_subsample - data_F_subsample_alt).tensor == 0).all()
        
        # Inverse operation is index_reduction (only possible when index is cartesian product)
        assert (index_expanded_F.is_reducible(batch_dims_FG))
        assert (index_reduced.tensor == index_expanded_F.reduce_to_dims(batch_dims_FG).tensor).all()
        assert (index_reduced.tensor == index_expanded_G.reduce_to_dims(batch_dims_FG).tensor).all()
        
        # Illustrate nonseparable case
        inseparable_index = CalipyIndex(torch.randint(10, [10,7,6,5,4]), data_dims_F)
        inseparable_index.is_reducible(batch_dims_FG)
        inseparable_index.reduce_to_dims(batch_dims_FG) # Produces a warning as it should

    """
    
    
    def __init__(self, tensor, dims, name = None):
        # Intialize abstract CalipyIndexer
        super().__init__(dims, name=name)
        
        # Integrate initial data
        self.name = name
        self.tensor = tensor
        self.tensor_dims = dims
        self.tensor_torchdims = dims.build_torchdims()
        self.tensor_named = tensor[self.tensor_torchdims]
        
        # Create index tensors
        self.local_index = self.create_local_index()
        
    @classmethod
    def create_block_subsample_indices(cls, batch_dims, tensor_shape, subsample_sizes):
        """
        Create a CalipyIndex that indexes only the specified batch_dims.

        :param batch_dims: DimTuple of batch dimensions to index.
        :param tensor_shape: List containing the sizes of the unsubsampled tensor
        :param subsample_sizes: Sizes for subsampling along each batch dimension.
        :return: A list of CalipyIndex instances indexing the batch_dims.
        """
        # Validate inputs
        if len(batch_dims) != len(subsample_sizes):
            raise ValueError("batch_dims and subsample_sizes must have the same length.")
        if len(batch_dims) != len(tensor_shape):
            raise ValueError("batch_dims and tensor_shape must have the same length.")

        # Create index ranges for subsampling
        generic_tensor = torch.zeros(tensor_shape)
        generic_tensor = CalipyTensor(generic_tensor, batch_dims, 'generic_indexer')
        _, block_subsample_indices = generic_tensor.indexer.block_subsample(batch_dims, subsample_sizes)

        return block_subsample_indices
 
    @classmethod
    def create_simple_subsample_indices(cls, batch_dim, batch_dim_size, subsample_size):
        """
        Create a CalipyIndex that indexes only the specified singular batch_dim.

        :param batch_dim: Element of DimTuple (typically CalipyDim) along which
            subbatching happens.
        :param batch_dim_size: The integer size of the unsubsampled tensor
        :param subsample_size: Single integer size determining length of batches to create.
        :return: A list of CalipyIndex instances indexing the batch_dim.
        """
        
        subsample_indices = cls.create_block_subsample_indices(DimTuple((batch_dim,)), [batch_dim_size], [subsample_size])
        
        return subsample_indices
                
    
    
    def create_local_index(self):
        """
        Create a local index tensor enumerating all possible indices for all the dims.
        The indices local_index_tensor are chosen such that they can be used for 
        indexing the tensor via value = tensor[i,j,k] = tensor[local_index_tensor[i,j,k,:]],
        i.e. the index at [i,j,k] is [i,j,k]. A more compact form of indexing
        is given by directly accessing the index tuples via tensor = tensor[local_index_tensor_tuple]

        :return: Writes torch tensors with indices representing all possible positions into the index
            local_index.tensor: index_tensor containing an index at each location of value in tensor
            local_index.tuple: index_tensor split into tuple for straightforward indexing
        """
        
        local_index = self._create_local_index(self.tensor_torchdims.sizes)
        return local_index
    
    def create_global_index(self, subsample_indextensor = None, data_source_name = None):
        """
        Create a global CalipyIndex object enumerating all possible indices for all the dims. The
        indices global_index_tensor are chosen such that they can be used to access the data
        in data_source with name data_source_name via self.tensor  = data_source[global_index_tensor_tuple] 
        
        :param subsample_indextensor: An index tensor that enumerates for all the entries of
            self.tensor which index needs to be used to access it in some global dataset.
        :param data_source_name: A string serving as info to record which object the global indices are indexing.
        
        :return: A CalipyIndex object global index containing indexing data that
            describes how the tensor is related to the superpopulation it has been
            sampled from.
        """
        
        global_index = self._create_global_index(subsample_indextensor, data_source_name)
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
        meshgrid = robust_meshgrid(indices_ordered_list, indexing='ij')
        block_index_tensor = torch.stack(meshgrid, dim=-1)
        block_index = CalipyIndex(block_index_tensor, block_index_dims, name = 'from_' + self.name)
        block_tensor = self.tensor[block_index.tuple]
        
        # Clarify global index
        block_tensor = CalipyTensor(block_tensor, self.tensor_dims, self.name)
        block_tensor.indexer.create_global_index(block_index.tensor, 'from_' + self.name)
        
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
        
        subsample_data, subsample_indices = self.block_subsample(DimTuple((batch_dim,)), [subsample_size])
        
        return subsample_data, subsample_indices
    
    def reorder(self, order_dimtuple):
        """
        Generate out of self.tensor a new tensor that is reordered to align with
        the order given in the order_dimtuple DimTuple object.

        :param order_dimtuple: DimTuple of CalipyDim objects whose sequence determines
            permutation and index binding of the produced tensor.
        :return: A tensor with an calipy.indexer where all ordering is aligned to order_dimtuple

        """
        # i) Check validity input
        if not set(order_dimtuple) == set(self.tensor_dims):
            raise Exception('CalipyDims in order_dimtuple and self.tensor_dims must '\
                            'be the same but are CalipyDims of order_dimtuple = {} '
                            'and CalipyDims of self.tensor_dims = {}'\
                            .format(set(order_dimtuple), set(self.tensor_dims)))
        
        # ii) Set up new tensor
        # preordered_index_dict = self.index_to_dim_dict
        preordered_dim_dict = self.dim_to_index_dict
        # reordered_dim_dict = self._create_dim_to_index_dict(order_dimtuple)
        reordered_index_dict = self._create_index_to_dim_dict(order_dimtuple)
        
        permutation_list = []
        for k in range(len(reordered_index_dict.keys())):
            dim_k = reordered_index_dict[k]
            new_index_dim_k = preordered_dim_dict[dim_k]
            permutation_list.append(new_index_dim_k) 
        reordered_tensor = self.tensor.permute(*permutation_list)
        reordered_tensor = CalipyTensor(reordered_tensor, order_dimtuple, self.name + '_reordered_({})'.format(order_dimtuple.names))
        
        return reordered_tensor
    
    def expand_index(self, index_reduced):
        """
        Expand the CalipyIndex index_reduced to align with the dimensions self.dims
        of the current tensor self.tensor.

        :param index_reduced: A CalipyIndex instance whosed dims are a subset of the
            dims of the current tensor.
        :return: A new CalipyIndex instance with the expanded index tensor.
        
        """
        # i) Check validity inpyt
        if not set(index_reduced.dims[0:-1]).issubset(set(self.tensor_dims)):
            raise Exception('CalipyDims in index_reduced.dims need to be contained '\
                            'in self.tensor_dims but are CalipyDims of index_reduced = {} '\
                            'and CalipyDims of self.tensor_dims = {}'\
                            .format(set(index_reduced.dims), set(self.tensor_dims)))
        
        # ii) Expand to match shape of current tensor
        expanded_dims = self.tensor_dims
        expanded_dim_sizes = [size if name not in index_reduced.dims.names else None
                              for name, size in zip(self.tensor_torchdims.names, self.tensor_torchdims.sizes)]
        expanded_index = index_reduced.expand_to_dims(expanded_dims, expanded_dim_sizes)
        return expanded_index
    
    
    def __repr__(self):
        dim_name_list = self.tensor_torchdims.names
        dim_sizes_list = self.tensor_torchdims.sizes
        repr_string = 'TensorIndexer for tensor with dims {} of sizes {}'.format(dim_name_list, dim_sizes_list)
        return repr_string
    
    
    
    
    
# CalipyTensor basic class 
    
# Preprocessing input arguments
def preprocess_args(args, kwargs):
    if kwargs is None:
            kwargs = {}

    # Unwrap CalipyTensors to get underlying tensors
    def unwrap(x):
        return x.tensor if isinstance(x, CalipyTensor) else x

    unwrapped_args = tuple(unwrap(a) for a in args)
    unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
    
        
    return unwrapped_args, unwrapped_kwargs
    

class CalipyTensor:
    """
    Class that wraps torch.Tensor objects and augments them with indexing operations
    and dimension upkeep functionality, while referring most torch functions to
    its wrapped torch.Tensor object. Can be sliced and indexed in the usual ways
    which produces another CalipyTensor whose indexer is inherited.
    
    :param tensor: The tensor which should be embedded into CalipyTensor
    :type tensor: torch.Tensor
    :param dims: A DimTuple containing the dimensions of the tensor or None
    :type dims: DimTuple
    :param name: A name for the CalipyTensor, useful for keeping track of derived CalipyTensor's.
        Default is None.
    :type name: string

    :return: An instance of CalipyTensor containing functionality for dimension
        upkeep, indexing, and function call referral.
    :rtype: CalipyTensor

    Example usage:

    .. code-block:: python
        
        # Imports and definitions
        import torch
        from calipy.core.tensor import CalipyTensor, TensorIndexer
        from calipy.core.utils import dim_assignment
        from calipy.core.data import DataTuple
    
        # Create CalipyTensors -----------------------------------------------
        #
        # Create DimTuples and tensors
        data_A_torch = torch.normal(0,1,[6,4,2])
        batch_dims_A = dim_assignment(dim_names = ['bd_1_A', 'bd_2_A'])
        event_dims_A = dim_assignment(dim_names = ['ed_1_A'])
        data_dims_A = batch_dims_A + event_dims_A
        data_A_cp = CalipyTensor(data_A_torch, data_dims_A, name = 'data_A')
        
        # Confirm that subsampling works as intended
        subtensor_1 = data_A_cp[0:1,0:3,...]
        subtensor_1.dims == data_A_cp.dims
        assert((subtensor_1.tensor - data_A_cp.tensor[0:1,0:3,...] == 0).all())
        # subsample has global_index that can be used for subsampling on tensors
        # and on CalipyTensors
        assert((data_A_cp.tensor[subtensor_1.indexer.global_index.tuple] 
                - data_A_cp.tensor[0:1,0:3,...] == 0).all())
        assert(((data_A_cp[subtensor_1.indexer.global_index] 
                - data_A_cp[0:1,0:3,...]).tensor == 0).all())
        
        # When using an integer, dims are kept; i.e. singleton dims are not reduced
        subtensor_2 = data_A_cp[0,0:3,...]
        assert((subtensor_2.tensor == data_A_cp[0,0:3,...].unsqueeze(0)).all())
        
        # Indexing of CalipyTensors via int, tuple, slice, and CalipyIndex
        data_A_cp[0,:]
        local_index = data_A_cp.indexer.local_index
        data_A_cp[local_index]
        # During addressing, appropriate indexers are built
        data_A_cp[0,:].indexer.global_index
        data_A_cp[local_index].indexer.global_index
        
        # CalipyTensors work well even when some dims are empty
        # Set up data and dimensions
        data_0dim = torch.ones([])
        data_1dim = torch.ones([5])
        data_2dim = torch.ones([5,2])
        
        batch_dim = dim_assignment(['bd'])
        event_dim = dim_assignment(['ed'])
        empty_dim = dim_assignment(['empty'], dim_sizes = [])
        
        data_0dim_cp = CalipyTensor(data_0dim, empty_dim)
        data_1dim_cp = CalipyTensor(data_1dim, batch_dim)
        data_1dim_cp = CalipyTensor(data_1dim, batch_dim + empty_dim)
        data_1dim_cp = CalipyTensor(data_1dim, empty_dim + batch_dim + empty_dim)
        
        data_2dim_cp = CalipyTensor(data_2dim, batch_dim + event_dim)
        data_2dim_cp = CalipyTensor(data_2dim, batch_dim + empty_dim + event_dim)
        
        # Indexing a scalar with an empty index just returns the scalar
        data_0dim_cp.indexer
        zerodim_index = data_0dim_cp.indexer.local_index
        zerodim_index.is_empty
        data_0dim_cp[zerodim_index]
        
        # # These produce errors or warnings as they should.
        # data_0dim_cp = CalipyTensor(data_0dim, batch_dim) # Trying to assign nonempty dim to scalar
        # data_1dim_cp = CalipyTensor(data_1dim, empty_dim) # Trying to assign empty dim to vector
        # data_2dim_cp = CalipyTensor(data_2dim, batch_dim + empty_dim) # Trying to assign empty dim to vector

        
        # CalipyTensor / DataTuple interaction ---------------------------------
        #
        # DataTuple and CalipyTensor interact well: In the following we showcase
        # that a DataTuple of CalipyTensors can be subsampled by providing a
        # DataTuple of CalipyIndexes or a single CalipyIndex that is automatically
        # distributed over the CalipyTensors for indexing.
        
        # Set up DataTuple of CalipyTensors
        batch_dims = dim_assignment(dim_names = ['bd_1'])
        event_dims_A = dim_assignment(dim_names = ['ed_1_A', 'ed_2_A'])
        data_dims_A = batch_dims + event_dims_A
        event_dims_B = dim_assignment(dim_names = ['ed_1_B'])
        data_dims_B = batch_dims + event_dims_B
        data_A_torch = torch.normal(0,1,[6,4,2])
        data_A_cp = CalipyTensor(data_A_torch, data_dims_A, 'data_A')
        data_B_torch = torch.normal(0,1,[6,3])
        data_B_cp = CalipyTensor(data_B_torch, data_dims_B, 'data_B')
        
        data_AB_tuple = DataTuple(['data_A_cp', 'data_B_cp'], [data_A_cp, data_B_cp])
        
        
        # Subsampling functionality -------------------------------------------
        #
        # subsample the data individually
        data_AB_subindices = TensorIndexer.create_simple_subsample_indices(batch_dims[0], data_A_cp.shape[0], 5)
        data_AB_subindex = data_AB_subindices[0]
        data_A_subindex = data_AB_subindex.expand_to_dims(data_dims_A, data_A_cp.shape)
        data_B_subindex = data_AB_subindex.expand_to_dims(data_dims_B, data_B_cp.shape)
        data_AB_sub_1 = DataTuple(['data_A_cp_sub', 'data_B_cp_sub'], [data_A_cp[data_A_subindex], data_B_cp[data_B_subindex]])
        
        # Use subsampling functionality for DataTuples, either by passing a DataTuple of
        # CalipyIndex or a single CalipyIndex that is broadcasted
        data_AB_subindex_tuple = DataTuple(['data_A_cp', 'data_B_cp'], [data_A_subindex, data_B_subindex])
        data_AB_sub_2 = data_AB_tuple.subsample(data_AB_subindex_tuple)
        data_AB_sub_3 = data_AB_tuple.subsample(data_AB_subindex)
        assert ((data_AB_sub_1[0] - data_AB_sub_2[0]).tensor == 0).all()
        assert ((data_AB_sub_2[0] - data_AB_sub_3[0]).tensor == 0).all()
        
        
        # Expansion and reordering -------------------------------------------
        #
        # Expand a tensor by copying it among some dimensions.
        data_dims_A = data_dims_A.bind([6,4,2])
        data_dims_B = data_dims_B.bind([6,3])
        data_dims_expanded = data_dims_A + data_dims_B[1:]
        data_A_expanded_cp = data_A_cp.expand_to_dims(data_dims_expanded)
        assert((data_A_expanded_cp[:,:,:,0].tensor.squeeze() - data_A_cp.tensor == 0).all())
        # Ordering of dims is also ordering of result
        data_dims_expanded_reordered = data_dims_A[1:] + data_dims_A[0:1] + data_dims_B[1:]
        data_A_expanded_reordered_cp = data_A_cp.expand_to_dims(data_dims_expanded_reordered)
        assert((data_A_expanded_reordered_cp.tensor -
                data_A_expanded_cp.tensor.permute([1,2,0,3]) == 0).all())
        
        # There also exists a CalipyTensor.reorder(dims) method
        data_dims_A_reordered = event_dims_A + batch_dims
        data_A_reordered_cp = data_A_cp.reorder(data_dims_A_reordered)
        assert((data_A_reordered_cp.tensor - data_A_cp.tensor.permute([1,2,0]) == 0).all())
        assert(data_A_reordered_cp.dims == data_dims_A_reordered)
        
    """
    
    __torch_function__ = True  # Not strictly necessary, but clarity

    def __init__(self, tensor, dims, name = None):
        
        # Input checks
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if dims is not None and len(dims) != tensor.ndim:
            warnings.warn("Number of dims in DimTuple does not match tensor.ndim, setting dims=None.")

        # Set initial attributes
        self.name = name
        self.tensor = tensor
        self.dims = dims
        self._indexer_construct(dims, name)
        
    def _indexer_construct(self, tensor_dims, name, silent = True):
        """Constructs a TensorIndexer for the tensor."""
        self.indexer = TensorIndexer(self.tensor, tensor_dims, name)
        return self if silent == False else None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        
        # Find first CalipyTensor in args
        self_instance = next((arg for arg in args if isinstance(arg, cls)), None)
        if self_instance is None:
            return NotImplemented
        
        
        # Call the original PyTorch function
        unwrapped_args, unwrapped_kwargs = preprocess_args(args, kwargs)
        result = func(*unwrapped_args, **unwrapped_kwargs)

        # Compute new dims
        new_dims = self_instance._compute_new_dims(func, args, kwargs, result)

        # Wrap result back into CalipyTensor if it's a Tensor
        if isinstance(result, torch.Tensor):
            return CalipyTensor(result, new_dims)
        elif isinstance(result, tuple):
            # If multiple Tensors, wrap each
            return tuple(CalipyTensor(r, new_dims) if isinstance(r, torch.Tensor) else r for r in result)
        else:
            return result

    def reorder(self, dims):
        """ Reorders the current CalipyTensor to another CalipyTensor with dims
        ordered as specified in argument dims.
        
        :param dims: A DimTuple instance that contains the dims of the current
            CalipyTensor and prescribes the dims of the reordered CalipyTensor
        :type dims: DimTuple
    
        :return: An instance of CalipyTensor reordered to match dims.
        :rtype: CalipyTensor
    
        Example usage:
    
        .. code-block:: python
        
            # Create DimTuples and tensors
            data_torch = torch.normal(0,1,[10,5,3])
            batch_dims = dim_assignment(dim_names = ['bd_1', 'bd_2'], dim_sizes = [10,5])
            event_dims = dim_assignment(dim_names = ['ed_1'], dim_sizes = [3])
            data_dims = batch_dims + event_dims
            data_cp = CalipyTensor(data_torch, data_dims, name = 'data')
            
            data_dims_reordered = event_dims + batch_dims
            data_reordered_cp = data_cp.reorder(data_dims_reordered)
            assert((data_reordered_cp.tensor - data_cp.tensor.permute([2,0,1]) == 0).all())
            assert(data_reordered_cp.dims == data_dims_reordered)
        """
        
        # Raise error if dims are mismatched
        if not set(dims.names) == set(self.dims.names):
            raise ValueError("Argument dims needs to be in bijection to self.dims " \
                             "but they are dims = {} and self.dims = {}"
                             .format(dims, self.dims))
        
        # Set up dims and indices
        current_tensor = self.tensor
        order_list = self.dims.find_indices(dims.names)
        
        # Reorder and package
        reordered_tensor = current_tensor.permute(order_list)
        reordered_tensor_cp = CalipyTensor(reordered_tensor, dims, name = self.name + '_reordered')
        return reordered_tensor_cp
        
        
        

    def expand_to_dims(self, dims):
        """ Expands the current CalipyTensor to another CalipyTensor with dims
        specified in argument dims. Returns a CalipyTensor with dims dims that
        consists of copies of self where expansion is necessary.
        
        :param dims: A DimTuple instance that contains the dims of the current
            CalipyTensor and prescribes the dims of the expanded CalipyTensor
        :type dims: DimTuple
    
        :return: An instance of CalipyTensor expanded to match dims.
        :rtype: CalipyTensor
    
        Example usage:
    
        .. code-block:: python
        
            # Create DimTuples and tensors
            data_torch = torch.normal(0,1,[10,3])
            batch_dims = dim_assignment(dim_names = ['bd_1'], dim_sizes = [10])
            event_dims = dim_assignment(dim_names = ['ed_1'], dim_sizes = [3])
            data_dims = batch_dims + event_dims
            data_cp = CalipyTensor(data_torch, data_dims, name = 'data')
            
            batch_dims_expanded = dim_assignment(dim_names = ['bd_1', 'bd_2'], dim_sizes = [10,5])
            data_dims_expanded = batch_dims_expanded + event_dims
            data_expanded_cp = data_cp.expand_to_dims(data_dims_expanded)
            assert((data_expanded_cp[:,0,:].tensor.squeeze() - data_cp.tensor == 0).all())
        """
        
        # i) Raise error if not all dims have sizes
        if None in dims.sizes:
            raise ValueError("All dims need to have integer sizes but sizes are {}"
                             .format(dims.sizes))
        
        # ii) Process dims
        current_dims = self.dims
        current_tensor = self.tensor
        expanded_dims = dims
        
        # new_dims = expanded_dims.delete_dims(current_dims.names)
        expansion_dims = expanded_dims.squeeze_dims(current_dims.names)
        expansion_tensor = torch.ones(expansion_dims.sizes)
        
        # iii) Construct expanded tensor
        # indices_current_dims = expanded_dims.find_indices(current_dims.names)
        # indices_new_dims = expanded_dims.find_indices(new_dims.names)
        
        input_dim_signature = ' '.join(current_dims.names)
        unsqueezed_dim_signature = [name if name in current_dims.names else '1' 
                                    for name in expanded_dims.names]
        output_dim_signature = ' '.join(unsqueezed_dim_signature)
        dim_signature = input_dim_signature + ' -> ' + output_dim_signature
        
        unsqueezed_tensor = einops.rearrange(current_tensor, dim_signature)
        expanded_tensor = expansion_tensor * unsqueezed_tensor
        expanded_tensor_cp = CalipyTensor(expanded_tensor, dims = expanded_dims,
                                          name = self.name + '_expanded')
        return expanded_tensor_cp

    def _compute_new_dims(self, func, orig_args, orig_kwargs, result):
        # A placeholder method that decides how dims change after an operation.
        # We'll handle a few common cases:
        # - Elementwise ops (torch.add): If broadcast occurred, attempt to broadcast dims.
        # - Reductions (torch.sum): Remove the reduced dimension.
        # - Otherwise: set dims=None by default.
        
        # List compatible function cases
        reduction_fun_list = ['sum', 'mean', 'prod', 'max', 'min']
        elementwise_fun_list = ['add', 'mul', 'sub', 'div']
        
        result_shape = result.shape

        if not isinstance(result, torch.Tensor):
            # Non-tensor result doesn't have dims
            return None

        # Extract dims from the first CalipyTensor in orig_args for reference
        input_dims = None
        for a in orig_args:
            if isinstance(a, CalipyTensor):
                input_dims = a.dims
                break

        # If no input had dims, no dims in output
        if input_dims is None:
            return_dims = dim_assignment(['return_dim'], dim_sizes = result_shape)

        # Example rules:
        func_name = func.__name__

        # Handle reduction-like ops:
        if func_name in reduction_fun_list:
            # If dim specified and it's a CalipyDim -> int conversion done
            # If dim not specified, all dims reduced -> dims=None
            dims_reduce_indices = orig_kwargs.get('dim', None)
            dims_reduce = self.dims[dims_reduce_indices]
            if dims_reduce is None:
                # Summation over all dims results in scalar or reduced shape with no dims
                return_dims = dim_assignment(['trivial_dim'], dim_sizes = [0])
            else:
                # If dim is CalipyDim, remove that dim from input_dims
                # If dim is DimTuple, remove all those dims from input_dims
                dims_reduce_names = [dims_reduce.name] if isinstance(dims_reduce, CalipyDim) else dims_reduce.names
                return_dims = input_dims.delete_dims(dims_reduce_names)
                


        # Handle elementwise ops like add, mul:
        elif func_name in elementwise_fun_list:
            # If multiple CalipyTensors involved, attempt to broadcast dims
            # Let's find all CalipyTensors and try broadcasting dims
            calipy_tensors = [a for a in orig_args if isinstance(a, CalipyTensor)]
            # For simplicity, assume two inputs:
            if len(calipy_tensors) == 2:
                dims1 = calipy_tensors[0].dims
                dims2 = calipy_tensors[1].dims
                new_dims = self._broadcast_dims(dims1, dims2, result.shape)
                return new_dims
            else:
                # If only one input with dims, keep them if shapes match, else None
                if input_dims is not None and len(input_dims) == result.ndim:
                    return input_dims
                else:
                    return None

        # By default, return None and possibly warn
        else: 
            warnings.warn(f"No dimension logic implemented for {func_name}, setting dims to None.")
        return return_dims

    def _broadcast_dims(self, dims1, dims2, result_shape):
        # Attempt to reconcile dims1 and dims2 according to broadcasting
        # Basic logic:
        # - If one of dim1, dims2 is None, copy the other
        # - If both have same length and each dimension matches, keep dims.
        # - If both have some common dims, inject missing dims of size 1.
        # - If shapes differ in ways that can't be mapped to dims easily, dims=None.
        
        # Case 1: one of dims is unspecified
        if dims1 is not None and dims2 is None:
            dims2 = dims1
        if dims1 is None and dims2 is not None:
            dims1 = dims2

        # Case 2: dims line up 1 to 1
        # Convert dims to lists
        d1, d2 = list(dims1), list(dims2)
        # Check length differences
        if len(d1) != len(d2):
            # Try to align by rightmost dimensions
            # For simplicity if rank differs, set dims=None
            return None

        # Check element-wise compatibility
        # If sizes differ and not broadcastable (one of them must be 1), dims=None
        for (dim_a, dim_b, size_r) in zip(d1, d2, result_shape):
            # If sizes differ and none is 1, dims=None
            if dim_a.size != dim_b.size:
                if dim_a.size != 1 and dim_b.size != 1:
                    return None
            # Update dim size to result size if ambiguous
            # If one dimension is 1, we can inherit the other's name and size
            # If both differ and one is 1, use the non-1 dimension's name and size
        # If we get here, let's pick dims from the first input or adapt sizes
        # This is simplistic: if broadcasting changed sizes, adapt them
        # Real logic might need to carefully assign names.
        # For now, assume result_shape matches after broadcast and assign dims from first input with updated sizes
        new_dims = []
        for i, d in enumerate(d1):
            new_dims.append(CalipyDim(d.name, size=result_shape[i]))
        return DimTuple(new_dims)
    
    def __getitem__(self, index):
        """ Returns new CalipyTensor based on either standard indexing quantities
        like slices or integers or a CalipyIndex. 
        
        :param index: Identifier for determining which elements oc self to compile
            to a new CalipyTensor
        :type index: Integer, tuple of ints,  slice, CalipyIndex
        :return: A new tensor with derived dimensions and same functionality as self
        :rtype: CalipyTensor
        """
        
        
        # Case 1: Standard indexing
        if type(index) in (int, tuple, slice):
            old_index = ensure_tuple(index)
            
            # Preserve singleton dimensions for integer indexing
            index = []
            for dim, idx in enumerate(old_index):
                if isinstance(idx, int):  # Prevent dimension collapse
                    index.append(torch.tensor([idx]))  # Convert to a tensor list
                else:
                    index.append(idx)
                   
            # Create new CalipyTensorby subsampling
            subtensor_cp = CalipyTensor(self.tensor[index], dims = self.dims,
                                        name = self.name)
            
            # Get indices corresponding to index
            mask = torch.zeros_like(self.tensor, dtype=torch.bool)
            mask[index] = True  # Apply the given index
            selected_indices = torch.nonzero(mask)
            
            # Reshape to proper indextensor
            indextensor_shape = list(subtensor_cp.tensor.shape) + [selected_indices.shape[1]]
            indextensor = selected_indices.view(*indextensor_shape)  

            subtensor_cp.indexer.create_global_index(subsample_indextensor = indextensor, 
                                                      data_source_name = self.name)
            return subtensor_cp
        
        # Case 2: If index is CalipyIndex, use index.tuple for subsampling
        elif type(index) is CalipyIndex:
            if index.is_empty:
                subtensor_cp = CalipyTensor(self.tensor, dims = self.dims,
                                            name = self.name)
            else:
                subtensor_cp = CalipyTensor(self.tensor[index.tuple], dims = self.dims,
                                            name = self.name)
            subtensor_cp.indexer.create_global_index(subsample_indextensor = index.tensor, 
                                          data_source_name = self.name)
            return subtensor_cp
        
        # Case 3: TorchDimTuple based indexing
        elif type(index) is TorchdimTuple:
            pass
        
        # Case 4: Raise an error for unsupported types
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")
        
        

    def __setitem__(self, index, value):
        self.tensor[index] = value    
        
    def __getattr__(self, name):
        # Prevent recursion for special methods
        # if name.startswith('__') and name.endswith('__'):
        #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Delegate attribute access to underlying tensor if not found
        return getattr(self.tensor, name)
    
    def __mul__(self, other):
        """ 
        Overloads the * operator to work on CalipyTensor objects.
        
        :param other: The CalipyTensor or torch.tensor to multiply.
        :type other: CalipyTensor or torch.tensor
        :return: A new Calipytensor with elements from each tensor multiplied elementwise.
            Operation supports broadcasting.
        :rtype: CalipyTensor
        :raises ValueError: If both self and other are not broadcastable.
        """
        if not isinstance(other, CalipyTensor):
            return NotImplemented

        if self.dims != other.dims:
            raise ValueError("Both CalpyTensors must have the same dims for elementwise addition.")

        result = torch.mul(self, other)
        
        return result
    
    def __div__(self, other):
        """ 
        Overloads the / operator to work on CalipyTensor objects.
        
        :param other: The CalipyTensor or torch.tensor used to divide self.
        :type other: CalipyTensor or torch.tensor
        :return: A new Calipytensor with elements from self divided elementwise by elements from other.
            Operation supports broadcasting.
        :rtype: CalipyTensor
        :raises ValueError: If both self and other are not broadcastable.
        """
        if not isinstance(other, CalipyTensor):
            return NotImplemented

        if self.dims != other.dims:
            raise ValueError("Both CalpyTensors must have the same dims for elementwise addition.")

        result = torch.div(self, other)
        
        return result
    
    
    def __add__(self, other):
        """ 
        Overloads the + operator to work on CalipyTensor objects.
        
        :param other: The CalipyTensor to add.
        :type other: CalipyTensor
        :return: A new Calipytensor with elements from each tensor added elementwise.
        :rtype: CalipyTensor
        :raises ValueError: If both self and other are not broadcastable.
        """
        if not isinstance(other, CalipyTensor):
            return NotImplemented

        if self.dims != other.dims:
            raise ValueError("Both CalpyTensors must have the same dims for elementwise addition.")

        result = torch.add(self, other)

        return result
    
    def __sub__(self, other):
        """ 
        Overloads the - operator to work on CalipyTensor objects.
        
        :param other: The CalipyTensor to add.
        :type other: CalipyTensor
        :return: A new Calipytensor with elements from each tensor added elementwise.
        :rtype: CalipyTensor
        :raises ValueError: If both self and other are not broadcastable.
        """
        if not isinstance(other, CalipyTensor):
            return NotImplemented

        if self.dims != other.dims:
            raise ValueError("Both CalpyTensors must have the same dims for elementwise addition.")

        result = torch.sub(self, other)

        return result
        
    def __repr__(self):
        return f"CalipyTensor({repr(self.tensor)}, \n shape = {self.shape}, \n dims={self.dims})"

    def __str__(self):
        return f"CalipyTensor({str(self.tensor)}, dims={self.dims})"
    