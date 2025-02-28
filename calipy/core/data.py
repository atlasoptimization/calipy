#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides basic functionality to represent and access data in a way
that interacts well with calipy's basic classes and methods. 

The classes and functions are
    DataTuple: A class for holding tuples of various objects with explicit names.
        is the basic object to be used for input variables, observations etc. as
        it makes explicit the meaning of the tensors passed or produced.
    
    sample:
  
The DataTuple class is often used to manage and package data, including for the
various forward() methods when activating CalipyNodes.
        

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


import torch
from calipy.core.tensor import CalipyTensor, CalipyIndex


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
        data_tuple_cp = data_tuple.calipytensor_construct(full_dims_datatuple)
        augmented_tensor = data_tuple_cp['tensor_A']
        augmented_tensor.indexer.local_index
        
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
        
        
        # DataTuple and CalipyTensor interact well: In the following we showcase
        # that a DataTuple of CalipyTensors can be subsampled by providing a
        # DataTuple of CalipyIndexes or a single CalipyIndex that is automatically
        distributed over the CalipyTensors for indexing.
        
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
                
    """
    def __init__(self, names, values):
        if len(names) != len(values):
            raise ValueError("Length of names must match length of values.")
        self._data_dict = {name: value for name, value in zip(names, values)}

    def __getitem__(self, key):
        if type(key) == int:
            keyname = list(self._data_dict.keys())
            return self._data_dict[keyname[key]]
        if type(key) == str:
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
    
    def __len__(self):
        return len(list(self.keys()))
    
    def as_dict(self):
        """ Returns the underlying dictionary linking names and values """
        self_dict = self._data_dict
        return self_dict
    
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
    
    def get_tensors(self):
        """ 
        Allows to extract .tensor attribute out of a DataTuple that contains
        CalipyTensors leaving other objects in the tuple unperturbed.
        
        :return: DataTuple containing for each key, value pair either the tensor
            subattribute value.tensor or the original value.        
        """
        new_dict = {}
        attr = 'tensor'
        for key, value in self._data_dict.items():
            if hasattr(value, attr):
                attribute = getattr(value, attr)
                new_dict[key] = attribute
            else:
                new_dict[key] = value
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

    def calipytensor_construct(self, dims_datatuple):
        """
        Applies construction of the TensorIndexer to build for each tensor in self
        the CalipyTensor construction used for indexing. Requires all elements
        of self to be tensors and requires dims_datatuple to be a DataTuple containing
        DimTuples.

        :param self: A DataTuple containing indexable tensors to be indexed
        :param dims_datatuple: A DataTuple containing the DimTuples used for indexing
        :type dim_datatuple: DataTuple
        :return: Nothing returned, calipy.indexer integrated into tensors in self
        :rtype: DataTuple
        :raises ValueError: If both DataTuples do not have matching keys.
        """
        # Check consistency
        if self.keys() != dims_datatuple.keys():
            raise ValueError("Both DataTuples self and dims_datatuple must have the same keys.")
        # # Apply indexer_construction
        # fun_dict = {}
        # for key, value in self.items():
        #     fun_dict[key] = lambda tensor, key = key: tensor.calipy.indexer_construct(dims_datatuple[key], name = key, silent = False)
        # self.apply_from_dict(fun_dict)
        key_list = []
        new_value_list = []
        for key, value in self.items():
            key_list.append(key)
            new_value = CalipyTensor(value, dims_datatuple[key], name = key)
            new_value_list.append(new_value)
        calipytensor_datatuple = DataTuple(key_list, new_value_list)
        
        return calipytensor_datatuple
    
    def subsample(self, datatuple_indices):
        """
        Subsamples a DataTuple containing CalipyTensors by applying to each of it
        the corresponding CalipyIndex object from the datatuple_indices. The arg
        datatuple_indices can also consist of just 1 entry of CalipyIndex that
        is then applied to all elements of self for subsampling. If a CalipyTensor
        in self does not feature the dim subsampled in CalipyIndex, then it is not 
        subsampled
        
        :param datatuple_indices: The DataTuple containing the CalipyIndex objects
            or a single CalipyIndex object.
        :type datatuple_indices: DataTuple or CalipyIndex
        :return: A new DataTuple with each CalipyTensor subsampled by the indices.
        :rtype: DataTuple
        :raises ValueError: If both DataTuples do not have matching keys.
        """
        
        # Argument checks
        # n_tensors = len(self)
        # n_indices = len(datatuple_indices)
        if not all([type(t) == CalipyTensor for t in self]):
            raise ValueError("Input DataTuple must be of type CalipyTensor")
        if isinstance(datatuple_indices, CalipyIndex):
            # if kk:
            #     raise ValueError("If a single CalipyIndex object is provided, it "\
            #      "must be a subsampling of a dim that is present in all")
            pass
            
        elif isinstance(datatuple_indices, DataTuple):
            if self.keys() != datatuple_indices.keys():
                raise ValueError("Input DataTuple and Index DataTuple must have the "\
                                 "same keys for subsampling or len datatuple_indices must be 1.")

            if not all([isinstance(i, CalipyIndex) for i in datatuple_indices]):
                raise ValueError("Elements of datatuple_indices must be of type CalipyIndex or subclass thereof")
        else:
            raise ValueError("datatuple_indices is of unsupported type: {type(datatuple_indices)}")
            
        # subsampling 
        list_names = []
        list_subsamples = []
        # when datatuple_indices is CalipyIndex
        if isinstance(datatuple_indices, CalipyIndex):
            for key in self.keys():
                list_names.append(key)
                current_tensor = self[key]
                current_index = datatuple_indices.expand_to_dims(dims = current_tensor.dims,
                                                                 dim_sizes = current_tensor.shape )
                list_subsamples.append(current_tensor[current_index])
            datatuple_subsampled = DataTuple(list_names, list_subsamples)
            
        # when datatuple_indices is DataTuple
        elif isinstance(datatuple_indices, DataTuple):
            for key in self.keys():
                list_names.append(key)
                list_subsamples.append(self[key][datatuple_indices[key]])
            datatuple_subsampled = DataTuple(list_names, list_subsamples)
        
        return datatuple_subsampled
    
    def rename_keys(self, rename_dict):
        """ 
        Renames current keys to the ones given by rename_dict[key].
        
        :param rename_dict: Dictionary s.t. for each key in rename_dict, key is in
            self.keys() with rename_dict[key] being the string that is the key
            in the newly produced DataTuple.
        :type rename_dict: dict
        :return: DataTuple the same values but with changed keys. 
        :rtype: DataTuple
        """
        
        new_dict = {}
        for key, value in self.items():
            if key in rename_dict.keys():
                new_dict[rename_dict[key]] = value
            else:
                new_dict[key] = value
                    
        key_list, value_list = zip(*new_dict.items())
        return DataTuple(key_list, value_list)

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
            if isinstance(v, (torch.Tensor, CalipyTensor)):
                repr_items.append(f"{k}: shape={v.shape}")
            else:
                repr_items.append(f"{k}: {v.__repr__()}")
        return f"DataTuple({', '.join(repr_items)})"
