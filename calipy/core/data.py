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
from typing import Any
from torch.utils.data import Dataset
from calipy.core.tensor import CalipyTensor, CalipyIndex, IOIndexer
from calipy.core.utils import dim_assignment, ensure_tuple
from calipy.core.funs import calipy_cat



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



# CalipyDict class providing standard form and wrapping of I/O data

class CalipyDict(dict):
    """
    A dictionary-like container that can store single or multiple items.
    If it contains exactly one item, calipy_dict.value can be called to retrieve
    it directly. Is meant as a convenient wrapper to DataTuple functionality 
    and is the basis  for the standard input/output/observation format CalipyIO
    handled inside of the CalipyNode objects. Is typically autowrapped around 
    dictionaries or single objects provided by the user towards e.g. the forward()
    method. Has idempotent property and leaves CalipyDict objects unchanged.
    
    CalipyDict allows heterogeneous tensor shapes for flexible datasets. Keys
    represent measurement identifiers ('mean', 'var', etc.); values are e.g. 
    CalipyTensors with potentially differing shapes across CalipyDict instances.
    
    :param data: The data being used to construct the CalipyDict. Data used for
        dictionary initialization can be:
          - None => empty dict
          - A dict {str -> item} => multi-item
          - A CalipyDict => Leave unchanged
          - A DataTuple => convert to dict
          - A single item => store under a default key '__single__'
    :type data: None, or dict, or CalipyDict, or DataTuple, or single object.
    
    :return: An instance of CalipyDict
    :rtype: CalipyDict

    Example usage:

    .. code-block:: python
        
        # Imports and definitions
        import torch
        from calipy.core.data import DataTuple, CalipyDict
           

        # Create data for CalipyDict initialization
        tensor_A = torch.ones(2, 3)
        tensor_B = torch.ones(4, 5)
        names = ['tensor_A', 'tensor_B']
        values = [tensor_A, tensor_B]
        data_tuple = DataTuple(names, values)
        data_dict = {'tensor_A': tensor_A, 'tensor_B' : tensor_B}
        
        # Create CalipyDict objects
        dict_from_none = CalipyDict()
        dict_from_dict = CalipyDict(data_dict)
        dict_from_tuple = CalipyDict(data_tuple)
        dict_from_calipy = CalipyDict(dict_from_dict)
        dict_from_single = CalipyDict(tensor_A)
        
        # Print contents and investigate 
        for cp_dict in [dict_from_none, dict_from_dict, dict_from_tuple, 
                        dict_from_calipy, dict_from_single]:
            print(cp_dict)
            
        dict_from_single.has_single_item
        dict_from_single.value
        dict_from_dict.as_datatuple()
    
    """

    def __init__(self, data=None):
        """
        Data used for dictionary initialization can be:
          - None => Dict containing None
          - A dict {str -> item} => multi-item
          - A single item => store under a default key '__single__'
          - A DataTuple => convert to dict
          - A CalipyDict => Leave unchanged
        """
        super().__init__()  # Initialize the underlying dict
                
        if isinstance(data, CalipyDict):
            # leave unchanged
            self.update(data)
        
        elif isinstance(data, DataTuple):
            # convert from DataTuple
            dt_dict = data.as_dict()
            self.update(dt_dict)
        
        elif isinstance(data, dict):
            # multiple items
            self.update(data)  # fill this dict with the items
        else:
            # for single item, store under default key
            self["__single__"] = data
    
    @property
    def value(self) -> Any:
        """
        If there's exactly one item in this CalipyDict, return it.
        Otherwise, raise an error. This property allows single-output usage.
        """
        if len(self) != 1:
            raise ValueError(
                "CalipyDict has {} items with keys {}, cannot use .value. Use"\
                    " bracket notation or iterate instead.".format(len(self), list(self.keys()))
            )
        # get the only key
        (k, v) = next(iter(self.items()))
        return v
    
    @property
    def is_null(self):
        """ Indicate if CalipyDict only has one element and that one is trivial"""
        null_indicator = self[list(self.keys())[0]] is  None and len(self) == 1
        return null_indicator

    @property
    def has_single_item(self) -> bool:
        """
        :return: True if exactly one item is in this dict, else False.
        """
        return len(self) == 1
    
    def as_datatuple(self) -> DataTuple:
        """
        Convert this CalipyDict into a DataTuple for dimension-aware operations
        or other advanced uses.
        """
        keys_list = list(self.keys())
        vals_list = list(self.values())
        return DataTuple(keys_list, vals_list)
    
    def rename_keys(self, rename_dict):
        """ 
        Renames current keys to the ones given by rename_dict[key].
        
        :param rename_dict: Dictionary s.t. for each key in rename_dict, key is in
            self.keys() with rename_dict[key] being the string that is the key
            in the newly produced CalipyDict.
        :type rename_dict: dict
        :return: CalipyDict with the same values but with changed keys. 
        :rtype: CalipyDict
        """
        
        renamed_data_tuple = self.as_datatuple().rename_keys(rename_dict)
        return CalipyDict(renamed_data_tuple)
    
    def subsample_tensors(self, dim, indices):
        """ Allows accessing CalipyTensor elements of CalipyDict by passing a
        list of integer indices and a single dimension along which all of the
        CalipyTensors in the dict are to be sliced.
        
        :param indices: List of integer indices that is used for indexing self
            in the dimension dim
        :type indices: list of int
        :param dim: A DimTuple containing a single CalipyDim object declaring
            which dim is to be subsampled
        :type dim: DimTuple
        :return: A new CalipyDict with keys of self and corresponding values =
            value[..., indices, ...] i.e. the values indexed by the indices in
            dimension dim.       
        :rtype: CalipyDict           
        """
        new_dict = {}
        for key, value in self.items():
            if isinstance(value, CalipyTensor):
                new_dict[key] = value.get_element(dim, indices)
        return CalipyDict(new_dict)
    
    def stack(self, other):
        """ 
        Overloads the + operator to return a new CalipyDicte when adding two 
        CalipyDict objects. Addition is defined 
        
        :param other: The CalipyDict to add.
        :type other: CalipyDict
        :return: A new CalipyDict with elements from each dict stacked.
        :rtype: CalipyDict
        :raises ValueError: If both DataTuples do not have matching keys.
        """
        
        # Basic IO check
        if not isinstance(other, CalipyDict):
            return NotImplemented

        if self.keys() != other.keys():
            raise ValueError("Both CalipyDicts must have the same keys for stacking.")

        # Stack elements based on their type
        stacked_dict = {}
        for key, value in self.items():
            val_1 = value
            val_2 = other[key]
            if type(val_1) != type(val_2):
                raise ValueError('Both values must be of the same type for stacking' \
                                 'but are {} and {}.'.format(type(val_1), type(val_2)))


        return stacked_dict
    

    def __getitem__(self, key):
        """ Allows accessing elements of CalipyDict by passing different types
        of arguments as keys, these include:
            - str => string is interpreted as key of dict; value[key] returned            
        """
        if type(key) == str:
            # Standard dictionary behavior for string keys
            return super().__getitem__(key)

    
    def __add__(self, other):
        """ 
        Overloads the + operator to return a new CalipyDicte when adding two 
        CalipyDict objects. Addition is defined elementwise for each key.
        
        :param other: The CalipyDict to add.
        :type other: CalipyDict
        :return: A new CalipyDict with elements from each dict added and keys as
            from self.
        :rtype: CalipyDict
        :raises ValueError: If both DataTuples do not have matching keys.
        
        Example usage:
    
        .. code-block:: python
            
            # Imports and definitions
            import torch
            from calipy.core.data import CalipyDict
               
    
            # Create data for CalipyDict initialization
            tensor_A = torch.ones(2, 3)
            tensor_B = torch.ones(2, 3)
            data_dict_1 = CalipyDict({'tensor': tensor_A})
            data_dict_2 = CalipyDict({'tensor' : tensor_B})
            
            # Create CalipyDict sum
            dict_sum = data_dict_1 + data_dict_2

        """
        if not isinstance(other, CalipyDict):
            return NotImplemented

        if self.keys() != other.keys():
            raise ValueError("Both CalipyDicts must have the same keys for elementwise addition.")

        data_tuple_1 = self.as_datatuple()
        data_tuple_2 = other.as_datatuple()
        
        data_tuple_sum = data_tuple_1 + data_tuple_2
        dict_sum = CalipyDict(data_tuple_sum)

        return dict_sum

    def __eq__(self, other):
        if isinstance(other, CalipyDict):
            return dict.__eq__(self, other)
        return NotImplemented
        
    def __repr__(self):
        # Use dict's __repr__ to  represent content
        base = dict.__repr__(self)
        return f"CalipyDict({base})"



# CalipyList class providing standard form and wrapping of I/O data

class CalipyList(list):
    """
    A list-like container that can store single or multiple dict like containers.
    Is meant as a convenient wrapper for homogeneous and inhomogeneous lists of
    data where each list element is a CalipyDict. CalipyList is a central element
    to the standard input/output/observation format handled inside of the CalipyNode
    objects. Is typically autowrapped around lists of dictionaries or single objects
    provided by the user towards e.g. the forward() method. Has idempotent property
    and leaves CalipyList objects unchanged; i.e. wrapping multiple times is equivalent
    to wrapping once. Ingredient to CalipyIO.
    
    
    :param data: The data being used to construct the CalipyList. Data used for
        dictionary initialization can be:
          - A single item => CalipyList containing single item
          - A list => CalipyList containg list of objects
    :type data: Any
    
    :return: An instance of CalipyList
    :rtype: CalipyList    
    """
    
    def __new__(cls, data = None):
        if isinstance(data, cls):
            return data  # Idempotent: if data is already CalipyList, return it directly.
        instance = super().__new__(cls)
        return instance


    def __init__(self, data = None):
        """
        Data used for CalipyList initialization can be:
          - A single item => CalipyList containing single item
          - A list => CalipyList containg list of objects
        """
        
        # Initialization already done, avoid overwriting
        if isinstance(data, CalipyList):
            return 
        
        # If list, wrap elements into CalipyDict
        elif isinstance(data, list):
            for datapoint in data:
                self.append(CalipyDict(datapoint)) 
        # Else assume single element and append
        else:
            self.append(CalipyDict(data))
        # self.data = CalipyDict(data)

    @property
    def value(self) -> Any:
        """
        If there's exactly one item in this CalipyList, return it.
        Otherwise, raise an error. This property allows single-output usage.
        """
        if len(self) != 1:
            raise ValueError("CalipyList has {} items, cannot use .value. Use"\
                    " bracket notation or iterate instead.".format(len(self)))
            
        return self[0]

    @property
    def is_null(self):
        """ Indicate if CalipyList only has one element and that one is trivial"""
        null_indicator = self[0].is_null and len(self) == 1
        return null_indicator
    
    @property
    def has_single_item(self) -> bool:
        """
        :return: True if exactly one item is in this dict, else False.
        """
        return len(self) == 1
    
    def __getitem__(self, idx):
        """ Returns new CalipyList based on integers or a list of integers. If
        the index is an integer k, returns the k-th element; otherwise a new
        CalipyList is created
        
        :param index: Identifier for determining which elements of self to compile
            to a new CalipyList
        :type index: Integer, list of Integers
        :return: A new datacontainer with same functionality as self
        :rtype: CalipyList
        """
        
        if isinstance(idx, int):
            return super().__getitem__(idx)
        
        elif isinstance(idx, (slice, tuple)):
            return CalipyList(super().__getitem__(idx))
        elif isinstance(idx, list):
            new_list = [self[key] for key in idx]
            return CalipyList(new_list)
            
    def __eq__(self, other):
        if isinstance(other, CalipyList):
            return list.__eq__(self, other)
        return NotImplemented
        
    def __repr__(self):
        # Use dict's __repr__ to  represent content
        base = list.__repr__(self)
        return f"CalipyList({base})"





# CalipyIO master class providing standard form and wrapping of I/O data

class CalipyIO:
    """
    A data container that can store single or multiple dict like containers.
    Is meant as a convenient wrapper for homogeneous and inhomogeneous lists of
    data where each list element is a CalipyDict. CalipyIO s the standard input
    /output/observation format handled inside of the CalipyNode objects. Is
    typically autowrapped around lists of dictionaries or single objects provided
    by the user towards e.g. the forward() method. Has idempotent property and 
    leaves CalipyIO objects unchanged; i.e. wrapping multiple times is equivalent
    to wrapping once. CalipyIO objects are also the output of iterating through
    InhomogeneousDataLoader objects; i.e. datasets and subbatched datasets are
    represented in this way.
    
    Special access rules: 
        - If calipy_io contains in its list a single dict, calipy_io.dict returns it
        - If calipy_io contains in its list a single dict, calipy_io[key] returns
            the corresponding value dict[key]
        - If calipy_io contains in its list a single dict and in that dict a single
            key, value pair, then calipy_io.value returns that value.
    
    :param data: The data being used to construct the CalipyDict. Data used for
        dictionary initialization can be:
          - None => empty dict
          - A dict {str -> item} => multi-item
          - A CalipyDict => Leave unchanged
          - A DataTuple => convert to dict
          - A single item => store under a default key '__single__'
    :type data: None, or dict, or CalipyDict, or DataTuple, or single object.
    
    :return: An instance of CalipyDict
    :rtype: CalipyDict

    Example usage:

    .. code-block:: python
        
        # Imports and definitions
        import torch
        from calipy.core.data import DataTuple, CalipyDict, CalipyList, CalipyIO
        from calipy.core.tensor import CalipyTensor
        from calipy.core.utils import dim_assignment
           
        
        # Create data for CalipyList
        calipy_list_empty = CalipyList()
        calipy_list = CalipyList(data = ['a','b'])
        calipy_same_list = CalipyList(calipy_list)
        
        
        # Pass data into CalipyIO and investigate
        
        # Legal input types are None, single object, dict, CalipyDict, DataTuple, list,
        # CalipyList, CalipyIO. 
        
        # Build inputs
        none_input = None
        single_input = torch.tensor([1.0])
        dict_input = {'a': 1, 'b' : 2}
        CalipyDict_input = CalipyDict(dict_input)
        DataTuple_input = CalipyDict_input.as_datatuple()
        list_input = [dict_input, {'c' : 3}, {'d' : 4}]
        CalipyList_input = CalipyList(list_input)
        CalipyIO_input = CalipyIO(dict_input)
        
        # Build CalipyIO's
        none_io = CalipyIO(none_input)
        single_io = CalipyIO(single_input)
        dict_io = CalipyIO(dict_input)
        CalipyDict_io = CalipyIO(CalipyDict_input)
        DataTuple_io = CalipyIO(DataTuple_input)
        list_io = CalipyIO(list_input, name = 'io_from_list')
        CalipyList_io = CalipyIO(CalipyList_input)
        CalipyIO_io = CalipyIO(CalipyIO_input)
        
        
        # Check properties
        none_io.is_null
        single_io.is_null
        print(single_io)
        
            
        # Functionality includes:
        #   1. Iteration
        #   2. Fetch by index
        #   3. Associated CalipyIndex
        #      -  Has global and local index
        #   4. Comes with collate function
        
        # 1. Iteration
        # Proceed to investigate one of the built calipy_io objects, here list_io
        for io in list_io:
            print(io)
            print(io.indexer.global_index)
        
        # 2. Fetch by index
        # Access values (special if list and dict only have 1 element)
        single_io.dict
        single_io.value
        single_io.calipy_dict
        single_io.calipy_list
        single_io.data_tuple
        single_io['__single__']
        list_io[0]['a']
        
        
        # 3. a) Associated Indexer
        # Content of indexer
        list_io.batch_dim_flattened
        list_io.indexer
        list_io.indexer.local_index
        list_io_sub = list_io[1:2]
        list_io_sub.indexer.data_source_name
        list_io_sub.indexer.index_tensor_dims
        
        # 3. b) Associated CalipyIndex
        # Content of specific IOIndexer
        list_io_sub.indexer.local_index.tuple
        list_io_sub.indexer.local_index.tensor
        list_io_sub.indexer.local_index.index_name_dict
        
        list_io_sub.indexer.global_index.tuple
        list_io_sub.indexer.global_index.tensor
        list_io_sub.indexer.global_index.index_name_dict
        
        # Iteration produces sub_io's
        for io in list_io:
            print(io.indexer.global_index)
            print(io.indexer.global_index.tensor)
        
        # 3. c) Index / IO interaction
        # subsampling and indexing: via intes, tuples, slices, and CalipyIndex
        sub_io_1 = list_io[0]
        sub_io_2 = list_io[1]
        sub_io_3 = list_io[1:3]
        
        sub_io_1.indexer.local_index
        sub_io_2.indexer.local_index
        sub_io_3.indexer.local_index
        
        sub_io_1.indexer.global_index
        sub_io_2.indexer.global_index
        sub_io_3.indexer.global_index
        
        global_index_1 = sub_io_1.indexer.global_index
        global_index_2 = sub_io_2.indexer.global_index
        global_index_3 = sub_io_3.indexer.global_index
        
        assert(list_io[global_index_1] == list_io[0])
        assert(list_io[global_index_2] == list_io[1])
        assert(list_io[global_index_3] == list_io[1:3])
        
        # 4. Collate function
        # Check collation functionality for autoreducing io s
        mean_dims = dim_assignment(['bd_1', 'ed_1'])
        var_dims = dim_assignment(['bd_1', 'ed_1'])
        
        mean_1 = CalipyTensor(torch.randn(3, 2), mean_dims)
        mean_2 = CalipyTensor(torch.randn(5, 2), mean_dims)
        var_1 = CalipyTensor(torch.randn(3, 2), var_dims)
        var_2 = CalipyTensor(torch.randn(5, 2), var_dims)
        
        io_obj = CalipyIO([
            CalipyDict({'mean': mean_1, 'var': var_1}),
            CalipyDict({'mean': mean_2, 'var': var_2})
        ])
        
        collated_io = io_obj.reduce_list()
        
        # Rename all entries in the dicts in CalipyIO
        rename_dict = {'a' : 'new_a', 'b' : 'new_b'}
        renamed_io = list_io.rename_keys(rename_dict)

    """
    
    def __new__(cls, data = None, name = 'io_noname'):
        if isinstance(data, cls):
            instance =  data  # Idempotency on CalipyIO
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, data=None, name = 'io_noname'):
        """
        Data used for CalipyIO initialization can be:
          - None => CalipyIO containing None
          - A single item => store under a default key '__single__'
          - A dict {str -> item} => multi-item
          - A list [obj_1, .. ob_n] => inhomogeneous multi-item
          - A DataTuple => convert to dict, then wrap in list
          - A CalipyDict => Wrap in list
          - A CalipyIO => Leave unchanged
        """
 
        if isinstance(data, CalipyIO):
            # Initialization already done, avoid overwriting
            return    
 
        # Build new instance depending on input type            
        if isinstance(data, (list, CalipyList)):
            self.calipy_dict = None
            self.data_tuple = None
            self.calipy_list = CalipyList(data)
            
        # elif isinstance(data, (type(None), dict, CalipyDict, DataTuple)):
        
        else:
            self.calipy_dict = CalipyDict(data)
            self.data_tuple = self.calipy_dict.as_datatuple()
            self.calipy_list = CalipyList(CalipyDict(data))
        
        # if self.is_reducible:
        #     self.reduce_list()
            
        self.name = name
        # self.data = data
        self.batch_dim_flattened = dim_assignment(['batch_dim_flattened'], 
            dim_descriptions = ['Flattened batch dimension used to index list elements'])
        
        self._indexer_construct(self.batch_dim_flattened, name)        
    
    
    @property
    def is_null(self):
        """ Indicate if CalipyIO only has one element and that one is trivial"""
        null_indicator = self.calipy_list.is_null
        return null_indicator
        
    @property
    def is_reducible(self):
        # Check compatibility
        first_dict = self.calipy_list[0]
        keys = first_dict.keys()
        indicator_bool = 1
    
        for d in self.calipy_list:
            if set(d.keys()) != set(keys):
                indicator_bool = False
                return indicator_bool
            
        for key in keys:
            tensors_to_concat = [d[key] for d in self.calipy_list]
            
            # Check shape compatibility (except first dimension)
            tensor_shapes = [t.shape[1:] if t is not None else None for t in tensors_to_concat]
            if len(set(tensor_shapes)) > 1:
                indicator_bool = False

        return indicator_bool
    
    @property
    def dict(self) -> Any:
        """
        If there's exactly one dict in this CalipyIO, return it.
        Otherwise, raise an error. This property allows single-output usage.
        """
        if len(self.calipy_list) != 1:
            raise ValueError("CalipyIO has {} items, cannot use .dict. Use"\
                    " bracket notation or iterate instead.".format(len(self)))
            
        cp_dict = self.calipy_list[0]
        return cp_dict
    
    @property
    def value(self) -> Any:
        """
        If there's exactly one dict in this CalipyIO and one entry in it return
        the entry. Otherwise, raise an error. This property allows single-output usage.
        """
        
        if not self.has_single_item:
            raise ValueError("CalipyIO has {} items, cannot use .value. Use"\
                    " bracket notation or iterate instead.".format(len(self)))
        if not self.dict.has_single_item:
            raise ValueError("CalipyIO.dict has {} items, cannot use .value. Use"\
                    " bracket notation or iterate instead.".format(len(self.dict)))
            
        value = self.calipy_list[0].value
        return value
        
    @property
    def has_single_item(self) -> bool:
        """
        :return: True if exactly one item is in this io, else False.
        """
        return len(self) == 1
    
    def rename_keys(self, rename_dict):
        """ 
        Renames current keys in all the dicts to the ones given by rename_dict[key].
        
        :param rename_dict: Dictionary s.t. for each key in rename_dict, key is in
            self.calipy_list[k]keys() with rename_dict[key] being the string that
            is the key in the newly produced CalipyDict.
        :type rename_dict: dict
        :return: CalipyIO with the same values but with changed keys. 
        :rtype: CalipyIO
        """
        list_renamed_datatuples = [self.calipy_list[k].as_datatuple().rename_keys(rename_dict)
                                   for k in range(len(self))]
        return CalipyIO(list_renamed_datatuples)
    
    def reduce_list(self):
        """
        Attempts to merge all CalipyDict elements in self.calipy_list into a single
        CalipyDict by concatenating tensors along the first dimension. This method 
        succeeds only if all CalipyDict elements have exactly matching keys and 
        tensor dimensions (excluding the first dimension).
        """
        
        if len(self.calipy_list) <= 1:
            # Nothing to reduce
            return self
    
        # Check compatibility
        first_dict = self.calipy_list[0]
        keys = first_dict.keys()
    
        for d in self.calipy_list:
            if set(d.keys()) != set(keys):
                raise ValueError("Cannot reduce CalipyIO: keys mismatch across CalipyDicts.")
    
        # Concatenate tensors along the first dimension
        reduced_dict = {}
        for key in keys:
            tensors_to_concat = [d[key] for d in self.calipy_list]
            
            # Check shape compatibility (except first dimension)
            tensor_shapes = [t.shape[1:] for t in tensors_to_concat]
            if len(set(tensor_shapes)) > 1:
                raise ValueError(f"Incompatible shapes for key '{key}': {tensor_shapes}")
    
            # Collate tensors
            reduced_tensor = calipy_cat(tensors_to_concat, dim=0)
            reduced_dict[key] = reduced_tensor
        
        return CalipyIO(reduced_dict)
    
    # Functionality for when only one dict is present
    def as_datatuple(self) -> DataTuple:
        """
        Convert this CalipyDict into a DataTuple for dimension-aware operations
        or other advanced uses.
        """
        
        return self.dict.as_datatuple()

    def preprocess_for_node(self, nodestructure):
        pass
    
    
    def _indexer_construct(self, io_dims, name, silent = True):
        """Constructs a TensorIndexer for the tensor."""
        self.indexer = IOIndexer(self.calipy_list, io_dims, name)
        return self if silent == False else None

    def __getitem__(self, index):
        """ Returns new CalipyIO based on either standard indexing quantities
        like slices or integers or a CalipyIndex. 
        
        :param index: Identifier for determining which elements of self to compile
            to a new CalipyIO
        :type index: Integer, tuple of ints,  slice, CalipyIndex
        :return: A new datacontainer with same functionality as self
        :rtype: CalipyIO
        """
        
        # If Null object, return Null object
        if self.is_null:
            return self
        
        # Case 1: Standard indexing with ints return the corresponding list element
        # if type(index) in (int):
        #     # Return list element, the output is not calipy_io
                    
        #     # Create new CalipyIO by subsampling
        #     list_element =self.calipy_list[index]
        #     return list_element
        
        # Case 1: Standard indexing
        if type(index) in (int, tuple, slice):
            old_index = ensure_tuple(index)
            
            # Preserve singleton dimensions for integer indexing
            index = []
            for dim, idx in enumerate(old_index):
                index.append(idx)
            index = ensure_tuple(index) # Added recently, check full compatibility
                    
            # Create new CalipyIO by subsampling
            sub_io = CalipyIO(self.calipy_list[index[0]])
            
            # Get indices corresponding to index
            mask = torch.zeros_like(torch.ones([len(self)]), dtype=torch.bool)
            mask[index] = True  # Apply the given index
            selected_indices = torch.nonzero(mask)
            
            # Reshape to proper indextensor
            indextensor_shape = [len(sub_io)] + [selected_indices.shape[1]]
            indextensor = selected_indices.view(*indextensor_shape)

            sub_io.indexer.create_global_index(subsample_indextensor = indextensor, 
                                                      data_source_name = self.name)
            
            return sub_io
        
        # Case 2: If index is CalipyIndex, use index.tensor for subsampling
        elif type(index) is CalipyIndex:
            if index.is_empty:
                sub_io = self
            else:
                index_list = index.tensor.view(-1).tolist()
                sub_io = CalipyIO(self.calipy_list[index_list], name = self.name)
        
        # Case 3: When index is string, try to access single dict
        elif type(index) is str:
            # Input check
            if len(self.calipy_list) != 1:
                raise ValueError("CalipyIO has {} items, cannot use dict-based "\
                                 "indexing. Use bracket notation or iterate instead."
                                 .format(len(self)))
            return self.dict[index]
        
        # Case 4: Passing None returns the full io
        elif type(index) is type(None) :
            return self
        
        # Case 5: Raise an error for unsupported types
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")
        
        return sub_io
        
    
    def __len__(self):
        return len(self.calipy_list)
    
    def __iter__(self):
        # Define iteration explicitly via indexing
        for idx in range(len(self)):
            yield self[idx]
    
    def __eq__(self, other):
        if isinstance(other, CalipyIO):
            return self.calipy_list == other.calipy_list
        return NotImplemented
    
    def __repr__(self):
        rep_string = self.calipy_list.__repr__()
        return f"CalipyIO({rep_string})"
    
    
    


# Build CalipyDataset class that allows DataLoader to batch over multiple dimensions

class CalipyDataset(Dataset):
    """
    CalipyDataset is a class mimicking the functionality of the Dataset class in
    torch.utils.data but providing some streamlined prebuilt functions needed
    in the context of calipy. This includes support for subsampling based on 
    CalipyDict objects. Is meant to be subclassed for augmenting user specified
    datasets with additional, calipy-ready functionality.    
    
    :param input_data: The input_data of the dataset reflecting the inputs to 
        the model that evoke the corresponding outputs. Can be:
          - None => No input data (no input)
          - CalipyTensor => Single tensor (single input)
          - CalipyDict => Dictionary containing CalipyTensors (multiple inputs)
          - CalipyIO => List containing CalipyDict containing CalipyTensors 
              (multiple inputs, possibly of inhomogeneous shape and type)
    :type input_data: NoneType, CalipyTensor, CalipyDict, CalipyIO
    
    :param output_data: The output_data of the dataset reflecting the outputs of 
        the model evoked by the corresponding inputs. Can be:
          - None => No output data (no output)
          - CalipyTensor => Single tensor (single output)
          - CalipyDict => Dictionary containing CalipyTensors (multiple outputs)
          - CalipyIO => List containing CalipyDict containing CalipyTensors 
              (multiple inputs, possibly of inhomogeneous shape and type)
    :type output_data: NoneType, CalipyTensor, CalipyDict, CalipyIO
    :param batch_dims: A DimTuple object defining the batch dimensions among 
        which flattening and subsampling is performed.
    :type batch_dims: DimTuple
    
    :return: An instance of CalipyDataset, suitable for accessing datasets and
        passing them to DataLoader objects.
    :rtype: CalipyDataset
    
    The following scenarios need to be evaluated are covered by the dataset 
    construction procedure:
        i) (Input, Ouptut) =  (None, CalipyTensor)
        ii) (Input, Ouptut) =  (Calipytensor, CalipyTensor)
        iii) (Input, Ouptut) =  (None, dict(CalipyTensor))
        iv) (Input, Output) = (dict(CalipyTensor), dict(CalipyTensor))
        v) (Input, Ouptut) =  (None, list(dict(CalipyTensor)))
        vi) (Input, Output) = (list(dict(CalipyTensor)), list(dict(CalipyTensor)))
        vii) (Input, Output) = (None, list_mixed)
        viii) (Input, Output) = (list_mixed, list_mixed)
        
    where list_mixed means a list of dicts with entries to keys sometimes being
    None or of nonmatching shapes.

    Example usage:

    .. code-block:: python
        

        # i) Imports and definitions
        import torch
        import pyro        
        from calipy.core.utils import dim_assignment
        from calipy.core.data import  CalipyDataset, io_collate
        from calipy.core.tensor import CalipyTensor
        from torch.utils.data import DataLoader
                
        # Definitions        
        n_meas = 2
        n_event = 1
        n_subbatch = 7
        
    
        # ii) Create data for dataset
        
        # Set up sample distributions
        mu_true = torch.tensor(0.0)
        sigma_true = torch.tensor(0.1)
        
        # Sample from distributions & wrap result
        data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
        data = data_distribution.sample([n_meas, n_event])
        data_dims = dim_assignment(['bd_data', 'ed_data'], dim_sizes = [n_meas, n_event])
        data_cp = CalipyTensor(data, data_dims, name = 'data')
        
        # dataset_inputs
        data_none = None
        data_ct = data_cp
        data_cd = {'a': data_cp, 'b' : data_cp}
        data_io = [data_cd, data_cd]
        data_io_mixed = [data_cd, {'a' : None, 'b' : data_cp} , {'a': data_cp, 'b':None}, data_cd]
        
        
        # iii) Build datasets
        
        # Build datasets and check
        dataset_none_none = CalipyDataset(input_data = data_none, output_data = data_none)
        dataset_none_ct = CalipyDataset(input_data = data_none, output_data = data_ct)
        dataset_none_cd = CalipyDataset(input_data = data_none, output_data = data_cd)
        dataset_none_io = CalipyDataset(input_data = data_none, output_data = data_io)
        dataset_none_iomixed = CalipyDataset(input_data = data_none, output_data = data_io_mixed)
        
        dataset_ct_ct = CalipyDataset(input_data = data_ct, output_data = data_ct)
        dataset_ct_cd = CalipyDataset(input_data = data_ct, output_data = data_cd)
        dataset_ct_io = CalipyDataset(input_data = data_ct, output_data = data_io)
        dataset_ct_iomixed = CalipyDataset(input_data = data_ct, output_data = data_io_mixed)
        
        dataset_cd_ct = CalipyDataset(input_data = data_cd, output_data = data_ct)
        dataset_cd_cd = CalipyDataset(input_data = data_cd, output_data = data_cd)
        dataset_cd_io = CalipyDataset(input_data = data_cd, output_data = data_io)
        dataset_cd_iomixed = CalipyDataset(input_data = data_cd, output_data = data_io_mixed)
        
        dataset_io_ct = CalipyDataset(input_data = data_io, output_data = data_ct)
        dataset_io_cd = CalipyDataset(input_data = data_io, output_data = data_cd)
        dataset_io_io = CalipyDataset(input_data = data_io, output_data = data_io)
        dataset_io_iomixed = CalipyDataset(input_data = data_io, output_data = data_io_mixed)
        
        dataset_iomixed_ct = CalipyDataset(input_data = data_io_mixed, output_data = data_ct)
        dataset_iomixed_cd = CalipyDataset(input_data = data_io_mixed, output_data = data_cd)
        dataset_iomixed_io = CalipyDataset(input_data = data_io_mixed, output_data = data_io)
        dataset_iomixed_iomixed = CalipyDataset(input_data = data_io_mixed, output_data = data_io_mixed)
        
        
        # iv) Build dataloader and subsample
        
        dataset = CalipyDataset(input_data = [None, data_ct, data_cd],
                                output_data = [None, data_ct, data_cd] )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=io_collate)
        
        # Iterate through the DataLoader
        for batch_input, batch_output, batch_index in dataloader:
            print(batch_input, batch_output, batch_index)
        
            
    """
    def __init__(self, input_data, output_data, homogeneous = False):
        
        # Preprocess I/O data
        self.input_type = type(input_data)
        self.output_type = type(output_data)
        self.input_data = CalipyIO(input_data)
        self.output_data = CalipyIO(output_data)
        self.homogeneous = homogeneous
        self.data = {'input_data' : input_data, 'output_data' :  output_data}
        
        # Error diagnostics
        self.valid_dataset_types = [CalipyTensor, CalipyDict, CalipyIO, type(None), list, dict]
        if self.input_type not in self.valid_dataset_types :
            raise(Exception('input_type must be one of {}; is {}'.format(self.valid_dataset_types, self.input_type)))
        if self.output_type not in self.valid_dataset_types :
            raise(Exception('output_type must be one of {}; is {}'.format(self.valid_dataset_types, self.output_type)))
            
    
    def infer_length(self, query_data):
        # Infer the batch length of input_data or output_data
        len_data = {key: query_data[key].shape[0] if query_data[key] is not None else None 
                    for key in query_data.keys()}
        return len_data
                    

    def __len__(self):
        # return the length of the dataset, i.e. the number of CalipyIO list entries
        return  max(len(self.input_data.calipy_list), len(self.output_data.calipy_list))
        

    def __getitem__(self, idx):
        # Handle the case where idx is a single integer
        input_data_idx = self.input_data[idx] if self.input_data is not None else None
        output_data_idx = self.output_data[idx] if self.output_data is not None else None
        data_dict = {'input' : CalipyIO(input_data_idx), 'output' : CalipyIO(output_data_idx),
                     'index' :  CalipyIO(torch.tensor([idx]))}
        
        return data_dict
    
    # def __repr__(self):
    #     input_rep = 
    #     output_rep = 
    #     len_rep = len(self)
    #     rep_string = "\n len : {len_rep}, \n input : {input_rep}, \n output : {output_rep}"
    #     return f"CalipyDataset({rep_string})"
        
    def __repr__(self):
        # Recursively format objects with neat indentation
        def short_repr(obj, indent=0, max_items=3):
            pad = '  ' * indent
    
            if isinstance(obj, (list, CalipyList)):
                n = len(obj)
                if n == 0:
                    return '[]'
                if n <= max_items:
                    elems = ',\n'.join(short_repr(el, indent + 1) for el in obj)
                else:
                    elems = ',\n'.join(short_repr(el, indent + 1) for el in obj[:2])
                    elems += f',\n{pad}  ...\n' + short_repr(obj[-1], indent + 1)
                return '[\n' + elems + f'\n{pad}]'
    
            elif isinstance(obj, (dict, CalipyDict)):
                items = []
                for idx, (k, v) in enumerate(obj.items()):
                    v_repr = short_repr(v, indent + 1)
                    key_repr = f"{pad}  {k}: {v_repr}" if idx > 0 else f"{k}: {v_repr}"
                    items.append(key_repr)
                return '    { ' + ', \n'.join(items) + ' }'
    
            elif isinstance(obj, torch.Tensor):
                return f'torch.Tensor(shape={list(obj.shape)})'
            elif isinstance(obj,  CalipyTensor):
                return f'CalipyTensor(shape={list(obj.shape)})'
    
            elif obj is None:
                return 'None'
    
            else:
                return type(obj).__name__
    
        input_rep = short_repr(self.input_data.calipy_list, indent=2)
        output_rep = short_repr(self.output_data.calipy_list, indent=2)
        len_rep = len(self)
    
        rep_string = (
            f"CalipyDataset(\n"
            f"  len: {len_rep},\n \n"
            f"  input: {input_rep},\n \n"
            f"  output: {output_rep}\n"
            f")"
        )
    
        return rep_string

# Custom collate function that collates ios together by concatenating contained
# list elements into a longer list aroung which a new new CalipyIO object is built.
# This new CalipyIO object contains a list of dicts.
# If reduce = True, list elements are aimed to be stacked themselves (e.g. tensors
# along their first dimensions) to create a single dict containing stacked elements.

def io_collate(batch, reduce = False):
    """ Custom collate function that collates ios together by concatenating contained
    list elements into a longer list aroung which a new new CalipyIO object is built.
    This new CalipyIO object contains a list of dicts.
    If reduce = True, list elements are aimed to be stacked themselves (e.g. tensors
                                                                        along their first dimensions) to create a single dict containing stacked elements.
    Used primarily as collate function for the DataLoader
    to perform automatized subsampling.
    :param batch: A list of CalipyDiIO containing info on input_vars, observations, and
        corresponding index that was used to produce them via dataset.__getitem__[idx]
    :type batch: list of CalipyIO
        
    :return: An instance of CalipyIO, where multiple CalipyDict objects are 
        collated together either into a list of dicts or into a single calipy_io
        containing stacked CalipyTensors.
    :rtype: CalipyIO
    """
    
    
        
    # If reduce = False, just concatenate the lists inside the IO's
    if reduce  == False:
        # Untangle batch list
        list_of_dicts = batch
        inputs_io = CalipyIO([batch_dict['input'].dict for batch_dict in list_of_dicts])
        outputs_io = CalipyIO([batch_dict['output'].dict for batch_dict in list_of_dicts] )
        indices_io = CalipyIO([batch_dict['index'].dict for batch_dict in list_of_dicts])        
    
    
    
    # If reduce = True, stack contained tensors along first dimension
    if reduce  == True:
    
        # Check input signatures
        bool_reduction_inputs = inputs_io.is_reducible()
        bool_reduction_outputs = outputs_io.is_reducible()
        bool_reduction_indices = indices_io.is_reducible()
        bool_reduction_data = bool_reduction_inputs*bool_reduction_outputs*bool_reduction_indices
        
        # compatibility reduce argument
        if bool_reduction_data == False:
            raise Exception("Attempting to reduce dataset io's to single dicts but not " \
                            "all of the are reducible. Reducibility: inputs: {}, outputs : {}, indices : {}"
                            .format(bool_reduction_inputs, bool_reduction_outputs, bool_reduction_indices) )
            
        # Construct new dictionaries
        output_dict = {}
        output_keys = outputs_io[0].dict.keys()
        for key in output_keys:
            tensors_to_concat = [d[key] for d in outputs_io]
            output_dict[key] = calipy_cat(tensors_to_concat, dim = 0)
            
        input_dict = {}
        input_keys = inputs_io[0].dict.keys()
        for key in input_keys:
            tensors_to_concat = [d[key] for d in inputs_io]
            input_dict[key] = calipy_cat(tensors_to_concat, dim = 0)
            
        
        flattened_batch_dim = list_of_indices[0][0]
        index_dim = dim_assignment(['index_dim'])
        indices_to_concat = torch.cat([d[1] for d in list_of_indices], dim = 0).reshape([-1,1])
        ssi = CalipyIndex(indices_to_concat, flattened_batch_dim + index_dim, name = 'subsample_index')
        
    
    return  inputs_io, outputs_io, indices_io


    

def preprocess_args(input_vars, observations, subsample_index):
    """ Function for preprocessing arguments to forward passes. Converts different
    forms of input to CalipyIO objects reflecting a standardized form of inputs
    and outputs. Typically just wraps input into CalipyIO.

    :param input_vars: Input input_vars to some .forward() method call. Specific
        contents depend on the node but typically None or a dict containing CalipyTensors
        with keys as specified in the nodes' input_vars_schema
    :type input_vars: None, single object, Dict, CalipyDict, list, CalipyList, 
        or CalipyIO containing CalipyTensors.
    :param observation:  Input observation to some .forward() method call. Specific
        contents depend on the node but typically a dict containing CalipyTensors
        with keys as specified in the nodes' observation_schema
    :type observations: None, single object, Dict, CalipyDict, list, CalipyList,
        or CalipyIO containing CalipyTensors.
    :param sbsample_index: Input subsample_index to some .forward() method call. Specific
        contents depend on the node but typically None if no subsampling happens
        or of type Dict containing CalipyIndex objects in case of subsampling. 
        The keys are as specified in the nodes' subsampling_schema
    :type subsample_index: None, single object, Dict, CalipyDict, list, CalipyList,
    
    :return: A tuple containing instances of CalipyIO that represent input_vars,
        observations, subsample_index in a way that forward methods can handle
        them well and they are easily passable between nodes.
    :rtype: tuple of CalipyIO

    Example usage:

    .. code-block:: python
        
        # Imports and definitions
        import torch
        from calipy.core.data import DataTuple, CalipyDict, CalipyList, CalipyIO
        from calipy.core.tensor import CalipyTensor
    """
    
    input_vars_io = CalipyIO(data = input_vars, name = 'input_vars_preprocessed')
    observations_io = CalipyIO(data = observations, name = 'observations_preprocessed')
    subsample_index_io = CalipyIO(data = subsample_index, name = 'subsample_index_preprocessed')
    
    return input_vars_io, observations_io, subsample_index_io
    
    
    

    
# ORIGINAL MULTIFUNCTIONAL DATASET VERSIONS


# # Build CalipyDataset class that allows DataLoader to batch over multiple dimensions

# class CalipyDataset(Dataset):
#     """
#     CalipyDataset is a class mimicking the functionality of the Dataset class in
#     torch.utils.data but providing some streamlined prebuilt functions needed
#     in the context of calipy. This includes support for subsampling based on 
#     CalipyDict objects. Is meant to be subclassed for augmenting user specified
#     datasets with additional, calipy-ready functionality.    
    
#     :param input_data: The input_data of the dataset reflecting the inputs to 
#         the model that evoke the corresponding outputs. Can be:
#           - None => No input data (no input)
#           - CalipyTensor => Single tensor (single input)
#           - CalipyDict => Dictionary containing CalipyTensors (multiple inputs)
#           - CalipyIO => List containing CalipyDict containing CalipyTensors 
#               (multiple inputs, possibly of inhomogeneous shape and type)
#     :type input_data: NoneType, CalipyTensor, CalipyDict, CalipyIO
    
#     :param output_data: The output_data of the dataset reflecting the outputs of 
#         the model evoked by the corresponding inputs. Can be:
#           - None => No output data (no output)
#           - CalipyTensor => Single tensor (single output)
#           - CalipyDict => Dictionary containing CalipyTensors (multiple outputs)
#           - CalipyIO => List containing CalipyDict containing CalipyTensors 
#               (multiple inputs, possibly of inhomogeneous shape and type)
#     :type output_data: NoneType, CalipyTensor, CalipyDict, CalipyIO
#     :param batch_dims: A DimTuple object defining the batch dimensions among 
#         which flattening and subsampling is performed.
#     :type batch_dims: DimTuple
    
#     :return: An instance of CalipyDataset, suitable for accessing datasets and
#         passing them to DataLoader objects.
#     :rtype: CalipyDataset

#     Example usage:

#     .. code-block:: python
        
#         # Imports and definitions
#         import torch
#         from calipy.core.data import CalipyDataset
           

#         # Create data for CalipyDict initialization

    
#     """
#     def __init__(self, input_data, output_data, batch_dims, homogeneous = False):
        
#         # Preprocess I/O data
#         self.batch_dims = batch_dims
#         self.input_type = type(input_data)
#         self.output_type = type(output_data)
#         self.input_data = CalipyIO(input_data)
#         self.output_data = CalipyIO(output_data)
#         self.homogeneous = homogeneous
#         self.data = {'input_data' : input_data, 'output_data' :  output_data}
        
#         # Error diagnostics
#         self.valid_dataset_types = [CalipyTensor, CalipyDict, CalipyIO, type(None)]
#         if self.input_type not in self.valid_dataset_types :
#             raise(Exception('input_type must be one of {}; is {}'.format(self.valid_dataset_types, self.input_type)))
#         if self.output_type not in self.valid_dataset_types :
#             raise(Exception('output_type must be one of {}; is {}'.format(self.valid_dataset_types, self.output_type)))
        
#         # # Lengths and flattened data
#         # self.flattened_input = self.flatten(self.input_data)
#         # self.flattened_output = self.flatten(self.output_data)
#         # self.len_input_data = self.infer_length(self.flattened_input)
#         # self.len_output_data = self.infer_length(self.flattened_output)
    
    
#     def infer_length(self, query_data):
#         # Infer the batch length of input_data or output_data
#         len_data = {key: query_data[key].shape[0] if query_data[key] is not None else None 
#                     for key in query_data.keys()}
#         return len_data
        
#     def flatten(self, io_data):
#         # This function returns a CalipyIO of tensors, where the first dimension
#         # is the (only) batch dimension and for each key in calipy_dict.keys(),
#         # CalipyDict[key][k, ...] is the kth datapoint in the dataset. The input
#         # arg io_data is a CalipyDict of CalipyTensors.
        
#         # i) Do the flattening
#         data_flattened = {}
#         for key, value in io_data.items():
#             data_flattened[key] = value.flatten(self.batch_dims, 'batch_dim_flattened') if value is not None else None
#         self.batch_dim_flattened = dim_assignment(['batch_dim_flattened'])
        
#         # ii) Check if flattening consistent
#         batch_dims_sizes = {key : value.dims[['batch_dim_flattened']].sizes if value is not None else [] 
#                              for key, value in data_flattened.items()}
#         first_dim_size = batch_dims_sizes[list(batch_dims_sizes.keys())[0]]
#         if not all([dim_sizes == first_dim_size for key, dim_sizes in batch_dims_sizes.items()]):
#             raise(Exception('For flattening, all DimTuples batch_dims must be of ' \
#                             'same size for all keys but are {} for keys {}'.format(batch_dims_sizes,list(batch_dims_sizes.keys()))))
                

#     def __len__(self):
#         # return the length of the dataset, i.e. the number of independent event samples
#         # return self.batch_length_total
#         return  self.batch_length[0]
        

#     def __getitem__(self, idx):
#         # Handle the case where idx is a single integer
#         bd_flat = self.batch_dim_flattened
#         input_data_idx = self.flattened_input.subsample_tensors(bd_flat, [idx]) if self.flattened_input is not None else None
#         output_data_idx = self.flattened_output.subsample_tensors(bd_flat, [idx]) if self.flattened_output is not None else None
#         data_dict = {'input' : CalipyIO(input_data_idx), 'output' : CalipyIO(output_data_idx),
#                      'index' : (bd_flat, torch.tensor([idx]))}
        
#         return data_dict



# # Custom collate function that collates dicts together by stacking contained 
# # tensors along their batch dimension (which is assumed to have the name of
# # 'batch_dim_flattened').

# def io_collate(batch, reduce = False):
#     """ Custom collate function that collates ios together by stacking contained 
#     tensors along their batch dimension (which is assumed to have the name of
#     'batch_dim_flattened'). Used primarily as collate function for the DataLoader
#     to perform automatized subsampling.
#     :param batch: A list of CalipyDict containing info on input_vars, observations, and
#         corresponding index that was used to produce them via dataset.__getitem__[idx]
#     :type batch: list of CalipyDict
        
#     :return: An instance of CalipyDict, where multiple CalipyDict objects are 
#         collated together into a calipy_dict containing stacked CalipyTensors.
#     :rtype: CalipyDict
#     """
    
#     # Untangle batch list
#     list_of_dicts = batch
#     inputs_io = CalipyIO([batch_dict['input'] for batch_dict in list_of_dicts])
#     outputs_io = CalipyIO([batch_dict['output'] for batch_dict in list_of_dicts] )
#     indices_io = CalipyIO([batch_dict['index'] for batch_dict in list_of_dicts])
    
#     # Check input signatures
#     bool_reduction_inputs = inputs_io.is_reducible()
#     bool_reduction_outputs = outputs_io.is_reducible()
#     bool_reduction_indices = indices_io.is_reducible()
#     bool_reduction_data = bool_reduction_inputs*bool_reduction_outputs*bool_reduction_indices
    
#     # compatibility reduce argument
#     if reduce == True and bool_reduction_data == False:
#         raise Exception("Attempting to reduce dataset io's to single dicts but not " \
#                         "all of the are reducible. Reducibility: inputs: {}, outputs : {}, indices : {}"
#                         .format(bool_reduction_inputs, bool_reduction_outputs, bool_reduction_indices) )
    
#     # If reduce = False, just concatenate the lists inside the IO's
#     if reduce  == False:
#         pass
        
#     # Construct new dictionaries
#     output_dict = {}
#     output_keys = list_of_outputs[0].keys()
#     for key in output_keys:
#         tensors_to_concat = [d[key] for d in list_of_outputs]
#         output_dict[key] = calipy_cat(tensors_to_concat, dim = 0)
        
#     input_dict = {}
#     input_keys = list_of_inputs[0].keys()
#     for key in input_keys:
#         tensors_to_concat = [d[key] for d in list_of_inputs]
#         input_dict[key] = calipy_cat(tensors_to_concat, dim = 0)
        
    
#     flattened_batch_dim = list_of_indices[0][0]
#     index_dim = dim_assignment(['index_dim'])
#     indices_to_concat = torch.cat([d[1] for d in list_of_indices], dim = 0).reshape([-1,1])
#     ssi = CalipyIndex(indices_to_concat, flattened_batch_dim + index_dim, name = 'subsample_index')
    
    
#     return  input_dict, output_dict, ssi
    
    
    
