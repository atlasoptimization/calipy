#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides support functionality related to logging, data organization,
preprocessing and other functions not directly related to calipy core domains.

The classes are
    CalipyRegistry: Dictionary type class that is used for tracking identity
        and uniqueness of objects created during a run and outputs warnings.

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

import torch
import pyro
import fnmatch
import contextlib
import networkx as nx
import matplotlib.pyplot as plt
import copy
import re
from functorch.dim import dims
import varname

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type
from collections.abc import Iterable


"""
    CalipyRegistry class
"""


# class CalipyRegistry:
#     def __init__(self):
#         self.registry = {}

#     def register(self, key, value):
#         if key in self.registry:
#             print(f"Warning: An item with the key '{key}' already exists.")
#         self.registry[key] = value
#         print(f"Item with key '{key}' has been registered.")

#     def get(self, key):
#         return self.registry.get(key, None)

#     def remove(self, key):
#         if key in self.registry:
#             del self.registry[key]
#             print(f"Item with key '{key}' has been removed.")
#         else:
#             print(f"Item with key '{key}' not found in registry.")

#     def clear(self):
#         self.registry.clear()
#         print("Registry has been cleared.")

#     def list_items(self):
#         for key, value in self.registry.items():
#             print(f"{key}: {value}")
            
            

"""
    Support functions
"""

def multi_unsqueeze(input_tensor, dims):
    output_tensor = input_tensor
    for dim in sorted(dims):
        output_tensor = output_tensor.unsqueeze(dim)
    return output_tensor

def robust_meshgrid(tensors, indexing = 'ij'):
    """ Ensures that meshgrid also works for empty inputs [] of sizes."""
    if tensors == []:
        output = (torch.tensor([]),)
        # tensors = [torch.tensor([0])]
    else:
        output = torch.meshgrid(*tensors, indexing = indexing)
    return output

# def ensure_tuple(item):
#     """Ensures the input is a tuple. Leaves tuples unchanged."""
#     return item if isinstance(item, tuple) else tuple(item)
def ensure_tuple(item):
    """Ensures the input is a tuple. Leaves tuples unchanged. Wraps non-iterables into a tuple."""
    if isinstance(item, tuple):
        return item
    elif isinstance(item, Iterable):
        return tuple(item)
    else:
        return (item,)

def format_mro(cls):
    # Get the MRO tuple and extract class names
    mro_names = [cls.__name__ for cls in cls.__mro__ if cls.__name__ not in ('object', 'ABC')]
    # Reverse the list to start from 'object' and move up to the most derived class
    mro_names.reverse()
    # Join the class names with underscores
    formatted_mro = '__'.join(mro_names)
    return formatted_mro

def get_params(name):
    pattern = "*__param_{}".format(name)
    matched_params = {name: value for name, value in pyro.get_param_store().items() if fnmatch.fnmatch(name, pattern)}
    return matched_params


@contextlib.contextmanager
def context_plate_stack(plate_stack):
    """
    Context manager to handle multiple nested pyro.plate contexts.
    
    Args:
    plate_stack (list): List where values are instances of pyro.plate.
    
    Yields:
    The combined context of all provided plates.
    """
    with contextlib.ExitStack() as stack:
        # Enter all plate contexts
        for plate in plate_stack:
            stack.enter_context(plate)
        yield  # Yield control back to the with-block calling this context manager
        
# Functions for dimension declaration per list

# restricted exec function
def restricted_exec(exec_string, allowed_locals):
    # Allow only simple assignment of `dims` using a regular expression
    if not re.match(r'^\w+\s*=\s*dims\(sizes=\[\d+(,\s*\d+)*\]\)$', exec_string) \
        and not re.match(r'^\w+\s*=\s*dims\(\)$', exec_string):
        raise ValueError("Invalid exec command")
    
    # Execute the command in a very limited scope
    allowed_globals = {"dims": dims}
    exec(exec_string, allowed_globals, allowed_locals)

# Safe eval function
def safe_eval(expr, allowed_globals=None, allowed_locals=None):
    if allowed_globals is None:
        allowed_globals = {}
    if allowed_locals is None:
        allowed_locals = {}
    return eval(expr, {"__builtins__": None, **allowed_globals}, allowed_locals)



# Classes for function documentation and type hinting

@dataclass
class InputSchema:
    required_keys: List[str]
    optional_keys: List[str] = None
    defaults: Dict[str, Any] = None
    key_types: Dict[str, Type] = None  # Maps keys to their types

    def __post_init__(self):
        self.optional_keys = self.optional_keys or []
        self.defaults = self.defaults or {}
        self.key_types = self.key_types or {}
        
    def __repr__(self):
        indent = "    "  # 4 spaces for clean indentation

        required_keys_str = "\n".join(f"{indent}- {key}" for key in self.required_keys) if self.required_keys else f"{indent}None"
        optional_keys_str = "\n".join(f"{indent}- {key}" for key in self.optional_keys) if self.optional_keys else f"{indent}None"
        defaults_str = "\n".join(f"{indent}- {k}: {v}" for k, v in self.defaults.items()) if self.defaults else f"{indent}None"
        key_types_str = "\n".join(f"{indent}- {k}: {v}" for k, v in self.key_types.items()) if self.key_types else f"{indent}None"

        return (
            f"InputSchema:\n"
            f"  Required Keys:\n{required_keys_str}\n"
            f"  Optional Keys:\n{optional_keys_str}\n"
            f"  Defaults:\n{defaults_str}\n"
            f"  Key Types:\n{key_types_str}"
        )
        

def check_schema(calipy_dict_obj, required_keys=None, optional_keys=None):
    """
    Example schema validation function:
      - required_keys: list of keys that must exist
      - optional_keys: list of recognized but optional keys
    Raises ValueError if some required keys are missing.
    """
    required_keys = required_keys or []
    optional_keys = optional_keys or []

    missing = []
    for key in required_keys:
        if key not in calipy_dict_obj:
            missing.append(key)
    if missing:
        raise ValueError(f"Missing required keys: {missing} in data {calipy_dict_obj}")

    return True



# CalipyDim and Dimension management

class CalipyDim:
    """ CalipyDim class contains information useful to manage dimensions and is
    the prime ingredient to DimTuple class which implements arithmentics on 
    dimensions. When initialized, it represents a dim primarily as a name and
    attaches a size to it - either a nonnegative integer number or None which
    represents the size of the dim bein undefined. Furthermore, a description
    of the dim can be provided.
    CalipyDim objects can be bound to tensors by their as_torchdim attribute 
    which converts accesses a representation in terms of functorch.dim Dim objects
    that allows indexing of tensors.
    
    :param name: A string representing the name by which the dimension is to be identified 
    :type name: str
    :param size: A nonnegative integer or None representing the size of this dimension; 
        None indicates an unbound dimension; a value of 0 indicates an empty dimension
    :type size: int or None
    :param description: A description describing this dimension; 
        None indicates absence of description.
    :type description: str or None
        
    :return: A CalipyDim object containing names, size, description of a dimension
    :rtype: CalipyDim

    Example usage:

    .. code-block:: python
        
        # Single dimension properties
        bd_1 = CalipyDim('bd_1', size = 5, description = 'batch dimension 1')
        bd_2 = CalipyDim('bd_2', size = None, description = 'batch dimension 2')
        A = torch.normal(0, 1, [5,3])
        A[bd_1.torchdim, bd_2.torchdim]
        
        # Typical use case with dim_assignment
        dim_names = ['d1', 'd2']
        dim_sizes = [10, 5]
        dim_tuple = dim_assignment(dim_names, dim_sizes)
        dim_tuple[0]    # Is CalipyDim d1

    """
    def __init__(self, name, size = None, description = None):
        self.name = name  # Unique identifier for the dimension
        self.size = size
        self.description = description
        # self.torchdim = self.convert_to_torchdim()

    @property
    def is_bound(self):
        if self.size is not None:
            bool_val = True
        else:
            bool_val = False
        return bool_val

    def build_torchdim(self, fix_size = False):
        """ Create a functorch.dim object that can e used for indexing by calling
        the functorch.dim.dims function.
        
        :param fix_size: Determines if the torchdims are initialized with fixed size
        :type fix_size: Boolean
        :return: A single functorch.dim.Dim with fixed or variable size
        :rtype: functorch.dim.Dim object
        
        """
        # Execute dimension initialization
        dims_locals = {}
        exec_string = f"{self.name} = dims()"
        
        if fix_size == True:
            if self.size is not None:
                exec_string = f"{self.name} = dims(sizes=[{self.size}])"
            else:
                exec_string = f"{self.name} = dims()"
        
        restricted_exec(exec_string, dims_locals)
        
        # Compile return
        eval_string = f"{self.name}"
        return_dim = safe_eval(eval_string, allowed_locals = dims_locals)
        return return_dim


    def __repr__(self):
        return 'Dim ' + self.name

    def __eq__(self, other):
        return isinstance(other, CalipyDim) and self.name == other.name

    def __hash__(self):
        return hash(self.name)




def dim_assignment(dim_names, dim_sizes=None, dim_descriptions=None):
    """
    dim_assignment dynamically assigns dimension objects to names and returns them as a DimTuple.

    This function creates `DimTuple` objects using the specified sizees in `dim_sizes` and assigns them to the
    names provided in `dim_names`. The function validates that the dimension sizes are positive integers
    or None (for unbound dimensions) and that the dimension names are valid Python identifiers. If only
    one name is provided with multiple shapes, the name is extended by indices (e.g., 'batch' -> 'batch_1',
    'batch_2', etc.). The function then returns a DimTuple of the created `CalipyDim` objects.

    :param dim_names: A list of strings representing the variable names to assign to each dimension. 
        These names must be valid Python identifiers. If only one name is provided and multiple shapes,
        the name will be broadcast with indices (e.g., ['batch'] -> ['batch_1', 'batch_2', ...]).
    :type dim_names: list of str
    :param dim_sizes: A list of nonnegative integers or None representing the sizes of each dimension; 
        None indicates an unbound dimension; a value of 0 indicates an empty dimension
    :type dim_sizes: list of int or None
    :param dim_descriptions: A list of descriptions describing each dimension; 
        None indicates absence of descriptions.
    :type dim_description: list of str or None
        
    :return: A DimTuple containing the `CalipyDim` objects assigned to the names in `dim_names`.
    :rtype: DimTuple

    Example usage:

    .. code-block:: python

        dim_names = ['batch_dim_1', 'batch_dim_2']
        dim_sizes = [10, 5]
        dim_tuple = dim_assignment(dim_names, dim_sizes)

        # Access the dimensions
        print(dim_tuple)  # Outputs: (batch_dim_1, batch_dim_2)
        print(dim_tuple[0].size)  # Outputs: 10
        print(dim_tuple[1].size)  # Outputs: 5
        
        # Example with broadcasting
        dim_tuple = dim_assignment(dim_names=['batch'], dim_sizes=[5, 2])
        print(dim_tuple)  # Outputs: (batch_1, batch_2)
        print(dim_tuple.sizes)  # Outputs: [5,2]
        
        # Example with bound and unbound dims
        dim_tuple = dim_assignment(dim_names=['batch_dim_1', 'batch_dim_2'], dim_sizes=[5, None])
        dim_tuple.sizes
        dim_tuple.filter_bound()
        dim_tuple.filter_unbound()
        
        # Example with a dimension skipped
        dim_tuple = dim_assignment(dim_names=['a'], dim_sizes=[0])
        print(dim_tuple)  # Outputs: DimTuple(())
    """
        
    # Broadcasting over dim_names if needed
    if len(dim_names) == 1 and dim_sizes is not None and len(dim_sizes) > 1:
        base_name = dim_names[0]
        dim_names = [f"{base_name}_{i+1}" for i in range(len(dim_sizes))]

    # Validate inputs    
    if not all(isinstance(name, str) and name.isidentifier() for name in dim_names):
        raise ValueError("All dimension names must be valid Python identifiers.")
    if dim_sizes is not None:
        if not all(size is None or (isinstance(size, int) and size >= 0) for size in dim_sizes):
            raise ValueError("All dimension sizes must be nonnegative integers or None.")
    
    # Broadcast Nones
    if dim_descriptions is None:
        dim_descriptions = [None] * len(dim_names)
    if dim_sizes is None:
        dim_sizes = [None] * len(dim_names)
    
    # Create and list Dims
    dim_list = []
    for name, size, description in zip(dim_names, dim_sizes, dim_descriptions):
        if size == 0:
            continue
        else:
         dim_list.append(CalipyDim(name, size, description))               
    dim_tuple = DimTuple(tuple(dim_list))
    
    return dim_tuple



# Generate trivial dimensions
def generate_trivial_dims(ndim):
    dim_names = ["trivial_dim_{}".format(k) for k in range(ndim)]
    trivial_dims = dim_assignment(dim_names, dim_sizes = [1 for name in dim_names])
    return trivial_dims
    
class TorchdimTuple(tuple):
    """ TorchdimTuple is a subclass of the Tuple class that allows esy handling
    of tuples build from functorchdim.dim.Dim objects. These tuples occur in the
    DimTuple class, which is the main class to represent dimensions.   
    
    :param input_tuple: A tuple of dimensions to be managed by TorchdimTuple.
    :type input_tuple: tuple of functorch.dim.Dim objects
    :param superior_dims: DimTuple object containing CalipyDim objects providing
        further info on the torchdims in the TorchdimTuple object.
    :type superior_dims: DimTuple object 
    
    :return: An instance of TorchdimTuple containing the dimension objects.
    :rtype: TorchdimTuple

    Example usage:

    .. code-block:: python

        # Create dimensions
        (bd,ed) = dims(2)
        torchdim_tuple = TorchdimTuple((bd,ed))
        torchdim_tuple.sizes
        
        # Bind dimensions
        A = torch.normal(0,1,[5,3])
        A_named = A[torchdim_tuple]
        torchdim_tuple.sizes
        
        # When being built from DimTuple, inherit info
        batch_dims = dim_assignment(dim_names=['bd_1', 'bd_2'], dim_sizes=[5, None])
        event_dims = dim_assignment(dim_names=['ed_1'])
        full_dims = batch_dims + event_dims
        full_torchdims = full_dims.build_torchdims()
        full_torchdims.sizes
        full_torchdims.names
        
        # Also allow for string-based and dim-based indexing
        full_torchdims[0]
        full_torchdims[['bd_1']]
        full_torchdims[batch_dims.names]
        full_torchdims[batch_dims]
        
        
    """
    
    def __new__(cls, input_tuple, superior_dims = None):
        obj = super(TorchdimTuple, cls).__new__(cls, input_tuple)
        obj.superior_dims = superior_dims
        if superior_dims is not None:
            obj.names = [d.name for d in superior_dims]
            obj_sizes = [d.size for d in superior_dims]
            obj.descriptions = [d.description for d in superior_dims]
        return obj
        
    @property
    def sizes(self):
        """ Returns a list of sizes for each of the dims in TorchdimTuple. 
        """
        sizes = []
        for d in self:
            try:
                sizes.append(d.size)  # Attempt to get the size
            except ValueError:
                sizes.append(None)  # Append None if the dimension is unbound
        return sizes
    
    def delete_dims(self, dim_keys):
        """ Computes a TorchdimTuple object reduced_dim_tuple with the dims referenced
        in dim_keys deleted from reduced_dim_tuple. Consequently, reduced_dim_tuple
        contains only those dims which are not mentioned in dim_keys.
        
        :param dim_keys: Identifier for determining which dimensions to select
        :type dim_names: DimTuple
        :return: A TorchdimTuple object with the selected dimensions removed
        :rtype: TorchdimTuple
        """
        unlisted_dims = []
        unlisted_dim_names = []
        for d, dname in zip(self, self.names):
            if dname not in dim_keys:
                unlisted_dims.append(d)        
                unlisted_dim_names.append(dname)
        sub_tuple = tuple(unlisted_dims)
        sub_superior_dims = self.superior_dims[unlisted_dim_names]
        return TorchdimTuple(sub_tuple, sub_superior_dims)
    
    def __getitem__(self, dim_keys):
        """ Returns torchdims based on either integer indices, a list of dim names
        or the contents of a DimTuple object. 
        
        :param dim_keys: Identifier for determining which dimensions to select
        :type dim_names: Integer, tuple of ints, slice, list of strings, DimTuple
        :return: A TorchdimTuple object with the selected dimensions included
        :rtype: TorchdimTuple
        """
        # Case 1: If dim_keys is an integer behave like a standard tuple
        if type(dim_keys) is int:
            return super().__getitem__(dim_keys)
        
        # Case 2: If dim_keys is slice, produce a TorchdimTuple
        elif type(dim_keys) is slice:
            return TorchdimTuple(ensure_tuple(super().__getitem__(dim_keys)), ensure_tuple(self.superior_dims[dim_keys]))
        
        # Case 3: If dim_keys is a tuple of ints, produce a DimTuple
        elif type(dim_keys) is tuple:
            sublist = []
            for dim_key in dim_keys:
                sublist.append(super().__getitem__(dim_key))
            sub_tuple = ensure_tuple(sublist)
            sub_superior_dims = self.superior_dims[dim_keys]
            return TorchdimTuple(sub_tuple, sub_superior_dims)
        
        # Case 4: If dim_keys is a list of names, get identically named elements of TorchdimTuple
        elif type(dim_keys) is list:
            sublist = []
            for key_dname in dim_keys:
                for dname, d in zip(self.names, self):
                    if dname == key_dname:
                        sublist.append(d) 
            sub_tuple = ensure_tuple(sublist)
            
            sub_superior_dims = self.superior_dims[dim_keys]
            return TorchdimTuple(sub_tuple, sub_superior_dims)
        
            
            # sublist = [d for dname, d in zip(self.names, self) if dname in dim_keys]
            # sub_tuple = tuple(sublist)
            # sub_superior_dims = self.superior_dims[dim_keys]
            # return TorchdimTuple(sub_tuple, sub_superior_dims)
        
        # Case 5: If dim_keys is an instance of DimTuple, look for identically named elements
        elif type(dim_keys) is DimTuple:
            torchdim_tuple = self[dim_keys.names]
            return torchdim_tuple
        
        # Case 6: Raise an error for unsupported types
        else:
            raise TypeError(f"Unsupported key type: {type(dim_keys)}")
            
    def __repr__(self):
        return f"TorchdimTuple({super().__repr__()})"

    def __add__(self, other):
        """ Overloads the + operator to return a new TorchdimTuple when adding two 
        TorchdimTuple objects.
        
        :param other: The TorchdimTuple to add.
        :type other: TorchdimTuple
        :return: A new TorchimTuple with the dimensions from both added tuples.
        :rtype: TorchdimTuple
        :raises NotImplemented: If other is not a TorchdimTuple.
        """
        # Overriding the + operator to return a TorchdimTuple when adding two 
        # TorchdimTuple objects
        if isinstance(other, TorchdimTuple):
            combined_dims = super().__add__(other)
            combined_superior_dims = self.superior_dims + other.superior_dims
            return TorchdimTuple(combined_dims, combined_superior_dims)
        return NotImplemented
        

class DimTuple(tuple):
    """ DimTuple is a custom subclass of Python's `tuple` designed to manage and manipulate tuples of 
    dimension objects, such as those from CalipyDim. This class provides enhanced functionality 
    specific to dimensions, allowing users to bind sizes, filter bound or unbound dimensions, 
    and perform other operations tailored to the handling of dimension objects.

    This class offers methods to bind dimension sizes selectively, retrieve sizes, and check 
    whether dimensions are bound or unbound. Additionally, DimTuple supports tuple-like operations 
    such as concatenation and repetition, while ensuring that the results remain within the DimTuple 
    structure.

    :param input_tuple: A tuple of dimension objects to be managed by DimTuple.
    :type input_tuple: tuple of Dim
    
    :return: An instance of DimTuple containing the dimension objects.
    :rtype: DimTuple

    Example usage:

    .. code-block:: python

        # Create dimensions
        bd_1 = CalipyDim('bd_1', size = 5)
        bd_2 = CalipyDim('bd_2')
        ed_1 = CalipyDim('ed_1')

        # Initialize DimTuples
        batch_dims = DimTuple((bd_1, bd_2))
        event_dims = DimTuple((ed_1,))
        
        # Equivalent command
        batch_dims = dim_assignment(dim_names=['bd_1', 'bd_2'], dim_sizes=[5, None])
        event_dims = dim_assignment(dim_names=['ed_1'])
        
        # Check sizes, names, properties
        batch_dims.names
        batch_dims.sizes
        batch_dims.filter_bound()
        batch_dims.filter_unbound()
        
        # Extract info
        batch_dims.find_indices(['bd_2'])
        batch_dims.find_relative_index('bd_1', 'bd_2')
        batch_dict = batch_dims.to_dict()

        # Change sizes for some dimensions
        bound_dims = batch_dims.bind([11, None])
        unbound_dims = batch_dims.unbind(['bd_1'])
        squeezed_dims = batch_dims.squeeze_dims(['bd_2'])
        
        # Add DimTuples
        full_dims = batch_dims + event_dims
        # raises an exception (as it should): batch_dims + bound_dims
                
        # Multiply DimTuples
        # Dimensions with size of 1 can be broadcasted over, names must match
        dt_factor_1 = dim_assignment(['d1', 'd2', 'd3'], dim_sizes = [5,1,None])
        dt_factor_2 = dim_assignment(['d1', 'd2', 'd3'], dim_sizes = [5,3,12])
        broadcasted_dims = dt_factor_1 * dt_factor_2        # sizes = [5,3,None]
        
        # Use torchdim functionality
        A = torch.normal(0,1, [5,3,2])
        torchdim_tuple = broadcasted_dims.build_torchdims()
        A_named = A[torchdim_tuple]
        
    """
    
    def __new__(cls, input_tuple):
        obj = super(DimTuple, cls).__new__(cls, input_tuple)
        obj.descriptions = [d.description for d in input_tuple]
        # obj.torchdim = obj.convert_to_torchdim()
        
        # Check if names unique
        names = [obj.name for obj in input_tuple]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates: raise ValueError(f"Duplicate names found: {', '.join(duplicates)}")
        return obj

    @property
    def sizes(self):
        """ Returns a list of sizes for each dimension in the DimTuple.
        If a dimension is unbound, None is returned in its place.
        :return: List of sizes corresponding to each dimension in the DimTuple.
        :rtype: list
        """
        sizes = [d.size for d in self]

        return sizes
    
    @property
    def names(self):
        """ Returns a list of names for each dimension in the DimTuple.
        :return: List of names corresponding to each dimension in the DimTuple.
        :rtype: list
        """
        names = [dim.name for dim in self]
        return names    
    
    def build_torchdims(self, fix_size = False):
        """Returns a tuple of functorch torchdims that can be bound and act as
        tensors allowing access to functorch functionality like implicit batching.
        If sizes is not none, torchdims are bound to these sizes. Dimensions 
        corresponding to a None value in the sizes list remain unbound.
        
        :param fix_size: Determines if the torchdims are initialized with fixed size
        :type fix_size: Boolean
        :return: Tuple of functorch.dim.Dim objects
        :rtype: Tuple
        """
        torchdim_tuple = TorchdimTuple(tuple([d.build_torchdim(fix_size = fix_size) for d in self]), superior_dims = self)

        return torchdim_tuple
    
    def find_indices(self, dim_names, from_right=False):
        """Returns a list of indices indicating the locations of the dimensions with names 
        specified in `dim_names` within the DimTuple. Raises an error if any dimension 
        is found multiple times.
    
        :param dim_names: The names of the dims to be located within DimTuple.
        :type dim_names: list of str
        :param from_right: If True, indices are counted from the right (e.g., -1, -2). 
                           If False, indices are counted from the left (e.g., 0, 1).
        :type from_right: bool
        :return: A list of indices where the dimensions are located in the DimTuple.
        :rtype: list of int
        :raises ValueError: If a dimension is found multiple times.
        """
        indices = []
        for name in dim_names:
            matching_indices = [i for i, d in enumerate(reversed(self)) if d.name == name]
            if len(matching_indices) > 1:
                raise ValueError(f"Dimension '{name}' is assigned multiple times.")
            elif not matching_indices:
                raise ValueError(f"Dimension '{name}' not found.")
            
            index = matching_indices[0]
            if from_right:
                indices.append(-1 - index)
            else:
                indices.append(len(self) - 1 - index)
        
        return indices
    
    
    def find_relative_index(self, dim_name, ref_dim_name):
        """Computes the index of `dim_name` relative to `ref_dim_name` within the DimTuple.
        The relative index is positive if `dim_name` is to the right of `ref_dim_name`, 
        and negative if it is to the left.
    
        :param dim_name: The name of the dimension whose relative index is to be computed.
        :type dim_name: str
        :param ref_dim_name: The name of the reference dimension.
        :type ref_dim_name: str
        :return: The relative index of `dim_name` with respect to `ref_dim_name`.
        :rtype: int
        :raises ValueError: If either dimension is not found or found multiple times.
        """
        dim_index = self.find_indices([dim_name], from_right=False)[0]
        ref_index = self.find_indices([ref_dim_name], from_right=False)[0]
        return dim_index - ref_index
    
    
    def delete_dims(self, dims_or_names):
        """ Computes a DimTuple object reduced_dim_tuple with the dims referenced
        in dim_names deleted from reduced_dim_tuple. Consequently, reduced_dim_tuple
        contains only those dims which are not mentioned in dim_names.
        
        :param dims_or_names: The names of the dims to be deleted from DimTuple or
            a DimTuple specifying the dims themselves.
        :type dims_or_names: list of str or DimTuple
        :return: A DimTuple object without the dims in dims_or_names.
        :rtype: DimTuple
        """
        # Check input type
        if isinstance(dims_or_names, DimTuple):
            dim_names = dims_or_names.names
        elif isinstance(dims_or_names, list):
            dim_names = dims_or_names
            
        # Rebuild DimTuple without indicated dims
        unlisted_dims = []
        for d in self:
            if d.name not in dim_names:
                unlisted_dims.append(d)        
        return DimTuple(tuple(unlisted_dims))
    
    def squeeze_dims(self, dim_names):
        """ Computes a DimTuple object squeezed_dim_tuple with the dims referenced
        in dim_names set to 1 in squeezed_dim_tuple. Consequently, squeezed_dim_tuple
        is suitable for broadcasting over the dims that have been squeezed.
        
        :param dim_names: The names of the dims to be located within DimTuple.
        :type dim_names: list of str
        :return: A DimTuple object without the dims in dim_names.
        :rtype: DimTuple
        """
        squeezed_dims = []
        for d in self:
            copied_d = copy.copy(d)
            if d.name not in dim_names:
                squeezed_dims.append(copied_d)
            else:
                copied_d.size =1
                squeezed_dims.append(copied_d)
        
        return DimTuple(tuple(squeezed_dims))
    
    @property
    def is_bound(self):
        """ Checks if all dimensions in the DimTuple are bound.
        :return: True if all dimensions are bound, False otherwise.
        :rtype: bool
        """
        # Returns True if all dimensions are bound, False otherwise
        return all([d.is_bound for d in self])

    def filter_bound(self):
        """ Returns a new DimTuple containing only the bound dimensions.
        :return: A DimTuple with only the bound dimensions.
        :rtype: DimTuple
        """
        # Returns a DimTuple containing only the bound dimensions
        bound_dims = [d for d in self if d.is_bound]
        return DimTuple(tuple(bound_dims))
    
    @property
    def is_unbound(self):
        """ Checks if all dimensions in the DimTuple are unbound.
        :return: True if all dimensions are unbound, False otherwise.
        :rtype: bool
        """
        # Returns True if all dimensions are unbound, False otherwise
        return all([not d.is_bound for d in self])

    def filter_unbound(self):
        """ Returns a new DimTuple containing only the unbound dimensions.
        :return: A DimTuple with only the unbound dimensions.
        :rtype: DimTuple
        """
        # Returns a DimTuple containing only the unbound dimensions
        unbound_dims = [d for d in self if not d.is_bound]
        return DimTuple(tuple(unbound_dims))

    def bind(self, sizes):
        """ Binds sizes to the dimensions in the DimTuple. Dimensions corresponding
        to a None value in the sizes list remain unbound. Raises a ValueError if the
        length of sizes does not match the number of dimensions.

        :param sizes: A list of sizes to bind to the dimensions. Use None to leave a dimension unbound.
        :type sizes: list
        :return: A new DimTuple with the specified sizes bound.
        :rtype: DimTuple
        :raises ValueError: If the number of sizes does not match the number of dimensions.
        """
        # Binds the sizes to the dimensions, returns a new DimTuple with bound dimensions
        if len(sizes) != len(self):
            raise ValueError("Sizes must match the number of dimensions, use None for unbound dims")
    
        new_dims = []
        for i,d in enumerate(self):
            new_dims.append(CalipyDim(d.name, sizes[i], d.description))
              
        return DimTuple(tuple(new_dims))
    
    # def hardbind(self, sizes):
    #     """ Binds sizes to the dimensions in the DimTuple and also binds the
    #     sizes of the interior TorchdimTuple. Dimensions corresponding to a None
    #     value in the sizes list remain unbound. Raises a ValueError if the length
    #     of sizes does not match the number of dimensions.

    #     :param sizes: A list of sizes to bind to the dimensions. Use None to leave a dimension unbound.
    #     :type sizes: list
    #     :return: A new DimTuple with the specified sizes bound.
    #     :rtype: DimTuple
    #     :raises ValueError: If the number of sizes does not match the number of dimensions.
    #     """
    #     # Binds the sizes to the dimensions, returns a new DimTuple with bound dimensions
    #     if len(sizes) != len(self):
    #         raise ValueError("Sizes must match the number of dimensions, use None for unbound dims")
    
    #     new_dims = []
    #     for i,d in enumerate(self):
    #         new_dims.append(CalipyDim(d.name, sizes[i], d.description))
              
    #     return DimTuple(tuple(new_dims))
    
    def unbind(self, dim_names = None):
        """ Returns a DimTuple with the dims corresponding to dim_names being cleared
        of any bindings. Effectively reversed the bind operation.
        
        :param dim_names: The names of the dims to be located within DimTuple. If
            left to default = None, all dims are set to unbound in new DimTuple.
        :type dim_names: list of str
        :return: A new DimTuple with the specified dims unbound.
        :rtype: DimTuple
        """
        
        new_dims = []
        for d in self:
            if d.name in dim_names:
                new_dims.append(CalipyDim(d.name, None, d.description))
            else:
                new_dims.append(CalipyDim(d.name, d.size, d.description))
        return DimTuple(tuple(new_dims))
        
    def reverse(self):
        """ Returns a new DimTuple with dimensions in reverse order.
        :return: A DimTuple with the dimensions reversed.
        :rtype: DimTuple
        """
        reversed_dims = tuple(reversed(self))
        return DimTuple(reversed_dims)
    
    def to_dict(self):
        """ Converts the DimTuple into a dictionary with dimension names as keys and sizes as values.
        If a dimension is unbound, the value in the dictionary is None.
        
        :return: A dictionary with dimension names as keys and sizes as values.
        :rtype: dict
        """
        # Returns a dictionary with dimension names as keys and sizes as values
        return {d.name: d.size for d in self}

    def __repr__(self):
        return f"DimTuple({super().__repr__()})"
    
    def __add__(self, other):
        """ Overloads the + operator to return a new DimTuple when adding two DimTuple objects.
        
        :param other: The DimTuple to add.
        :type other: DimTuple
        :return: A new DimTuple with the dimensions from both added tuples.
        :rtype: DimTuple
        :raises NotImplemented: If other is not a DimTuple.
        """
        # Overriding the + operator to return a DimTuple when adding two DimTuple objects
        if isinstance(other, DimTuple):
            combined_dims = super().__add__(other)
            return DimTuple(combined_dims)
        return NotImplemented

    def __mul__(self, other_tuple):
        """ Overloads the * operator to return a new DimTuple when multiplying two DimTuple objects.
        Multiplication of two DimTuples is possible when their dims line up exactly and
        the sizes of the dims are broadcasteable: For each pair of dims, sizes need to be identical,
        one of both must be 1, or one of both must be None.
        
        :param other: The DimTuple to multiply.
        :type other: DimTuple
        :return: A new DimTuple with the dimensions matching both DimTuples.
        :rtype: DimTuple
        :raises NotImplemented: If other is not a DimTuple.
        """
        
        # Check if dim names line up
        if not other_tuple.names == self.names:
            raise Exception('Names of dimensions of both DimTuples need to line up.'\
                            ' But are {} vs {}'.format(self.names, other_tuple.names))

        # Perform multiplication
        d_new_list = []
        for d1, d2 in zip(self, other_tuple):
            if d1.size == None or d2.size == None:
                d_new_size = None
            elif d1.size == 1 or d2.size == 1:
                d_new_size = d1.size*d2.size
            elif d1.size == d2.size:
                d_new_size = d1.size
            else:
                raise Exception('Dim sizes are incompatible with d1.size = {} and d2.size = {}'.format(d1.size, d2.size))
        
            d_new_list.append(CalipyDim(d1.name, d_new_size, d1.description))
        return DimTuple(tuple(d_new_list))

    def __getitem__(self, dim_keys):
        """ Returns DimTuple based on either integer indices, a list of dim names
        or the contents of a DimTuple object. 
        
        :param dim_keys: Identifier for determining which dimensions to select
        :type dim_keys: Integer, tuple of ints,  slice, list of strings, DimTuple
        :return: A DimTuple object with the selected dimensions included
        :rtype: DimTuple
        """
        # Case 1: If dim_keys is an integer behave like a standard tuple
        if type(dim_keys) is int:
            return super().__getitem__(dim_keys)
        
        # Case 2: If dim_keys is slice, produce a DimTuple
        elif type(dim_keys) is slice:
            return DimTuple(ensure_tuple(super().__getitem__(dim_keys)))
        
        # Case 3: If dim_keys is a tuple of ints, produce a DimTuple
        elif type(dim_keys) is tuple:
            sublist = []
            for dim_key in dim_keys:
                sublist.append(super().__getitem__(dim_key))
            return DimTuple(ensure_tuple(sublist))
        
        # Case 4: If dim_keys is a list of names, get identically named elements of DimTuple
        elif type(dim_keys) is list:
            sublist = []
            for key_dname in dim_keys:
                for d in self:
                    if d.name == key_dname:
                        sublist.append(d) 
                
            # sublist = [d for dname, d in zip(self.names, self) if dname in dim_keys]
            return DimTuple(tuple(sublist))
        
        # Case 5: If dim_keys is an instance of DimTuple, look for identically named elements
        elif type(dim_keys) is DimTuple:
            dim_tuple = self[dim_keys.names]
            return dim_tuple
        
        # Case 6: Raise an error for unsupported types
        else:
            raise TypeError(f"Unsupported key type: {type(dim_keys)}")



