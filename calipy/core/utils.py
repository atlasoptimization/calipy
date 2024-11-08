#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides support functionality related to logging, data organization,
preprocessing and other functions not directly related to calipy core domains.

The classes are
    CalipyRegistry: Dictionary type class that is used for tracking identity
        and uniqueness of objects created during a run and outputs warnings.

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

import pyro
import fnmatch
import contextlib
import networkx as nx
import matplotlib.pyplot as plt
import copy
import re
from functorch.dim import dims
import varname


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


class CalipyDim:
    def __init__(self, name, size = None, description = None):
        self.name = name  # Unique identifier for the dimension
        self.size = size
        self.description = description

    @property
    def is_bound(self):
        if self.size is not None:
            bool_val = True
        else:
            bool_val = False
        return bool_val

    def __repr__(self):
        return self.name

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
        bd_1, bd_2 = dims(2)
        ed_1 = dims(1)

        # Initialize DimTuples
        batch_dims = DimTuple((bd_1, bd_2))
        event_dims = DimTuple((ed_1,))

        # Bind sizes to some dimensions
        batch_dims.bind([10, None])

        # Combine DimTuples
        full_dims = batch_dims + event_dims

        # Accessing the sizes
        print(full_dims.sizes)  # Outputs: [10, None, None]

        # Check if all dimensions are bound
        print(full_dims.is_bound())  # Outputs: False

        # Filter bound dimensions & show dict
        print(full_dims.filter_bound())  # Outputs: DimTuple((bd_1,))
        full_dims.to_dict()
        
        # Broadcasting dimensions
        # Dimensions with shapes of 1 can be broadcasted over
        bc_dims_1 = CalipyDim(['bc_dim_1'])
    """
    
    def __new__(cls, input_tuple):
        obj = super(DimTuple, cls).__new__(cls, input_tuple)
        obj.descriptions = [d.description for d in input_tuple]
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
        sizes = []
        for d in self:
            try:
                sizes.append(d.size)  # Attempt to get the size
            except ValueError:
                sizes.append(None)  # Append None if the dimension is unbound
        return sizes
    
    @property
    def names(self):
        """ Returns a list of names for each dimension in the DimTuple.
        :return: List of names corresponding to each dimension in the DimTuple.
        :rtype: list
        """
        names = [dim.name for dim in self]
        return names    
    
    def find_indices(self, dim_names, from_right=True):
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
    
    
    def delete_dims(self, dim_names):
        """ Computes a DimTuple object reduced_dim_tuple with the dims referenced
        in dim_names deleted from reduced_dim_tuple. Consequently, reduced_dim_tuple
        contains only those dims which are not mentioned in dim_names.
        
        :param dim_names: The names of the dims to be deleted from DimTuple.
        :type dim_names: list of str
        :return: A DimTuple object without the dims in dim_names.
        :rtype: DimTuple
        """
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

    def __mul__(self, n):
        # Allows repeating the DimTuple n times
        return DimTuple(super().__mul__(n))


dim_tuple = dim_assignment(dim_names=['batch_dim_1', 'batch_dim_2'], dim_sizes=[5, None])
dim_tuple.is_bound
dim_tuple.is_unbound
dim_tuple.filter_bound()
dim_tuple.filter_unbound()
dim_tuple.find_indices(['batch_dim_2'])
dim_tuple.find_relative_index('batch_dim_1', 'batch_dim_2')
bound_tuple = dim_tuple.bind([11, None])
unbound_tuple = dim_tuple.unbind(['batch_dim_1'])
squeezed_tuple = dim_tuple.squeeze_dims(['batch_dim_2'])
reversed_tuple = dim_tuple.reverse()
dim_dict = dim_tuple.to_dict()

dim_tuple_2 = dim_assignment(dim_names=['event_dim'])
combined_tuple = dim_tuple+dim_tuple_2
# raises an exception (as it should): dim_tuple+bound_tuple


# OLD BEFORE INTRO OF CALIPYDIM

# Functions for dimension declaration per list

# # restricted exec function
# def restricted_exec(exec_string, allowed_locals):
#     # Allow only simple assignment of `dims` using a regular expression
#     if not re.match(r'^\w+\s*=\s*dims\(sizes=\[\d+(,\s*\d+)*\]\)$', exec_string) \
#         and not re.match(r'^\w+\s*=\s*dims\(\)$', exec_string):
#         raise ValueError("Invalid exec command")
    
#     # Execute the command in a very limited scope
#     allowed_globals = {"dims": dims}
#     exec(exec_string, allowed_globals, allowed_locals)

# # Safe eval function
# def safe_eval(expr, allowed_globals=None, allowed_locals=None):
#     if allowed_globals is None:
#         allowed_globals = {}
#     if allowed_locals is None:
#         allowed_locals = {}
#     return eval(expr, {"__builtins__": None, **allowed_globals}, allowed_locals)



# def dim_assignment(dim_names, dim_shapes=None, dim_descriptions=None):
#     """
#     dim_assignment dynamically assigns dimension objects to names and returns them as a DimTuple.

#     This function creates `Dim` objects using the specified shapes in `dim_shapes` and assigns them to the
#     names provided in `dim_names`. The function validates that the dimension shapes are positive integers
#     or None (for unbound dimensions) and that the dimension names are valid Python identifiers. If only
#     one name is provided with multiple shapes, the name is extended by indices (e.g., 'batch' -> 'batch_1',
#     'batch_2', etc.). The function then executes these assignments in a restricted environment to ensure
#     safety and returns a DimTuple of the created `Dim` objects.

#     :param dim_names: A list of strings representing the variable names to assign to each dimension. 
#         These names must be valid Python identifiers. If only one name is provided and multiple shapes,
#         the name will be broadcast with indices (e.g., ['batch'] -> ['batch_1', 'batch_2', ...]).
#     :type dim_names: list of str
#     :param dim_shapes: A list of nonnegative integers or None representing the sizes of each dimension; 
#         None indicates an unbound dimension; a value of 0 indicates an empty dimension
#     :type dim_shapes: list of int or None
#     :param dim_descriptions: A list of descriptions describing each dimension; 
#         None indicates absence of descriptions.
#     :type dim_shapes: list of str or None
        
#     :return: A DimTuple containing the `Dim` objects assigned to the names in `dim_names`.
#     :rtype: DimTuple

#     Example usage:

#     .. code-block:: python

#         dim_names = ['batch_dim_1', 'batch_dim_2']
#         dim_shapes = [10, 5]
#         dim_tuple = dim_assignment(dim_names, dim_shapes)

#         # Access the dimensions
#         print(dim_tuple)  # Outputs: (batch_dim_1, batch_dim_2)
#         print(dim_tuple[0].size)  # Outputs: 10
#         print(dim_tuple[1].size)  # Outputs: 5
        
#         # Example with broadcasting
#         dim_tuple = dim_assignment(dim_names=['batch'], dim_shapes=[5, 2])
#         print(dim_tuple)  # Outputs: (batch_1, batch_2)
#         print(dim_tuple.size)  # Outputs: [5,2]
        
#         # Example with bound and unbound dims
#         dim_tuple = dim_assignment(dim_names=['batch_dim_1', 'batch_dim_2'], dim_shapes=[5, None])
#         dim_tuple.get_sizes()
#         dim_tuple.filter_bound()
#         dim_tuple.filter_unbound()
        
#         # Example with a dimension skipped
#         dim_tuple = dim_assignment(dim_names=['a'], dim_shapes=[0])
#         print(dim_tuple)  # Outputs: DimTuple(())
#     """
        
#     # Broadcasting over dim_names if needed
#     if len(dim_names) == 1 and dim_shapes is not None and len(dim_shapes) > 1:
#         base_name = dim_names[0]
#         dim_names = [f"{base_name}_{i+1}" for i in range(len(dim_shapes))]

#     # Validate inputs    
#     if not all(isinstance(name, str) and name.isidentifier() for name in dim_names):
#         raise ValueError("All dimension names must be valid Python identifiers.")
#     if dim_shapes is not None:
#         if not all(shape is None or (isinstance(shape, int) and shape >= 0) for shape in dim_shapes):
#             raise ValueError("All dimension shapes must be nonnegative integers or None.")
    
#     # Create a local environment to hold the assigned dimensions
#     dims_locals = {}
#     valid_dim_names = []
#     valid_dim_descriptions = []
    
#     if dim_descriptions is None:
#         dim_descriptions = [None] * len(dim_names)
#     if dim_shapes is None:
#         dim_shapes = [None] * len(dim_names)
    
#     for name, shape, description in zip(dim_names, dim_shapes, dim_descriptions):
#         if shape == 0:
#             continue  # Skip this dimension entirely
#         if shape is not None:
#             exec_string = f"{name} = dims(sizes=[{shape}])"
#         else:
#             exec_string = f"{name} = dims()"
#         restricted_exec(exec_string, dims_locals)
#         valid_dim_names.append(name)
#         valid_dim_descriptions.append(description)

#     # If no valid dimensions were created, return an empty DimTuple
#     if not valid_dim_names:
#         return DimTuple(())
    
#     # Create a tuple of dimensions
#     if len(valid_dim_names) == 1:
#         eval_string = f"({valid_dim_names[0]},)"  # Ensure single element tuple with a trailing comma
#     else:
#         eval_string = f"({', '.join(valid_dim_names)})"
    
#     dim_tuple = DimTuple(safe_eval(eval_string, allowed_locals=dims_locals), descriptions=valid_dim_descriptions)
    
#     return dim_tuple



# # Generate trivial dimensions
# def generate_trivial_dims(ndim):
#     dim_names = ["trivial_dim_{}".format(k) for k in range(ndim)]
#     trivial_dims = dim_assignment(dim_names, dim_shapes = [1 for name in dim_names])
#     return trivial_dims
    

# class DimTuple(tuple):
#     """ DimTuple is a custom subclass of Python's `tuple` designed to manage and manipulate tuples of 
#     dimension objects, such as those from `functorch`. This class provides enhanced functionality 
#     specific to dimensions, allowing users to bind sizes, filter bound or unbound dimensions, 
#     and perform other operations tailored to the handling of dimension objects.

#     This class offers methods to bind dimension sizes selectively, retrieve sizes, and check 
#     whether dimensions are bound or unbound. Additionally, DimTuple supports tuple-like operations 
#     such as concatenation and repetition, while ensuring that the results remain within the DimTuple 
#     structure.

#     :param input_tuple: A tuple of dimension objects to be managed by DimTuple.
#     :type input_tuple: tuple of Dim
    
#     :return: An instance of DimTuple containing the dimension objects.
#     :rtype: DimTuple

#     Example usage:

#     .. code-block:: python

#         # Create dimensions
#         bd_1, bd_2 = dims(2)
#         ed_1 = dims(1)

#         # Initialize DimTuples
#         batch_dims = DimTuple((bd_1, bd_2))
#         event_dims = DimTuple((ed_1,))

#         # Bind sizes to some dimensions
#         batch_dims.bind([10, None])

#         # Combine DimTuples
#         full_dims = batch_dims + event_dims

#         # Accessing the sizes
#         print(full_dims.sizes)  # Outputs: [10, None, None]

#         # Check if all dimensions are bound
#         print(full_dims.is_bound())  # Outputs: False

#         # Filter bound dimensions & show dict
#         print(full_dims.filter_bound())  # Outputs: DimTuple((bd_1,))
#         full_dims.to_dict()
#     """
#     # def __new__(cls, input_tuple):
#     #     # __new__ is used for immutable types like tuple
#     #     return super(DimTuple, cls).__new__(cls, input_tuple)
    
#     def __new__(cls, input_tuple, descriptions=None):
#         obj = super(DimTuple, cls).__new__(cls, input_tuple)
#         obj.descriptions = descriptions if descriptions is not None else [None] * len(input_tuple)
#         if len(obj.descriptions) != len(input_tuple):
#             raise ValueError("Length of descriptions must match length of input_tuple")
#         return obj

#     @property
#     def sizes(self):
#         """ Returns a list of sizes for each dimension in the DimTuple.
#         If a dimension is unbound, None is returned in its place.
#         :return: List of sizes corresponding to each dimension in the DimTuple.
#         :rtype: list
#         """
#         sizes = []
#         for d in self:
#             try:
#                 sizes.append(d.size)  # Attempt to get the size
#             except ValueError:
#                 sizes.append(None)  # Append None if the dimension is unbound
#         return sizes
    
#     @property
#     def names(self):
#         """ Returns a list of names for each dimension in the DimTuple.
#         :return: List of names corresponding to each dimension in the DimTuple.
#         :rtype: list
#         """
#         names = ['{}'.format(dim.__repr__()) for dim in self]
#         return names
    
#     def get_local_copy(self):
#         """ Returns a new DimTuple with the same content as source DimTuple that 
#         can be bound and used locally without affecting source DimTuple.
#         :return: A DimTuple with same econtent but different identiy.
#         :rtype: DimTuple
#         """
#         # Returns a DimTuple containing the same content
#         dim_names = self.names
#         dim_shapes = self.sizes
#         dim_descriptions = self.descriptions
#         dim_tuple = dim_assignment(dim_names, dim_shapes, dim_descriptions)
#         return dim_tuple
    
    
#     def find_indices(self, dim_names, from_right=True):
#         """Returns a list of indices indicating the locations of the dimensions with names 
#         specified in `dim_names` within the DimTuple. Raises an error if any dimension 
#         is found multiple times.
    
#         :param dim_names: The names of the dims to be located within DimTuple.
#         :type dim_names: list of str
#         :param from_right: If True, indices are counted from the right (e.g., -1, -2). 
#                            If False, indices are counted from the left (e.g., 0, 1).
#         :type from_right: bool
#         :return: A list of indices where the dimensions are located in the DimTuple.
#         :rtype: list of int
#         :raises ValueError: If a dimension is found multiple times.
#         """
#         indices = []
#         for name in dim_names:
#             matching_indices = [i for i, d in enumerate(reversed(self)) if d.__repr__() == name]
#             if len(matching_indices) > 1:
#                 raise ValueError(f"Dimension '{name}' is assigned multiple times.")
#             elif not matching_indices:
#                 raise ValueError(f"Dimension '{name}' not found.")
            
#             index = matching_indices[0]
#             if from_right:
#                 indices.append(-1 - index)
#             else:
#                 indices.append(len(self) - 1 - index)
        
#         return indices
    
    
#     def find_relative_index(self, dim_name, ref_dim_name):
#         """Computes the index of `dim_name` relative to `ref_dim_name` within the DimTuple.
#         The relative index is positive if `dim_name` is to the right of `ref_dim_name`, 
#         and negative if it is to the left.
    
#         :param dim_name: The name of the dimension whose relative index is to be computed.
#         :type dim_name: str
#         :param ref_dim_name: The name of the reference dimension.
#         :type ref_dim_name: str
#         :return: The relative index of `dim_name` with respect to `ref_dim_name`.
#         :rtype: int
#         :raises ValueError: If either dimension is not found or found multiple times.
#         """
#         dim_index = self.find_indices([dim_name], from_right=False)[0]
#         ref_index = self.find_indices([ref_dim_name], from_right=False)[0]
#         return dim_index - ref_index
    
#     def reduce_dims(self, dim_names):
#         """ Computes a DimTuple object reduced_dim_tuple with the dims referenced
#         in dim_names deleted from reduced_dim_tuple. Consequently, reduced_dim_tuple
#         contains only those dims which are not mentioned in dim_names.
        
#         :param dim_names: The names of the dims to be located within DimTuple.
#         :type dim_names: list of str
#         :return: A DimTuple object without the dims in dim_names.
#         :rtype: DimTuple
#         """
#         unlisted_dims = []
#         unlisted_dims_descriptions = []
#         for d, desc in zip(self, self.descriptions):
#             if d.__repr__() not in dim_names:
#                 unlisted_dims.append(d)
#                 unlisted_dims_descriptions.append(desc)
        
#         return DimTuple(tuple(unlisted_dims), descriptions = unlisted_dims_descriptions )
    
#     def is_bound(self):
#         """ Checks if all dimensions in the DimTuple are bound.
#         :return: True if all dimensions are bound, False otherwise.
#         :rtype: bool
#         """
#         # Returns True if all dimensions are bound, False otherwise
#         return all(d.is_bound for d in self)

#     def filter_bound(self):
#         """ Returns a new DimTuple containing only the bound dimensions.
#         :return: A DimTuple with only the bound dimensions.
#         :rtype: DimTuple
#         """
#         # Returns a DimTuple containing only the bound dimensions
#         bound_dims = [d for d in self if d.is_bound]
#         bound_descriptions = [desc for d, desc in zip(self, self.descriptions) if d.is_bound]
#         return DimTuple(tuple(bound_dims), descriptions=bound_descriptions)
    
#     def is_unbound(self):
#         """ Checks if all dimensions in the DimTuple are unbound.
#         :return: True if all dimensions are unbound, False otherwise.
#         :rtype: bool
#         """
#         # Returns True if all dimensions are unbound, False otherwise
#         return all(not d.is_bound for d in self)

#     def filter_unbound(self):
#         """ Returns a new DimTuple containing only the unbound dimensions.
#         :return: A DimTuple with only the unbound dimensions.
#         :rtype: DimTuple
#         """
#         # Returns a DimTuple containing only the unbound dimensions
#         unbound_dims = [d for d in self if not d.is_bound]
#         unbound_descriptions = [desc for d, desc in zip(self, self.descriptions) if not d.is_bound]
#         return DimTuple(tuple(unbound_dims), descriptions=unbound_descriptions)

#     def bind(self, sizes):
#         """ Binds sizes to the dimensions in the DimTuple. Dimensions corresponding
#         to a None value in the sizes list remain unbound. Raises a ValueError if the
#         length of sizes does not match the number of dimensions.

#         :param sizes: A list of sizes to bind to the dimensions. Use None to leave a dimension unbound.
#         :type sizes: list
#         :return: A new DimTuple with the specified sizes bound.
#         :rtype: DimTuple
#         :raises ValueError: If the number of sizes does not match the number of dimensions.
#         """
#         # Binds the sizes to the dimensions, returns a new DimTuple with bound dimensions
#         if len(sizes) != len(self):
#             raise ValueError("Sizes must match the number of dimensions, use None for unbound dims")
#         [setattr(d, 'size', size) for d, size in zip(self, sizes) if size is not None]
    
#     def unbind(self, dim_names = None):
#         """ Returns a DimTuple with the dims corresponding to dim_names being cleared
#         of any bindings. Effectively reversed the bind operation.
        
#         :param dim_names: The names of the dims to be located within DimTuple. If
#             left to default = None, all dims are set to unbound in new DimTuple.
#         :type dim_names: list of str
#         :return: A new DimTuple with the specified dims unbound.
#         :rtype: DimTuple
#         """
        
#         # Build new shapes
#         dim_new_shapes = []
#         for dn, ds in zip(self.names, self.sizes):
#             if dim_names is None or dn in dim_names:
#                 dim_new_shapes.append(None)
#             else:
#                 dim_new_shapes.append(ds)
                    
#         dim_tuple = dim_assignment(self.names, dim_new_shapes, self.descriptions)
#         return dim_tuple
        
        
    
#     def reverse(self):
#         """ Returns a new DimTuple with dimensions in reverse order.
#         :return: A DimTuple with the dimensions reversed.
#         :rtype: DimTuple
#         """
#         reversed_dims = tuple(reversed(self))
#         reversed_descriptions = list(reversed(self.descriptions))
#         return DimTuple(reversed_dims, descriptions=reversed_descriptions)
    
#     def to_dict(self):
#         """ Converts the DimTuple into a dictionary with dimension names as keys and sizes as values.
#         If a dimension is unbound, the value in the dictionary is None.
        
#         :return: A dictionary with dimension names as keys and sizes as values.
#         :rtype: dict
#         """
#         # Returns a dictionary with dimension names as keys and sizes as values
#         return {d.__repr__(): d.size if d.is_bound else None 
#             for i, d in enumerate(self)}

#     def __repr__(self):
#         return f"DimTuple({super().__repr__()})"
    
#     def __add__(self, other):
#         """ Overloads the + operator to return a new DimTuple when adding two DimTuple objects.
        
#         :param other: The DimTuple to add.
#         :type other: DimTuple
#         :return: A new DimTuple with the dimensions from both added tuples.
#         :rtype: DimTuple
#         :raises NotImplemented: If other is not a DimTuple.
#         """
#         # Overriding the + operator to return a DimTuple when adding two DimTuple objects
#         if isinstance(other, DimTuple):
#             combined_dims = super().__add__(other)
#             combined_descriptions = self.descriptions + other.descriptions
#             return DimTuple(combined_dims, descriptions=combined_descriptions)
#         return NotImplemented

#     def __mul__(self, n):
#         # Allows repeating the DimTuple n times
#         return DimTuple(super().__mul__(n))


