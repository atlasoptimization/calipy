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





# def illustrate_trace(trace):
    
#     # Create a directed graph
#     G = nx.DiGraph()
    
#     # Add nodes and edges based on the trace
#     for node_name, node_info in trace.nodes.items():
#         if node_info["type"] == "sample":
#             G.add_node(node_name, **node_info)
#             if node_info["is_observed"] == False:
#                 parent_name = node_info["fn"].base_dist.loc if hasattr(node_info["fn"], 'base_dist') else node_info["fn"].loc
#                 if isinstance(parent_name, str) and parent_name in trace.nodes:
#                     G.add_edge(parent_name, node_name)
    
#     # Draw the network graph
#     pos = nx.spring_layout(G)  # positions for all nodes
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=4000, edge_color='k', linewidths=1, font_size=15)
    
#     # Draw node labels
#     labels = {node: node for node in G.nodes()}
#     nx.draw_networkx_labels(G, pos, labels, font_size=16)
    
#     plt.title("Pyro Model Trace Graph")
#     plt.show()
    
    









