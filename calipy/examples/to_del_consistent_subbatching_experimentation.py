#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We want to test here how to do consistent subsampling within pyro over different
even when pyro.plate is called at different locations
"""
 
import torch
import pyro
import pyro.distributions as dist
# Consistens subsampling?
 
# The following does not work! Plates are not synchronized between different calls.   
# def model():
#     with pyro.plate('batch_plate', size = 10, subsample_size = 3, dim =-1) as ind:
#         print(ind)
#     with pyro.plate('batch_plate', size = 10, subsample_size = 3, dim =-1) as ind:
#         print(ind)
        
    
# def model_2():
#     for ind in pyro.plate('batch_plate_2', size = 10, subsample_size = 3, dim =-1):
#         print(ind)
#     for ind in pyro.plate('batch_plate_2', size = 10, subsample_size = 3, dim =-1):
#         print(ind)
        
# def guide():
#     for i in pyro.plate("locals", 10, subsample_size=5):
#         # sample the local RVs
#         print(i)
        
# def model(data):
#     batch_size = 10
#     subsample_size = 3

#     # Named plate ensures the same subsample indices are used across the model
#     with pyro.plate('batch_plate', size=batch_size, subsample_size=subsample_size, dim=-1) as ind:
#         # Use the subsample indices to subsample data
#         subsampled_data = data[ind]
#         print(ind)
        
#         # First operation using subsample
#         mu = pyro.sample("mu", dist.Normal(0, 1).expand([subsample_size]))

#         # Compute some statistics based on subsample
#         mean_value = subsampled_data.mean()

#     # Second operation using the same subsample
#     with pyro.plate('batch_plate', size=batch_size, subsample_size=subsample_size, dim=-1) as ind:
#         pyro.sample("obs", dist.Normal(mu + mean_value, 1), obs=subsampled_data)
#         print(ind)

#     return
# # Example data
# data = torch.randn(10)
# model(data)
    

def model(data):
    batch_size = 10
    subsample_size = 3
    
    # Manually generate subsample indices
    with pyro.plate('batch_plate', size=batch_size, subsample_size=subsample_size, dim=-1) as ind:
        # Subsample the data using the generated indices
        subsampled_data = data[ind]
        print("First plate indices:", ind)
        
        # Perform the first operation using the subsample
        mu = pyro.sample("mu", dist.Normal(0, 1).expand([subsample_size]))
        
        # Compute some statistics based on the subsample
        mean_value = subsampled_data.mean()

    # Reuse the same indices for the second operation
    with pyro.plate('batch_plate', size=batch_size, subsample=ind, dim=-1):
        pyro.sample("obs", dist.Normal(mu + mean_value, 1), obs=subsampled_data)
        print("Second plate indices:", ind)

# Example data
data = torch.randn(10)
model(data)