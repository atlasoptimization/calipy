#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We want to test here how to do consistent subsampling within pyro over different
even when pyro.plate is called at different locations.

We use the Dataset and DataLoader class
"""
 
import torch
import pyro
import pyro.distributions as dist
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Assuming data is a list or array of samples

    def __len__(self):
        return len(self.data)  # Number of samples

    def __getitem__(self, idx):
        # Return both the data and its index
        return self.data[idx], idx


from torch.utils.data import DataLoader

# Assuming MyDataset is defined as above
dataset = MyDataset(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Iterate through the DataLoader
for batch, index in dataloader:
    print(batch, index)
