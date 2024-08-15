#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:30:03 2024

@author: jemil
"""

# Import total station class
from calipy.core.instruments 
    import TotalStationTS60

# Import axis deviation effect
from calipy.core.effects
    import AxisDeviation
    

# Initiate unknown params
sigma = sigma_object.forward()
mu = mu_object.forward()

# Measurement model and data link
measurement = noise_addition.forward(
    input_vars = (mu,sigma),
    observations = data)



total_station = TotalStationTS60()
total_station.train()

epoch: 0 ; loss : 3.9739323
epoch: 100 ; loss : -2.6217
...
epoch: 800 ; loss : 0.05679
epoch: 900 ; loss : 0.06500