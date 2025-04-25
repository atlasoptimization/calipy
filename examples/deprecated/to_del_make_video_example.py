#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:24:54 2025

@author: jemil
"""


import numpy as np
from calipy.core.illustrator import TrainingVideoMaker

import numpy as np

n_epochs = 100

def train_simulation(num_epochs= n_epochs):
    """
    Fake 'training' that returns (param, loss, obs_list) for each epoch.
    """
    for epoch in range(num_epochs):
        param = np.sin(epoch * 0.1) + 0.1 * np.random.randn()
        loss = np.exp(-epoch * 0.05) + 0.01 * np.random.randn()
        obs_list = param + 0.3 * np.random.randn(5)  # 5 noisy observations
        yield epoch, param, loss, obs_list

if __name__ == "__main__":
    animator = TrainingVideoMaker(
        fixed_axes=True,           # lock axes
        xlim=(0, n_epochs),      # epochs as x-axis
        ylim_param=(-2, 2),        # param range
        ylim_obs=(-3, 3),          # obs range
        ylim_loss=(0, 2)           # loss range
    )

    for (epoch, param, loss, obs_list) in train_simulation(num_epochs = n_epochs):
        animator.add_data(epoch, param, loss, obs_list)

    # Save as GIF
    animator.finalize("../illustrations/training_sim.gif")



# def train_simulation(num_epochs=100):
#     """
#     Fake 'training' that returns (param, loss, obs_list) for each epoch.
#     """
#     for epoch in range(num_epochs):
#         param = np.sin(epoch * 0.1) + 0.1 * np.random.randn()
#         loss = np.exp(-epoch * 0.05) + 0.01 * np.random.randn()
#         obs_list = param + 0.3 * np.random.randn(5)  # 5 noisy observations
#         yield epoch, param, loss, obs_list

# if __name__ == "__main__":
    

#     # 1. Instantiate the animator
#     animator = TrainingVideoMaker()

#     # 2. Run your "training"
#     for (epoch, param, loss, obs_list) in train_simulation(num_epochs=50):
#         animator.add_data(epoch, param, loss, obs_list)

#     # 3. Finalize the animation (saving to mp4)
#     animator.finalize("./illustrations/training_sim.mp4")
