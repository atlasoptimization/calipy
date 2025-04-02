#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:42:23 2024

@author: jemil
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TrainingVideoMaker:
    """
    A reusable class for creating and saving training animations, with an option
    to fix your axes from the start.

    Usage:
        animator = TrainingVideoMaker(
            fixed_axes=True, 
            xlim=(0, 100), 
            ylim_param=(-2, 2),
            ylim_obs=(-5, 5),
            ylim_loss=(0, 1)
        )
        
        # inside training loop
        for epoch in range(num_epochs):
            param, loss, obs_list = ...
            animator.add_data(epoch, param, loss, obs_list)
        
        # after training finishes
        animator.finalize("training_progress.gif")  # or .mp4, if FFmpeg is installed
    """
    def __init__(
        self,
        figsize=(8, 8),
        sharex=True,
        fixed_axes=False,
        xlim=None,         # e.g., (0, 1000)
        ylim_param=None,   # e.g., (-2, 2)
        ylim_obs=None,     # e.g., (-5, 5)
        ylim_loss=None     # e.g., (0, 1)
    ):
        # Create figure and axes
        self.fig, self.axes = plt.subplots(3, 1, figsize=figsize, sharex=sharex)
        self.param_ax, self.obs_ax, self.loss_ax = self.axes

        # Prepare lines / scatter references
        self.param_line, = self.param_ax.plot([], [], label="Parameter Value")
        self.loss_line, = self.loss_ax.plot([], [], label="Loss")
        # For scatter, initialize an empty Nx2 array.
        self.obs_scatter = self.obs_ax.scatter([], [], alpha=0.5, label="Noisy Obs")

        # Data containers
        self.epochs = []
        self.param_data = []
        self.loss_data = []
        self.obs_data = []   # list of lists/arrays of noisy observations

        # Store the config
        self.fixed_axes = fixed_axes
        self.xlim = xlim
        self.ylim_param = ylim_param
        self.ylim_obs   = ylim_obs
        self.ylim_loss  = ylim_loss

        # Axis labels, etc.
        self.param_ax.set_ylabel("Parameter")
        self.obs_ax.set_ylabel("Noisy Observations")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_xlabel("Epoch")

        # If user provided axis limits, set them right away
        if xlim is not None:
            self.param_ax.set_xlim(*xlim)
            self.obs_ax.set_xlim(*xlim)
            self.loss_ax.set_xlim(*xlim)

        if ylim_param is not None:
            self.param_ax.set_ylim(*ylim_param)

        if ylim_obs is not None:
            self.obs_ax.set_ylim(*ylim_obs)

        if ylim_loss is not None:
            self.loss_ax.set_ylim(*ylim_loss)

        for ax in self.axes:
            ax.grid(True)
            ax.legend(loc="upper right")

    def add_data(self, epoch, param, loss, obs_list):
        """
        Store data for each training epoch.

        epoch: scalar (e.g. int)
        param: scalar for the parameter
        loss:  scalar for the loss
        obs_list: list or array of noisy observations
        """
        self.epochs.append(epoch)
        self.param_data.append(param)
        self.loss_data.append(loss)
        self.obs_data.append(obs_list)

    def _init_func(self):
        """
        Required by FuncAnimation for initialization.
        """
        self.param_line.set_data([], [])
        self.loss_line.set_data([], [])
        # Clear scatter data with empty Nx2 array
        self.obs_scatter.set_offsets(np.empty((0, 2)))
        return (self.param_line, self.obs_scatter, self.loss_line)

    def _update_func(self, frame_idx):
        """
        Update function for FuncAnimation. Gets called for each frame (epoch).
        """
        # Slice data up to frame_idx
        current_epochs = self.epochs[:frame_idx+1]
        current_params = self.param_data[:frame_idx+1]
        current_losses = self.loss_data[:frame_idx+1]

        # Flatten all noisy observations up to frame_idx
        xs = []
        ys = []
        for i, obs_list in enumerate(self.obs_data[:frame_idx+1]):
            xs.extend([self.epochs[i]] * len(obs_list))
            ys.extend(obs_list)

        # Update lines
        self.param_line.set_data(current_epochs, current_params)
        self.loss_line.set_data(current_epochs, current_losses)

        # Update scatter
        if len(xs) > 0:
            coords = np.column_stack((xs, ys))
        else:
            coords = np.empty((0, 2))
        self.obs_scatter.set_offsets(coords)

        # Autoscale only if user wants dynamic axes
        if not self.fixed_axes:
            self.param_ax.relim(); self.param_ax.autoscale_view()
            self.obs_ax.relim();   self.obs_ax.autoscale_view()
            self.loss_ax.relim();  self.loss_ax.autoscale_view()

        return (self.param_line, self.obs_scatter, self.loss_line)

    def finalize(self, output_filename=None, fps=15, interval=50):
        """
        Create the animation and optionally save it.

        output_filename: e.g. "training_progress.mp4" or ".gif"
        fps: frames per second
        interval: delay between frames in ms (for live preview)
        """
        frames = len(self.epochs)
        if frames == 0:
            print("No data was added, skipping animation.")
            return

        self.anim = FuncAnimation(
            self.fig,
            func=self._update_func,
            init_func=self._init_func,
            frames=frames,
            blit=False,
            interval=interval,
            repeat=False
        )

        if output_filename:
            # self.anim.save(output_filename, fps=fps, dpi=150)
            self.anim.save(output_filename, writer="pillow", fps=10, dpi=150)
        else:
            plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# class TrainingVideoMaker:
#     """
#     A reusable class for creating and saving training animations.

#     Usage:
#         animator = TrainingVideoMaker()
#         # inside training loop
#         for epoch in range(num_epochs):
#             param, loss, obs = ...
#             animator.add_data(epoch, param, loss, obs)
#         # after training finishes
#         animator.finalize("training_progress.mp4")
#     """

#     def __init__(self, figsize=(8, 8), sharex=True):
#         # Create figure and axes
#         self.fig, self.axes = plt.subplots(3, 1, figsize=figsize, sharex=sharex)
#         self.param_ax, self.obs_ax, self.loss_ax = self.axes

#         # Prepare lines / scatter references
#         self.param_line, = self.param_ax.plot([], [], label="Parameter Value")
#         self.loss_line, = self.loss_ax.plot([], [], label="Loss")
#         # For scatter, we initialize with no points (empty Nx2 array).
#         self.obs_scatter = self.obs_ax.scatter([], [], alpha=0.5, label="Noisy Obs")

#         # Data containers
#         self.epochs = []
#         self.param_data = []
#         self.loss_data = []
#         # obs_data is a list of lists/arrays of observations at each epoch
#         self.obs_data = []

#         # Axis labels, etc.
#         self.param_ax.set_ylabel("Parameter")
#         self.obs_ax.set_ylabel("Noisy Observations")
#         self.loss_ax.set_ylabel("Loss")
#         self.loss_ax.set_xlabel("Epoch")

#         for ax in self.axes:
#             ax.grid(True)
#             ax.legend(loc="upper right")

#     def add_data(self, epoch, param, loss, obs_list):
#         """
#         Store data for each training epoch.

#         epoch: scalar (e.g. int)
#         param: scalar for the parameter
#         loss:  scalar for the loss
#         obs_list: list or array of noisy observations
#         """
#         self.epochs.append(epoch)
#         self.param_data.append(param)
#         self.loss_data.append(loss)
#         self.obs_data.append(obs_list)

#     def _init_func(self):
#         """
#         Required by FuncAnimation for initialization.
#         """
#         # Clear line data
#         self.param_line.set_data([], [])
#         self.loss_line.set_data([], [])
#         # Clear scatter data, using an empty (0,2) array instead of []
#         self.obs_scatter.set_offsets(np.empty((0, 2)))
#         return (self.param_line, self.obs_scatter, self.loss_line)

#     def _update_func(self, frame_idx):
#         """
#         Update function for FuncAnimation. Gets called for each frame (epoch) in the final video.
#         """
#         # Slice data up to frame_idx
#         current_epochs = self.epochs[:frame_idx+1]
#         current_params = self.param_data[:frame_idx+1]
#         current_losses = self.loss_data[:frame_idx+1]

#         # Flatten all noisy observations up to frame_idx
#         xs = []
#         ys = []
#         for i, obs_list in enumerate(self.obs_data[:frame_idx+1]):
#             # Each observation in obs_list is plotted at x = self.epochs[i]
#             xs.extend([self.epochs[i]] * len(obs_list))
#             ys.extend(obs_list)

#         # Update lines
#         self.param_line.set_data(current_epochs, current_params)
#         self.loss_line.set_data(current_epochs, current_losses)

#         # Update scatter
#         if len(xs) == 0:
#             coords = np.empty((0, 2))
#         else:
#             coords = np.column_stack((xs, ys))
#         self.obs_scatter.set_offsets(coords)

#         # Autoscale
#         self.param_ax.relim(); self.param_ax.autoscale_view()
#         self.obs_ax.relim();   self.obs_ax.autoscale_view()
#         self.loss_ax.relim();  self.loss_ax.autoscale_view()

#         return (self.param_line, self.obs_scatter, self.loss_line)

#     def finalize(self, output_filename=None, fps=30, interval=50):
#         """
#         Create the animation and optionally save it to disk.

#         output_filename: (str) path to output file (e.g. "training_progress.mp4").
#         fps:  frames per second for the saved video.
#         interval:  delay between frames in ms (only affects live preview).
#         """
#         frames = len(self.epochs)
#         if frames == 0:
#             print("No data was added, skipping animation.")
#             return

#         self.anim = FuncAnimation(
#             self.fig,
#             func=self._update_func,
#             init_func=self._init_func,
#             frames=frames,
#             blit=False,
#             interval=interval,
#             repeat=False
#         )

#         if output_filename:
#             # Saving requires a working writer (FFMPEG or others).
#             # self.anim.save(output_filename, fps=fps, dpi=150)
#             self.anim.save("output_filename.gif", writer="pillow", fps=10, dpi=150)

#         else:
#             plt.show()
