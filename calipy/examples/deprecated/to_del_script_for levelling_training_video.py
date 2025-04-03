#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a simple two-peg level test
as dealt with in section 4.2 of the paper: "Building and Solving Probabilistic 
Instrument Models with CaliPy" presented at JISDM 2025 in Karlsruhe. The overall
measurement process consists in setting up a levelling instrument in some arbitrary
distances l_A, l_B to two levelling rods, then reading out the height measurements,
and then setting up the levelling instrument at some other location. The goal is 
to estimate the collimation angle alpha from these observations y.
The corresponding probabilistic model is given by the following expression for y 
as      y_A ~ N(h_I + l_A tan(alpha))
        y_B ~ N(h_I - DeltaH + l_B tan(alpha))
where l_A, l_B are the distances between levelling instrument and rods A, B, and
y_A_true = h_I, y_B_true = h_I-DeltaH are the true readings. N is the Gaussian
distribution. DeltaH is the heigh difference between A and B and h_I is the 
# instruments height.
Here l_A, l_B and sigma are assumed known, y is observed, and alpha is to be inferred
while DeltaH, h_I are unknowns we do not care about. The true readings for rod A and 
rod B are connected via y_A_true = h_I, y_B_true = h_I - DeltaH where h_I is the height of
the instrument in each configuration and DeltaH is the height difference between 
A and B. We want to infer alpha from observations y without performing any further
manual computations.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Load and customize effects
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import pyro
import matplotlib.pyplot as plt

# calipy
import calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.tensor import CalipyTensor
from calipy.core.data import CalipyIO, preprocess_args


# ii) Definitions

n_config = 2 # number of configurations
def set_seed(seed=42):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

set_seed(123)



"""
    2. Simulate some data
"""


# i) Set up sample distributions
# Note the model y_A ~ N(h_I + l_A tan(alpha))
#                y_B ~ N(h_I - DeltaH + l_B tan(alpha))
# where h_I is unknown different for each config and DeltaH and alpha are global
# scalar unknowns.

# Global instrument params
alpha_true = torch.tensor(0.001)
dh_true = torch.tensor(0.5)
sigma_true = torch.tensor(0.001)

# Config specific params
hI_true = torch.normal(1, 0.1, [n_config])
l_A = torch.tensor([[30], [0]])
l_B = torch.tensor([[30], [60]])
l_mat = torch.hstack([l_A, l_B])

# Distribution params
y_A_true = hI_true
y_B_true = hI_true - dh_true
y_true = torch.vstack([y_A_true, y_B_true]).T

l_impact = torch.tan(alpha_true) * l_mat
y_biased = y_true + l_impact


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(y_biased, sigma_true)
data = data_distribution.sample()

# The data now is a tensor of shape [n_meas,2] and reflects biased measurements being
# taken of a two-rod measurement configuration.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.

n_gt = 1000

ground_truth_residuals = data_distribution.sample([n_gt]).numpy()
gt_resid_00 = ground_truth_residuals[:,0,0]



"""
    3. Load and customize effects
"""


# i) Set up dimensions

dim_1 = dim_assignment(['dim_1'], dim_sizes = [n_config])
dim_2 = dim_assignment(['dim_2'], dim_sizes = [2])
dim_3 = dim_assignment(['dim_3'], dim_sizes = [])

# ii) Set up dimensions parameters

# alpha setup
alpha_ns = NodeStructure(UnknownParameter)
alpha_ns.set_dims(batch_dims = dim_1 + dim_2, param_dims = dim_3)
alpha_object = UnknownParameter(alpha_ns, name = 'alpha', init_tensor = torch.tensor(0.01))


# hI setup
hI_ns = NodeStructure(UnknownParameter)
hI_ns.set_dims(batch_dims = dim_2, param_dims = dim_1)
hI_object = UnknownParameter(hI_ns, name = 'hI')


# dh setup
dh_ns = NodeStructure(UnknownParameter)
dh_ns.set_dims(batch_dims = dim_1 + dim_2, param_dims = dim_3)
dh_object = UnknownParameter(dh_ns, name = 'dh')


# iii) Set up the dimensions for noise addition
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = dim_1 + dim_2, event_dims = dim_3)
noise_object = NoiseAddition(noise_ns, name = 'noise')




"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.alpha_object = alpha_object
        self.hI_object = hI_object
        self.dh_object = dh_object
        self.noise_object = noise_object 
        
    # Define model by forward passing, input_vars = lengths [l_A, l_B]
    def model(self, input_vars, observations = None):
        l_mat = input_vars.value
        alpha = self.alpha_object.forward()
        hI = self.hI_object.forward().T
        dh = self.dh_object.forward()
        
        scaler = torch.hstack([torch.zeros([n_config,1]), torch.ones([n_config,1])])
        y_true = hI - scaler * dh
        y_biased = y_true + torch.tan(alpha) * l_mat

        inputs = {'mean': y_biased, 'standard_deviation': sigma_true} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations)
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars, observations = None):
        pass
    
demo_probmodel = DemoProbModel()




"""
    5. Perform inference
"""
    

# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 1})
elbo = pyro.infer.Trace_ELBO()
n_steps = 300

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) Train the model

input_data = l_mat
data_cp = CalipyTensor(data, dims = dim_1 + dim_2)
# optim_results = demo_probmodel.train(input_data, data_cp, optim_opts = optim_opts)


output_data = data_cp


# Loop for optimization
# Wrap input_data and output_data into CalipyDict
input_data_io, output_data_io, subsample_index_io = preprocess_args(input_data,
                                output_data, subsample_index = None)

# Visualize
import torchviz
graphical_model = pyro.render_model(model = demo_probmodel.model,
                                    model_args= (input_data_io,),
                                    render_distributions=True, render_params=True)

output = demo_probmodel.model(input_data_io).value.tensor
comp_graph = torchviz.make_dot(output)


# Fetch optional arguments
lr = optim_opts.get('learning_rate', 0.01)
loss = pyro.infer.Trace_ELBO()
n_steps_report = 10

# Set optimizer and initialize training
svi = pyro.infer.SVI(demo_probmodel.model, demo_probmodel.guide, adam, loss)

epochs = []
loss_sequence = []
param_sequence = []
noisy_obs_sequence = []
noisy_obs_sequence_many = []

n_simu = 100
# Handle direct data input case
for step in range(n_steps):
    loss = svi.step(input_vars=input_data_io, observations=output_data_io)
    param_alpha_val = pyro.get_param_store()['Node_1__param_alpha'].clone().detach()
    param_dh_val = pyro.get_param_store()['Node_3__param_dh'].clone().detach()
    
    sim_vals = demo_probmodel.model(input_data_io).value.tensor.clone().detach()
    sim_vals_many = [demo_probmodel.model(input_data_io).value.tensor.clone().detach()[0,0] for k in range(n_simu)]
    
    if step % n_steps_report == 0:
        print(f'epoch: {step} ; loss : {loss}')
    
    
    epochs.append(step)
    loss_sequence.append(loss)
    param_sequence.append(param_alpha_val.numpy())
    noisy_obs_sequence.append(sim_vals)
    noisy_obs_sequence_many.append(torch.stack(sim_vals_many).numpy())

# noisy_obs_sequence_many_00  = [noisy_obs_sequence_many[k][0,0] for k in range(n_simu)]

# iii) Solve via handcrafted equations

dh_ls = data[0,0] - data[0,1]
tan_a_ls = (1/60)*(dh_ls - (data[1,0] - data[1,1]))
alpha_ls = torch.atan(tan_a_ls)
hI_ls = torch.tensor([[data[0,0] - tan_a_ls * l_A[0]],
                      [data[1,0] - tan_a_ls * l_A[1]]])


"""
    6. Analyse results and illustrate
"""


# i)  Plot loss

plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.title('ELBO loss')
plt.xlabel('epoch')

# ii) Print  parameters

for param, value in pyro.get_param_store().items():
    print(param, '\n', value)
    
print('True values \n alpha : {} \n dh : {} \n hI : {}'.format(alpha_true, dh_true, hI_true))
print('Values estimated by least squares \n alpha : {} \n dh : {} \n hI : {}'.format(alpha_ls,  dh_ls, hI_ls))


train_data = []
for k in range(n_steps):
    train_data.append([epochs[k], param_sequence[k], loss_sequence[k],
                       noisy_obs_sequence[k].flatten(), noisy_obs_sequence_many[k]])


import numpy as np



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

def make_training_video(train_data, gt_residuals, alpha_ls, n_steps, output_path="training_sim.gif"):
    # Plot config
    hist_bins = 50
    hist_xlim = (1.01, 1.04)
    bin_edges = np.linspace(hist_xlim[0], hist_xlim[1], hist_bins + 1)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=False)
    param_ax, obs_ax, loss_ax, hist_ax = axes

    # Titles and labels
    param_ax.set_title("Parameter Evolution")
    obs_ax.set_title("Noisy Observations")
    loss_ax.set_title("ELBO Loss")
    hist_ax.set_title("Obs distribution true and model")
    param_ax.set_ylabel("Parameter alpha")
    obs_ax.set_ylabel("Noisy Obs")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_xlabel("Epoch")
    hist_ax.set_ylabel("Density")
    hist_ax.set_xlabel("Observation")
    hist_ax.set_xlim(hist_xlim)

    # Plot ground truth alpha
    param_ax.axhline(alpha_ls, color="orange", label="Ground Truth alpha")

    # Prepare plot lines and scatter
    param_line, = param_ax.plot([], [], label="Parameter Value")
    loss_line, = loss_ax.plot([], [], label="Loss")
    obs_scatter = obs_ax.scatter([], [], alpha=0.5, label="Noisy Obs")

    # Plot static ground truth histogram
    hist_ax.hist(gt_residuals, bins=bin_edges, alpha=0.5, density=True, color="orange", label="Ground Truth")

    # Initialize model histogram bars (initially empty)
    _, _, model_patches = hist_ax.hist([], bins=bin_edges, alpha=0.5, density=True, color="blue")

    # Set axis limits
    param_ax.set_xlim(0, n_steps)
    param_ax.set_ylim(-0.1, 0.1)
    obs_ax.set_xlim(0, n_steps)
    obs_ax.set_ylim(0, 3)
    loss_ax.set_xlim(0, n_steps)
    loss_ax.set_ylim(-1e3, 1e4)
    hist_ax.set_ylim(0, 400)  # adjust for density scale

    # Legends
    param_ax.legend(loc="upper right")
    obs_ax.legend(loc="upper right")
    loss_ax.legend(loc="upper right")
    hist_ax.legend(handles=[
        Patch(facecolor="orange", alpha=0.5, label="Ground Truth"),
        Patch(facecolor="blue", alpha=0.5, label="Model"),
    ])

    fig.tight_layout(pad=1.5)

    # Unpack training data
    epochs, param_seq, loss_seq, obs_seq, resid_seq = zip(*train_data)

    def init():
        param_line.set_data([], [])
        loss_line.set_data([], [])
        obs_scatter.set_offsets(np.empty((0, 2)))
        for patch in model_patches:
            patch.set_height(0)
        return param_line, loss_line, obs_scatter, *model_patches

    def update(frame):
        # Epoch-wise data
        epoch = epochs[frame]
        param_line.set_data(epochs[:frame+1], param_seq[:frame+1])
        loss_line.set_data(epochs[:frame+1], loss_seq[:frame+1])

        xs = []
        ys = []
        for i in range(frame + 1):
            xs.extend([epochs[i]] * len(obs_seq[i]))
            ys.extend(obs_seq[i])
        if xs:
            obs_scatter.set_offsets(np.column_stack((xs, ys)))
        else:
            obs_scatter.set_offsets(np.empty((0, 2)))

        # Histogram update
        model_resids = resid_seq[frame]
        in_range = (model_resids >= bin_edges[0]) & (model_resids <= bin_edges[-1])
        filtered = model_resids[in_range]
        if len(filtered) > 0:
            counts, _ = np.histogram(filtered, bins=bin_edges, density=True)
        else:
            counts = np.zeros(len(bin_edges) - 1)
        for height, patch in zip(counts, model_patches):
            patch.set_height(height)

        return param_line, loss_line, obs_scatter, *model_patches

    anim = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                         blit=False, interval=50, repeat=False)

    anim.save(output_path, writer="pillow", fps=10, dpi=150)
    print(f"Saved animation to {output_path}")
    
make_training_video(
    train_data=train_data,
    gt_residuals=gt_resid_00,
    alpha_ls=alpha_ls,
    n_steps=n_steps,
    output_path="../illustrations/training_sim.gif"
)   




# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# class TrainingVideoMaker:
#     """
#     A reusable class for creating and saving training animations, with an option
#     to fix your axes from the start.

#     Usage:
#         animator = TrainingVideoMaker(
#             fixed_axes=True, 
#             xlim=(0, 100), 
#             ylim_param=(-2, 2),
#             ylim_obs=(-5, 5),
#             ylim_loss=(0, 1)
#         )
        
#         # inside training loop
#         for epoch in range(num_epochs):
#             param, loss, obs_list = ...
#             animator.add_data(epoch, param, loss, obs_list)
        
#         # after training finishes
#         animator.finalize("training_progress.gif")  # or .mp4, if FFmpeg is installed
#     """
#     def __init__(
#         self,
#         figsize=(6, 10),
#         sharex=False,
#         fixed_axes=False,
#         xlim=None,         # e.g., (0, 1000)
#         ylim_param=None,   # e.g., (-2, 2)
#         ylim_obs=None,     # e.g., (-5, 5)
#         ylim_loss=None,     # e.g., (0, 1)
#         ylim_hist=None     # e.g., (0, 1)
#     ):
#         # Create figure and axes
#         self.hist_bins = 20
#         # self.hist_xlim = (1, 1.05)
#         self.hist_xlim = (1, 50)
#         self.bin_edges = np.linspace(self.hist_xlim[0], self.hist_xlim[1], self.hist_bins+1)
        
#         self.fig, self.axes = plt.subplots(4, 1, figsize=figsize, sharex=sharex)
#         self.param_ax, self.obs_ax, self.loss_ax, self.hist_ax = self.axes
#         self.param_ax.set_title("Parameter Evolution")
#         self.obs_ax.set_title("Noisy Observations")
#         self.loss_ax.set_title("ELBO Loss")
#         _, _, self.patches_model = self.hist_ax.hist([], bins=self.bin_edges, alpha=0.5, color="blue", density=True)
#         self.hist_ax.set_xlim(self.hist_xlim)
#         self.hist_ax.set_title("Residuals true and model")
#         self.fig.tight_layout(pad=1.5)

#         # Prepare lines / scatter references
#         self.param_line, = self.param_ax.plot([], [], label="Parameter Value")
#         self.loss_line, = self.loss_ax.plot([], [], label="Loss")
#         # For scatter, initialize an empty Nx2 array.
#         self.obs_scatter = self.obs_ax.scatter([], [], alpha=0.5, label="Noisy Obs")

#         # Data containers
#         self.epochs = []
#         self.param_data = []
#         self.loss_data = []
#         self.obs_data = []   # list of lists/arrays of noisy observations

#         # For histogram data:
#         self.model_residuals_by_epoch = []  # store model residuals per epoch
#         self.ground_truth_residuals = gt_resid_00
#         self.model_patches = None
#         self.model_bin_edges = None

#         # Store the config
#         self.fixed_axes = fixed_axes
#         self.xlim = xlim
#         self.ylim_param = ylim_param
#         self.ylim_obs   = ylim_obs
#         self.ylim_loss  = ylim_loss

#         # Axis labels, etc.
#         self.param_ax.set_ylabel("Parameter alpha")
#         self.obs_ax.set_ylabel("Noisy Observations")
#         self.loss_ax.set_ylabel("ELBO Loss")
#         self.loss_ax.set_xlabel("Epoch")
        
#         self.param_ax.axhline(
#             alpha_ls, 
#             label="Ground Truth alpha",
#             color="orange"
#             )
        
#         self.hist_ax.set_ylabel("Count")
#         self.hist_ax.set_xlabel("Simulated obs")

#         # If user provided axis limits, set them right away
#         if xlim is not None:
#             self.param_ax.set_xlim(*xlim)
#             self.obs_ax.set_xlim(*xlim)
#             self.loss_ax.set_xlim(*xlim)

#         if ylim_param is not None:
#             self.param_ax.set_ylim(*ylim_param)

#         if ylim_obs is not None:
#             self.obs_ax.set_ylim(*ylim_obs)

#         if ylim_loss is not None:
#             self.loss_ax.set_ylim(*ylim_loss)

#         if ylim_hist is not None:
#             self.hist_ax.set_ylim(*ylim_hist)
            
#         # self.hist_ax.set_xlim([1.0, 1.05])

#         for ax in self.axes:
#             ax.grid(True)
#             ax.legend(loc="upper right")

#     def add_data(self, epoch, param, loss, obs_list, model_residuals = None):
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
        
#         # If we have residual data for this epoch, store it
#         if model_residuals is not None:
#             self.model_residuals_by_epoch.append(model_residuals)
#         else:
#             self.model_residuals_by_epoch.append([])
            
#     def _init_func(self):
#         """
#         Required by FuncAnimation for initialization.
#         """
#         # Clear line data
#         self.param_line.set_data([], [])
#         self.loss_line.set_data([], [])
#         self.obs_scatter.set_offsets(np.empty((0, 2)))

#         # Plot ground truth histogram once if it exists
#         if self.ground_truth_residuals is not None:
#             _ = self.hist_ax.hist(
#                 self.ground_truth_residuals,
#                 bins=self.hist_bins,
#                 alpha=0.5,
#                 label="Ground Truth",
#                 density=True   # normalize
#             )

#         # Initialize an empty histogram for the model
#         # We'll keep references to the patches/bins so we can update them.
#         counts, bin_edges, patches = self.hist_ax.hist(
#             [],  # no data yet
#             bins=self.hist_bins,
#             alpha=0.5,
#             label="Model",
#             density=True   # normalize
#         )
#         self.model_bin_edges = bin_edges
#         self.model_patches = patches

#         self.hist_ax.legend()

#         return (self.param_line, self.obs_scatter, self.loss_line, *patches)


#     def _update_func(self, frame_idx):
#         """
#         Update function for FuncAnimation, called for each frame.
#         """
#         # 1) Lines / scatter
#         current_epochs = self.epochs[:frame_idx+1]
#         current_params = self.param_data[:frame_idx+1]
#         current_losses = self.loss_data[:frame_idx+1]

#         # Flatten all noisy observations up to frame_idx
#         xs = []
#         ys = []
#         for i, obs_list in enumerate(self.obs_data[:frame_idx+1]):
#             xs.extend([self.epochs[i]] * len(obs_list))
#             ys.extend(obs_list)

#         self.param_line.set_data(current_epochs, current_params)
#         self.loss_line.set_data(current_epochs, current_losses)

#         if len(xs) > 0:
#             coords = np.column_stack((xs, ys))
#         else:
#             coords = np.empty((0, 2))
#         self.obs_scatter.set_offsets(coords)

#         # 2) Hist of model residuals up to current frame
#         #    Flatten everything from epoch 0..frame_idx
#         # model_resids_so_far = []
#         # for r in self.model_residuals_by_epoch[:frame_idx+1]:
#         #     model_resids_so_far.extend(r)
        
#         model_resids = self.model_residuals_by_epoch[frame_idx]
        
#         mask = (model_resids >= self.bin_edges[0]) & (model_resids <= self.bin_edges[-1])
#         filtered_resids = model_resids[mask]
        
#         if len(filtered_resids) > 0:
#             counts, _ = np.histogram(filtered_resids, bins=self.bin_edges, density=True)
#         else:
#             counts = np.zeros(len(self.bin_edges) - 1)
    
                
#         # model_resids = self.model_residuals_by_epoch[frame_idx]
#         # Now compute histogram
#         # counts, _ = np.histogram(model_resids, bins=self.model_bin_edges)

#         # Update patch heights
#         for patch, height in zip(self.patches_model, counts):
#             patch.set_height(height)
#             patch.set_visible(True) 
        
        
#         # self.hist_ax.hist(
#         #         model_resids,
#         #         bins=self.hist_bins,
#         #         alpha=0.5,
#         #         label="Model",
#         #         density=True   # normalize
#         #     )

#         # 3) Auto-scale only if user wants dynamic axes
#         if not self.fixed_axes:
#             self.param_ax.relim(); self.param_ax.autoscale_view()
#             self.obs_ax.relim();   self.obs_ax.autoscale_view()
#             self.loss_ax.relim();  self.loss_ax.autoscale_view()
#             self.hist_ax.relim();  self.hist_ax.autoscale_view()

#         return (self.param_line, self.obs_scatter, self.loss_line, *self.model_patches)


#     def finalize(self, output_filename=None, fps=15, interval=50):
#         """
#         Create the animation and optionally save it.

#         output_filename: e.g. "training_progress.mp4" or ".gif"
#         fps: frames per second
#         interval: delay between frames in ms (for live preview)
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
#             # self.anim.save(output_filename, fps=fps, dpi=150)
#             self.anim.save(output_filename, writer="pillow", fps=10, dpi=150)
#         else:
#             plt.show()



# if __name__ == "__main__":
#     animator = TrainingVideoMaker(
#         fixed_axes=True,           # lock axes
#         xlim=(0, n_steps),         # epochs as x-axis
#         ylim_param=(-0.1, 0.1),        # param range
#         ylim_obs=(0, 3),          # obs range
#         ylim_loss=(-1e3, 1e4),           # loss range
#         ylim_hist=(0, 500)           # loss range
#     )

#     for (epoch, param, loss, obs_list, model_resids) in train_data:
#         animator.add_data(epoch, param, loss, obs_list, model_resids)

#     # Save as GIF
#     animator.finalize("../illustrations/training_sim.gif")












