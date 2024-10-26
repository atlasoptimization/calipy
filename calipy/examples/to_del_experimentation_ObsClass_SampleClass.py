import pyro
import torch
import contextlib
import itertools

class CalipyObservation:
    """
    Stores observations along with batch and event dimensions.
    Each observation has a unique name for reference during sampling.
    """
    def __init__(self, observations, batch_shape, event_shape):
        self.observations = observations  # Should be a tensor
        self.batch_shape = batch_shape
        self.event_shape = event_shape
        self.names = self._generate_unique_names()

    def _generate_unique_names(self):
        # Generate unique names for each observation based on indices
        indices = list(itertools.product(*[range(size) for size in self.batch_shape]))
        names = ['obs_' + '_'.join(map(str, idx)) for idx in indices]
        return names

    def get_observation(self, idx):
        # Retrieve observation tensor and name for a given index
        obs = self.observations[idx]
        name = self.names[self._flat_index(idx)]
        return name, obs

    def _flat_index(self, idx):
        # Convert multi-dimensional index to flat index
        flat_idx = 0
        for i, ind in enumerate(idx):
            flat_idx *= self.batch_shape[i]
            flat_idx += ind
        return flat_idx

class CalipySample:
    """
    Holds sampled data, preserving batch and event dimensions.
    Provides methods for data access and manipulation.
    """
    def __init__(self, data, batch_shape, event_shape, vectorizable):
        self.data = data
        self.batch_shape = batch_shape
        self.event_shape = event_shape
        self.vectorizable = vectorizable

    def __getitem__(self, idx):
        return self.data[idx]

    def as_tensor(self):
        return self.data

def calipy_sample(name, dist, plate_names, plate_sizes, vectorizable=True, obs=None, subsample_indices=None):
    """
    Flexible sampling function handling multiple plates and four cases based on obs and subsample_indices.

    Parameters:
    -----------
    name : str
        Base name for the sample site.
    dist : pyro.distributions.Distribution
        The distribution to sample from.
    plate_names : list of str
        Names of the plates (batch dimensions).
    plate_sizes : list of int
        Sizes of the plates (batch dimensions).
    vectorizable : bool, optional
        If True, uses vectorized sampling. If False, uses sequential sampling. Default is True.
    obs : CalipyObservation or None, optional
        Observations wrapped in CalipyObservation. If provided, sampling is conditioned on these observations.
    subsample_indices : list of torch.Tensor or None, optional
        Subsample indices for each plate dimension. If provided, sampling is performed over these indices.

    Returns:
    --------
    CalipySample
        The sampled data, preserving batch and event dimensions.
    """
    event_shape = dist.event_shape
    n_plates = len(plate_sizes)

    if vectorizable:
        # Vectorized sampling using pyro.plate
        with contextlib.ExitStack() as stack:
            # Determine dimensions for plates
            dims = [-(n_plates + len(event_shape)) + i for i in range(n_plates)]
            current_obs = obs.observations if obs is not None else None

            # Handle multiple plates
            for i, (plate_name, plate_size, dim) in enumerate(zip(plate_names, plate_sizes, dims)):
                subsample = subsample_indices[i] if subsample_indices is not None else None
                size = plate_size
                stack.enter_context(pyro.plate(plate_name, size=size, subsample=subsample, dim=dim))

                # Index observations if subsampling
                if current_obs is not None and subsample is not None:
                    current_obs = torch.index_select(current_obs, dim=dim, index=subsample)

            # Sample data
            data = pyro.sample(name, dist, obs=current_obs)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=True)
    else:
        # Non-vectorized sampling
        samples = []
        batch_ranges = []

        # Determine ranges for each plate
        if subsample_indices is not None:
            for idx in subsample_indices:
                batch_ranges.append(idx.tolist())
        else:
            for size in plate_sizes:
                batch_ranges.append(list(range(size)))

        # Iterate over all combinations of indices
        for idx in itertools.product(*batch_ranges):
            idx_dict = {plate_names[i]: idx[i] for i in range(n_plates)}
            idx_str = '_'.join(f"{plate_names[i]}_{idx[i]}" for i in range(n_plates))

            # Get observation if available
            if obs is not None:
                obs_name, obs_value = obs.get_observation(idx)
            else:
                obs_name, obs_value = None, None

            # Sample with unique name
            sample_name = f"{name}_{idx_str}"
            sample_value = pyro.sample(sample_name, dist, obs=obs_value)
            samples.append((idx, sample_value))

        # Construct tensor from samples
        # Assuming samples are in order; otherwise, sorting or indexing may be needed
        sample_values = [s[1] for s in samples]
        data = torch.stack(sample_values).reshape(*[len(r) for r in batch_ranges], *event_shape)
        batch_shape = data.shape[:n_plates]
        return CalipySample(data, batch_shape, event_shape, vectorizable=False)


# Define the distribution
mu = torch.zeros(2)
sigma = torch.eye(2)
dist = pyro.distributions.MultivariateNormal(mu, sigma)

# Observations (if any)
observations = torch.randn(5, 2)  # Assuming batch size of 5 and event size of 2
obs = CalipyObservation(observations, batch_shape=[5], event_shape=[2])

# Subsample indices (if any)
subsample_indices = [torch.tensor([0, 2, 4])]

# Plate names and sizes
plate_names = ['batch']
plate_sizes = [5]

# Sample using calipy_sample
sample_result = calipy_sample('my_sample', dist, plate_names, plate_sizes, vectorizable=True, obs=obs, subsample_indices=subsample_indices)

# Access the sampled data
sampled_data = sample_result.as_tensor()
print(sampled_data.shape)  # Should reflect the subsampled batch size and event shape
