import torch
import numpy as np

class SamplesDataset(torch.utils.data.Dataset):

    def __init__(self, path, input_length_sequence):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        self._data = [item for _, item in np.load(path, allow_pickle=True).items()]

        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        # input_length_sequence is 6 if 5 timesteps are used to predict the 6th step
        self._input_length_sequence = input_length_sequence
        self._dimension = self._data[0][0].shape[-1]  # particle dim, 2 for xy, 3 for xyz
        
        # Qilin, skip particle types and strains in self._data
        self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _, _ in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths)+1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training data.
        # Given input_length_sequency=6, we take positions of past 6 steps
        # and use it to compute past 5 velocities
        positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        particle_type = np.full(positions.shape[0], self._data[trajectory_idx][1], dtype=int)
        n_particles_per_example = positions.shape[0]
        next_position = self._data[trajectory_idx][0][time_idx]
        next_strain = self._data[trajectory_idx][2][time_idx]

        # data sample as a diectionary consisting of input, output, meta
        data_sample = {'input': {}, 'output': {}, 'meta': {}}
        data_sample['input']['positions'] = positions
        data_sample['input']['particle_type'] = particle_type
        data_sample['input']['n_particles_per_example'] = n_particles_per_example
        data_sample['output']['next_position'] = next_position
        data_sample['output']['next_strain'] = next_strain
        data_sample['meta']['trajectory_idx'] = trajectory_idx
        data_sample['meta']['time_idx'] = time_idx
        
        return data_sample 

def collate_fn(batch):
    # collate data to a batch
    position_list = []
    particle_type_list = []
    n_particles_per_example_list = []
    next_position_list = []
    next_strain_list = []
    trajectory_idx_list = []
    time_idx_list = []

    for data_sample in batch:
        position_list.append(data_sample['input']['positions'])
        particle_type_list.append(data_sample['input']['particle_type'])
        n_particles_per_example_list.append(data_sample['input']['n_particles_per_example'])
        next_position_list.append(data_sample['output']['next_position'])
        next_strain_list.append(data_sample['output']['next_strain'])
        trajectory_idx_list.append(data_sample['meta']['trajectory_idx'])
        time_idx_list.append(data_sample['meta']['time_idx'])

    # data batch as a diectionary consisting of input, output, meta
    # of type torch.float32 or int (particle_type, n_particles_per_example)
    data_batch = {'input': {}, 'output': {}, 'meta': {}}
    data_batch['input']['positions'] = torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous()            # shape (nparticles*batch_size, input_sequence_length, dimension)
    data_batch['input']['particle_type'] = torch.tensor(np.concatenate(particle_type_list)).contiguous()                # shape (nparticles*batch_size,)
    data_batch['input']['n_particles_per_example'] = torch.tensor(n_particles_per_example_list).contiguous()            # shape (batch_size,)
    data_batch['output']['next_position'] = torch.tensor(np.vstack(next_position_list)).to(torch.float32).contiguous()  # shape (nparticles*batch_size, dimension)
    data_batch['output']['next_strain'] = torch.tensor(np.concatenate(next_strain_list)).to(torch.float32).contiguous() # shape (nparticles*batch_size,)
    data_batch['meta']['trajectory_idx'] = torch.tensor(trajectory_idx_list).contiguous()                               # shape (batch_size,)
    data_batch['meta']['time_idx'] = torch.tensor(time_idx_list).contiguous()                                           # shape (batch_size,)
    
    return data_batch

class TrajectoriesDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO (jpv): allow_pickle=True is potential security risk. See docs.
        self._data = [item for _, item in np.load(path, allow_pickle=True).items()]
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        positions, _particle_type, strains = self._data[idx]   # qilin, with strain
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, timesteps, dimension
        particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
        n_particles_per_example = positions.shape[0]
        
        # data trajectory as a dictionary
        data_traj = {}
        # shape (nparticles, timesteps, dimension)
        data_traj['positions'] = torch.tensor(positions).to(torch.float32).contiguous() 
        # shape (nparticles,)
        data_traj['particle_type'] = torch.tensor(particle_type).contiguous()  
        # scalar == nparticles
        data_traj['n_particles_per_example'] = torch.tensor(int(n_particles_per_example), 
                                                            dtype=torch.int32)
        # shape (nparticles,)
        data_traj['strains'] = torch.tensor(strains).to(torch.float32).contiguous()
         # scalar
        data_traj['trajectory_idx'] = idx                                              
    
        return data_traj


def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)

def get_data_loader_by_trajectories(path):
    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)
