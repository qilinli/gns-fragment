import torch
from gns import learned_simulator


# def get_random_walk_noise_for_position_sequence(
#     position_sequence: torch.tensor,
#     noise_std_last_step):
#     """Returns random-walk noise in the velocity applied to the position.

#     Args: 
#         position_sequence: A sequence of particle positions. Shape is
#         (nparticles, 6, dim). Includes current + last 5 positions.
#         noise_std_last_step: Standard deviation of noise in the last step.

#     """
#     velocity_sequence = learned_simulator.time_diff(position_sequence)
    
#     # We want the noise scale in the velocity at the last step to be fixed.
#     # Because we are going to compose noise at each step using a random_walk:
#     # std_last_step**2 = num_velocities * std_each_step**2
#     # so to keep `std_last_step` fixed, we apply at each step:
#     # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
#     num_velocities = velocity_sequence.shape[1]
#     velocity_sequence_noise = torch.randn(
#         list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

#     # Apply the random walk.
#     velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

#     # Integrate the noise in the velocity to the positions, assuming
#     # an Euler intergrator and a dt = 1, and adding no noise to the very first
#     # position (since that will only be used to calculate the first position
#     # change).
#     position_sequence_noise = torch.cat([
#         torch.zeros_like(velocity_sequence_noise[:, 0:1]),
#         torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

#     return position_sequence_noise



def get_random_walk_noise_for_position_sequence(position_sequence, down_scale_factor=10):
    velocity_sequence = learned_simulator.time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]

    # Define noise scales (std) for each dimension differently
    # defined by scalling down data velocity std by a factor, e.g., 10
    # down_scale_factor=10
    # noise_std_last_step_xyz = [x / down_scale_factor for x in [0.08, 0.08, 0.54]] 
    noise_std_last_step_xyz = [0.0005, 0.0005, 0.01]
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(
        list(velocity_sequence.shape)) * (1.0 / (num_velocities**0.5))

    # Add different standard deviations for each dimension
    for i in range(velocity_sequence_noise.shape[-1]):
        velocity_sequence_noise[..., i] *= noise_std_last_step_xyz[i]

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)
    
    return position_sequence_noise



