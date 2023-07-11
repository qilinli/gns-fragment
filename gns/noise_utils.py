import torch
from gns import learned_simulator


def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.tensor,
    noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position.

    Args: 
        position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
        noise_std_last_step: Standard deviation of noise in the last step.

    """
    velocity_sequence = learned_simulator.time_diff(position_sequence)

    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(
        list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

    # Apply the random walk.
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate the noise in the velocity to the positions, assuming
    # an Euler intergrator and a dt = 1, and adding no noise to the very first
    # position (since that will only be used to calculate the first position
    # change).
    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise


# Apply noise of different scales to x,y,z

# # Define noise scales for each dimension
# noise_scales = [0.1, 0.2, 0.3]  # Adjust these values according to your desired noise scales

# # Generate random noise tensors for each dimension
# noise_tensors = []
# for scale in noise_scales:
#     noise = torch.randn(list(velocity_sequence.shape)) * (scale / num_velocities ** 0.5)
#     noise_tensors.append(noise)

# # Apply the random walk to each dimension separately
# velocity_sequence_noise = torch.stack(noise_tensors, dim=-1).cumsum(dim=1)

# # Integrate the noise in the velocity to the positions
# position_sequence_noise = torch.cat([
#     torch.zeros_like(velocity_sequence_noise[:, 0:1]),
#     velocity_sequence_noise.cumsum(dim=1)], dim=1)
