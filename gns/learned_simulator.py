import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: Dict,
          nparticle_types: int,
          particle_type_embedding_size,
          device="cpu"):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types
    self._particle_dimensions = particle_dimensions

    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions + 1, # qilin, with one auxiliary output
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius

    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Specify examples id for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)
    
    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        node_features, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=32)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          meta_feature: torch.tensor):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (nparticles, 3)
    velocity_sequence = time_diff(position_sequence)

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    
    # Debug graph connection
    # print("==There are {}-{}-{}-{}-{} edges per node.".format(sum(receivers == 17120), sum(receivers == 1500),sum(receivers == 1820), sum(receivers==2691), sum(receivers==100520)))
    # for i in range(len(receivers)):
    #     if receivers[i] == 1820:
    #         print(f"r==1712: {senders[i]} -> {receivers[i]}")
    # print(f"knn={sum(receivers == 17120)}")
            
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
    node_features.append(flat_velocity_sequence)
    
    # Normalised position of particles
    position_stats = self._normalization_stats["position"]
    norm_current_position = (most_recent_position - position_stats['mean']) / position_stats['std']
    node_features.append(norm_current_position)
    
    # Normalised meta feature TNTx(0),TNTy(1),TNTz(2),l(3),r(4),m(5),s(6),t(7)
    col_to_keep = [0,1,2,5,6]
    meta_feature = meta_feature[col_to_keep]
    meta_feature = torch.tile(meta_feature, (flat_velocity_sequence.shape[0], 1))
    meta_feature[particle_types==1, -1] = 10 
    node_features.append(meta_feature)
    
    # Particle type
    if self._nparticle_types > 1:
        particle_type_embeddings = self._particle_type_embedding(particle_types)
        node_features.append(particle_type_embeddings)
    # Final node_features shape (nparticles, 30) for 2D
    # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding

    # Collect edge features.
    edge_features = []

    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius
    
    # Add relative displacement between two particles as an edge feature
    # with shape (nedges, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nedges, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor) -> torch.tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          meta_feature: torch.tensor) -> torch.tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, INPUT_SEQUENCE_LENGTH, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        current_positions, nparticles_per_example, particle_types, meta_feature)
    pred = self._encode_process_decode(
        node_features, edge_index, edge_features)
    predicted_normalized_acceleration = pred[:, :self._particle_dimensions] # qilin, the first 2 or 3 dims are accelrations
    predicted_strain = pred[:, -1]  # the last dimension are strains
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    
    return next_positions, predicted_strain

  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          meta_feature: torch.tensor):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        noisy_position_sequence, nparticles_per_example, particle_types, meta_feature)
    pred = self._encode_process_decode(
        node_features, edge_index, edge_features)
    predicted_normalized_acceleration = pred[:, :self._particle_dimensions] # qilin, the first 2 or 3 dims are accelrations
    predicted_strain = pred[:, -1]  # the rest dimension are strains

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration, predicted_strain

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']

    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
