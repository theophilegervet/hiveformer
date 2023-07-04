import numpy as np
import einops
import torch


@torch.no_grad()
def find_cylinder_points(start, end, num_points, point_cloud):
    """
    start: (B, 3)
    end: (B, 3)
    num_points: int
    point_cloud: (B, P, 3)
    """
    device = end.device
    # Neighborhood size
    size = (end - start).abs().max(1).values  # (B,)

    # Compute line (B, num_points, 3)
    slope = (end - start) / (num_points - 1)  # (B, 3)
    line = (
        slope[:, None] * torch.arange(num_points).to(device)[None, :, None]
        + start[:, None]
    )

    # Initialize empty repository of cylinder points (B, P)
    in_cylinder = torch.zeros(point_cloud.shape[:2], device=end.device)
    in_cylinder = in_cylinder.bool()

    # Loop over line points and add neighborhoods to repository
    for p in range(num_points):
        point = line[:, p]  # (B, 3)
        dists = ((point[:, None] - point_cloud) ** 2).sum(-1).sqrt()
        in_cylinder = in_cylinder | (dists <= size[:, None])
    return in_cylinder  # (B, P)


@torch.no_grad()
def find_traj_nn(trajectory, point_cloud, nn_=64):
    """
    trajectory: (B, L, 3)
    point_cloud: (B, P, 3)
    """
    dists = ((trajectory[:, :, None] - point_cloud[:, None]) ** 2).sum(-1)
    min_dists = dists.min(1).values  # B P
    lt = trajectory.shape[1]
    indices = min_dists.topk(k=nn_ * lt, dim=-1, largest=False).indices
    return indices  # # B nn_


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def sample_ghost_points_grid(bounds, num_points_per_dim=10):
    x_ = np.linspace(bounds[0][0], bounds[1][0], num_points_per_dim)
    y_ = np.linspace(bounds[0][1], bounds[1][1], num_points_per_dim)
    z_ = np.linspace(bounds[0][2], bounds[1][2], num_points_per_dim)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    ghost_points = einops.rearrange(np.stack([x, y, z]), "n x y z -> (x y z) n")
    return ghost_points


def sample_ghost_points_uniform_cube(bounds, num_points=1000):
    x = np.random.uniform(bounds[0][0], bounds[1][0], num_points)
    y = np.random.uniform(bounds[0][1], bounds[1][1], num_points)
    z = np.random.uniform(bounds[0][2], bounds[1][2], num_points)
    ghost_points = np.stack([x, y, z], axis=1)
    return ghost_points


def sample_ghost_points_uniform_sphere(center, radius, bounds, num_points=1000):
    """Sample points uniformly within a sphere through rejection sampling."""
    ghost_points = np.empty((0, 3))
    while ghost_points.shape[0] < num_points:
        points = sample_ghost_points_uniform_cube(bounds, num_points)
        l2 = np.linalg.norm(points - center, axis=1)
        ghost_points = np.concatenate([ghost_points, points[l2 < radius]])
    ghost_points = ghost_points[:num_points]
    return ghost_points


"""
Below is a continuous 6D rotation representation adapted from
On the Continuity of Rotation Representations in Neural Networks
https://arxiv.org/pdf/1812.07035.pdf
https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
"""


def normalize_vector(v, return_mag=False):
    device = v.device
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix
