import torch
import numpy as np
import torch.nn.functional as F
import pytorch3d.transforms

import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['figure.dpi'] = 128
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PIL.Image as Image
import cv2

from model.utils.utils import (
    cross_product,
    compute_rotation_matrix_from_ortho6d
)


############# Utility functions and Layers #############
# Offset from the gripper center to three gripper points before any action
GRIPPER_DELTAS = torch.tensor([
    [0, 0, 0,],
    [0, -0.04, 0.00514],
    [0, 0.04, 0.00514],
])
GRIPPER_DELTAS_FOR_VIS = torch.tensor([
    [0, 0, 0,],
    [0, -0.08, 0.08],
    [0, 0.08, 0.08],
])


def get_gripper_matrix_from_action(action: torch.Tensor,
                                   rotation_param="quat_from_query"):
    """Converts an action to a transformation matrix.

    Args:
        action: A N-D tensor of shape (batch_size, ..., 8) if rotation is
                parameterized as quaternion.  Otherwise, we assume to have
                a 9D rotation vector (3x3 flattened).

    """
    dtype = action.dtype
    device = action.device

    position = action[..., :3]

    if "quat" in rotation_param:
        quaternion = action[..., 3:7]
        rotation = pytorch3d.transforms.quaternion_to_matrix(quaternion)
    else:
        rotation = compute_rotation_matrix_from_ortho6d(action[..., 3:9])

    shape = list(action.shape[:-1]) + [4, 4]
    gripper_matrix = torch.zeros(shape, dtype=dtype, device=device)
    gripper_matrix[..., :3, :3] = rotation
    gripper_matrix[..., :3, 3] = position
    gripper_matrix[..., 3, 3] = 1

    return gripper_matrix


def get_three_points_from_curr_action(gripper: torch.Tensor,
                                      rotation_param="quat_from_query",
                                      for_vis=False):
    gripper_matrices = get_gripper_matrix_from_action(gripper, rotation_param)
    bs = gripper.shape[0]
    if for_vis:
        pcd = GRIPPER_DELTAS_FOR_VIS.unsqueeze(0).repeat(bs, 1, 1).to(gripper.device)
    else:
        pcd = GRIPPER_DELTAS.unsqueeze(0).repeat(bs, 1, 1).to(gripper.device)

    pcd = torch.cat([pcd, torch.ones_like(pcd[..., :1])], dim=-1)
    pcd = pcd.permute(0, 2, 1)

    pcd = (gripper_matrices @ pcd).permute(0, 2, 1)
    pcd = pcd[..., :3]

    return pcd


def inverse_transform_pcd_with_action(pcd: torch.Tensor, action: torch.Tensor,
                                      rotation_param: str = "quat_from_query"):
    mat = get_gripper_matrix_from_action(action, rotation_param).inverse()

    pcd = torch.cat([pcd, torch.ones_like(pcd[..., :1])], dim=-1)
    pcd = pcd.permute(0, 2, 1)

    pcd = (mat @ pcd).permute(0, 2, 1)
    pcd = pcd[..., :3]

    return pcd


############# Visualization utility functions #############
def visualize_actions_and_point_clouds_video(visible_pcd, visible_rgb,
                                             gt_pose, noisy_poses, pred_poses,
                                             save=True, rotation_param="quat_from_query"):
    """Visualize by plotting the point clouds and gripper pose as video.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gt_pose: A tensor of shape (B, 8)
        noisy_poses: A list of tensors of shape (B, 8)
        pred_poses: A list of tensors of shape (B, 8)
    """
    images, rand_inds = [], None
    for noisy_pose, pred_pose in zip(noisy_poses, pred_poses):
        image, rand_inds = visualize_actions_and_point_clouds(
            visible_pcd, visible_rgb,
            [gt_pose, noisy_pose, pred_pose],
            ["gt", "noisy", "pred"],
            ["d", "o", "*"],
            save=False,
            rotation_param=rotation_param,
            rand_inds=rand_inds,
        )
        images.append(image)
    if save:
        pil_images = []
        for img in images:
            pil_images.extend([Image.fromarray(img)] * 2)
        pil_images[0].save("diff_trajs.gif", save_all=True,
                           append_images=pil_images[1:], duration=1, loop=0)
    video = np.stack(images, axis=0)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
    return video

def visualize_actions_and_point_clouds(visible_pcd, visible_rgb,
                                       gripper_pose_trajs, legends=[], markers=[],
                                       save=True, rotation_param="quat_from_query",
                                       rand_inds=None):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gripper_pose_trajs: A list of tensors of shape (B, 8)
        legends: A list of strings indicating the legend for each trajectory
    """
    gripper_pose_trajs = [t.data.cpu() for t in gripper_pose_trajs]

    cur_vis_pcd = visible_pcd[0].permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    cur_vis_rgb = visible_rgb[0].permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    if rand_inds is None:
        rand_inds = torch.randperm(cur_vis_pcd.shape[0]).data.cpu().numpy()[:5000]
        rand_inds = rand_inds[cur_vis_pcd[rand_inds, 2] >= 0.25] # FIXME <>
    fig = plt.figure()
    canvas = fig.canvas
    # ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(cur_vis_pcd[rand_inds, 0],
               cur_vis_pcd[rand_inds, 1],
               cur_vis_pcd[rand_inds, 2],
               c=cur_vis_rgb[rand_inds], s=1)

    # predicted gripper pose
    cont_range_inds = np.linspace(0, 1, len(gripper_pose_trajs)).astype(np.float32)
    cm = plt.get_cmap('brg')
    colors = cm(cont_range_inds)
    legends = (legends if len(legends) == len(gripper_pose_trajs)
               else [""] * len(gripper_pose_trajs))
    markers = (markers if len(markers) == len(gripper_pose_trajs)
                else ["*"] * len(gripper_pose_trajs))
    for gripper_pose, color, legend, marker in (
        zip(gripper_pose_trajs, colors, legends, markers)
        ):
        gripper_pcd = get_three_points_from_curr_action(
            gripper_pose, rotation_param=rotation_param, for_vis=True
        )
        ax.plot(gripper_pcd[0, [1, 0, 2], 0],
                gripper_pcd[0, [1, 0, 2], 1],
                gripper_pcd[0, [1, 0, 2], 2],
                c=color,
                markersize=1, marker=marker,
                linestyle='--', linewidth=1,
                label=legend)
        polygons = compute_rectangle_polygons(gripper_pcd[0])
        for poly_ind, polygon in enumerate(polygons):
            polygon = Poly3DCollection(polygon, facecolors=color)
            alpha = 0.5 if poly_ind == 0 else 1.3
            polygon.set_edgecolor([min(c * alpha, 1.0) for c in color])
            ax.add_collection3d(polygon)

    fig.tight_layout()
    ax.legend(loc="lower center", ncol=len(gripper_pose_trajs))
    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                          [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110] # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1)
    ], axis=0)
    if save:
        Image.fromarray(images, mode='RGB').save('diff_traj.png')

    plt.close()

    return images, rand_inds


def visualize_point_clouds(pcd_list, rgbs=[], legends=[], save=True):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        pcd_list: A list of tensor of shape (B, npoints, 3)
        legends: A list of strings indicating the legend for each trajectory
    """
    pcd_list = [p.data.cpu() for p in pcd_list]

    # predicted gripper pose
    cont_range_inds = np.linspace(0, 1, len(pcd_list)).astype(np.float32)
    cm = plt.get_cmap('rainbow')
    colors = cm(cont_range_inds)
    legends = (legends if len(legends) == len(pcd_list)
               else [""] * len(pcd_list))
    rgbs = (rgbs if len(rgbs) == len(pcd_list)
               else [None] * len(pcd_list))

    fig = plt.figure()
    canvas = fig.canvas
    ax = fig.add_subplot(projection='3d')
    for i in range(len(pcd_list)):
        cur_pcd = pcd_list[i][0]
        num_points = cur_pcd.shape[0]
        if num_points > 1000:
            inds = torch.randperm(num_points).data.cpu().numpy()[:1000]
        else:
            inds = torch.arange(num_points).data.cpu().numpy()

        # inds = inds[cur_pcd[inds, 2] >= 0.25] # FIXME <>
        s = 10 if legends[i] == "gripper" else 3
        if rgbs[i] is None:
            color = colors[i]
        else:
            color = rgbs[i][0].data.cpu().numpy()[inds]
        ax.scatter(cur_pcd[inds, 0],
                   cur_pcd[inds, 1],
                   cur_pcd[inds, 2],
                   c=color, s=s, label=legends[i])

    fig.tight_layout()
    ax.legend()
    ax.view_init(elev=15, azim=0, roll=0)
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    if save:
        Image.fromarray(image, mode='RGB').save('point_clouds.png')

    plt.close()

    return image


def visualize_diffusion_process(action, noise_scheduler, norm_fn,
                                visible_rgb, visible_pcd,
                                rotation_param="quat_from_query"):
    """Visualize diffusion process
    """
    action = action[[0]]

    # Visualize Diffusion process
    images = []
    # for t in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        noisy_actions = []
        for i in range(25):
            noise = torch.randn(action.shape, device=action.device)
            noisy_action = noise_scheduler.add_noise(
                action, noise,
                torch.ones((1, ), dtype=torch.long, device=action.device).mul(t)
            )
            noisy_action = norm_fn(noisy_action)
            noisy_actions.append(noisy_action)

        image = visualize_actions_and_point_clouds(
            visible_pcd, visible_rgb, noisy_actions, save=False,
            rotation_param=rotation_param
        )
        images.append(image)
    images = np.concatenate(images, axis=1)
    Image.fromarray(images, mode='RGB').save('diffusion_steps.png')


def build_rectangle_points(center, axis_h, axis_w, axis_d, h, w, d):
    def _helper(cur_points, axis, size):
        points = []
        for p in cur_points:
            points.append(p + axis * size / 2)
        for p in cur_points:
            points.append(p - axis * size / 2)
        return points

    points = _helper([center], axis_h, h)
    points = _helper(points, axis_w, w)
    points = _helper(points, axis_d, d)

    return points


def make_polygons(points):
    """Make polygons from 8 side points of a rectangle
    """
    def _helper(four_points):
        center = four_points.mean(axis=0, keepdims=True)
        five_points = np.concatenate([four_points, center], axis=0)
        return [five_points[[0,1,-1]],
                five_points[[0,2,-1]],
                five_points[[0,3,-1]],
                five_points[[1,2,-1]],
                five_points[[1,3,-1]],
                five_points[[2,3,-1]]]

    polygons = (
        _helper(points[:4])
        + _helper(points[-4:])
        + _helper(points[[0,1,4,5]])
        + _helper(points[[2,3,6,7]])
        + _helper(points[[0,2,4,6]])
        + _helper(points[[1,3,5,7]])
    )
    return polygons


def compute_rectangle_polygons(points):
    p1, p2, p3 = points.chunk(3, 0)

    line12 = p2 - p1
    line13 = p3 - p1

    axis_d = F.normalize(cross_product(line12, line13))
    axis_w = F.normalize(p3 - p2)
    axis_h = F.normalize(cross_product(axis_d, axis_w))
    
    length23 = torch.norm(p3 - p2, dim=-1)
    length13 = (line13 * axis_h).sum(-1).abs()
    rectangle1 = build_rectangle_points(p1, axis_d, axis_w, axis_h,
                                        0.03, length23, length13 / 2)
    rectangle2 = build_rectangle_points(p2, axis_d, axis_w, axis_h,
                                        0.03, length23 / 4, length13 * 2)
    rectangle3 = build_rectangle_points(p3, axis_d, axis_w, axis_h,
                                        0.03, length23 / 4 , length13 * 2)

    rectangle1 = torch.cat(rectangle1, dim=0).data.cpu().numpy()
    rectangle2 = torch.cat(rectangle2, dim=0).data.cpu().numpy()
    rectangle3 = torch.cat(rectangle3, dim=0).data.cpu().numpy()

    polygon1 = make_polygons(rectangle1)
    polygon2 = make_polygons(rectangle2)
    polygon3 = make_polygons(rectangle3)

    return polygon1, polygon2, polygon3
