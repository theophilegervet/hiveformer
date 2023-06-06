import torch
import numpy as np
rgb = np.load('/home/zhouxian/Downloads/figure_materials/start_kinect_rgb.npy')
points = np.load('/home/zhouxian/Downloads/figure_materials/start_kinect_pc.npy')
points = points.reshape([-1, 3])
rgb = rgb.reshape([-1, 3])
original_colors = torch.tensor(rgb) / 255
N = 5000
torch.manual_seed(0)
points_ = torch.rand((N, 3)) * torch.tensor([[1, 0.5, 1]]) - 0.5
from scipy.spatial.transform import Rotation as R
rot = torch.tensor(R.from_rotvec(np.pi/180*45 * np.array([1, 0, 0])).as_matrix())
points_ = torch.transpose(torch.matmul(rot.float(), torch.transpose(points_, 0, 1)), 0, 1) + 0.5 + torch.tensor([[-0.5, -0.6, 0.3]])


import open3d as o3d
colors = original_colors * 0.4 + 0.6
# colors = original_colors * 0.3 + 0.7
colors = original_colors
# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# opt = vis.get_render_option()
# opt.background_color = np.asarray([235.0, 235.0, 235.0]) / 255

# Create an Open3D point cloud object
colors_clean = colors[points[:, 0]< 8]
points_clean = points[points[:, 0]< 8]
colors_clean = colors_clean[points_clean[:, 2]<1.1]
points_clean = points_clean[points_clean[:, 2]<1.1]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_clean)
pcd.colors = o3d.utility.Vector3dVector(colors_clean)

# Add the point cloud to the visualization
vis.add_geometry(pcd)

vis_ghost_pcd = True
vis_ghost_pcd = False
if vis_ghost_pcd:
    pcd_ghost = o3d.geometry.PointCloud()
    pcd_ghost.points = o3d.utility.Vector3dVector(points_)
    pcd_ghost.colors = o3d.utility.Vector3dVector(
        torch.tile(torch.tensor([[0.8, 0.2, 0.0]]), [N, 1])
        # torch.tile(torch.tensor([[0.9, 0.65, 0.65]]), [N, 1])
    )
    vis.add_geometry(pcd_ghost)

# Set the camera view
vis.get_render_option().point_size = 2

vis.get_view_control().convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters('open3d_pose2.json'), allow_arbitrary=True)
vis.run()
vis.destroy_window()
