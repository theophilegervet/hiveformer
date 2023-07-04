import torch
from torch import nn
from torch.nn import functional as F

from model.utils.utils import normalise_quat
from .regression_head import TrajectoryHead


class TrajectoryRegressor(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 use_instruction=False,
                 use_goal=False,
                 use_goal_at_test=True,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 weight_tying=False,
                 gripper_loc_bounds=None,
                 diffusion_head=None):
        super().__init__()
        self.prediction_head = TrajectoryHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            use_instruction=use_instruction,
            use_goal=use_goal,
            feat_scales_to_use=feat_scales_to_use,
            attn_rounds=attn_rounds,
            weight_tying=weight_tying
        )
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def policy_forward_pass(self, fixed_inputs):
        # Parse inputs
        (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        ) = fixed_inputs

        trajectory = self.prediction_head(
            trajectory_mask,
            visible_rgb=rgb_obs,
            visible_pcd=pcd_obs,
            curr_gripper=curr_gripper,
            goal_gripper=goal_gripper,
            instruction=instruction
        )
        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        goal_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        )

        # Regress the trajectory
        pred = self.policy_forward_pass(fixed_inputs)
        trajectory = pred[-1]  # last head's output
        # trajectory[:, -1] = goal_gripper

        # Normalize quaternion
        trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        goal_gripper,
        run_inference=False
    ):
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                goal_gripper
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        )

        # Regress the trajectory
        pred = self.policy_forward_pass(fixed_inputs)

        # Padding mask (Don't compute loss on pad)
        pad_mask = torch.zeros_like(gt_trajectory)
        for d in range(len(pad_mask)):
            neg_len_ = -trajectory_mask[d].sum().long()
            if neg_len_ < 0:
                pad_mask[d][neg_len_:] = 1
        pad_mask = pad_mask.bool()

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            loss = F.mse_loss(layer_pred, gt_trajectory, reduction='none')
            loss_mask = ~pad_mask
            loss = loss * loss_mask.type(loss.dtype)
            loss = loss = loss.sum() / loss_mask.sum()
            total_loss = total_loss + loss
        return total_loss
