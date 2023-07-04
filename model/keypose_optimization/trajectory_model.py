import torch
from torch import nn
from torch.nn import functional as F

from model.utils.utils import normalise_quat
from .trajectory_head import TrajectoryHead


class TrajectoryHLRegressor(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 num_sampling_level=3,
                 use_instruction=False,
                 use_rgb=True,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 gripper_loc_bounds=None,
                 diffusion_head="simple"):
        super().__init__()
        self.prediction_head = TrajectoryHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            num_sampling_level=num_sampling_level,
            use_instruction=use_instruction,
            use_rgb=use_rgb,
            feat_scales_to_use=feat_scales_to_use,
            attn_rounds=attn_rounds
        )
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def policy_forward_pass(self, fixed_inputs):
        # Parse inputs
        (
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper
        ) = fixed_inputs

        keypose = self.prediction_head(
            visible_rgb=rgb_obs,
            visible_pcd=pcd_obs,
            curr_gripper=curr_gripper,
            instruction=instruction
        )
        return keypose

    def compute_action(
        self,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper
    ):
        # Sample
        keypose = self.forward(
            torch.zeros_like(curr_gripper),
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            return_loss=False
        )

        # Normalize quaternion
        keypose[..., 3:7] = normalise_quat(keypose[..., 3:7])
        # unnormalize position
        # keypose[..., :3] = self.unnormalize_pos(keypose[..., :3])

        return keypose

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
        gt_action,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        return_loss=True,
        run_inference=False
    ):
        if run_inference:
            return self.compute_action(
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper
            )
        # normalize all pos
        # gt_action[..., :3] = self.normalize_pos(gt_action[..., :3])
        # pcd_obs = torch.permute(
        #     self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2])),
        #     [0, 1, 4, 2, 3]
        # )
        # curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Prepare inputs
        fixed_inputs = (
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper[..., :3]
        )

        # Regress the trajectory
        pred = self.policy_forward_pass(fixed_inputs)
        if not return_loss:
            return pred[-1]

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            total_loss = (
                total_loss
                + F.mse_loss(layer_pred[:, :3], gt_action[:, :3])
                + F.mse_loss(layer_pred[:, 3:7], gt_action[:, 3:7])
                + F.mse_loss(layer_pred[:, 7:8], gt_action[:, 7:8])
            )
        return total_loss
