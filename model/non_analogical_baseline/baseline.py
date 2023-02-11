import einops
import torch
import torch.nn as nn

from .prediction_head import PredictionHead
from .utils import norm_tensor


class Baseline(nn.Module):
    def __init__(self,
                 position_loss="ce",
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 rotation_pooling_gaussian_spread=0.01,
                 use_ground_truth_position_for_sampling=False,
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 coarse_to_fine_sampling=True,
                 fine_sampling_cube_size=0.1):
        super().__init__()

        self.prediction_head = PredictionHead(
            loss=position_loss,
            embedding_dim=embedding_dim,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            rotation_pooling_gaussian_spread=rotation_pooling_gaussian_spread,
            num_ghost_points=num_ghost_points,
            use_ground_truth_position_for_sampling=use_ground_truth_position_for_sampling,
            coarse_to_fine_sampling=coarse_to_fine_sampling,
            fine_sampling_cube_size=fine_sampling_cube_size,
            gripper_loc_bounds=gripper_loc_bounds
        )

    def compute_action(self, pred) -> torch.Tensor:
        rotation = norm_tensor(pred["rotation"])
        return torch.cat(
            [pred["position"], rotation, pred["gripper"]],
            dim=1,
        )

    def forward(self,
                rgb_obs,
                pcd_obs,
                padding_mask,
                instruction,
                gripper,
                gt_action=None):
        visible_pcd = pcd_obs[padding_mask]

        # Undo pre-processing to feed RGB to pre-trained ResNet (from [-1, 1] to [0, 255])
        visible_rgb = einops.rearrange(rgb_obs, "b t n d h w -> (b t) n d h w")
        visible_rgb = (visible_rgb / 2 + 0.5) * 255
        visible_rgb = visible_rgb[:, :, :3, :, :]

        curr_gripper = einops.rearrange(gripper, "b t c -> (b t) c")[:, :3]

        pred = self.prediction_head(
            visible_rgb=visible_rgb,
            visible_pcd=visible_pcd,
            curr_gripper=curr_gripper,
            gt_action=gt_action,
        )

        return {
            # Action
            "position": pred["position"],
            "rotation": pred["rotation"],
            "gripper": pred["gripper"],
            # Auxiliary outputs used to compute the loss
            "visible_rgb_masks": pred["visible_rgb_masks"],
            "ghost_pcd_masks": pred["ghost_pcd_masks"],
            "all_masks": pred["all_masks"],
            "all_features": pred["all_features"],
            "all_pcd": pred["all_pcd"],
            "task": None,
        }