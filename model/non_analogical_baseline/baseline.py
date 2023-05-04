import torch
import torch.nn as nn

from .prediction_head import PredictionHead
from model.utils.utils import norm_tensor


class Baseline(nn.Module):
    def __init__(self,
                 backbone="resnet",
                 image_size=(128, 128),
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 num_vis_ins_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 num_ghost_points_val=1000,
                 weight_tying=False,
                 gp_emb_tying=False,
                 simplify=False,
                 simplify_ins=False,
                 ins_pos_emb=False,
                 vis_ins_att=False,
                 vis_ins_att_complex=False,
                 disc_rot=False,
                 disc_rot_res=5.0,
                 num_sampling_level=2,
                 fine_sampling_ball_diameter=0.08,
                 regress_position_offset=False,
                 visualize_rgb_attn=False,
                 use_instruction=False,
                 task_specific_biases=False,
                 task_ids=[]):
        super().__init__()
        self.disc_rot = disc_rot
        self.disc_rot_res = disc_rot_res

        self.prediction_head = PredictionHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            rotation_parametrization=rotation_parametrization,
            num_ghost_points=num_ghost_points,
            num_ghost_points_val=num_ghost_points_val,
            weight_tying=weight_tying,
            gp_emb_tying=gp_emb_tying,
            simplify=simplify,
            simplify_ins=simplify_ins,
            ins_pos_emb=ins_pos_emb,
            vis_ins_att=vis_ins_att,
            vis_ins_att_complex=vis_ins_att_complex,
            disc_rot=disc_rot,
            disc_rot_res=disc_rot_res,
            num_sampling_level=num_sampling_level,
            fine_sampling_ball_diameter=fine_sampling_ball_diameter,
            gripper_loc_bounds=gripper_loc_bounds,
            regress_position_offset=regress_position_offset,
            visualize_rgb_attn=visualize_rgb_attn,
            use_instruction=use_instruction,
            task_specific_biases=task_specific_biases,
            task_ids=task_ids,
        )

    def compute_action(self, pred) -> torch.Tensor:
        if self.disc_rot:
            from scipy.spatial.transform import Rotation
            device = pred["rotation"].device
            n_rot_bin = int(360 // self.disc_rot_res)
            pred_euler = []
            for ax in range(3):
                euler_ax = torch.argmax(pred["rotation"][:, n_rot_bin*ax:n_rot_bin*(ax+1)], dim=-1) * self.disc_rot_res + self.disc_rot_res*0.5
                pred_euler.append(euler_ax)
            pred_euler = torch.permute(torch.stack(pred_euler), (1, 0))
            rotation = torch.tensor(Rotation.from_euler('zxy', pred_euler.cpu(), degrees=True).as_quat(), device=device)

        else:
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
                task_id,
                gt_action=None):

        history_length = rgb_obs.shape[1]
        instruction = instruction.unsqueeze(1).repeat(1, history_length, 1, 1)[padding_mask]
        task_id = task_id.unsqueeze(1).repeat(1, history_length)[padding_mask]
        visible_pcd = pcd_obs[padding_mask]
        visible_rgb = rgb_obs[padding_mask]
        curr_gripper = gripper[padding_mask][:, :3]
        if gt_action is not None:
            gt_action = gt_action[padding_mask]

        # Undo pre-processing to feed RGB to pre-trained backbone (from [-1, 1] to [0, 1])
        visible_rgb = (visible_rgb / 2 + 0.5)
        visible_rgb = visible_rgb[:, :, :3, :, :]

        pred = self.prediction_head(
            visible_rgb=visible_rgb,
            visible_pcd=visible_pcd,
            curr_gripper=curr_gripper,
            instruction=instruction,
            task_id=task_id,
            gt_action=gt_action,
        )
        pred["task"] = None
        return pred
