import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from model.utils.utils import (
    normalise_quat,
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    orthonormalize_by_gram_schmidt,
)
from model.utils.position_encodings import RotaryPositionEncoding3D
from model.utils.layers import RelativeCrossAttentionModule
from .act3d_diffusion_head import Act3dDiffusionHeadv3
from .act3d import Baseline
from .act3d_diffusion_utils import (
    GRIPPER_DELTAS,
    get_three_points_from_curr_action,
    visualize_actions_and_point_clouds,
    visualize_actions_and_point_clouds_video,
    visualize_diffusion_process,
    visualize_point_clouds,
)


############ Diffusion Model #############
class Act3dDiffusion(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 num_vis_ins_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 num_ghost_points_val=10000,
                 weight_tying=True,
                 gp_emb_tying=True,
                 ins_pos_emb=False,
                 num_sampling_level=3,
                 fine_sampling_ball_diameter=0.16,
                 regress_position_offset=False,
                 use_instruction=False,
                 diffusion_head="simple",
                 noise_scheduler="ddpm",
                 num_train_timesteps=10,
                #  beta_schedule="squaredcos_cap_v2",
                 beta_schedule="scaled_linear",
    ):
        super().__init__()
        self.act3d = Baseline(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            # rotation_parametrization=rotation_parametrization,
            rotation_parametrization="quat_from_query", # DEBUG <>
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=num_ghost_points,
            num_ghost_points_val=num_ghost_points_val,
            weight_tying=weight_tying,
            gp_emb_tying=gp_emb_tying,
            ins_pos_emb=ins_pos_emb,
            num_sampling_level=num_sampling_level,
            fine_sampling_ball_diameter=fine_sampling_ball_diameter,
            regress_position_offset=regress_position_offset,
            use_instruction=use_instruction
        )
        # for m in self.act3d.parameters():
        #     m.requires_grad = False

        self.rotation_parametrization = rotation_parametrization
        self.rotation_dim = 4 if "quat" in rotation_parametrization else 6
        self.action_dim = 3 + self.rotation_dim
        if diffusion_head == "simple":
            self.curr_gripper_context_head = RelativeCrossAttentionModule(
                embedding_dim, num_attn_heads, num_layers=3
            )
            self.prediction_head = Act3dDiffusionHeadv3(
                rotation_dim=self.rotation_dim,
                position_dim=3,
                embedding_dim=embedding_dim,
                num_attn_heads=num_attn_heads,
                num_cross_attn_layers=2,
                num_self_attn_layers=6,
            )
        print(self.prediction_head)
        if noise_scheduler == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                prediction_type="sample",
            )
        self.n_steps = self.noise_scheduler.config.num_train_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)
        self.register_buffer(
            'gripper_loc_bounds_for_normalize',
            torch.tensor(gripper_loc_bounds, dtype=torch.float)
        )

        # Final output layers
        self.final_gripper_embed = nn.Embedding(
            GRIPPER_DELTAS.shape[0], embedding_dim
        )
        self.curr_gripper_embed = nn.Embedding(
            1, embedding_dim
        )

        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)


    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def prepare_valid_action(self, action, threshold=True):
        """Unnormalize the position and rotation, and constraint the quaternion
        """
        if "quat" in self.rotation_parametrization:
            action = torch.cat(
                [
                    self.unnormalize_pos(action[..., :3]),
                    normalise_quat(action[..., 3:])
                ], dim=-1
            )
        else:
            action = torch.cat(
                [
                    self.unnormalize_pos(action[..., :3]), action[..., 3:]
                ], dim=-1
            )
        if threshold:
            pos_min = self.gripper_loc_bounds[0].float().to(action.device)
            pos_max = self.gripper_loc_bounds[1].float().to(action.device)
            position = action[..., :3]
            position = torch.maximum(position, pos_min)
            position = torch.minimum(position, pos_max)
            action = torch.cat([position, action[..., 3:]], dim=-1)
        return action

    def _diffusion_step_to_sampling_level(self, timestep):
        """Sampling fine (coarse) features and ghost points when T is small (large).
        """
        ratio = [(i + 1) ** 2 for i in range(self.act3d.num_sampling_level)]
        ratio = [int(i / sum(ratio) * self.n_steps) for i in ratio]
        ratio[-1] = ratio[-1] + (self.n_steps - sum(ratio))

        sampling_levels = torch.repeat_interleave(
            torch.arange(self.act3d.num_sampling_level, 0, -1) - 1,
            torch.tensor(ratio)
        ).to(timestep.device)

        # DEBUG <>
        sampling_levels = torch.clamp(sampling_levels, 0, self.act3d.num_sampling_level - 2)

        sampling_levels = sampling_levels[timestep]

        return sampling_levels

    def _sample_context_points(self, pcd, levels,
                               visible_rgb_features_pyramid,
                               visible_pcd_pyramid,
                               instruction_features,
                               visible_rgb=None):
        """
        Args:
            pcd: A tensor of shape (B, 3)
            levels: A tensor of shape (B,) indicating the sampling level
            visible_rgb_features_pyramid: A list of tensors of shape (B, ncam, C, H, W)
            visible_pcd_pyramid: A list of tensors of shape (B, ncam * H * W, 3)
            instruction_features: A tensor of shape (B, ntoken, C)
            visible_rgb: A tensor of shape (B, ncam, C, H, W) for visualization
        """

        # We assume weight tying to reuse the same module across sampling levels
        assert self.act3d.weight_tying, f"Weight tying must be enabled"

        ########### Follow Act3D to sample visible and ghost points ###########
        batch_size, num_cameras = visible_rgb_features_pyramid[0].shape[:2]
        device = visible_rgb_features_pyramid[0].device

        context_features, context_pcd, context_rgb = [], [], []
        # Sample visible points
        for i, cur_lvl in enumerate(levels):
            l2_pred_pos = (
                (pcd[i][None, :] - visible_pcd_pyramid[cur_lvl][i]) ** 2
            ).sum(-1).sqrt()
            indices = l2_pred_pos.topk(
                k=32 * 32 * num_cameras, dim=-1, largest=False
            ).indices

            visible_rgb_features_i = einops.rearrange(
                visible_rgb_features_pyramid[cur_lvl][i],
                "ncam c h w -> (ncam h w) c"
            )
            visible_rgb_features_i = visible_rgb_features_i[indices]
            visible_pcd_i = visible_pcd_pyramid[cur_lvl][i][indices]
            context_features.append(visible_rgb_features_i)
            context_pcd.append(visible_pcd_i)

            if visible_rgb is not None:
                visible_rgb_i = F.interpolate(
                    visible_rgb[i],
                    scale_factor=1. / self.act3d.downscaling_factor_pyramid[cur_lvl],
                    mode='bilinear'
                )
                visible_rgb_i = einops.rearrange(
                    visible_rgb_i, "ncam c h w -> (ncam h w) c"
                )[indices]
            else:
                visible_rgb_i = torch.ones((indices.shape[0], 3),
                                           dtype=torch.float, device=device)
                visible_rgb_i[:, 1:] = 0.0
                visible_rgb_i[:, :1] = 0.8
            context_rgb.append(visible_rgb_i)

        context_features = torch.stack(context_features, dim=0)
        context_features = einops.rearrange(context_features, "b npts c -> npts b c")
        context_pcd = torch.stack(context_pcd, dim=0)
        context_rgb = torch.stack(context_rgb, dim=0)

        # Contextualize with instructions if specified
        if self.act3d.use_instruction:
            # Assume weight tying so that we can reuse the same attn module
            # across different sampling levels
            context_features = self.vis_ins_attn_pyramid[-1](
                query=context_features, value=instruction_features,
                query_pos=None, value_pos=None
            )[-1]
            
            context_features = torch.cat(
                [context_features, instruction_features], dim=0)
            instruction_dummy_pcd = torch.zeros(
                (batch_size, 1, 3)).to(device)
            context_pcd = torch.cat(
                [context_pcd, instruction_dummy_pcd], dim=1)
            instruction_dummy_rgb = torch.zeros(
                (batch_size, 1, 3)).to(device)
            context_rgb = torch.cat(
                [context_rgb, instruction_dummy_rgb], dim=1)

        # Sample ghost points
        # ghost_pcd, ghost_rgb = [], []
        # for i, cur_lvl in enumerate(levels):
        #     ghost_pcd_i = self.act3d._sample_ghost_points(
        #         1, device, level=cur_lvl, anchor=pcd[i][None, None, :]
        #     )
        #     ghost_pcd.append(ghost_pcd_i)

        #     ghost_rgb_i = torch.ones_like(ghost_pcd_i) * 0.5
        #     ghost_rgb.append(ghost_rgb_i)
        # ghost_pcd = torch.cat(ghost_pcd, dim=0)
        # ghost_rgb = torch.cat(ghost_rgb, dim=0)

        # VISUALIZE <>
        # visualize_point_clouds([context_pcd, ghost_pcd, pcd[:, None, :]],
        #                        legends=["context", "ghost", "gripper"])

        # Contextualize ghost points with visible points
        # context_pos = self.act3d.relative_pe_layer(context_pcd)
        # Assume weight tying so that we can reuse the same attn module
        # across different sampling levels
        # ghost_features, _, _ = self.act3d._compute_ghost_point_features(
        #     ghost_pcd, context_features, context_pos,
        #     batch_size, level=-1
        # )

        return context_features, context_pcd, context_rgb

    def _contextualize_curr_gripper(self, curr_gripper, visible_rgb_features_pyramid,
                                    visible_pcd_pyramid, instruction_features=None):
        """Contextualize current gripper using image features.

        Args:
            curr_gripper: A tensor of shape (B, 8)
            visible_rgb_features_pyramid: A list of tensors of shape (B, ncam, C, H, W)
            visible_pcd_pyramid: A list of tensors of shape (B, ncam * H * W, 3)
            instruction_features: A tensor of shape (B, ntoken, C)

        Returns:
            curr_gripper_features: A tensor of shape (B, C)
            curr_gripper_pcd: A tensor of shape (B, 3)
        """

        # We assume weight tying to reuse the same module across sampling levels
        assert self.act3d.weight_tying, f"Weight tying must be enabled"

        ########### Follow Act3D to sample visible and ghost points ###########
        batch_size, num_cameras = visible_rgb_features_pyramid[0].shape[:2]
        device = visible_rgb_features_pyramid[0].device

        # Prepare curr gripper features and pcd
        curr_gripper_features = (
            self.curr_gripper_embed.weight.unsqueeze(1).repeat(1, curr_gripper.shape[0], 1)
        )
        curr_gripper_pcd = curr_gripper[:, :3]

        # Use the coarsest features for contextualization
        level = 0
        context_features = visible_rgb_features_pyramid[level]
        context_pcd = visible_pcd_pyramid[level]

        context_features = einops.rearrange(
            context_features, "b ncam c h w -> (ncam h w) b c"
        )

        # Contextualize with instructions if specified
        if self.act3d.use_instruction:
            # Assume weight tying so that we can reuse the same attn module
            # across different sampling levels
            context_features = self.vis_ins_attn_pyramid[-1](
                query=context_features, value=instruction_features,
                query_pos=None, value_pos=None
            )[-1]
            
            context_features = torch.cat(
                [context_features, instruction_features], dim=0)
            instruction_dummy_pcd = torch.zeros(
                (batch_size, 1, 3)).to(device)
            context_pcd = torch.cat(
                [context_pcd, instruction_dummy_pcd], dim=1)

        # Feed-forward cross-attention
        curr_gripper_pos = self.relative_pe_layer(curr_gripper_pcd[:, None, :])
        context_pos = self.relative_pe_layer(context_pcd)

        curr_gripper_features = self.curr_gripper_context_head(
            query=curr_gripper_features, value=context_features,
            query_pos=curr_gripper_pos, value_pos=context_pos
        )[-1]

        curr_gripper_features = curr_gripper_features.squeeze(0)

        return curr_gripper_features, curr_gripper_pcd

    def policy_forward_pass(self, action, curr_gripper, visible_rgb_features_pyramid,
                            visible_pcd_pyramid, timestep,
                            instruction_features=None, visible_rgb=None):
        """Run reverse diffusion.

        Args:
            action: A tensor of shape (B, 8) if rotation is parameterized as
                    quaternion.  Otherwise, we assume to have a 9D rotation
                    vector (3x3 flattened).
            curr_gripper: A tensor of shape (B, 8)
            visible_rgb_features_pyramid: A list of tensors of shape (B, ncam, C, H, W)
            visible_pcd_pyramid: A list of tensors of shape (B, ncam * H * W, 3)
            query_features: A tensor of shape (B, C)
            timestep: A tensor of shape (B,) indicating the diffusion timestep
            instruction_features: A tensor of shape (B, ntoken, C)
        """
        gripper_pcd = get_three_points_from_curr_action(
            action, self.rotation_parametrization
        )
        sampling_levels = self._diffusion_step_to_sampling_level(timestep)

        # Contextualize curr_gripper
        curr_gripper_features, _ = (
            self._contextualize_curr_gripper(
                curr_gripper=curr_gripper,
                visible_rgb_features_pyramid=visible_rgb_features_pyramid,
                visible_pcd_pyramid=visible_pcd_pyramid,
                instruction_features=instruction_features,
            )
        )

        # Sample context points
        context_features, context_pcd, _ = (
            self._sample_context_points(
                pcd=gripper_pcd[:, 0, :], # gripper center
                levels=sampling_levels,
                visible_rgb_features_pyramid=visible_rgb_features_pyramid,
                visible_pcd_pyramid=visible_pcd_pyramid,
                instruction_features=instruction_features,
            )
        )
        gripper_features = (
            self.final_gripper_embed.weight.unsqueeze(1).repeat(1, gripper_pcd.shape[0], 1)
        )

        # The diffusion model predicts delta position and rotation w.r.t to the
        # gripper action.
        position = action[..., :3]
        gripper_pcd = gripper_pcd - position[:, None, :]
        context_pcd = context_pcd - position[:, None, :]

        delta_position, delta_rotation = self.prediction_head(
            gripper_pcd, gripper_features,
            context_pcd, context_features,
            timestep, curr_gripper_features
        )

        # DEBUG <>
        # delta_position = torch.stack([
        #     delta_position[:, 0],
        #     delta_position[:, 1] + delta_position[:, 0],
        #     delta_position[:, 2] + delta_position[:, 0],
        # ], dim=1)
        clean_gripper = gripper_pcd + position[:, None, :] + delta_position
        clean_position = clean_gripper[:, 0]

        # clean_position = position + delta_position
        if "quat" in self.rotation_parametrization:
            clean_rotation = pytorch3d.transforms.quaternion_multiply(
                action[..., 3:7], normalise_quat(delta_rotation)
            )
            clean_rotation = normalise_quat(clean_rotation)
        else:
            clean_rotation_mat = (
                compute_rotation_matrix_from_ortho6d(delta_rotation) @
                compute_rotation_matrix_from_ortho6d(action[..., 3:9])
            )
            clean_rotation = get_ortho6d_from_rotation_matrix(clean_rotation_mat)
            assert (compute_rotation_matrix_from_ortho6d(clean_rotation) - clean_rotation_mat).abs().max() < 1e-3

        clean_action = torch.cat([clean_position, clean_rotation], dim=-1)

        return clean_action, clean_gripper, delta_rotation

    @torch.inference_mode()
    def compute_action(self, curr_gripper, visible_rgb_features_pyramid,
                       visible_pcd_pyramid,
                       instruction_features=None,
                       start_action=None, timesteps=None):
        """One-step denoising for computing loss.

        Args:
            curr_gripper: A tensor of shape (B, 8)
            visible_rgb_features_pyramid: A list of tensors of shape (B, ncam, C, H, W)
            visible_pcd_pyramid: A list of tensors of shape (B, ncam * H * W, 3)
            query_features: A tensor of shape (B, C)
            instruction_features: A tensor of shape (B, ntoken, C)
        """

        batch_size = visible_rgb_features_pyramid[0].shape[0]
        device = visible_rgb_features_pyramid[0].device
 
        self.noise_scheduler.set_timesteps(self.n_steps)

        # Sample random noise
        if start_action is None:
            normed_action = torch.randn((batch_size, self.action_dim), device=device)
            timesteps = self.noise_scheduler.timesteps
            timesteps = torch.tensor(
                timesteps.tolist() + [0], # DEBUG <>
                device=timesteps.device, dtype=timesteps.dtype
            )
        else:
            normed_action = torch.cat([
                self.normalize_pos(start_action[..., :3]),
                start_action[..., 3:]
            ], dim=-1)
            assert timesteps is not None

        # Reverse diffusion
        noisy_actions, pred_actions = [], []
        for t in timesteps:
            # Unnormalize the position and rotation, and constraint the quaternion
            unnormed_action = self.prepare_valid_action(normed_action, threshold=True)
            noisy_actions.append(unnormed_action)

            # Predict clean normed pose from unnormed noisy pose
            unnormed_out, _, _ = self.policy_forward_pass(
                action=unnormed_action,
                curr_gripper=curr_gripper,
                visible_rgb_features_pyramid=visible_rgb_features_pyramid,
                visible_pcd_pyramid=visible_pcd_pyramid,
                timestep=t * torch.ones(batch_size).to(device).long(),
                instruction_features=instruction_features,
            )

            # Our noise is added on normalized positions which range from [-1, 1].
            normed_out = torch.cat([
                self.normalize_pos(unnormed_out[..., :3]),
                unnormed_out[..., 3:]
            ], dim=-1)
            if t > 0:
                normed_action = self.noise_scheduler.step(
                    normed_out, t, normed_action
                ).prev_sample
            else:
                normed_action = normed_out
            pred_actions.append(self.prepare_valid_action(normed_out, threshold=True))

        # Unnormalize the position and rotation, and constraint the quaternion
        action = self.prepare_valid_action(normed_action, threshold=True)
        position = action[..., :3]
        rotation = action[..., 3:]

        return position, rotation, noisy_actions, pred_actions
 
    def denoise_for_loss(self, action, curr_gripper, visible_rgb_features_pyramid,
                         visible_pcd_pyramid, instruction_features=None,
                         visible_rgb=None, visible_pcd=None):
        """One-step denoising for computing loss.

        Args:
            action: A tensor of shape (B, 3 + 4) or (B, 3 + 6)
            curr_gripper: A tensor of shape (B, 8)
            visible_rgb_features_pyramid: A list of tensors of shape (B, ncam, C, H, W)
            visible_pcd_pyramid: A list of tensors of shape (B, ncam * H * W, 3)
            query_features: A tensor of shape (B, C)
            instruction_features: A tensor of shape (B, ntoken, C)
        """

        # Normalize the position and rotation.  Rescale the position to [-1, 1],
        # and constraint the quaternion to be unit norm
        normed_action = torch.cat(
            [self.normalize_pos(action[..., :3]), action[..., 3:]], dim=-1
        )

        # Sample noise
        noise = torch.randn(normed_action.shape, device=normed_action.device)

        # Sample a random timestep
        diff_timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean pose
        noisy_action= self.noise_scheduler.add_noise(
            normed_action, noise, diff_timesteps
        )

        # Unnormalize the position and rotation, and constraint the quaternion
        noisy_action = self.prepare_valid_action(noisy_action, threshold=True)

        # Predict clean normed pose from unnormed noisy pose
        pred_action, pred_gripper, pred_delta_rotation = self.policy_forward_pass(
            action=noisy_action,
            curr_gripper=curr_gripper,
            visible_rgb_features_pyramid=visible_rgb_features_pyramid,
            visible_pcd_pyramid=visible_pcd_pyramid,
            timestep=diff_timesteps,
            instruction_features=instruction_features,
            visible_rgb=visible_rgb,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        assert pred_type == "sample", f"Unsupported prediction type {pred_type}"

        position = pred_action[..., :3]
        rotation = pred_action[..., 3:]
        return position, rotation, noisy_action, pred_action, pred_gripper, pred_delta_rotation

    def forward(self, visible_rgb, visible_pcd, instruction, curr_gripper,
                gt_action, visualize=None, inference=False):
        # Compute visual features
        num_cameras = visible_rgb.shape[1]
        visible_rgb_features_pyramid, _, visible_pcd_pyramid = (
            self.act3d._compute_visual_features(
                visible_rgb, visible_pcd, num_cameras
            )
        )
        # DEBUG <>
        if "quat" not in self.rotation_parametrization:
            # ground-truth rotation is always 4D quaternion
            gt_position = gt_action[..., :3]
            gt_rotation = pytorch3d.transforms.quaternion_to_matrix(gt_action[..., 3:7])
            gt_rotation_6d = get_ortho6d_from_rotation_matrix(gt_rotation)
            assert (compute_rotation_matrix_from_ortho6d(gt_rotation_6d) - gt_rotation).abs().max() < 1e-3
            gt_action = torch.cat([
                gt_position, gt_rotation_6d
            ], dim=-1)

        predictions = {}
        if inference:
            # Sample keypose from pure noise
            position, rotation, noisy_actions, pred_actions = self.compute_action(
                curr_gripper=curr_gripper,
                visible_rgb_features_pyramid=visible_rgb_features_pyramid,
                visible_pcd_pyramid=visible_pcd_pyramid,
            )

            # VISUALIZE <>
            # if visualize:
            #     # Visualize reverse diffusion trajectories
            #     predictions["video/trajectories"] = (
            #         visualize_actions_and_point_clouds_video(
            #             visible_pcd, visible_rgb, gt_action, noisy_actions, pred_actions,
            #             save=True, rotation_param=self.rotation_parametrization
            #         )
            #     )
        else:
            # Denoise for computing loss
            position, rotation, noisy_action, _, pred_gripper, pred_delta_rotation = self.denoise_for_loss(
                action=gt_action,
                curr_gripper=curr_gripper,
                visible_rgb_features_pyramid=visible_rgb_features_pyramid,
                visible_pcd_pyramid=visible_pcd_pyramid,
                visible_rgb=visible_rgb, visible_pcd=visible_pcd,
            )
            gt_gripper_pcd = get_three_points_from_curr_action(
                gt_action, self.rotation_parametrization
            )
            predictions["gripper_position"] = pred_gripper
            predictions["gt_gripper_position"] = gt_gripper_pcd

            if "quat" not in self.rotation_parametrization:
                noisy_rotation = compute_rotation_matrix_from_ortho6d(
                    noisy_action[:, 3:9]
                )
                gt_rotation = compute_rotation_matrix_from_ortho6d(
                    gt_action[:, 3:9]
                )
                gt_delta_rotation = gt_rotation @ noisy_rotation.transpose(1, 2)
                gt_delta_rotation_6d = get_ortho6d_from_rotation_matrix(gt_delta_rotation)
                predictions["delta_rotation"] = pred_delta_rotation
                predictions["gt_delta_rotation"] = gt_delta_rotation_6d

            # VISUALIZE <>
            # normed_gt_action = torch.cat(
            #     [self.normalize_pos(gt_action[..., :3]), gt_action[..., 3:]], dim=-1
            # )
            # visualize_diffusion_process(normed_gt_action, self.noise_scheduler,
            #                             self.prepare_valid_action,
            #                             visible_rgb, visible_pcd,
            #                             self.rotation_parametrization)


            # VISUALIZE <>
            # if visualize:
            #     predictions["image/trajectories"] = (
            #         visualize_actions_and_point_clouds(
            #             visible_pcd, visible_rgb,
            #             [curr_gripper, gt_action, noisy_action, pred_action],
            #             ["input", "gt", "noisy", "diffusion"],
            #             self.rotation_parametrization
            #         )
            #     )

        predictions["position"] = position
        predictions["rotation"] = rotation

        return predictions
