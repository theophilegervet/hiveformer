from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
from torch import nn
from torch.nn import functional as F

from model.utils.utils import normalise_quat


class DiffusionHLPlanner(nn.Module):

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
        if diffusion_head == "simple":
            from .diffusion_head import DiffusionHead
            self.prediction_head = DiffusionHead(
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
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
        )
        self.n_steps = self.noise_scheduler.config.num_train_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def policy_forward_pass(self, keypose, timestep, fixed_inputs):
        # Parse inputs
        (
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            gt_action
        ) = fixed_inputs

        noise = self.prediction_head(
            keypose,
            timestep,
            visible_rgb=rgb_obs,
            visible_pcd=pcd_obs,
            curr_gripper=curr_gripper,
            instruction=instruction,
            gt_action=gt_action
        )
        return noise

    def conditional_sample(self, condition_data, fixed_inputs):
        # Random keypose
        keypose = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )

        # Iterative denoising
        self.noise_scheduler.set_timesteps(self.n_steps)
        for t in self.noise_scheduler.timesteps:
            out = self.policy_forward_pass(
                keypose,
                t * torch.ones(len(keypose)).to(keypose.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            keypose = self.noise_scheduler.step(
                out, t, keypose
            ).prev_sample

        return keypose

    def compute_action(
        self,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        gt_action
    ):
        # normalize all pos
        pcd_obs = torch.permute(
            self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2])),
            [0, 1, 4, 2, 3]
        )
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        if gt_action is not None:
            gt_action[..., :3] = self.normalize_pos(gt_action[..., :3])

        # Prepare inputs
        fixed_inputs = (
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper[..., :3],
            gt_action
        )

        # Sample
        keypose = self.conditional_sample(
            torch.zeros(curr_gripper.shape, device=rgb_obs.device),
            fixed_inputs
        )

        # Normalize quaternion
        keypose[..., 3:7] = normalise_quat(keypose[..., 3:7])
        # unnormalize position
        keypose[..., :3] = self.unnormalize_pos(keypose[..., :3])

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
        run_inference=False
    ):
        if run_inference:
            return self.compute_action(
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                gt_action
            )
        # normalize all pos
        gt_action[..., :3] = self.normalize_pos(gt_action[..., :3])
        pcd_obs = torch.permute(
            self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2])),
            [0, 1, 4, 2, 3]
        )
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Prepare inputs
        fixed_inputs = (
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper[..., :3],
            gt_action
        )

        # Sample noise
        noise = torch.randn(gt_action.shape, device=gt_action.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean keyposes
        noisy_keypose = self.noise_scheduler.add_noise(
            gt_action, noise, timesteps
        )

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_keypose, timesteps,
            fixed_inputs
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = gt_action
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            total_loss = total_loss + F.mse_loss(layer_pred, target)
        return total_loss
