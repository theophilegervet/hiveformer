from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
from torch import nn
from torch.nn import functional as F

from model.utils.utils import normalise_quat


class DiffusionPlanner(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 ins_pos_emb=False,
                 num_sampling_level=3,
                 use_instruction=False,
                 use_goal=False,
                 use_goal_at_test=True,
                 use_rgb=True,
                 gripper_loc_bounds=None,
                 positional_features="none",
                 diffusion_head="simple"):
        super().__init__()
        self._use_goal = use_goal
        self._use_goal_at_test = use_goal_at_test
        if diffusion_head == "simple":
            from .diffusion_head_simple import DiffusionHead
            self.prediction_head = DiffusionHead(
                backbone=backbone,
                image_size=image_size,
                embedding_dim=embedding_dim,
                output_dim=output_dim,
                num_vis_ins_attn_layers=num_vis_ins_attn_layers,
                num_query_cross_attn_layers=num_query_cross_attn_layers,
                ins_pos_emb=ins_pos_emb,
                num_sampling_level=num_sampling_level,
                use_instruction=use_instruction,
                positional_features=positional_features,
                use_goal=use_goal,
                use_rgb=use_rgb
            )
        elif diffusion_head == "unconditional":
            from .diffusion_head_unconditional import DiffusionHead
            self.prediction_head = DiffusionHead(
                backbone=backbone,
                image_size=image_size,
                embedding_dim=embedding_dim,
                output_dim=output_dim,
                num_vis_ins_attn_layers=num_vis_ins_attn_layers,
                ins_pos_emb=ins_pos_emb,
                num_sampling_level=num_sampling_level,
                use_instruction=use_instruction,
                positional_features=positional_features,
                use_goal=use_goal
            )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            # clip_sample=False,
            beta_schedule="squaredcos_cap_v2",
        )
        self.n_steps = self.noise_scheduler.config.num_train_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        ) = fixed_inputs

        # Undo pre-processing to feed RGB to pre-trained backbone (from [-1, 1] to [0, 1])
        rgb_obs = (rgb_obs / 2 + 0.5)
        rgb_obs = rgb_obs[:, :, :3, :, :]

        noise = self.prediction_head(
            trajectory,
            trajectory_mask,
            timestep,
            visible_rgb=rgb_obs,
            visible_pcd=pcd_obs,
            curr_gripper=curr_gripper,
            goal_gripper=goal_gripper,
            instruction=instruction
        )
        return noise

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        # Random trajectory, conditioned on start-end
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        trajectory[condition_mask] = condition_data[condition_mask]

        # Iterative denoising
        self.noise_scheduler.set_timesteps(self.n_steps)
        for t in self.noise_scheduler.timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            trajectory = self.noise_scheduler.step(
                out, t, trajectory
            ).prev_sample
            trajectory[condition_mask] = condition_data[condition_mask]

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
        # normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2])), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper[..., :3],
            goal_gripper[..., :3]
        )

        # Condition on start-end pose
        B, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        # start pose
        cond_data[:, 0] = curr_gripper
        cond_mask[:, 0] = 1
        # end pose
        if self._use_goal_at_test:
            for d in range(len(cond_data)):
                neg_len_ = -trajectory_mask[d].sum().long()
                cond_data[d][neg_len_ - 1] = goal_gripper[d]
                cond_mask[d][neg_len_ - 1:] = 1
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

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
        goal_gripper
    ):
        # normalize all pos
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2])), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper[..., :3],
            goal_gripper[..., :3]
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        # start pose
        cond_data[:, 0] = curr_gripper
        cond_mask[:, 0] = 1
        # end pose
        if self._use_goal:
            for d in range(len(cond_data)):
                neg_len_ = -trajectory_mask[d].sum().long()
                cond_data[d][neg_len_ - 1] = goal_gripper[d]
                cond_mask[d][neg_len_ - 1:] = 1
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        noisy_trajectory = self.noise_scheduler.add_noise(
            gt_trajectory, noise, timesteps
        )
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        
        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, 
            fixed_inputs
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = gt_trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Compute loss
        loss = F.mse_loss(pred, target, reduction='none')  # (B, L, 7+)
        loss_mask = ~cond_mask
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.sum() / loss_mask.sum()
        return loss
