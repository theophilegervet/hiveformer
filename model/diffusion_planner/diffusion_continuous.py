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
                use_rgb=use_rgb,
                use_sigma=True
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
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

        # Denoising stuff
        self.p_mean = -1.2
        self.p_std = 1.2
        self.T = 10
        self.sigma_data = 1.0
        sigma_min = 2e-3
        sigma_max = 8e1
        rho = 7
        sigmas = (
            sigma_max ** (1 / rho)
            + (torch.arange(self.T+1) / (self.T-1))
            * (sigma_min**(1 / rho) - sigma_max**(1 / rho))
        ) ** rho
        sigmas[-1] = 0
        self.register_buffer('ts', sigmas)

    def sigma(self, t):
        return t

    def policy_forward_pass(self, trajectory, sigma, fixed_inputs):
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
            sigma,
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
        trajectory = trajectory * self.sigma(self.ts[0])
        trajectory[condition_mask] = condition_data[condition_mask]

        # Heun's 2nd order method (algorithm 1): https://arxiv.org/pdf/2206.00364
        B = len(trajectory)
        for i in range(self.T):
            t = self.ts[i]
            t_next = self.ts[i+1]
            sigma = self.sigma(t)[None, None].repeat(B, 1)
            sigma_next = self.sigma(t_next)[None, None].repeat(B, 1)

            denoised = self.denoise_with_preconditioning(
                trajectory,
                sigma,
                fixed_inputs
            )
            d_i = (1 / sigma[..., None]) * (trajectory - denoised)
            trajectory_next = trajectory + (t_next - t) * d_i
            trajectory_next[condition_mask] = condition_data[condition_mask]
            
            # Apply second order correction
            if (sigma_next != 0).all():
                denoised_next = self.denoise_with_preconditioning(
                    trajectory_next, 
                    sigma_next,
                    fixed_inputs
                )
                d_ip = (1 / sigma_next[..., None]) * (trajectory_next - denoised_next)
                trajectory_next = trajectory + (t_next - t) * 0.5 * (d_i + d_ip)
            
            trajectory = trajectory_next
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
        _, L_, A_ = gt_trajectory.shape
        sigma = (
            torch.randn(len(gt_trajectory), device=gt_trajectory.device)
            * self.p_std
            + self.p_mean
        ).exp()

        # Add noise to the clean trajectories
        noisy_trajectory = torch.normal(
            gt_trajectory,
            sigma[:, None, None].repeat(1, L_, A_)
        )
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition

        # Predict denoised trajectory
        pred = self.denoise_with_preconditioning(
            noisy_trajectory,
            sigma,
            fixed_inputs
        )
        loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2

        # Compute loss
        target = gt_trajectory
        loss = F.mse_loss(pred, target, reduction='none')  # (B, L, 7+)
        loss_mask = ~cond_mask
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss * loss_weight.reshape(len(sigma), -1, 1).clamp(0, 1000)
        loss = loss.sum() / loss_mask.sum()
        return loss
        
    def denoise_with_preconditioning(self, trajectory, sigma, fixed_inputs):
        """
        Denoise ego_trajectory with noise level sigma
        Equivalent to evaluating D_theta here: https://arxiv.org/pdf/2206.00364
        Returns denoised trajectory, not the residual noise
        """
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (
            sigma * self.sigma_data
            / torch.sqrt(sigma**2 + self.sigma_data**2)
        )
        c_noise = .25 * torch.log(sigma)

        out = (
            c_skip.reshape(len(trajectory), 1, 1) * trajectory
            + c_out.reshape(len(trajectory), 1, 1) * self.policy_forward_pass(
                c_in.reshape(len(trajectory), 1, 1) * trajectory,
                c_noise.reshape(len(trajectory),),
                fixed_inputs
            )
        )
        return out
