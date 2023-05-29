import einops
import torch
import torch.nn as nn

from model.diffusion_planner.diffusion_head_simple import DiffusionHead


class TrajectoryHead(DiffusionHead):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_attn_heads=4,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 ins_pos_emb=False,
                 num_sampling_level=3,
                 use_instruction=False,
                 use_goal=False,
                 positional_features="none",
                 use_rgb=True):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_attn_heads=num_attn_heads,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            ins_pos_emb=ins_pos_emb,
            num_sampling_level=num_sampling_level,
            use_instruction=use_instruction,
            use_goal=True,
            positional_features=positional_features,
            use_rgb=use_rgb
        )
        self.traj_encoder = nn.Linear(embedding_dim, embedding_dim)
        self.curr_gripper_encoder = nn.Linear(output_dim, embedding_dim)
        self.goal_gripper_encoder = nn.Linear(output_dim, embedding_dim)

    def forward(self, trajectory_mask,
                visible_rgb, visible_pcd, curr_gripper, goal_gripper,
                instruction):
        """
        Arguments:
            trajectory_mask: (batch, trajectory_length)
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, output_dim)
            goal_gripper: (batch x history, output_dim)
            instruction: (batch x history, max_instruction_length, 512)
        """
        total_timesteps, num_cameras, _, _, _ = visible_rgb.shape
        device = visible_rgb.device

        # Trajectory features
        traj_time_pos = self.time_emb(
            torch.arange(0, trajectory_mask.size(1), device=device)
        )[None].repeat(len(trajectory_mask), 1, 1)  # (B, L, F)
        traj_feats = self.traj_encoder(traj_time_pos)  # (B, L, F)
        traj_pos = self.relative_pe_layer(torch.zeros(
            len(traj_feats), traj_feats.shape[1], 3,
            device=device
        ))

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, rgb_pos_pyramid, pcd_pyramid = self._compute_visual_features(
            visible_rgb, visible_pcd, num_cameras
        )
        # rgb_feats_pyramid: [(B, n_cameras, F, H_i, W_i)]
        # rgb_pos_pyramid [(B, n_cameras*H_i*W_i, F, 2)]
        # pcd_pyramid [(B, n_cameras*H_i*W_i, 3)]

        # Encode instruction (B, 53, F)
        instr_feats, instr_dummy_pos = self._encode_instruction(instruction)

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.relative_pe_layer(curr_gripper[..., :3].unsqueeze(1))
        curr_gripper_feats = self.curr_gripper_encoder(curr_gripper).unsqueeze(1)

        # Compute goal gripper position features and positional embeddings
        if self.use_goal:
            goal_gripper_pos = self.relative_pe_layer(goal_gripper[..., :3][:, None])
            goal_gripper_feats = self.goal_gripper_encoder(goal_gripper).unsqueeze(1)

        # Attention layers
        # Visual context
        if self.use_rgb:
            context_feats = einops.rearrange(
                rgb_feats_pyramid[0],
                "b ncam c h w -> b (ncam h w) c"
            )
        else:
            context_feats = self.occ_encoder(pcd_pyramid[0])
        context_pos = rgb_pos_pyramid[0]

        # Language context
        if self.use_instruction:
            context_feats = self.vis_ins_attn_pyramid[0](
                query=context_feats.transpose(0, 1),
                value=instr_feats.transpose(0, 1),
                query_pos=None, value_pos=None
            )[-1].transpose(0, 1)
            
            context_feats = torch.cat([context_feats, instr_feats], dim=1)
            context_pos = torch.cat([context_pos, instr_dummy_pos], dim=1)

        # Concatenate rest of context (grippers, time)
        context_feats = torch.cat([
            context_feats,
            curr_gripper_feats
        ], dim=1)
        context_pos = torch.cat([
            context_pos,
            curr_gripper_pos
        ], dim=1)

        # Concatenate goal gripper if used
        if self.use_goal:
            context_feats = torch.cat([context_feats, goal_gripper_feats], 1)
            context_pos = torch.cat([context_pos, goal_gripper_pos], 1)

        # Trajectory features cross-attend to context features
        for layer in self.traj_attention:
            traj_feats, _ = layer(
                seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
                seq2=context_feats, seq2_key_padding_mask=None,
                seq1_pos=traj_pos, seq2_pos=context_pos,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )

        # Regress trajectory
        trajectory = self.noise_regressor(traj_feats)  # (B, L, output_dim)

        return trajectory
