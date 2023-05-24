import torch
import torch.nn as nn

from model.utils.position_encodings import (
    SinusoidalPosEmb,
    RotaryPositionEncoding3D
)
from model.utils.layers import ParallelAttentionLayer


class DiffusionHead(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_attn_heads=4,
                 num_vis_ins_attn_layers=8,
                 ins_pos_emb=False,
                 num_sampling_level=3,
                 use_instruction=False,
                 use_goal=False,
                 positional_features="none"):
        super().__init__()
        self.positional_features = positional_features

        # Trajectory encoder
        self.traj_encoder = nn.Linear(output_dim, embedding_dim)

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(embedding_dim)

        # Trajectory cross/self-attention
        self.traj_attention = nn.ModuleList()
        for _ in range(num_vis_ins_attn_layers):
            self.traj_attention.append(ParallelAttentionLayer(
                d_model=embedding_dim, n_heads=num_attn_heads,
                cross_attention1=False,
                self_attention2=False, cross_attention2=False,
                rotary_pe=True
            ))

        # Noise regression
        self.noise_regressor = nn.Linear(embedding_dim, output_dim)

    def forward(self, trajectory, trajectory_mask, timestep,
                visible_rgb, visible_pcd, curr_gripper, goal_gripper,
                instruction):
        """
        Arguments:
            trajectory: (batch, trajectory_length, output_dim)
            trajectory_mask: (batch, trajectory_length)
            timestep: (B, 1)
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 3)
            goal_gripper: (batch x history, 3)
            instruction: (batch x history, max_instruction_length, 512)
        """
        # Relative to current gripper location
        if curr_gripper is None:
            curr_gripper = trajectory[:, 0]
        trajectory[..., :3] = trajectory[..., :3] - curr_gripper[:, None]

        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)
        traj_pos = self.relative_pe_layer(trajectory[..., :3])

        # Diffusion timestep features
        time_feats = self.time_emb(timestep).unsqueeze(1)  # (B, 1, F)

        # Trajectory timestep features
        traj_time_pos = self.time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)

        # Add the two time positional embeddings
        time_pos = time_feats + traj_time_pos  # (B, L, F)

        # Trajectory features cross-attend to context features
        for layer in self.traj_attention:
            traj_feats, _ = layer(
                seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
                seq2=None, seq2_key_padding_mask=None,
                seq1_pos=traj_pos, seq2_pos=None,
                seq1_sem_pos=time_pos, seq2_sem_pos=None
            )

        # Regress noise
        noise = self.noise_regressor(traj_feats)  # (B, L, output_dim)

        return noise
