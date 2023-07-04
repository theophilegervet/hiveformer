import einops
import torch
from torch import nn

from model.utils.layers import ParallelAttention
from model.utils.encoder import Encoder


class DiffusionHead(Encoder):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_attn_heads=4,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 use_instruction=False,
                 use_goal=False,
                 use_sigma=False,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 weight_tying=False):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=feat_scales_to_use,
            use_sigma=use_sigma
        )
        self.use_instruction = use_instruction
        self.use_goal = use_goal
        self.attn_rounds = attn_rounds
        self.feat_scales = feat_scales_to_use

        # Trajectory encoder
        self.traj_encoder = nn.Linear(output_dim, embedding_dim)

        # Attention from vision to language
        if use_instruction and weight_tying:
            layer = ParallelAttention(
                num_layers=num_vis_ins_attn_layers,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False
            )
            self.vl_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        elif use_instruction:
            self.vl_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=num_vis_ins_attn_layers,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=False, self_attention2=False,
                    cross_attention1=True, cross_attention2=False
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Attention from trajectory queries to language
        if weight_tying:
            layer = ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
            self.traj_lang_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        else:
            self.traj_lang_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=1,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=False, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=False, apply_ffn=False
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Attention from trajectory queries to context
        if weight_tying:
            layer = ParallelAttention(
                num_layers=num_query_cross_attn_layers,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True
            )
            self.traj_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        else:
            self.traj_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=num_query_cross_attn_layers,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Noise regression after every attention to a scale
        self.traj_regressor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, output_dim)
            )
            for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])

    def forward(self, trajectory, trajectory_mask, timestep,
                visible_rgb, visible_pcd, curr_gripper, goal_gripper,
                instruction):
        """
        Arguments:
            trajectory: (B, trajectory_length, output_dim)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            visible_rgb: (B, num_cameras, 3, H, W) in [0, 1]
            visible_pcd: (B, num_cameras, 3, H, W) in world coordinates
            curr_gripper: (B, 3)
            goal_gripper: (B, 3)
            instruction: (B, max_instruction_length, 512)
        """
        device = visible_rgb.device

        # Trajectory features (B, L, F)
        traj_feats = self.traj_encoder(trajectory)
        if self.use_goal:
            n_points = trajectory_mask.shape[-1]
            slope = (goal_gripper - curr_gripper) / (n_points - 1)
            traj_line = (
                slope[:, None]
                * torch.arange(n_points)[None, :, None].to(device)
                + curr_gripper[:, None]
            )
            traj_line[..., 3:] = 0
            traj_pos = self.relative_pe_layer(traj_line)
        else:
            traj_pos = self.relative_pe_layer(trajectory[..., :3])

        # Timestep features (B, 1, F)
        time_feats, time_pos = self.encode_denoising_timestep(timestep)

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, rgb_pos_pyramid, pcd_pyramid = self.encode_images(
            visible_rgb, visible_pcd
        )

        # Encode instruction (B, 53, F)
        instr_feats, instr_pos = None, None
        if self.use_instruction:
            instr_feats, instr_pos = self.encode_instruction(instruction)

        # Encode current gripper (B, 1, F)
        curr_gripper_feats, curr_gripper_pos = self.encode_curr_gripper(
            curr_gripper, batch_size=len(traj_feats)
        )

        # Encode goal gripper (B, 1, F)
        goal_gripper_feats, goal_gripper_pos = None, None
        if self.use_goal:
            goal_gripper_feats, goal_gripper_pos = self.encode_goal_gripper(
                goal_gripper, batch_size=len(traj_feats)
            )

        # Attention layers
        noise = []
        for attn_round in range(self.attn_rounds):
            for scale in range(self.feat_scales):
                noise.append(self._one_attention_round(
                    rgb_feats_pyramid, pcd_pyramid, rgb_pos_pyramid,  # visual
                    instr_feats, instr_pos,  # language
                    curr_gripper_feats, curr_gripper_pos,  # current gripper
                    goal_gripper_feats, goal_gripper_pos,  # goal gripper
                    time_feats, time_pos,  # time
                    traj_feats, traj_pos, trajectory_mask,  # trajectory
                    attn_round, scale
                ))
        return noise

    def _one_attention_round(
        self,
        rgb_feats_pyramid, pcd_pyramid, rgb_pos_pyramid,  # visual
        instr_feats, instr_pos,  # language
        curr_gripper_feats, curr_gripper_pos,  # current gripper
        goal_gripper_feats, goal_gripper_pos,  # goal gripper
        time_feats, time_pos,  # time
        traj_feats, traj_pos, trajectory_mask,  # trajectory
        attn_round, scale
    ):
        # Visual context
        context_feats = einops.rearrange(
            rgb_feats_pyramid[scale],
            "b ncam c h w -> b (ncam h w) c"
        )
        context_pos = rgb_pos_pyramid[scale]

        # Language context
        if self.use_instruction:
            # Attention from vision to language
            l_offset = attn_round * self.feat_scales + scale
            context_feats, _ = self.vl_attention[l_offset](
                seq1=context_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )

        # Concatenate rest of context (gripper, time)
        context_feats = torch.cat([
            context_feats,
            curr_gripper_feats,
            time_feats
        ], dim=1)
        context_pos = torch.cat([
            context_pos,
            curr_gripper_pos,
            time_pos
        ], dim=1)

        # Concatenate goal gripper if used
        if self.use_goal:
            context_feats = torch.cat([context_feats, goal_gripper_feats], 1)
            context_pos = torch.cat([context_pos, goal_gripper_pos], 1)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        l_offset = attn_round * self.feat_scales + scale
        traj_feats, _ = self.traj_lang_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )
        traj_feats, _ = self.traj_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )

        # Regress trajectory
        return self.traj_regressor[l_offset](traj_feats)  # (B, L, output_dim)
