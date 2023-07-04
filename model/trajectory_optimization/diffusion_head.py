import math

import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from model.utils.position_encodings import (
    RotaryPositionEncoding3D,
    LearnedAbsolutePositionEncoding3D
)
from model.utils.layers import (
    RelativeCrossAttentionModule,
    TaskSpecificRelativeCrossAttentionModule
)
from model.utils.resnet import load_resnet50
from model.utils.clip import load_clip


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionHead(nn.Module):

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
                 visualize_rgb_attn=False,
                 use_instruction=False,
                 task_specific_biases=False,
                 positional_features="none",
                 task_ids=[]):
        super().__init__()
        assert backbone in ["resnet", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert rotation_parametrization in [
            "quat_from_top_ghost", "quat_from_query", "6D_from_top_ghost", "6D_from_query"]
        assert num_sampling_level in [1, 2, 3, 4]
        assert visualize_rgb_attn in [False], "Temporarily disabled"
        assert positional_features in ["xyz_concat", "z_concat", "xyz_add", "z_add", "none"]

        self.image_size = image_size
        self.rotation_parametrization = rotation_parametrization
        self.num_ghost_points = num_ghost_points // num_sampling_level
        self.num_ghost_points_val = num_ghost_points_val // num_sampling_level
        self.num_sampling_level = num_sampling_level
        self.sampling_ball_diameter_pyramid = [
            None,
            fine_sampling_ball_diameter,
            fine_sampling_ball_diameter / 4.0,
            fine_sampling_ball_diameter / 16.0
        ]
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)
        self.regress_position_offset = regress_position_offset
        self.visualize_rgb_attn = visualize_rgb_attn
        self.weight_tying = weight_tying
        self.gp_emb_tying = gp_emb_tying
        self.positional_features = positional_features
        self.ins_pos_emb = ins_pos_emb

        # Trajectory encoder
        self.traj_encoder = nn.Linear(8, embedding_dim)

        # Frozen backbone
        if backbone == "resnet":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim)
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(embedding_dim)

        # Ghost points learnable initial features
        self.ghost_points_embed_pyramid = nn.ModuleList()
        if self.gp_emb_tying:
            gp_emb = nn.Embedding(1, embedding_dim)
            for _ in range(self.num_sampling_level):
                self.ghost_points_embed_pyramid.append(gp_emb)
        else:
            for _ in range(self.num_sampling_level):
                self.ghost_points_embed_pyramid.append(nn.Embedding(1, embedding_dim))

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Gaol gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Query learnable features
        self.query_embed = nn.Embedding(1, embedding_dim)

        # Ghost point cross-attention to visual features and current gripper position
        self.task_specific_biases = task_specific_biases
        if self.task_specific_biases:
            self.ghost_point_cross_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                ghost_point_cross_attn = TaskSpecificRelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers, task_ids)
                for _ in range(self.num_sampling_level):
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
            else:
                for _ in range(self.num_sampling_level):
                    ghost_point_cross_attn = TaskSpecificRelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers, task_ids)
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
        else:
            self.ghost_point_cross_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                ghost_point_cross_attn = RelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers)
                for _ in range(self.num_sampling_level):
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
            else:
                for _ in range(self.num_sampling_level):
                    ghost_point_cross_attn = RelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers)
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)

        self.use_instruction = use_instruction
        # Visual tokens cross-attention to language instructions
        if self.use_instruction:
            self.vis_ins_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                vis_ins_cross_attn = RelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_vis_ins_attn_layers)
                for i in range(self.num_sampling_level):
                    self.vis_ins_attn_pyramid.append(vis_ins_cross_attn)
            else:
                for i in range(self.num_sampling_level):
                    vis_ins_cross_attn = RelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_vis_ins_attn_layers)
                    self.vis_ins_attn_pyramid.append(vis_ins_cross_attn)

        # Query cross-attention to visual features, ghost points, and the current gripper position
        if self.task_specific_biases:
            self.query_cross_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                coarse_query_cross_attn = TaskSpecificRelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_query_cross_attn_layers, task_ids)
                for i in range(self.num_sampling_level):
                    self.query_cross_attn_pyramid.append(coarse_query_cross_attn)
            else:
                for i in range(self.num_sampling_level):
                    coarse_query_cross_attn = TaskSpecificRelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_query_cross_attn_layers, task_ids)
                    self.query_cross_attn_pyramid.append(coarse_query_cross_attn)
        else:
            self.query_cross_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                coarse_query_cross_attn = RelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_query_cross_attn_layers)
                for i in range(self.num_sampling_level):
                    self.query_cross_attn_pyramid.append(coarse_query_cross_attn)
            else:
                for i in range(self.num_sampling_level):
                    coarse_query_cross_attn = RelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_query_cross_attn_layers)
                    self.query_cross_attn_pyramid.append(coarse_query_cross_attn)

        # Ghost point offset prediction
        if self.regress_position_offset:
            self.ghost_point_offset_predictor = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 3)
            )

        # Gripper rotation (quaternion) and binary opening prediction
        if "quat" in self.rotation_parametrization:
            self.rotation_dim = 4
        elif "6D" in self.rotation_parametrization:
            self.rotation_dim = 6
        self.gripper_state_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.rotation_dim + 1)
        )

        # Noise regression
        self.noise_regressor = nn.Linear(embedding_dim, 8)

        # Instruction encoder
        if self.use_instruction:
            self.instruction_encoder = nn.Linear(512, embedding_dim)
            if self.ins_pos_emb:
                self._num_words = 53
                self.instr_position_embedding = nn.Embedding(self._num_words, embedding_dim)
                self.instr_position_norm = nn.LayerNorm(embedding_dim)

    def forward(self, trajectory, timestep, visible_rgb, visible_pcd, curr_gripper, goal_gripper, instruction, task_id):
        """
        Arguments:
            trajectory: (batch, trajectory_length, 8)
            timestep: (B, 1)
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 3)
            goal_gripper: (batch x history, 3)
            instruction: (batch x history, max_instruction_length, 512)
            task_id: (batch x history)
            gt_action: (batch x history, 8) in world coordinates
        """
        total_timesteps, num_cameras, _, _, _ = visible_rgb.shape

        # Trajectory features
        traj_feats = self.traj_encoder(trajectory).transpose(0, 1)  # (L, B, F)
        traj_pos = self.relative_pe_layer(trajectory[..., :3])

        # Timestep features
        time_feats = self.time_emb(timestep).unsqueeze(0)  # (1, B, F)
        time_pos = torch.zeros(len(timestep), 1, 3, device=timestep.device)
        time_pos = self.relative_pe_layer(time_pos)

        # Compute visual features at different scales and their positional embeddings
        visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid = self._compute_visual_features(
            visible_rgb, visible_pcd, num_cameras)
        # visible_rgb_features_pyramid: [(B, n_cameras, F, H_i, W_i)]
        # visible_rgb_pos_pyramid [(B, n_cameras*H_i*W_i, F, 2)]
        # visible_pcd_pyramid [(B, n_cameras*H_i*W_i, 3)]

        # Encode instruction (53, B, F)
        instruction_features, instruction_dummy_pos = self._encode_instruction(instruction)

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.relative_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        # Compute goal gripper position features and positional embeddings
        goal_gripper_pos = self.relative_pe_layer(goal_gripper.unsqueeze(1))
        goal_gripper_features = self.goal_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        # Iterative attention layers
        for i in range(self.num_sampling_level):
            # Visual context
            masks = self._find_cylinder_points(
                curr_gripper, goal_gripper, trajectory.size(1),
                visible_pcd_pyramid[i]
            )
            visible_rgb_features_i = einops.rearrange(
                visible_rgb_features_pyramid[i],
                "b ncam c h w -> b (ncam h w) c"
            )
            context_pos_i = visible_rgb_pos_pyramid[i]
            context_feats_i = einops.rearrange(
                visible_rgb_features_i,
                "b npts c -> npts b c"
            )

            # Language context
            if self.use_instruction:
                context_feats_i = self.vis_ins_attn_pyramid[i](
                    query=context_feats_i, value=instruction_features,
                    query_pos=None, value_pos=None
                )[-1]
                
                context_feats_i = torch.cat(
                    [context_feats_i, instruction_features], dim=0)
                context_pos_i = torch.cat(
                    [context_pos_i, instruction_dummy_pos], dim=1)

            # Concatenate rest of context (grippers, time)
            context_feats_i = torch.cat([
                context_feats_i,
                curr_gripper_features,
                goal_gripper_features,
                time_feats
            ], dim=0)
            context_pos_i = torch.cat([
                context_pos_i,
                curr_gripper_pos,
                goal_gripper_pos,
                time_pos
            ], dim=1)
            masks = torch.cat((
                masks,
                torch.ones(
                    (len(masks), len(context_feats_i) - masks.size(1)),
                    device=masks.device,
                    dtype=masks.dtype
                )
            ), dim=1)

            # Trajectory features cross-attend to context features
            traj_feats = self._compute_query_features(
                traj_feats, context_feats_i,
                traj_pos, context_pos_i,
                level=i, task_id=task_id,
                pad_mask=~masks
            )

        # Regress noise
        noise = self.noise_regressor(traj_feats).transpose(0, 1)  # (B, L, 8)

        return noise

    def _compute_visual_features(self, visible_rgb, visible_pcd, num_cameras):
        """Compute visual features at different scales and their positional embeddings."""
        ncam = visible_rgb.shape[1]

        # Pass each view independently through backbone
        visible_rgb = einops.rearrange(visible_rgb, "bt ncam c h w -> (bt ncam) c h w")
        visible_rgb = self.normalize(visible_rgb)
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "bt ncam c h w -> (bt ncam) c h w")

        visible_rgb_features_pyramid = []
        visible_rgb_pos_pyramid = []
        visible_pcd_pyramid = []

        for i in range(self.num_sampling_level):
            visible_rgb_features_i = visible_rgb_features[self.feature_map_pyramid[i]]
            visible_pcd_i = F.interpolate(
                visible_pcd, scale_factor=1. / self.downscaling_factor_pyramid[i], mode='bilinear')
            h, w = visible_pcd_i.shape[-2:]
            visible_pcd_i = einops.rearrange(
                visible_pcd_i, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras)
            visible_rgb_pos_i = self.relative_pe_layer(visible_pcd_i)
            visible_rgb_features_i = einops.rearrange(
                visible_rgb_features_i, "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras)

            visible_rgb_features_pyramid.append(visible_rgb_features_i)
            visible_rgb_pos_pyramid.append(visible_rgb_pos_i)
            visible_pcd_pyramid.append(visible_pcd_i)

        return visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid

    def _encode_instruction(self, instruction):
        if self.use_instruction:
            instruction_features = self.instruction_encoder(instruction)

            if self.ins_pos_emb:
                position = torch.arange(self._num_words)
                position = position.unsqueeze(0).to(instruction_features.device)

                pos_emb = self.instr_position_embedding(position)
                pos_emb = self.instr_position_norm(pos_emb)
                pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=instruction_features.shape[0])

                instruction_features += pos_emb

            instruction_features = einops.rearrange(instruction_features, "bt l c -> l bt c")
            instruction_dummy_pos = torch.zeros(len(instruction), instruction_features.shape[0], 3, device=instruction.device)
            instruction_dummy_pos = self.relative_pe_layer(instruction_dummy_pos)
        else:
            instruction_features = None
            instruction_dummy_pos = None
        return instruction_features, instruction_dummy_pos

    @torch.no_grad()
    @staticmethod
    def _find_cylinder_points(start, end, num_points, point_cloud):
        """
        start: (B, 3)
        end: (B, 3)
        num_points: int
        point_cloud: (B, P, 3)
        """
        # Neighborhood size
        size = (end - start).abs().max(1).values  # (B,)

        # Compute line (B, num_points, 3)
        slope = (end - start) / (num_points - 1)  # (B, 3)
        line = slope[:, None] * np.arange(num_points)[None, :, None] + start

        # Initialize empty repository of cylinder points (B, P)
        in_cylinder = torch.zeros(point_cloud.shape[:2], device=end.device)
        in_cylinder = in_cylinder.bool()

        # Loop over line points and add neighborhoods to repository
        for p in range(num_points):
            point = line[:, p]  # (B, 3)
            dists = ((point[:, None] - point_cloud) ** 2).sum(-1).sqrt()
            in_cylinder = in_cylinder | (dists <= size)
        return in_cylinder  # (B, P)

    def _compute_query_features(self,
                                query_features, context_features,
                                query_pos, context_pos,
                                level, task_id, pad_mask=None):
        """The query cross-attends to context features (visual features, instruction features,
        and current gripper position)."""
        attn_layers = self.query_cross_attn_pyramid[level]

        if self.task_specific_biases:
            query_features = attn_layers(
                task_id=task_id,
                query=query_features, value=context_features,
                query_pos=query_pos, value_pos=context_pos
            )
        else:
            query_features = attn_layers(
                query=query_features, value=context_features,
                query_pos=query_pos, value_pos=context_pos,
                pad_mask=pad_mask
            )

        return query_features
