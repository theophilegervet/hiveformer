import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from model.utils.position_encodings import (
    RotaryPositionEncoding3D,
    LearnedAbsolutePositionEncoding3D,
    SinusoidalPosEmb
)
from model.utils.layers import (
    RelativeCrossAttentionModule,
    ParallelAttentionLayer
)
from model.utils.resnet import load_resnet50
from model.utils.clip import load_clip


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
                 positional_features="none",
                 predict_length=False,
                 use_rgb=True):
        super().__init__()
        assert backbone in ["resnet", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]
        assert positional_features in [
            "xyz_concat", "z_concat", "xyz_add", "z_add", "none"
        ]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.positional_features = positional_features
        self.ins_pos_emb = ins_pos_emb
        self.use_instruction = use_instruction
        self.use_goal = use_goal
        self.use_rgb = use_rgb
        self.predict_length = predict_length

        # Trajectory encoder
        self.traj_encoder = nn.Linear(output_dim, embedding_dim)

        # Frozen backbone
        if backbone == "resnet":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        if self.positional_features in ["xyz_concat", "z_concat"]:
            self.feature_pyramid = FeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048], embedding_dim - embedding_dim // 10
            )
        else:
            self.feature_pyramid = FeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048], embedding_dim
            )
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

        # If not use_rgb, then use an occupancy embedding
        if not use_rgb:
            self.occ_encoder = nn.Linear(3, embedding_dim)

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # 3D absolute positional embeddings (only used for positional features, if any)
        if self.positional_features == "xyz_concat":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(3, embedding_dim // 10)
        elif self.positional_features == "z_concat":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(1, embedding_dim // 10)
        if self.positional_features == "xyz_add":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(3, embedding_dim)
        elif self.positional_features == "z_add":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(1, embedding_dim)

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Visual tokens cross-attention to language instructions
        if self.use_instruction:
            self.vis_ins_attn_pyramid = nn.ModuleList()
            for _ in range(self.num_sampling_level):
                self.vis_ins_attn_pyramid.append(RelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_vis_ins_attn_layers
                ))

        # Trajectory cross/self-attention
        self.traj_attention = nn.ModuleList()
        for _ in range(num_vis_ins_attn_layers):
            self.traj_attention.append(ParallelAttentionLayer(
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention2=False, cross_attention2=False,
                rotary_pe=True
            ))

        if self.predict_length:
            self.length_predictor_attn = nn.ModuleList()
            for _ in range(num_vis_ins_attn_layers):
                self.length_predictor_attn.append(ParallelAttentionLayer(
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention2=False, cross_attention2=False,
                    rotary_pe=True
                ))
            self.length_query_embed = nn.Embedding(1, embedding_dim)
            self.length_regressor = nn.Linear(embedding_dim, 1)

        # Noise regression
        self.noise_regressor = nn.Linear(embedding_dim, output_dim)

        # Instruction encoder
        if self.use_instruction:
            self.instruction_encoder = nn.Linear(512, embedding_dim)
            if self.ins_pos_emb:
                self._num_words = 53
                self.instr_position_embedding = nn.Embedding(
                    self._num_words, embedding_dim
                )
                self.instr_position_norm = nn.LayerNorm(embedding_dim)

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
        total_timesteps, num_cameras, _, _, _ = visible_rgb.shape

        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)
        traj_pos = self.relative_pe_layer(trajectory[..., :3])

        # Timestep features
        time_feats = self.time_emb(timestep).unsqueeze(1)  # (B, 1, F)
        time_pos = torch.zeros(len(timestep), 1, 3, device=timestep.device)
        time_pos = self.relative_pe_layer(time_pos)

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
        curr_gripper_pos = self.relative_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_feats = self.curr_gripper_embed.weight.repeat(
            total_timesteps, 1
        ).unsqueeze(1)

        # Compute goal gripper position features and positional embeddings
        if self.use_goal:
            goal_gripper_pos = self.relative_pe_layer(goal_gripper[:, None])
            goal_gripper_feats = self.goal_gripper_embed.weight.repeat(
                total_timesteps, 1
            ).unsqueeze(1)

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
        for layer in self.traj_attention:
            traj_feats, _ = layer(
                seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
                seq2=context_feats, seq2_key_padding_mask=None,
                seq1_pos=traj_pos, seq2_pos=context_pos,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )

        # Length prediction
        if self.predict_length:
            query_feats = self.length_query_embed.weight.unsqueeze(0).repeat(traj_feats.shape[0], 1, 1)
            query_dummy_pos = self.relative_pe_layer(torch.zeros([query_feats.shape[0], query_feats.shape[1], 3], device=instruction.device))
            for layer in self.length_predictor_attn:
                query_feats, _ = layer(
                    seq1=query_feats, seq1_key_padding_mask=None,
                    seq2=context_feats, seq2_key_padding_mask=None,
                    seq1_pos=query_dummy_pos, seq2_pos=context_pos,
                    seq1_sem_pos=None, seq2_sem_pos=None
                )
            length = self.length_regressor(query_feats).squeeze()
        else:
            length = None

        # Regress noise
        noise = self.noise_regressor(traj_feats)  # (B, L, output_dim)

        return noise, length

    def _compute_visual_features(self, visible_rgb, visible_pcd, num_cameras):
        """Compute visual features/pos embeddings at different scales."""
        ncam = visible_rgb.shape[1]

        # Pass each view independently through backbone
        visible_rgb = einops.rearrange(visible_rgb, "bt ncam c h w -> (bt ncam) c h w")
        visible_rgb = self.normalize(visible_rgb)
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        rgb_pos_pyramid = []
        pcd_pyramid = []

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

            if self.positional_features in ["xyz_concat", "xyz_add"]:
                visible_rgb_pos_features_i = self.absolute_pe_layer(visible_pcd_i)
                visible_rgb_pos_features_i = einops.rearrange(
                    visible_rgb_pos_features_i, "bt (ncam h w) c -> bt ncam c h w", ncam=ncam, h=h, w=w)
            elif self.positional_features in ["z_concat", "z_add"]:
                visible_rgb_pos_features_i = self.absolute_pe_layer(visible_pcd_i[:, :, 2:3])
                visible_rgb_pos_features_i = einops.rearrange(
                    visible_rgb_pos_features_i, "bt (ncam h w) c -> bt ncam c h w", ncam=ncam, h=h, w=w)

            if self.positional_features in ["xyz_concat", "z_concat"]:
                visible_rgb_features_i = torch.cat([visible_rgb_features_i, visible_rgb_pos_features_i], dim=2)
            elif self.positional_features in ["xyz_add", "z_add"]:
                visible_rgb_features_i = visible_rgb_features_i + visible_rgb_pos_features_i

            rgb_feats_pyramid.append(visible_rgb_features_i)
            rgb_pos_pyramid.append(visible_rgb_pos_i)
            pcd_pyramid.append(visible_pcd_i)

        return rgb_feats_pyramid, rgb_pos_pyramid, pcd_pyramid

    def _encode_instruction(self, instruction):
        if self.use_instruction:
            instr_feats = self.instruction_encoder(instruction)

            if self.ins_pos_emb:
                position = torch.arange(self._num_words)
                position = position[None].to(instr_feats.device)

                pos_emb = self.instr_position_embedding(position)
                pos_emb = self.instr_position_norm(pos_emb)
                pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=len(instr_feats))

                instr_feats += pos_emb

            instr_dummy_pos = torch.zeros(
                len(instruction), instr_feats.shape[1], 3,
                device=instruction.device
            )
            instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        else:
            instr_feats = None
            instr_dummy_pos = None
        return instr_feats, instr_dummy_pos
