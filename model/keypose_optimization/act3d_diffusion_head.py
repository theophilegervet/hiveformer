"""Define Pose-Diffusion Model.

For training, we perturb the ground-truth keypose, and transform the gripper
w.r.t the perturbation.  The denoising model predict the clean keypose.

For testing, we sample clean keypose from pure noise.
"""
import torch
from torch import nn
from torch.nn import functional as F
import einops
import dgl.geometry as dgl_geo

from model.utils.position_encodings import (
    RotaryPositionEncoding3D,
    LearnedAbsolutePositionEncoding3Dv2,
    SinusoidalPosEmb,
)
from model.utils.layers import (
    MultiheadCustomAttention,
)


class Act3dDiffusionHead(nn.Module):

    def __init__(self,
                 #output_dim=7,
                 rotation_dim=4,
                 position_dim=3,
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_cross_attn_layers=8,
                 num_gripper_points=3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.absolute_pe_layer = LearnedAbsolutePositionEncoding3Dv2(3, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim * 2)

        # Output layers
        self.feature_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim * num_gripper_points, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim * num_gripper_points, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, position_dim)
        )

        # Attention layers
        self.cross_attn = StagedRelativeCrossAttentionModule(
            embedding_dim * 2, num_attn_heads, num_cross_attn_layers
        )

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )

    def forward(self,
                gripper_pcd, gripper_features,
                ghost_pcd, ghost_features,
                context_pcd, context_features,
                timesteps):
        """Compute the predicted action (position, rotation, opening) from the 
        gripper's visual features.  We contextualize the features by cross-
        attending to ghost point and local context features.  To obatin geometrical
        outputs, we add visual features with absolute positional encodings.

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, C)
            ghost_pcd: A tensor of shape (B, N, 3)
            ghost_features: A tensor of shape (N, B, C)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, C)
            timesteps: A tensor of shape (B, ) indicating the diffusion step
        
        Returns:
            A tensor of shape (B, self.output_dim)
        """

        time_embs = self.encode_denoising_timestep(timesteps)

        abs_gripper_pos = self.absolute_pe_layer(gripper_pcd)
        abs_ghost_pos = self.absolute_pe_layer(ghost_pcd)
        abs_context_pos = self.absolute_pe_layer(context_pcd)

        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_ghost_pos = self.relative_pe_layer(ghost_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        abs_ghost_pos = einops.rearrange(abs_ghost_pos, "b npts c -> npts b c")
        abs_context_pos = einops.rearrange(abs_context_pos, "b npts c -> npts b c")
        abs_gripper_pos = einops.rearrange(abs_gripper_pos, "b npts c -> npts b c")

        gripper_features_with_abs_pos = torch.cat([gripper_features, abs_gripper_pos], dim=-1)
        ghost_context_features_with_abs_pos = torch.cat([context_features, abs_context_pos], dim=-1)
        ghost_features_with_abs_pos = torch.cat([ghost_features, abs_ghost_pos], dim=-1)
        gripper_features_with_abs_pos = self.cross_attn(
            query=gripper_features_with_abs_pos,
            value_1=ghost_context_features_with_abs_pos,
            value_2=ghost_features_with_abs_pos,
            query_pos=rel_gripper_pos,
            value_pos_1=rel_context_pos,
            value_pos_2=rel_ghost_pos,
            diff_ts=time_embs,
        )[-1]

        features = einops.rearrange(
            gripper_features_with_abs_pos, "npts b c -> b npts c"
        )
        features = self.feature_proj(features) # (B, N, C * 2) -> (B, N, C)
        features = features.flatten(1)

        rotation = self.rotation_predictor(features)
        position = self.position_predictor(features)

        return position, rotation

    def encode_denoising_timestep(self, timestep):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        return time_feats


class Act3dDiffusionHeadv3(nn.Module):

    def __init__(self,
                 #output_dim=7,
                 rotation_dim=4,
                 position_dim=3,
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_cross_attn_layers=8,
                 num_self_attn_layers=8,
                 num_gripper_points=3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.absolute_pe_layer = LearnedAbsolutePositionEncoding3Dv2(3, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim * 2)

        # Output layers
        self.feature_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim * num_gripper_points, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, position_dim)
        )

        # Attention layers
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim * 2, num_attn_heads, num_cross_attn_layers
        )

        self.self_attn = FFWRelativeSelfAttentionModule(
            embedding_dim * 2, num_attn_heads, num_self_attn_layers
        )

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )

        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )

    def forward(self,
                gripper_pcd, gripper_features,
                context_pcd, context_features,
                timesteps, curr_gripper_features):
        """Compute the predicted action (position, rotation, opening) from the 
        gripper's visual features.  We contextualize the features by cross-
        attending to ghost point and local context features.  To obatin geometrical
        outputs, we add visual features with absolute positional encodings.

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, C)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, C)
            timesteps: A tensor of shape (B, ) indicating the diffusion step
            curr_gripper_features: A tensor of shape (B, C)
        
        Returns:
            A tensor of shape (B, self.output_dim)
        """
        time_embs = self.encode_denoising_timestep(timesteps, curr_gripper_features)

        abs_gripper_pos = self.absolute_pe_layer(gripper_pcd)
        abs_context_pos = self.absolute_pe_layer(context_pcd)

        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        abs_context_pos = einops.rearrange(abs_context_pos, "b npts c -> npts b c")
        abs_gripper_pos = einops.rearrange(abs_gripper_pos, "b npts c -> npts b c")

        gripper_features_with_abs_pos = torch.cat([gripper_features, abs_gripper_pos], dim=-1)
        context_features_with_abs_pos = torch.cat([context_features, abs_context_pos], dim=-1)

        # Cross attention from gripper to context
        gripper_features_with_abs_pos = self.cross_attn(
            query=gripper_features_with_abs_pos,
            value=context_features_with_abs_pos,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs,
        )[-1]

        # Pick out the current gripper token
        context_features_with_abs_pos, curr_gripper_features_with_abs_pos = (
            context_features_with_abs_pos[:-1], context_features_with_abs_pos[-1:]
        )
        rel_context_pos, rel_curr_gripper_pos = (
            rel_context_pos[:, :-1], rel_context_pos[:, -1:]
        )

        # Sample context points
        npts, bs, ch = context_features_with_abs_pos.shape
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(context_features_with_abs_pos, "npts b c -> b npts c").to(torch.float64),
            max(npts // 5, 1), 0
        ).long()
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features_with_abs_pos = torch.gather(
            context_features_with_abs_pos,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c") 
        )
        sampled_context_features_with_abs_pos = torch.cat([
            sampled_context_features_with_abs_pos,
            curr_gripper_features_with_abs_pos
        ], dim=0)

        _, _, ch, npos = rel_context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_rel_context_pos = torch.gather(
            rel_context_pos, 1, expanded_sampled_inds
        )
        sampled_rel_context_pos = torch.cat([
            sampled_rel_context_pos,
            rel_curr_gripper_pos
        ], dim=1)

        # Self attention among gripper and sampled context
        features_with_abs_pos = torch.cat([
            gripper_features_with_abs_pos,
            sampled_context_features_with_abs_pos
        ], dim=0)
        rel_pos = torch.cat([
            rel_gripper_pos,
            sampled_rel_context_pos
        ], dim=1)
        features_with_abs_pos = self.self_attn(
            query=features_with_abs_pos,
            query_pos=rel_pos,
            diff_ts=time_embs,
        )[-1]

        num_gripper = gripper_features_with_abs_pos.shape[0]
        gripper_features_with_abs_pos = features_with_abs_pos[:num_gripper]
        features = einops.rearrange(
            gripper_features_with_abs_pos, "npts b c -> b npts c"
        )
        features = self.feature_proj(features) # (B, N, C * 2) -> (B, N, C)

        rotation = self.rotation_predictor(features.flatten(1))
        position = self.position_predictor(features)

        return position, rotation


    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats


class Act3dDiffusionHeadv2(nn.Module):

    def __init__(self,
                 #output_dim=7,
                 rotation_dim=4,
                 position_dim=3,
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_cross_attn_layers=8,
                 num_gripper_points=3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.absolute_pe_layer = LearnedAbsolutePositionEncoding3Dv2(3, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim * 2)

        # Output layers
        self.feature_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim * num_gripper_points, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, position_dim)
        )

        # Attention layers
        self.cross_attn = StagedRelativeCrossAttentionModule(
            embedding_dim * 2, num_attn_heads, num_cross_attn_layers
        )
        self.self_attn = nn.ModuleList()
        for _ in range(2):
            self.self_attn.append(
                SelfAttentionLayer(embedding_dim * 2, num_attn_heads)
            )

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )

    def forward(self,
                gripper_pcd, gripper_features,
                ghost_pcd, ghost_features,
                context_pcd, context_features,
                timesteps):
        """Compute the predicted action (position, rotation, opening) from the 
        gripper's visual features.  We contextualize the features by cross-
        attending to ghost point and local context features.  To obatin geometrical
        outputs, we add visual features with absolute positional encodings.

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, C)
            ghost_pcd: A tensor of shape (B, N, 3)
            ghost_features: A tensor of shape (N, B, C)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, C)
            timesteps: A tensor of shape (B, ) indicating the diffusion step
        
        Returns:
            A tensor of shape (B, self.output_dim)
        """

        time_embs = self.encode_denoising_timestep(timesteps)

        abs_gripper_pos = self.absolute_pe_layer(gripper_pcd)
        abs_ghost_pos = self.absolute_pe_layer(ghost_pcd)
        abs_context_pos = self.absolute_pe_layer(context_pcd)

        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_ghost_pos = self.relative_pe_layer(ghost_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        abs_ghost_pos = einops.rearrange(abs_ghost_pos, "b npts c -> npts b c")
        abs_context_pos = einops.rearrange(abs_context_pos, "b npts c -> npts b c")
        abs_gripper_pos = einops.rearrange(abs_gripper_pos, "b npts c -> npts b c")

        gripper_features_with_abs_pos = torch.cat([gripper_features, abs_gripper_pos], dim=-1)
        ghost_context_features_with_abs_pos = torch.cat([context_features, abs_context_pos], dim=-1)
        ghost_features_with_abs_pos = torch.cat([ghost_features, abs_ghost_pos], dim=-1)
        gripper_features_with_abs_pos = self.cross_attn(
            query=gripper_features_with_abs_pos,
            value_1=ghost_context_features_with_abs_pos,
            value_2=ghost_features_with_abs_pos,
            query_pos=rel_gripper_pos,
            value_pos_1=rel_context_pos,
            value_pos_2=rel_ghost_pos,
            diff_ts=time_embs,
        )[-1]

        for self_attn_mod in self.self_attn:
            gripper_features_with_abs_pos, _ = self_attn_mod(
                query=gripper_features_with_abs_pos,
                diff_ts=time_embs,
            )

        features = einops.rearrange(
            gripper_features_with_abs_pos, "npts b c -> b npts c"
        )
        features = self.feature_proj(features) # (B, N, C * 2) -> (B, N, C)

        rotation = self.rotation_predictor(features.flatten(1))
        position = self.position_predictor(features)

        return position, rotation

    def encode_denoising_timestep(self, timestep):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        return time_feats


class Act3dDiffusionSingleHead(nn.Module):

    def __init__(self,
                 output_dim=3,
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_cross_attn_layers=8,
                 num_gripper_points=3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.absolute_pe_layer = LearnedAbsolutePositionEncoding3Dv2(3, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim * 2)

        # Output layers
        self.feature_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * num_gripper_points, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )

        # Attention layers
        self.cross_attn = StagedRelativeCrossAttentionModule(
            embedding_dim * 2, num_attn_heads, num_cross_attn_layers
        )

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )

    def forward(self,
                gripper_pcd, gripper_features,
                ghost_pcd, ghost_features,
                context_pcd, context_features,
                timesteps):
        """Compute the predicted action (position, rotation, opening) from the 
        gripper's visual features.  We contextualize the features by cross-
        attending to ghost point and local context features.  To obatin geometrical
        outputs, we add visual features with absolute positional encodings.

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, C)
            ghost_pcd: A tensor of shape (B, N, 3)
            ghost_features: A tensor of shape (N, B, C)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, C)
            timesteps: A tensor of shape (B, ) indicating the diffusion step
        
        Returns:
            A tensor of shape (B, self.output_dim)
        """

        time_embs = self.encode_denoising_timestep(timesteps)

        abs_gripper_pos = self.absolute_pe_layer(gripper_pcd)
        abs_ghost_pos = self.absolute_pe_layer(ghost_pcd)
        abs_context_pos = self.absolute_pe_layer(context_pcd)

        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_ghost_pos = self.relative_pe_layer(ghost_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        abs_ghost_pos = einops.rearrange(abs_ghost_pos, "b npts c -> npts b c")
        abs_context_pos = einops.rearrange(abs_context_pos, "b npts c -> npts b c")
        abs_gripper_pos = einops.rearrange(abs_gripper_pos, "b npts c -> npts b c")

        gripper_features_with_abs_pos = torch.cat([gripper_features, abs_gripper_pos], dim=-1)
        ghost_context_features_with_abs_pos = torch.cat([context_features, abs_context_pos], dim=-1)
        ghost_features_with_abs_pos = torch.cat([ghost_features, abs_ghost_pos], dim=-1)
        gripper_features_with_abs_pos = self.cross_attn(
            query=gripper_features_with_abs_pos,
            value_1=ghost_context_features_with_abs_pos,
            value_2=ghost_features_with_abs_pos,
            query_pos=rel_gripper_pos,
            value_pos_1=rel_context_pos,
            value_pos_2=rel_ghost_pos,
            diff_ts=time_embs,
        )[-1]

        features = einops.rearrange(
            gripper_features_with_abs_pos, "npts b c -> b npts c"
        )
        features = self.feature_proj(features) # (B, N, C * 2) -> (B, N, C)
        features = features.flatten(1)

        return self.predictor(features)

    def encode_denoising_timestep(self, timestep):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        return time_feats


############## Attention Layers for Diffusion ################
class AdaLN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
             nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)
    
    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (N, B, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1) # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

        self.adaln = AdaLN(embedding_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, diff_ts):
        x = self.adaln(x, diff_ts)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class RelativeCrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.adaln = AdaLN(embedding_dim)

    def forward(self, query, value, diff_ts, query_pos=None, value_pos=None, pad_mask=None):
        adaln_query = self.adaln(query, diff_ts)
        attn_output, attn_output_weights = self.multihead_attn(
            query=adaln_query,
            key=value,
            value=value,
            rotary_pe=(query_pos, value_pos) if query_pos is not None else None,
            key_padding_mask=pad_mask
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)


class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.adaln = AdaLN(embedding_dim)

    def forward(self, query, diff_ts, query_pos=None, value_pos=None, pad_mask=None):
        adaln_query = self.adaln(query, diff_ts)
        attn_output, attn_output_weights = self.multihead_attn(
            query=adaln_query,
            key=adaln_query,
            value=adaln_query,
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)


class StagedRelativeCrossAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers_1 = nn.ModuleList()
        self.attn_layers_2 = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers_1.append(
                RelativeCrossAttentionLayer(embedding_dim, num_attn_heads)
            )
            self.attn_layers_2.append(
                RelativeCrossAttentionLayer(embedding_dim, num_attn_heads)
            )
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, query, value_1, value_2, diff_ts,
                query_pos=None, value_pos_1=None, value_pos_2=None):
        output = []
        for i in range(self.num_layers):
            query, _ = self.attn_layers_1[i](
                query, value_1, diff_ts, query_pos, value_pos_1
            )
            query, _ = self.attn_layers_2[i](
                query, value_2, diff_ts, query_pos, value_pos_2
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output


class FFWRelativeCrossAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                RelativeCrossAttentionLayer(embedding_dim, num_attn_heads)
            )
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, query, value, diff_ts,
                query_pos=None, value_pos=None):
        output = []
        for i in range(self.num_layers):
            query, _ = self.attn_layers[i](
                query, value, diff_ts, query_pos, value_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output


class FFWRelativeSelfAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                RelativeCrossAttentionLayer(embedding_dim, num_attn_heads)
            )
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, query, diff_ts, query_pos=None, ):
        output = []
        for i in range(self.num_layers):
            query, _ = self.attn_layers[i](
                query, query, diff_ts, query_pos, query_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output