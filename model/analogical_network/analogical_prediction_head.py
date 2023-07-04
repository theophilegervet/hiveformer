import einops
import torch
from torch import nn

from model.keypose_optimization.baseline import PredictionHead


class AnalogicalPredictionHead(PredictionHead):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 action_dim=8,
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
                 use_instruction=False):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            rotation_parametrization=rotation_parametrization,
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
        self.mem_query_encoder = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self,
                rgb, pcd, curr_gripper, instruction,
                mem_rgb, mem_pcd, mem_action, gt_action_for_sampling=None):
        """
        Arguments:
            rgb: (batch, 1 + support, history, num_cameras, 3, height, width) in [0, 1]
            pcd: (batch, 1 + support, history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch, 1 + support, history, 3)
            instruction: (batch, 1 + support, history, max_instruction_length, 512)
            padding_mask: (batch, 1 + support, history)
            gt_action_for_support: ground-truth action used as the support set
             of shape (batch, 1 + support, history, 8) in world coordinates
            gt_action_for_sampling: ground-truth action used to guide ghost point sampling
             of shape (batch, 1 + support, history, 8) in world coordinates

        Use of (1 + support) dimension:
        - During training, all demos in the set come from the same train split, and we use
           each demo in this dimension for training with all other demos as the support set
        - During evaluation, only the first demo in this dimension comes from the val split while
           others come from the train split and act as the support set
        """
        total_timesteps, num_cameras, _, height, width = rgb.shape
        device = rgb.device
        if gt_action_for_sampling is not None:
            gt_position = gt_action_for_sampling[:, :3].unsqueeze(1).detach()
        else:
            gt_position = None
        
        # MEMORY PART: initialize query by attending to memory point cloud
        # 1. Featurize gt memory action
        mem_query = self.mem_query_encoder(mem_action)[:, None]  # B 1 F
        mem_pos = self.relative_pe_layer(mem_action[:, None, :3])  # B 1 F 2

        # 2. Compute memory visual features
        rgb_feats_mem, rgb_pos_mem, pcd_mem = self._compute_visual_features(
            mem_rgb, mem_pcd, num_cameras)
        # rgb_feats: [(B, n_cameras, F, H_i, W_i)]
        # rgb_pos [(B, n_cameras*H_i*W_i, F, 2)]
        # pcd [(B, n_cameras*H_i*W_i, 3)]

        for i in range(len(rgb_feats_pyramid)):
            # Local fine RGB features
            l2_pred_pos = ((mem_action[:, None, :3] - pcd_mem[i]) ** 2).sum(-1).sqrt()
            indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

            rgb_features_i = einops.rearrange(
                rgb_feats_pyramid[i], "b ncam c h w -> b (ncam h w) c")
            rgb_features_i = torch.stack([
                f[i] for (f, i) in zip(rgb_features_i, indices)])
            rgb_pos_i = torch.stack([
                f[i] for (f, i) in zip(rgb_pos_pyramid[i], indices)])
            ghost_pcd_context_features_i = einops.rearrange(
                rgb_features_i, "b npts c -> npts b c")

            # Compute ghost point features and their positional embeddings by attending to visual
            # features

            # Initialize query features
            if i == 0:
                query_features = self.query_embed.weight.unsqueeze(1).repeat(1, total_timesteps, 1)

            query_context_features_i = ghost_pcd_context_features_i
            query_context_pos_i = ghost_pcd_context_pos_i

            # Now that the query is localized, we use positional embeddings
            query_pos_i = self.relative_pe_layer(position_pyramid[-1])
            context_pos_i = query_context_pos_i

            # The query cross-attends to context features (visual features and the current gripper position)
            query_features = self._compute_query_features(
                query_features, query_context_features_i,
                query_pos_i, context_pos_i,
                level=i
            )

        # Compute visual features at different scales and their positional embeddings
        rgb_feats_pyramid, rgb_pos_pyramid, pcd_pyramid = self._compute_visual_features(
            rgb, pcd, num_cameras)

        # Encode instruction
        if self.use_instruction:
            instr_feats = self.instruction_encoder(instruction)

            if self.ins_pos_emb:
                position = torch.arange(self._num_words)
                position = position.unsqueeze(0).to(instr_feats.device)

                pos_emb = self.instr_position_embedding(position)
                pos_emb = self.instr_position_norm(pos_emb)
                pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=instr_feats.shape[0])

                instr_feats += pos_emb

            instr_feats = einops.rearrange(instr_feats, "bt l c -> l bt c")
            instr_pos = torch.zeros(total_timesteps, instr_feats.shape[0], 3, device=device)
            instr_pos = self.relative_pe_layer(instr_pos)
        else:
            instr_feats = None
            instr_pos = None

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.relative_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        ghost_pcd_features_pyramid = []
        ghost_pcd_pyramid = []
        position_pyramid = []
        rgb_mask_pyramid = []
        ghost_pcd_masks_pyramid = []

        for i in range(self.num_sampling_level):
            # Sample ghost points
            if i == 0:
                anchor = None
            else:
                anchor = gt_position if gt_position is not None else position_pyramid[-1]
            ghost_pcd_i = self._sample_ghost_points(total_timesteps, device, level=i, anchor=anchor)

            if i == 0:
                # Coarse RGB features
                rgb_features_i = rgb_feats_pyramid[i]
                rgb_pos_i = rgb_pos_pyramid[i]
                ghost_pcd_context_features_i = einops.rearrange(
                    rgb_features_i, "b ncam c h w -> (ncam h w) b c")
            else:
                # Local fine RGB features
                l2_pred_pos = ((position_pyramid[-1] - pcd_pyramid[i]) ** 2).sum(-1).sqrt()
                indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

                rgb_features_i = einops.rearrange(
                    rgb_feats_pyramid[i], "b ncam c h w -> b (ncam h w) c")
                rgb_features_i = torch.stack([
                    f[i] for (f, i) in zip(rgb_features_i, indices)])
                rgb_pos_i = torch.stack([
                    f[i] for (f, i) in zip(rgb_pos_pyramid[i], indices)])
                ghost_pcd_context_features_i = einops.rearrange(
                    rgb_features_i, "b npts c -> npts b c")

            # Compute ghost point features and their positional embeddings by attending to visual
            # features and current gripper position
            ghost_pcd_context_features_i = torch.cat(
                [ghost_pcd_context_features_i, curr_gripper_features], dim=0)
            ghost_pcd_context_pos_i = torch.cat([rgb_pos_i, curr_gripper_pos], dim=1)
            if self.use_instruction:
                ghost_pcd_context_features_i = self.vis_ins_attn_pyramid[i](
                    query=ghost_pcd_context_features_i, value=instr_feats,
                    query_pos=None, value_pos=None
                )[-1]
                
                ghost_pcd_context_features_i = torch.cat(
                    [ghost_pcd_context_features_i, instr_feats], dim=0)
                ghost_pcd_context_pos_i = torch.cat(
                    [ghost_pcd_context_pos_i, instr_pos], dim=1)
            (
                ghost_pcd_features_i,
                ghost_pcd_pos_i,
                ghost_pcd_to_rgb_attn_i
            ) = self._compute_ghost_point_features(
                ghost_pcd_i, ghost_pcd_context_features_i, ghost_pcd_context_pos_i,
                total_timesteps, level=i
            )

            # Initialize query features
            if i == 0:
                query_features = self.query_embed.weight.unsqueeze(1).repeat(1, total_timesteps, 1)

            query_context_features_i = ghost_pcd_context_features_i
            query_context_pos_i = ghost_pcd_context_pos_i
            
            if i == 0:
                # Given the query is not localized yet, we don't use positional embeddings
                query_pos_i = None
                context_pos_i = None
            else:
                # Now that the query is localized, we use positional embeddings
                query_pos_i = self.relative_pe_layer(position_pyramid[-1])
                context_pos_i = query_context_pos_i

            # The query cross-attends to context features (visual features and the current gripper position)
            query_features = self._compute_query_features(
                query_features, query_context_features_i,
                query_pos_i, context_pos_i,
                level=i
            )

            # The query decodes a mask over ghost points (used to predict the gripper position) and over visual
            # features (for visualization only)
            ghost_pcd_masks_i, rgb_mask_i = self._decode_mask(
                query_features,
                ghost_pcd_features_i, ghost_pcd_to_rgb_attn_i,
                height, width, level=i
            )
            query_features = query_features[-1]

            top_idx = torch.max(ghost_pcd_masks_i[-1], dim=-1).indices
            ghost_pcd_i = einops.rearrange(ghost_pcd_i, "b npts c -> b c npts")
            position_i = ghost_pcd_i[torch.arange(total_timesteps), :, top_idx].unsqueeze(1)

            ghost_pcd_pyramid.append(ghost_pcd_i)
            ghost_pcd_features_pyramid.append(ghost_pcd_features_i)
            position_pyramid.append(position_i)
            rgb_mask_pyramid.append(rgb_mask_i)
            ghost_pcd_masks_pyramid.append(ghost_pcd_masks_i)

        # Regress an offset from the ghost point's position to the predicted position
        if self.regress_position_offset:
            fine_ghost_pcd_offsets = self.ghost_point_offset_predictor(ghost_pcd_features_i)
            fine_ghost_pcd_offsets = einops.rearrange(fine_ghost_pcd_offsets, "npts b c -> b c npts")
        else:
            fine_ghost_pcd_offsets = None

        ghost_pcd = ghost_pcd_i
        ghost_pcd_masks = ghost_pcd_masks_i
        ghost_pcd_features = ghost_pcd_features_i

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_masks[-1], ghost_pcd, ghost_pcd_features, query_features, total_timesteps,
            fine_ghost_pcd_offsets if self.regress_position_offset else None
        )

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "position_pyramid": position_pyramid,
            "visible_rgb_mask_pyramid": rgb_mask_pyramid,
            "ghost_pcd_masks_pyramid":  ghost_pcd_masks_pyramid,
            "ghost_pcd_pyramid": ghost_pcd_pyramid,
            "fine_ghost_pcd_offsets": fine_ghost_pcd_offsets if self.regress_position_offset else None,
        }
