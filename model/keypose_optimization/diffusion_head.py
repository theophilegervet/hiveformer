import einops
import torch
from torch import nn

from model.utils.layers import ParallelAttentionLayer
from model.utils.encoder import Encoder


class DiffusionHead(Encoder):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_attn_heads=4,
                 num_query_cross_attn_layers=8,
                 num_sampling_level=3,
                 use_instruction=False,
                 use_rgb=True,
                 use_sigma=False,
                 feat_scales_to_use=1,
                 attn_rounds=1):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=num_sampling_level,
            use_sigma=use_sigma
        )
        self.num_cross_attn_layers = num_query_cross_attn_layers
        self.use_instruction = use_instruction
        self.use_rgb = use_rgb

        # Keypose encoder
        self.keypose_encoder = nn.Linear(output_dim, embedding_dim)

        # If not use_rgb, then use an occupancy embedding
        if not use_rgb:
            self.occ_encoder = nn.Linear(3, embedding_dim)

        # Keypose cross-attention to context
        self.keypose_attention = nn.ModuleList()
        self.attn_rounds = attn_rounds
        self.feat_scales = feat_scales_to_use
        self.keypose_attention = nn.ModuleList([
            ParallelAttentionLayer(
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False,
                self_attention2=False, cross_attention2=False,
                rotary_pe=True
            )
            for _ in range(self.num_cross_attn_layers)
            # for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])

        # Noise regression
        self.noise_regressor = nn.Linear(embedding_dim, output_dim)

    def forward(self, keypose, timestep,
                visible_rgb, visible_pcd, curr_gripper, instruction,
                gt_action=None):
        """
        Arguments:
            keypose: (B, output_dim)
            timestep: (B, 1)
            visible_rgb: (B, num_cameras, 3, H, W) in [0, 1]
            visible_pcd: (B, num_cameras, 3, H, W) in world coordinates
            curr_gripper: (B, 3)
            instruction: (B, max_instruction_length, 512)
        """
        # Keypose features (B, 1, F)
        keypose = keypose[:, None]
        keypose_feats = self.keypose_encoder(keypose)
        keypose_pos = self.relative_pe_layer(keypose[..., :3])

        # Timestep features (B, 1, F)
        time_feats, time_pos = self.encode_denoising_timestep(timestep)

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, rgb_pos_pyramid, pcd_pyramid = self.encode_images(
            visible_rgb, visible_pcd
        )
        # rgb_feats_pyramid: [(B, n_cameras, F, H_i, W_i)]
        # rgb_pos_pyramid [(B, n_cameras*H_i*W_i, F, 2)]
        # pcd_pyramid [(B, n_cameras*H_i*W_i, 3)]
        if gt_action is not None:
            local_rgb_feats_pyramid = []
            local_rgb_pos_pyramid = []
            local_pcd_pyramid = []
            for i in range((len(rgb_feats_pyramid))):
                l2_pred_pos = ((gt_action[:, None, :3] - pcd_pyramid[i]) ** 2).sum(-1).sqrt()
                indices = l2_pred_pos.topk(
                    k=32 * 32 * rgb_feats_pyramid[i].shape[1],
                    dim=-1, largest=False
                ).indices

                rgb_feats_i = einops.rearrange(
                    rgb_feats_pyramid[i], "b ncam c h w -> b (ncam h w) c"
                )
                local_rgb_feats_pyramid.append(torch.stack([
                    f[i] for (f, i) in zip(rgb_feats_i, indices)])
                )
                local_rgb_pos_pyramid.append(torch.stack([
                    f[i] for (f, i) in zip(rgb_pos_pyramid[i], indices)])
                )
                local_pcd_pyramid.append(torch.stack([
                    f[i] for (f, i) in zip(pcd_pyramid[i], indices)])
                )
            rgb_feats_pyramid = [
                local_rgb_feats_pyramid[i].reshape(
                    len(rgb_feats_pyramid[i]),
                    rgb_feats_pyramid[i].shape[1],
                    32, 32, -1
                ).permute(0, 1, 4, 2, 3)
                for i in range((len(rgb_feats_pyramid)))
            ]
            rgb_pos_pyramid = local_rgb_pos_pyramid
            pcd_pyramid = local_pcd_pyramid

        # Encode instruction (B, 53, F)
        instr_feats, instr_pos = None, None
        if self.use_instruction:
            instr_feats, instr_pos = self.encode_instruction(instruction)

        # Encode current gripper (B, 1, F)
        curr_gripper_feats, curr_gripper_pos = self.encode_curr_gripper(
            curr_gripper, batch_size=len(keypose_feats)
        )

        # Attention layers
        noise = []
        # for attn_round in range(self.attn_rounds):
        for scale in range(self.feat_scales):
            noise.append(self._one_attention_round(
                rgb_feats_pyramid, pcd_pyramid, rgb_pos_pyramid,  # visual
                instr_feats, instr_pos,  # language
                curr_gripper_feats, curr_gripper_pos,  # current gripper
                time_feats, time_pos,  # time
                keypose_feats, keypose_pos,  # keypose
                attn_round=0, scale=scale
            ))
        return noise

    def _one_attention_round(
        self,
        rgb_feats_pyramid, pcd_pyramid, rgb_pos_pyramid,  # visual
        instr_feats, instr_pos,  # language
        curr_gripper_feats, curr_gripper_pos,  # current gripper
        time_feats, time_pos,  # time
        keypose_feats, keypose_pos,  # keypose
        attn_round, scale
    ):
        # Visual context
        if self.use_rgb:
            context_feats = einops.rearrange(
                rgb_feats_pyramid[scale],
                "b ncam c h w -> b (ncam h w) c"
            )
        else:
            context_feats = self.occ_encoder(pcd_pyramid[scale])
        context_pos = rgb_pos_pyramid[scale]

        # Language context
        if self.use_instruction:
            context_feats = torch.cat([context_feats, instr_feats], dim=1)
            context_pos = torch.cat([context_pos, instr_pos], dim=1)

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

        # Keypose features cross-attend to context features
        l_offset = (
            attn_round * (self.num_cross_attn_layers * self.feat_scales)
            + scale * self.num_cross_attn_layers
        )
        for layer in range(l_offset, l_offset + self.num_cross_attn_layers):
            keypose_feats, _ = self.keypose_attention[layer](
                seq1=keypose_feats, seq1_key_padding_mask=None,
                seq2=context_feats, seq2_key_padding_mask=None,
                seq1_pos=keypose_pos, seq2_pos=context_pos,
                seq1_sem_pos=None, seq2_sem_pos=None
            )

        # Regress noise
        noise = self.noise_regressor(keypose_feats)  # (B, 1, output_dim)

        return noise.squeeze(1)
