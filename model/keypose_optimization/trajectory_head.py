import einops
import torch
from torch import nn

from model.utils.layers import ParallelAttentionLayer
from model.utils.encoder import Encoder


class TrajectoryHead(Encoder):

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

        # Keypose query
        self.keypose_query = nn.Embedding(1, embedding_dim)

        # If not use_rgb, then use an occupancy embedding
        if not use_rgb:
            self.occ_encoder = nn.Linear(3, embedding_dim)

        # Positional embeddings
        self.rgb_pos = nn.Linear(3, embedding_dim)
        self.curr_gripper_feats = nn.Linear(3, embedding_dim)

        # Keypose cross-attention to context
        self.keypose_attention = nn.ModuleList()
        self.attn_rounds = attn_rounds
        self.feat_scales = feat_scales_to_use
        self.keypose_attention = nn.ModuleList([
            ParallelAttentionLayer(
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False,
                self_attention2=False, cross_attention2=False,
                rotary_pe=False
            )
            for _ in range(self.num_cross_attn_layers)
            for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])

        # Noise regression
        self.noise_regressor = nn.Linear(embedding_dim, output_dim)

    def forward(self, visible_rgb, visible_pcd, curr_gripper, instruction):
        """
        Arguments:
            visible_rgb: (B, num_cameras, 3, H, W) in [0, 1]
            visible_pcd: (B, num_cameras, 3, H, W) in world coordinates
            curr_gripper: (B, 3)
            instruction: (B, max_instruction_length, 512)
        """
        # Keypose features (B, 1, F)
        keypose_feats = self.keypose_query.weight.repeat(
            len(curr_gripper), 1
        ).unsqueeze(1)

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, _, pcd_pyramid = self.encode_images(
            visible_rgb, visible_pcd
        )
        # rgb_feats_pyramid: [(B, n_cameras, F, H_i, W_i)]
        # rgb_pos_pyramid [(B, n_cameras*H_i*W_i, F, 2)]
        # pcd_pyramid [(B, n_cameras*H_i*W_i, 3)]
        rgb_pos_pyramid = [self.rgb_pos(pcd) for pcd in pcd_pyramid]

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encode_instruction(instruction)

        # Encode current gripper (B, 1, F)
        curr_gripper_emb, _ = self.encode_curr_gripper(
            curr_gripper, batch_size=len(keypose_feats)
        )
        curr_gripper_feats = self.curr_gripper_feats(curr_gripper)[:, None]
        curr_gripper_feats = curr_gripper_feats + curr_gripper_emb

        # Attention layers
        noise = []
        for attn_round in range(self.attn_rounds):
            for scale in range(self.feat_scales):
                noise.append(self._one_attention_round(
                    rgb_feats_pyramid, pcd_pyramid, rgb_pos_pyramid,  # visual
                    instr_feats, None,  # language
                    curr_gripper_feats, None,  # current gripper
                    keypose_feats, None,  # keypose
                    attn_round, scale
                ))
        return noise

    def _one_attention_round(
        self,
        rgb_feats_pyramid, pcd_pyramid, rgb_pos_pyramid,  # visual
        instr_feats, instr_pos,  # language
        curr_gripper_feats, curr_gripper_pos,  # current gripper
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
        context_feats = context_feats + rgb_pos_pyramid[scale]
        context_pos = None

        # Language context
        if self.use_instruction:
            context_feats = torch.cat([context_feats, instr_feats], dim=1)

        # Concatenate rest of context (grippers, time)
        context_feats = torch.cat([
            context_feats,
            curr_gripper_feats
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
