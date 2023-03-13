import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        raise NotImplementedError


class RotaryPositionEncoding3D(RotaryPositionEncoding):
    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    def forward(self,  XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        vox = XYZ
        x_position, y_position, z_position = vox[..., 0:1], vox[...,1:2], vox[...,2:3]
        div_term = torch.exp( torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device) *  (-math.log(10000.0) / (self.feature_dim // 3)))
        div_term = div_term.view( 1,1, -1) # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term) # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map( lambda  feat:torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
                [ sinx, cosx, siny, cosy, sinz, cosz] )
        sin_pos = torch.cat([sinx,siny,sinz], dim=-1)
        cos_pos = torch.cat([cosx,cosy,cosz], dim=-1)
        position_code = torch.stack( [cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class LearnedAbsolutePositionEncoding3D(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)
