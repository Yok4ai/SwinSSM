"""VITMamba: Axial-ViT encoder (stages 0-1) + Mamba encoder (stages 2-3) + UNETR decoder.

Imports shared building blocks from swinmamba to avoid duplication.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath

from .swinmamba import MambaLayer, MlpChannel, GSC


# ---------------------------------------------------------------------------
# VITTransformerBlock
# ---------------------------------------------------------------------------

class VITTransformerBlock(nn.Module):
    """Axial Vision Transformer block.

    Uses 3 sequential 1D attentions along D, H, W axes instead of global O(L²)
    or windowed attention. O(L × max_dim) complexity — feasible at stage-0 (64³).
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn_d = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_h = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_w = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        mlp_hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, C, D, H, W] channels-first
        B, C, D, H, W = x.shape
        skip = x

        # Pre-norm (operate in [B, L, C] then reshape back)
        x_flat = x.reshape(B, C, -1).transpose(1, 2)        # [B, L, C]
        x_norm = self.norm1(x_flat).transpose(1, 2).reshape(B, C, D, H, W)

        # --- Axial attention: depth axis ---
        # Group tokens by (H, W) position, attend over D
        xd = x_norm.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        xd, _ = self.attn_d(xd, xd, xd)
        x_axial = xd.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)  # [B,C,D,H,W]

        # --- Axial attention: height axis ---
        xh = x_axial.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C)
        xh, _ = self.attn_h(xh, xh, xh)
        x_axial = xh.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2)  # [B,C,D,H,W]

        # --- Axial attention: width axis ---
        xw = x_axial.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C)
        xw, _ = self.attn_w(xw, xw, xw)
        x_axial = xw.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B,C,D,H,W]

        x = skip + self.drop_path(x_axial)

        # MLP with pre-norm
        x_flat = x.reshape(B, C, -1).transpose(1, 2)             # [B, L, C]
        x_mlp = self.fc2(self.act(self.fc1(self.norm2(x_flat))))  # [B, L, C]
        x = x + self.drop_path(x_mlp).transpose(1, 2).reshape(B, C, D, H, W)

        return x


# ---------------------------------------------------------------------------
# VITMambaEncoder
# ---------------------------------------------------------------------------

class VITMambaEncoder(nn.Module):
    """Same 4-stage encoder as SwinMambaEncoder.
    Stages 0-1: VITTransformerBlock (axial attention).
    Stages 2-3: MambaLayer (multi-directional SSM).
    No GSC for ViT stages (conv bias not needed with global context).
    """
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3], mlp_ratio=4.0,
                 num_heads=[3, 6, 12, 24], d_state=16, d_conv=3, expand=1):
        super().__init__()

        # Conv stem + 3 downsamples (identical to SwinMambaEncoder)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(dims[0], eps=1e-4),
            nn.ReLU(),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm3d(dims[i], eps=1e-4),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        num_slices_list = [64, 32, 16, 8]

        self.stages = nn.ModuleList()
        self.vit_stages = nn.ModuleList()
        self.conv_stages = nn.ModuleList()
        cur = 0

        for i in range(4):
            if i < 2:  # ViT stages
                vit_stage = nn.ModuleList([
                    VITTransformerBlock(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[cur + j],
                        norm_layer=nn.LayerNorm,
                    ) for j in range(depths[i])
                ])
                self.stages.append(None)
                self.vit_stages.append(vit_stage)
                self.conv_stages.append(None)
            else:  # Mamba stages
                self.stages.append(nn.Sequential(
                    *[MambaLayer(dim=dims[i], d_state=d_state, d_conv=d_conv,
                                 expand=expand, num_slices=num_slices_list[i],
                                 layer_scale=layer_scale_init_value)
                      for j in range(depths[i])]
                ))
                self.vit_stages.append(None)
                self.conv_stages.append(GSC(dims[i]))
            cur += depths[i]

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i in range(4):
            self.add_module(f'norm{i}', nn.BatchNorm3d(dims[i], eps=1e-4))
            self.mlps.append(MlpChannel(dims[i], 2 * dims[i]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.vit_stages[i] is not None:
                for blk in self.vit_stages[i]:
                    x = blk(x)
            else:
                x = self.conv_stages[i](x)
                x = self.stages[i](x)
            if i in self.out_indices:
                x_out = getattr(self, f'norm{i}')(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


# ---------------------------------------------------------------------------
# VITMambaV2
# ---------------------------------------------------------------------------

class VITMambaV2(nn.Module):
    """VITMamba: Axial-ViT encoder (stages 0-1) + Mamba encoder (stages 2-3)
    + identical UNETR decoder to SwinMamba.

    Drop-in replacement for SwinMamba — same constructor signature minus
    window_size/num_heads list (uses its own defaults).
    """
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        mlp_ratio=4.0,
        num_heads=[3, 6, 12, 24],
        d_state=16,
        d_conv=3,
        expand=1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.spatial_dims = spatial_dims

        self.vit = VITMambaEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Decoder — identical to SwinMamba
        self.encoder1 = UnetrBasicBlock(spatial_dims, in_chans, feat_size[0], 3, 1, norm_name, res_block)
        self.encoder2 = UnetrBasicBlock(spatial_dims, feat_size[0], feat_size[1], 3, 1, norm_name, res_block)
        self.encoder3 = UnetrBasicBlock(spatial_dims, feat_size[1], feat_size[2], 3, 1, norm_name, res_block)
        self.encoder4 = UnetrBasicBlock(spatial_dims, feat_size[2], feat_size[3], 3, 1, norm_name, res_block)
        self.encoder5 = UnetrBasicBlock(spatial_dims, feat_size[3], hidden_size, 3, 1, norm_name, res_block)
        self.decoder5 = UnetrUpBlock(spatial_dims, hidden_size,   feat_size[3], 3, 2, norm_name, res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims, feat_size[3],  feat_size[2], 3, 2, norm_name, res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims, feat_size[2],  feat_size[1], 3, 2, norm_name, res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims, feat_size[1],  feat_size[0], 3, 2, norm_name, res_block)
        self.decoder1 = UnetrBasicBlock(spatial_dims, feat_size[0], feat_size[0], 3, 1, norm_name, res_block)
        self.out = UnetOutBlock(spatial_dims, feat_size[0], out_chans)

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out  = self.decoder1(dec0)
        return self.out(out)
