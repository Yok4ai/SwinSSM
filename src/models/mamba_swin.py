"""MambaSwinV2: Reverse of SwinMamba.

Encoder stage layout:
  Stage 0 (64³):  GSC + Mamba
  Stage 1 (32³):  GSC + Mamba
  Stage 2 (16³):  Swin
  Stage 3 (8³):   Swin

Same UNETR decoder as SwinMamba.
Hypothesis: Mamba first captures long-range context at fine resolution;
Swin then refines semantics at coarse resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .swinunetr import SwinTransformerBlock
from .swinmamba import MambaLayer, MlpChannel, GSC


class MambaSwinEncoder(nn.Module):
    """4-stage encoder: Mamba (stages 0-1) → Swin (stages 2-3)."""

    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3], window_size=[7, 7, 7],
                 mlp_ratio=4.0, num_heads=[3, 6, 12, 24],
                 d_state=16, d_conv=3, expand=1):
        super().__init__()

        # Conv stem + 3 downsamples (identical to SwinMambaEncoder)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(dims[0], eps=1e-4),
            nn.ReLU(),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm3d(dims[i], eps=1e-4),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        num_slices_list = [64, 32, 16, 8]

        self.stages = nn.ModuleList()
        self.swin_stages = nn.ModuleList()
        self.conv_stages = nn.ModuleList()
        cur = 0

        for i in range(4):
            use_mamba = (i < 2)  # Mamba for early stages, Swin for later

            if use_mamba:
                self.stages.append(nn.Sequential(
                    *[MambaLayer(dim=dims[i], d_state=d_state, d_conv=d_conv,
                                 expand=expand, num_slices=num_slices_list[i],
                                 layer_scale=layer_scale_init_value)
                      for _ in range(depths[i])]
                ))
                self.conv_stages.append(GSC(dims[i]))
                self.swin_stages.append(None)
            else:
                swin_stage = nn.ModuleList([
                    SwinTransformerBlock(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=[0, 0, 0] if j % 2 == 0 else [w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[cur + j],
                        norm_layer=nn.LayerNorm,
                    ) for j in range(depths[i])
                ])
                self.swin_stages.append(swin_stage)
                self.stages.append(None)
                self.conv_stages.append(None)
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

            if self.swin_stages[i] is not None:
                x_swin = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
                for blk in self.swin_stages[i]:
                    x_swin = blk(x_swin, mask_matrix=None)
                x = x_swin.permute(0, 4, 1, 2, 3)
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


class MambaSwinV2(nn.Module):
    """MambaSwinV2: reverse of SwinMamba.
    Mamba (stages 0-1, fine resolution) + Swin (stages 2-3, coarse resolution)
    + identical UNETR decoder.
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
        window_size=[7, 7, 7],
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

        self.vit = MambaSwinEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            window_size=window_size,
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
