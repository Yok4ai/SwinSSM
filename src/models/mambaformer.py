"""MambaFormer: NNFormer windowed-Swin encoder (stages 0-1) + Mamba encoder (stages 2-3) + UNETR decoder.

NNFormer building blocks are inlined here (no nnformer package required).
Replaces:  timm.DropPath → monai.networks.layers.DropPath
           trunc_normal_ → nn.init.trunc_normal_
           to_3tuple     → _to_3tuple (local helper)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath

from .swinmamba import MambaLayer, MlpChannel, GSC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


# ---------------------------------------------------------------------------
# NNFormer building blocks (inlined)
# ---------------------------------------------------------------------------

class NNMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def nn_window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size,
                   H // window_size, window_size,
                   W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, window_size, window_size, window_size, C)


def nn_window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size ** 3))
    x = windows.view(B, S // window_size, H // window_size, W // window_size,
                     window_size, window_size, window_size, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)


class NNWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = _to_3tuple(window_size)  # (ws, ws, ws)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * ws[0] - 1) * (2 * ws[1] - 1) * (2 * ws[2] - 1),
                num_heads))

        coords_s = torch.arange(ws[0])
        coords_h = torch.arange(ws[1])
        coords_w = torch.arange(ws[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w], indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws[0] - 1
        rel[:, :, 1] += ws[1] - 1
        rel[:, :, 2] += ws[2] - 1
        rel[:, :, 0] *= (3 * ws[1] - 1)
        rel[:, :, 1] *= (2 * ws[1] - 1)
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, pos_embed=None):
        B_, N, C = x.shape
        ws = self.window_size
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws[0]*ws[1]*ws[2], ws[0]*ws[1]*ws[2], -1).permute(2, 0, 1)
        attn = attn + rpb.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if pos_embed is not None:
            x = x + pos_embed
        return self.proj_drop(self.proj(x))


class NNSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (S, H, W) — fixed at init
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = NNWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = NNMlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix, S, H, W):
        # S, H, W passed dynamically — supports variable sliding-window crops
        B, L, C = x.shape
        assert L == S * H * W, f"input size mismatch: L={L} vs S*H*W={S*H*W}"

        shortcut = x
        x = self.norm1(x).view(B, S, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size,) * 3, dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            attn_mask = None

        x_win = nn_window_partition(x, self.window_size)
        x_win = x_win.view(-1, self.window_size ** 3, C)
        attn_win = self.attn(x_win, mask=attn_mask)
        attn_win = attn_win.view(-1, self.window_size, self.window_size, self.window_size, C)
        x = nn_window_reverse(attn_win, self.window_size, Sp, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size,) * 3, dims=(1, 2, 3))
        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NNBasicLayer(nn.Module):
    """NNFormer's BasicLayer — windowed Swin attention with SW-MSA mask.
    Always use downsample=None (downsampling handled by NNFormerMambaEncoder's
    downsample_layers).
    """
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.blocks = nn.ModuleList([
            NNSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = None

    def forward(self, x, S, H, W):
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for s in slices:
            for h in slices:
                for w in slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1
        mask_windows = nn_window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size ** 3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)

        for blk in self.blocks:
            x = blk(x, attn_mask, S, H, W)
        # Return 8-tuple to match original nnFormer API (no-op downsample)
        return x, S, H, W, x, S, H, W


# ---------------------------------------------------------------------------
# NNFormerMambaEncoder
# ---------------------------------------------------------------------------

class NNFormerMambaEncoder(nn.Module):
    """Same 4-stage encoder pattern.
    Stages 0-1: NNFormer windowed Swin (NNBasicLayer, fixed input_resolution).
    Stages 2-3: MambaLayer (multi-directional SSM).

    Requires input_img_size (int) to compute per-stage spatial resolutions.
    Assumes cubic input: (input_img_size,) × 3.
    """
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3], window_size=4, mlp_ratio=4.0,
                 num_heads=[3, 6, 12, 24], d_state=16, d_conv=3, expand=1,
                 input_img_size=128):
        super().__init__()

        # Stage spatial resolutions (stem stride-2, each downsample stride-2)
        stage_res = [input_img_size >> (i + 1) for i in range(4)]

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

        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, sum(depths))]
        num_slices_list = [64, 32, 16, 8]

        self.stages = nn.ModuleList()
        self.nn_stages = nn.ModuleList()
        self.conv_stages = nn.ModuleList()
        cur = 0

        for i in range(4):
            if i < 2:  # NNFormer Swin stages
                res = stage_res[i]
                self.nn_stages.append(NNBasicLayer(
                    dim=dims[i],
                    input_resolution=(res, res, res),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[cur:cur + depths[i]],
                    norm_layer=nn.LayerNorm,
                    downsample=None,
                ))
                self.stages.append(None)
                self.conv_stages.append(None)
            else:  # Mamba stages
                self.stages.append(nn.Sequential(
                    *[MambaLayer(dim=dims[i], d_state=d_state, d_conv=d_conv,
                                 expand=expand, num_slices=num_slices_list[i],
                                 layer_scale=layer_scale_init_value)
                      for _ in range(depths[i])]
                ))
                self.nn_stages.append(None)
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
            if self.nn_stages[i] is not None:
                B, C, D, H, W = x.shape
                x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, L, C]
                result = self.nn_stages[i](x_flat, D, H, W)
                x_out = result[0]  # [B, L, C]
                x = x_out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
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
# MambaFormer
# ---------------------------------------------------------------------------

class MambaFormer(nn.Module):
    """MambaFormer: NNFormer windowed-Swin encoder (stages 0-1) +
    Mamba encoder (stages 2-3) + identical UNETR decoder to SwinMamba.

    Key difference from SwinMamba: stages 0-1 use NNFormer's Swin variant
    (fixed input_resolution at init, token [B,L,C] format internally,
    window_size is a scalar int).

    Args:
        window_size: scalar int (NNFormer API), default 4.
        input_img_size: cubic input spatial size (default 128 for BraTS 128³).
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
        window_size=4,
        mlp_ratio=4.0,
        num_heads=[3, 6, 12, 24],
        d_state=16,
        d_conv=3,
        expand=1,
        input_img_size=128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.spatial_dims = spatial_dims

        self.vit = NNFormerMambaEncoder(
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
            input_img_size=input_img_size,
        )

        # Decoder — identical to SwinMamba and VITMambaV2
        self.encoder1 = UnetrBasicBlock(spatial_dims, in_chans, feat_size[0], 3, 1, norm_name, res_block)
        self.encoder2 = UnetrBasicBlock(spatial_dims, feat_size[0], feat_size[1], 3, 1, norm_name, res_block)
        self.encoder3 = UnetrBasicBlock(spatial_dims, feat_size[1], feat_size[2], 3, 1, norm_name, res_block)
        self.encoder4 = UnetrBasicBlock(spatial_dims, feat_size[2], feat_size[3], 3, 1, norm_name, res_block)
        self.encoder5 = UnetrBasicBlock(spatial_dims, feat_size[3], hidden_size, 3, 1, norm_name, res_block)
        self.decoder5 = UnetrUpBlock(spatial_dims, hidden_size,  feat_size[3], 3, 2, norm_name, res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims, feat_size[3], feat_size[2], 3, 2, norm_name, res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims, feat_size[2], feat_size[1], 3, 2, norm_name, res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims, feat_size[1], feat_size[0], 3, 2, norm_name, res_block)
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
