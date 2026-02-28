from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial
import math

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from .swinunetr import SwinTransformerBlock, PatchMerging
# from mamba_ssm import Mamba  # Unused - replaced by MambaVisionMixer3D
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class MambaVisionMixer3D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
        num_directions=3,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.num_directions = num_directions
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * self.num_directions, bias=bias, **factory_kwargs)    
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(self.num_directions)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
            for _ in range(self.num_directions)
        ])
        
        # Initialize dt projection for each direction
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        for dt_proj in self.dt_proj:
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError
                
            dt = torch.exp(
                torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True
        
        # Create separate A and D parameters for each direction
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n", d=self.d_inner//2
            ).contiguous()))
            for _ in range(self.num_directions)
        ])
        for A_log in self.A_log:
            A_log._no_weight_decay = True
            
        self.D = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner//2, device=device))
            for _ in range(self.num_directions)
        ])
        for D in self.D:
            D._no_weight_decay = True
            
        self.out_proj = nn.Linear(self.d_inner * self.num_directions, self.d_model, bias=bias, **factory_kwargs)
        
        # Separate conv layers for each direction
        self.conv1d_x = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner//2,
                padding=(d_conv-1)//2,
                **factory_kwargs,
            ) for _ in range(self.num_directions)
        ])
        self.conv1d_z = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner//2,
                padding=(d_conv-1)//2,
                **factory_kwargs,
            ) for _ in range(self.num_directions)
        ])

    def forward(self, hidden_states, spatial_shape=None):
        """Forward pass with multi-directional 3D scanning"""
        B, L, D = hidden_states.shape
        
        # Project input
        xz = self.in_proj(hidden_states)  # [B, L, d_inner * num_directions]
        xz = rearrange(xz, "b l (n d) -> b n l d", n=self.num_directions)
        
        direction_outputs = []
        
        for i in range(self.num_directions):
            # Get direction-specific features
            xz_dir = xz[:, i]  # [B, L, d_inner]
            xz_dir = rearrange(xz_dir, "b l d -> b d l")
            x, z = xz_dir.chunk(2, dim=1)  # Each: [B, d_inner//2, L]
            
            # Apply direction-specific transformations
            if spatial_shape is not None:
                x, z = self._apply_3d_scanning(x, z, spatial_shape, direction=i)
            
            # Apply convolutions
            x = F.silu(self.conv1d_x[i](x))
            z = F.silu(self.conv1d_z[i](z))
            
            actual_seqlen = x.shape[2]
            
            # SSM parameters
            A = -torch.exp(self.A_log[i].float())
            x_dbl = self.x_proj[i](rearrange(x, "b d l -> (b l) d"))
            dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = rearrange(self.dt_proj[i](dt), "(b l) d -> b d l", l=actual_seqlen)
            B_param = rearrange(B_param, "(b l) dstate -> b dstate l", l=actual_seqlen).contiguous()
            C_param = rearrange(C_param, "(b l) dstate -> b dstate l", l=actual_seqlen).contiguous()
            
            # Selective scan
            y = selective_scan_fn(x, 
                                  dt, 
                                  A, 
                                  B_param, 
                                  C_param, 
                                  self.D[i].float(), 
                                  z=None, 
                                  delta_bias=self.dt_proj[i].bias.float(), 
                                  delta_softplus=True, 
                                  return_last_state=None)
            
            # Combine with z and reverse spatial scanning if applied
            y = torch.cat([y, z], dim=1)  # [B, d_inner, L]
            if spatial_shape is not None:
                y = self._reverse_3d_scanning(y, spatial_shape, direction=i)
            
            direction_outputs.append(y)
        
        # Concatenate all directions
        y_combined = torch.cat(direction_outputs, dim=1)  # [B, d_inner * num_directions, L]
        y_combined = rearrange(y_combined, "b d l -> b l d")  # [B, L, d_inner * num_directions]
        
        # Final projection
        out = self.out_proj(y_combined)  # [B, L, d_model]
        return out
    
    def _apply_3d_scanning(self, x, z, spatial_shape, direction):
        """Apply direction-specific 3D spatial scanning patterns"""
        D, H, W = spatial_shape
        B = x.shape[0]
        C = x.shape[1]
        
        x = rearrange(x, "b c (d h w) -> b c d h w", d=D, h=H, w=W)
        z = rearrange(z, "b c (d h w) -> b c d h w", d=D, h=H, w=W)
        
        if direction == 0:  # Axial scanning (depth-first)
            x = rearrange(x, "b c d h w -> b c (d h w)")
            z = rearrange(z, "b c d h w -> b c (d h w)")
        elif direction == 1:  # Sagittal scanning (width-first)  
            x = rearrange(x, "b c d h w -> b c (w d h)")
            z = rearrange(z, "b c d h w -> b c (w d h)")
        elif direction == 2:  # Coronal scanning (height-first)
            x = rearrange(x, "b c d h w -> b c (h w d)")
            z = rearrange(z, "b c d h w -> b c (h w d)")
            
        return x, z
    
    def _reverse_3d_scanning(self, y, spatial_shape, direction):
        """Reverse the spatial scanning to original order"""
        D, H, W = spatial_shape
        B = y.shape[0]
        C = y.shape[1]
        
        if direction == 0:  # Reverse axial
            y = rearrange(y, "b c (d h w) -> b c d h w", d=D, h=H, w=W)
        elif direction == 1:  # Reverse sagittal
            y = rearrange(y, "b c (w d h) -> b c d h w", d=D, h=H, w=W)
        elif direction == 2:  # Reverse coronal
            y = rearrange(y, "b c (h w d) -> b c d h w", d=D, h=H, w=W)
            
        y = rearrange(y, "b c d h w -> b c (d h w)")
        return y

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=3, expand=1, num_slices=None, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # Use enhanced MambaVision3DMixer instead of basic Mamba
        self.mamba = MambaVisionMixer3D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_directions=3,  # Enable multi-directional scanning
        )
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.use_layer_scale = True
        else:
            self.use_layer_scale = False
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        # Prepare spatial shape for 3D scanning
        spatial_shape = img_dims if len(img_dims) == 3 else None
        
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        
        # Pass spatial shape to enable multi-directional scanning
        x_mamba = self.mamba(x_norm, spatial_shape=spatial_shape)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        if self.use_layer_scale:
            out = self.gamma.view(1, -1, 1, 1, 1) * out
            
        out = out + x_skip
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

# OptionalAttention removed - not needed in clean Swin→Mamba hybrid

class SwinMambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 window_size=[7, 7, 7], mlp_ratio=4.0, num_heads=[3, 6, 12, 24],
                 d_state=16, d_conv=3, expand=1):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              nn.BatchNorm3d(dims[0], eps=1e-4),
              nn.ReLU()
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dims[i], eps=1e-4),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Create drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        self.swin_stages = nn.ModuleList()
        self.conv_stages = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        
        for i in range(4):
            # Use Swin for early stages (0-1), Mamba for later stages (2-3)
            use_swin = (i == 0 or i == 1)
            
            if use_swin:
                # Create Swin Transformer blocks for early stages
                swin_stage = nn.ModuleList([
                    SwinTransformerBlock(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=[0, 0, 0] if j % 2 == 0 else [w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[cur + j] if isinstance(dpr, list) else dpr,
                        norm_layer=nn.LayerNorm,
                    ) for j in range(depths[i])
                ])
                stage = None
                conv_stage = None
            else:
                # Use Mamba blocks for later stages
                stage = nn.Sequential(
                    *[MambaLayer(dim=dims[i], 
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,
                                num_slices=num_slices_list[i],
                                layer_scale=layer_scale_init_value) for j in range(depths[i])]
                )
                conv_stage = GSC(dims[i])  # Keep GSC for Mamba stages
                swin_stage = None

            self.stages.append(stage)
            self.swin_stages.append(swin_stage)
            self.conv_stages.append(conv_stage)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.BatchNorm3d(dims[i_layer], eps=1e-4)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            
            # Apply stage-specific processing
            if self.swin_stages[i] is not None:
                # Swin Transformer blocks (early stages)
                # Convert from [B, C, D, H, W] to [B, D, H, W, C] for Swin blocks
                B, C, D, H, W = x.shape
                x_swin = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
                
                for swin_block in self.swin_stages[i]:
                    x_swin = swin_block(x_swin, mask_matrix=None)
                
                # Convert back to [B, C, D, H, W]
                x = x_swin.permute(0, 4, 1, 2, 3)
            else:
                # Mamba blocks (later stages)
                if self.conv_stages[i] is not None:
                    x = self.conv_stages[i](x)
                x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SwinMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,  # Kept for compatibility
        res_block: bool = True,
        spatial_dims=3,
        window_size=[7, 7, 7],
        mlp_ratio=4.0,
        num_heads=[3, 6, 12, 24],
        # Mamba parameters
        d_state=16,
        d_conv=3,
        expand=1,
        # use_optional_attention removed - clean hybrid design
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        # Store Mamba parameters
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.spatial_dims = spatial_dims
        self.vit = SwinMambaEncoder(in_chans, 
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
        
        # Standard UNETR decoder blocks
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
                
        return self.out(out)