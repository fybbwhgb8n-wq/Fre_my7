"""
GSFANet with Robust Capon Frequency Fusion (RCF)

This module extends the GSFANet architecture with Robust Capon Frequency Fusion
for improved infrared small target detection with robustness to guidance
vector mismatch.

The Robust Capon Fusion replaces or augments the standard frequency fusion
at decoder stages, providing:
    - Principled robustness to target signature variations
    - Convex per-pixel SOCP optimization with theoretical guarantees
    - One-shot closed-form solver (ADMM-free)

Key innovation (TRUE SCHEME 3 IMPLEMENTATION):
    For each pixel p, solve the robust SOCP:
        min_w   w^T R(p) w                        (minimize output power)
        s.t.    ||A(p)^T w||_2 <= a0(p)^T w - 1   (robust distortionless)
    
    where:
        R(p): Sample covariance from local neighborhood
        a0(p): Nominal guidance/steering vector
        A(p) = diag(sigma(p)): Uncertainty matrix
    
    One-shot closed-form solution:
        u = (R + gamma * A A^T)^{-1} a0
        margin = a0^T u - ||A^T u||_2
        w = u / margin

Author: Implementation for IRSTD with TRUE robust SOCP frequency fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import numbers

sys.path.append('../')
from utils.pooling import PWD2d
from model.fusion import Freq_Fusion, LowPassConvGenerator, HighPassConvGenerator
from model.robust_capon_fusion import (
    RobustCaponFrequencyFusion,
    RobustCaponFusionBlock,
)
from model.freq_decompose import FrequencyDecomposer
from einops import rearrange


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class RobustFreqFusion(nn.Module):
    """
    Robust Frequency Fusion module for decoder stages.
    
    TRUE SCHEME 3 IMPLEMENTATION: Uses one-shot closed-form SOCP solver.
    
    For each pixel p, solves:
        min_w   w^T R(p) w                        (minimize output power)
        s.t.    ||A(p)^T w||_2 <= a0(p)^T w - 1   (robust distortionless)
    
    One-shot solution:
        u = (R + gamma * A A^T)^{-1} a0
        margin = a0^T u - ||A^T u||_2
        w = u / margin
    
    Args:
        c_high: High-level (deeper) feature channels
        c_low: Low-level (shallower) feature channels
        c_out: Output channels
        K: Number of frequency streams (2 or 4)
        decompose_method: Frequency decomposition method
        gamma: Robustness loading strength (default: 1.0)
        cov_shrink: Covariance shrinkage (default: 0.1)
        sigma_max: Max uncertainty (default: 0.3)
        a0_mode: Guidance vector mode ('fixed', 'stage', 'pixel')
        sigma_mode: Uncertainty mode ('fixed', 'stage', 'pixel')
        patch_size: Local neighborhood size for covariance estimation
    """
    
    def __init__(
        self,
        c_high: int,
        c_low: int,
        c_out: int,
        K: int = 2,
        decompose_method: str = 'low_high',
        # TRUE SOCP solver parameters
        gamma: float = 1.0,
        cov_shrink: float = 0.1,
        cov_delta: float = 1e-3,
        sigma_max: float = 0.3,
        a0_mode: str = 'stage',
        sigma_mode: str = 'stage',
        patch_size: int = 3,
        use_double: bool = True,
        use_slack: bool = True,
        slack_threshold: float = 0.1,
        # Legacy parameters (ignored, kept for API compatibility)
        guidance_mode: str = 'stage',  # Alias for a0_mode
        kernel_size: int = 3,
        n_iters: int = 5,
        rho: float = 10.0,
        uncertainty_mode: str = 'diagonal',
        detach_solver: bool = True,
    ):
        super(RobustFreqFusion, self).__init__()
        assert c_high == c_low * 2, 'c_high must be 2 * c_low'
        
        self.c_high = c_high
        self.c_low = c_low
        self.c_out = c_out
        self.K = K
        
        # Use guidance_mode as alias for a0_mode if a0_mode not explicitly set
        if a0_mode == 'stage' and guidance_mode != 'stage':
            a0_mode = guidance_mode
        
        # Channel projection for high-level features
        self.high_proj = nn.Sequential(
            nn.Conv2d(c_high, c_low, 1, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        
        # Concatenation fusion (fallback path)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(c_high + c_low, c_high + c_low, 3, 1, padding='same', groups=c_high + c_low),
            nn.BatchNorm2d(c_high + c_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_high + c_low, c_low, 1, 1, padding='same'),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        
        # Frequency decomposition for the combined features
        self.decomposer = FrequencyDecomposer(
            in_channels=c_low,
            method=decompose_method,
            K=K,
        )
        self.K = self.decomposer.K  # May be updated by decomposer
        
        # TRUE ROBUST CAPON FREQUENCY FUSION (SOCP SOLVER)
        self.robust_fusion = RobustCaponFrequencyFusion(
            K=self.K,
            in_channels=c_low,
            patch_size=patch_size,
            a0_mode=a0_mode,
            sigma_mode=sigma_mode,
            sigma_max=sigma_max,
            gamma=gamma,
            cov_shrink=cov_shrink,
            cov_delta=cov_delta,
            use_double=use_double,
            use_slack=use_slack,
            slack_threshold=slack_threshold,
        )
        
        # Output convolution
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_low, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x_low, y_high, return_weights=False):
        """
        Fuse high-level and low-level features using TRUE robust Capon SOCP fusion.
        
        Args:
            x_low: [B, c_low, H, W] low-level (shallow) features
            y_high: [B, c_high, H/2, W/2] high-level (deep) features
            return_weights: Whether to return fusion weights and debug info
            
        Returns:
            fused: [B, c_out, H, W] fused features
            (optional) debug_dict: Solver debug information
        """
        # Upsample high-level features
        y_high_up = F.interpolate(y_high, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Project high-level features to match low-level channels
        y_high_proj = self.high_proj(y_high_up)
        
        # Concatenation path (global context)
        cat_fuse = torch.cat([x_low, y_high_up], dim=1)
        cat_fuse = self.cat_conv(cat_fuse)
        
        # Combine low-level and projected high-level
        combined = x_low + y_high_proj
        
        # Decompose into frequency streams
        freq_streams = self.decomposer(combined)  # [B, K, C, H, W]
        
        # Apply TRUE robust Capon SOCP fusion
        if return_weights:
            fused, debug_dict = self.robust_fusion(freq_streams, return_debug=True)
        else:
            fused = self.robust_fusion(freq_streams)
        
        # Add concatenation path and apply output convolution
        fused = fused + cat_fuse
        fused = self.out_conv(fused)
        
        # Safety: sanitize output to prevent NaN propagation
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        
        if return_weights:
            return fused, debug_dict
        return fused
    
    def get_regularization_loss(self):
        """Get regularization loss from robust fusion module."""
        return self.robust_fusion.get_regularization_loss()


class GSFANet_RCF(nn.Module):
    """
    GSFANet with TRUE Robust Capon Frequency Fusion (Scheme 3 SOCP).
    
    Implements the full robust MVDR/SOCP formulation:
        For each pixel p, solves:
            min_w   w^T R(p) w                        (minimize output power)
            s.t.    ||A(p)^T w||_2 <= a0(p)^T w - 1   (robust distortionless)
    
    Key features:
        - Explicit covariance R(p) estimation from local patches
        - Learnable guidance vector a0(p) and uncertainty sigma(p)
        - One-shot closed-form SOCP solution (ADMM-free)
        - Smooth gradients, stable training with slack mechanism
    
    Mathematical formulation:
        u = (R + gamma * diag(sigma^2))^{-1} a0
        margin = a0^T u - ||sigma * u||_2
        w = u / margin
    
    Args:
        size: Input image size
        input_channels: Number of input channels (1 for grayscale IR)
        K: Number of frequency streams for fusion
        decompose_method: Frequency decomposition method
        gamma: Robustness loading strength (default: 1.0)
        cov_shrink: Covariance shrinkage (default: 0.1)
        sigma_max: Max uncertainty for robustness envelope (default: 0.3)
        a0_mode: Guidance vector mode ('fixed', 'stage', 'pixel')
        sigma_mode: Uncertainty mode ('fixed', 'stage', 'pixel')
        patch_size: Local neighborhood size for covariance estimation
        use_rcf_stages: Which decoder stages use RCF
    """
    
    def __init__(
        self,
        size: int,
        input_channels: int,
        block=ResNet,
        K: int = 2,
        decompose_method: str = 'low_high',
        # TRUE SOCP solver parameters
        gamma: float = 1.0,
        cov_shrink: float = 0.1,
        cov_delta: float = 1e-3,
        sigma_max: float = 0.3,
        a0_mode: str = 'stage',
        sigma_mode: str = 'stage',
        patch_size: int = 3,
        use_double: bool = True,
        use_slack: bool = True,
        slack_threshold: float = 0.1,
        use_rcf_stages: list = None,
        # Legacy parameters (ignored, kept for API compatibility)
        guidance_mode: str = 'stage',  # Alias for a0_mode
        n_iters: int = 5,
        rho: float = 10.0,
        uncertainty_mode: str = 'diagonal',
        detach_solver: bool = True,
    ):
        super().__init__()
        
        param_channels = [16, 32, 64, 128]
        param_blocks = [2, 2, 2, 2]
        
        self.K = K
        self.use_rcf_stages = use_rcf_stages if use_rcf_stages is not None else [0, 1, 2]
        
        # Pooling layers (wavelet-based downsampling)
        self.pool1 = PWD2d(param_channels[0], param_channels[0] * 4)
        self.pool2 = PWD2d(param_channels[1], param_channels[1] * 4)
        self.pool3 = PWD2d(param_channels[2], param_channels[2] * 4)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        # Initial convolution
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        
        # Encoder stages
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block, param_blocks[0])
        self.encoder_1 = self._make_layer(param_channels[0] * 2, param_channels[1], block, param_blocks[1])
        self.encoder_2 = self._make_layer(param_channels[1] * 2, param_channels[2], block, param_blocks[2])
        
        # Bottleneck
        self.middle_layer = self._make_layer(param_channels[2] * 2, param_channels[3], block, param_blocks[3])
        
        # Multi-Level Attention
        self.mla = MLA(
            channel_num=[param_channels[0], param_channels[1], param_channels[2], param_channels[3]],
            patchSize=[32, 16, 8, 4], 
            layer_num=2, 
            size=size
        )
        
        # Decoder stages with TRUE Robust Capon Fusion (SOCP SOLVER)
        # Use guidance_mode as alias for a0_mode if needed
        if a0_mode == 'stage' and guidance_mode != 'stage':
            a0_mode = guidance_mode
        
        rcf_kwargs = {
            'K': K,
            'decompose_method': decompose_method,
            'gamma': gamma,
            'cov_shrink': cov_shrink,
            'cov_delta': cov_delta,
            'sigma_max': sigma_max,
            'a0_mode': a0_mode,
            'sigma_mode': sigma_mode,
            'patch_size': patch_size,
            'use_double': use_double,
            'use_slack': use_slack,
            'slack_threshold': slack_threshold,
        }
        
        if 2 in self.use_rcf_stages:
            self.decoder_2 = RobustFreqFusion(param_channels[3], param_channels[2], param_channels[2], **rcf_kwargs)
        else:
            self.decoder_2 = Freq_Fusion(param_channels[3], param_channels[2], param_channels[2])
        
        if 1 in self.use_rcf_stages:
            self.decoder_1 = RobustFreqFusion(param_channels[2], param_channels[1], param_channels[1], **rcf_kwargs)
        else:
            self.decoder_1 = Freq_Fusion(param_channels[2], param_channels[1], param_channels[1])
        
        if 0 in self.use_rcf_stages:
            self.decoder_0 = RobustFreqFusion(param_channels[1], param_channels[0], param_channels[0], **rcf_kwargs)
        else:
            self.decoder_0 = Freq_Fusion(param_channels[1], param_channels[0], param_channels[0])
        
        # Output heads
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.final = nn.Conv2d(3, 1, 3, 1, 1)
    
    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)
    
    def forward(self, x, tag=True, return_weights=False):
        """
        Forward pass with optional fusion weight output.
        
        Args:
            x: [B, 1, H, W] input IR image
            tag: Whether to return deep supervision outputs
            return_weights: Whether to return fusion weights (for visualization)
            
        Returns:
            masks: List of intermediate predictions (if tag=True)
            output: Final segmentation prediction
            (optional) weights: Dict of fusion weights per stage
        """
        # Encoder
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool1(x_e0))
        x_e2 = self.encoder_2(self.pool2(x_e1))
        
        # Bottleneck
        x_m = self.middle_layer(self.pool3(x_e2))
        
        # Multi-Level Attention
        x_e0, x_e1, x_e2, x_m, _ = self.mla(x_e0, x_e1, x_e2, x_m)
        
        # Decoder with optional weight collection
        weights = {}
        
        if return_weights and isinstance(self.decoder_2, RobustFreqFusion):
            x_d2, w2 = self.decoder_2(x_e2, x_m, return_weights=True)
            weights['stage2'] = w2
        else:
            x_d2 = self.decoder_2(x_e2, x_m)
        
        if return_weights and isinstance(self.decoder_1, RobustFreqFusion):
            x_d1, w1 = self.decoder_1(x_e1, x_d2, return_weights=True)
            weights['stage1'] = w1
        else:
            x_d1 = self.decoder_1(x_e1, x_d2)
        
        if return_weights and isinstance(self.decoder_0, RobustFreqFusion):
            x_d0, w0 = self.decoder_0(x_e0, x_d1, return_weights=True)
            weights['stage0'] = w0
        else:
            x_d0 = self.decoder_0(x_e0, x_d1)
        
        # Output
        if tag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2)], dim=1))
            
            if return_weights:
                return [mask0, mask1, mask2], output, weights
            return [mask0, mask1, mask2], output
        else:
            output = self.output_0(x_d0)
            if return_weights:
                return [], output, weights
            return [], output
    
    def get_regularization_loss(self):
        """Get total regularization loss from all RCF modules."""
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for decoder in [self.decoder_0, self.decoder_1, self.decoder_2]:
            if isinstance(decoder, RobustFreqFusion):
                reg_loss = reg_loss + decoder.get_regularization_loss()
        
        return reg_loss


# ============== Supporting Modules (from original GSFANet) ==============

class MLA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], patchSize=[32, 16, 8, 4], layer_num=1, size=256):
        super().__init__()
        patchSize = [size // 8, size // 16, size // 32, size // 64]
        overlap = [p // 2 for p in patchSize]
        self.embeddings_1 = Channel_Embeddings(patchSize[0], in_channels=channel_num[0], overlap=overlap[0])
        self.embeddings_2 = Channel_Embeddings(patchSize[1], in_channels=channel_num[1], overlap=overlap[1])
        self.embeddings_3 = Channel_Embeddings(patchSize[2], in_channels=channel_num[2], overlap=overlap[2])
        self.embeddings_4 = Channel_Embeddings(patchSize[3], in_channels=channel_num[3], overlap=overlap[3])

        self.layer = nn.ModuleList()
        for _ in range(layer_num):
            layer = MLA_Block(channel_num, mode='kernel')
            self.layer.append(copy.deepcopy(layer))

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1)
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1)
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1)
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1)

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, sa_weight = layer_block(emb1, emb2, emb3, emb4)

        org_size = [(en1.shape[2], en1.shape[3]), (en2.shape[2], en2.shape[3]), 
                    (en3.shape[2], en3.shape[3]), (en4.shape[2], en4.shape[3])]
        emb1 = self.reconstruct_1(emb1, size=org_size[0]) if emb1 is not None else None
        emb2 = self.reconstruct_2(emb2, size=org_size[1]) if emb2 is not None else None
        emb3 = self.reconstruct_3(emb3, size=org_size[2]) if emb3 is not None else None
        emb4 = self.reconstruct_4(emb4, size=org_size[3]) if emb4 is not None else None

        out1 = emb1 + en1 if en1 is not None else None
        out2 = emb2 + en2 if en2 is not None else None
        out3 = emb3 + en3 if en3 is not None else None
        out4 = emb4 + en4 if en4 is not None else None

        return out1, out2, out3, out4, sa_weight


class MLA_Block(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel'):
        super().__init__()
        self.msa = MSA(channel_num)
        self.mca = MCA(channel_num, mode=mode)
        self.mca_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.mca_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.mca_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.mca_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

    def forward(self, emb1, emb2, emb3, emb4):
        res1, res2, res3, res4 = emb1, emb2, emb3, emb4
        ca1, ca2, ca3, ca4 = self.mca(emb1, emb2, emb3, emb4)
        ca1 = ca1 + res1
        ca2 = ca2 + res2
        ca3 = ca3 + res3
        ca4 = ca4 + res4

        ca1 = self.mca_norm1(ca1) if ca1 is not None else None
        ca2 = self.mca_norm2(ca2) if ca2 is not None else None
        ca3 = self.mca_norm3(ca3) if ca3 is not None else None
        ca4 = self.mca_norm4(ca4) if ca4 is not None else None

        sa1, sa2, sa3, sa4, sa_weight = self.msa(ca1, ca2, ca3, ca4)

        return sa1, sa2, sa3, sa4, sa_weight


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, size=None):
        if x is None:
            return None
        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Channel_Embeddings(nn.Module):
    def __init__(self, patchsize, in_channels, overlap):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                          kernel_size=patchsize, stride=patchsize - overlap)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class MSA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128]):
        super().__init__()
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]
        self.conv_emb = nn.Conv2d(self.channel_size, self.channel_size // 8, 1, 1, padding='same')
        self.emb_norm = nn.BatchNorm2d(self.channel_size // 8)
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(channel_num[3], channel_num[3] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[3] // 8),
            nn.ReLU()
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(channel_num[2], channel_num[2], 3, 1, padding='same', groups=channel_num[2]),
            nn.Conv2d(channel_num[2], channel_num[2] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[2] // 8),
            nn.ReLU()
        )
        self.Conv_5 = nn.Sequential(
            nn.Conv2d(channel_num[1], channel_num[1], 5, 1, padding='same', groups=channel_num[1]),
            nn.Conv2d(channel_num[1], channel_num[1] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[1] // 8),
            nn.ReLU()
        )
        self.Conv_7 = nn.Sequential(
            nn.Conv2d(channel_num[0], channel_num[0], 7, 1, padding='same', groups=channel_num[0]),
            nn.Conv2d(channel_num[0], channel_num[0] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[0] // 8),
            nn.ReLU()
        )
        self.se = SE_Block(self.channel_size // 8 * 2, ratio=10)
        self.final_Conv = nn.Conv2d(self.channel_size // 8 * 2, 1, 1, 1, padding='same')
        self.sig = nn.Sigmoid()

    def forward(self, emb1, emb2, emb3, emb4):
        emb = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        emb = self.conv_emb(emb)
        emb = self.emb_norm(emb)
        emb_1 = self.Conv_7(emb1)
        emb_3 = self.Conv_5(emb2)
        emb_5 = self.Conv_3(emb3)
        emb_7 = self.Conv_1(emb4)

        sa = self.final_Conv(self.se(torch.cat([emb_1, emb_3, emb_5, emb_7, emb], dim=1)))
        sa = self.sig(sa)

        emb_1 = sa * emb1
        emb_2 = sa * emb2
        emb_3 = sa * emb3
        emb_4 = sa * emb4

        emb_1 = emb_1 + emb1
        emb_2 = emb_2 + emb2
        emb_3 = emb_3 + emb3
        emb_4 = emb_4 + emb4

        return emb_1, emb_2, emb_3, emb_4, sa


class MCA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel'):
        super().__init__()
        self.mode = mode
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(self.channel_size, LayerNorm_type='WithBias')

        if self.mode == 'kernel':
            self.q1_Conv = nn.Conv2d(channel_num[0], channel_num[0], 1, 1)
            self.k1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)
            self.v1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)

            self.q2_Conv = nn.Conv2d(channel_num[1], channel_num[1], 1, 1)
            self.k2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)
            self.v2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)

            self.q3_Conv = nn.Conv2d(channel_num[2], channel_num[2], 1, 1)
            self.k3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)
            self.v3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)

            self.q4_Conv = nn.Conv2d(channel_num[3], channel_num[3], 1, 1)
            self.k4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)
            self.v4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)

            self.out_Conv1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
            self.out_Conv2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
            self.out_Conv3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
            self.out_Conv4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

            self.softmax = nn.Softmax(dim=1)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)

    def kernel_similarity(self, Gq, Gk, sigma=None):
        assert Gq.shape == Gk.shape
        diff = Gq - Gk
        if sigma is None:
            sigma = torch.std(diff, dim=2).unsqueeze(-1)
        distance = torch.norm(diff, p=2, dim=2).unsqueeze(-1)
        similarity = torch.exp(-distance ** 2 / (2 * sigma ** 2))
        return similarity

    def forward(self, emb1, emb2, emb3, emb4):
        b, c, h, w = emb1.shape
        emb_all = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        emb1_norm = self.attn_norm1(emb1) if emb1 is not None else None
        emb2_norm = self.attn_norm2(emb2) if emb2 is not None else None
        emb3_norm = self.attn_norm3(emb3) if emb3 is not None else None
        emb4_norm = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)

        if self.mode == 'kernel':
            emb1_gq = self.q1_Conv(emb1_norm)
            emb_all_gk1 = self.k1_Conv(emb_all)
            emb_all_gv1 = self.v1_Conv(emb_all)
            emb1_gq = rearrange(emb1_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk1 = rearrange(emb_all_gk1, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv1 = rearrange(emb_all_gv1, 'b c h w -> b c (h w)', h=h, w=w)
            emb1_gq, emb_all_gk1 = (F.normalize(emb1_gq, dim=-1), F.normalize(emb_all_gk1, dim=-1))

            emb2_gq = self.q2_Conv(emb2_norm)
            emb_all_gk2 = self.k2_Conv(emb_all)
            emb_all_gv2 = self.v2_Conv(emb_all)
            emb2_gq = rearrange(emb2_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk2 = rearrange(emb_all_gk2, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv2 = rearrange(emb_all_gv2, 'b c h w -> b c (h w)', h=h, w=w)
            emb2_gq, emb_all_gk2 = (F.normalize(emb2_gq, dim=-1), F.normalize(emb_all_gk2, dim=-1))

            emb3_gq = self.q3_Conv(emb3_norm)
            emb_all_gk3 = self.k3_Conv(emb_all)
            emb_all_gv3 = self.v3_Conv(emb_all)
            emb3_gq = rearrange(emb3_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk3 = rearrange(emb_all_gk3, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv3 = rearrange(emb_all_gv3, 'b c h w -> b c (h w)', h=h, w=w)
            emb3_gq, emb_all_gk3 = (F.normalize(emb3_gq, dim=-1), F.normalize(emb_all_gk3, dim=-1))

            emb4_gq = self.q4_Conv(emb4_norm)
            emb_all_gk4 = self.k4_Conv(emb_all)
            emb_all_gv4 = self.v4_Conv(emb_all)
            emb4_gq = rearrange(emb4_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk4 = rearrange(emb_all_gk4, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv4 = rearrange(emb_all_gv4, 'b c h w -> b c (h w)', h=h, w=w)
            emb4_gq, emb_all_gk4 = (F.normalize(emb4_gq, dim=-1), F.normalize(emb_all_gk4, dim=-1))

            sim1 = self.kernel_similarity(emb1_gq, emb_all_gk1)
            sim2 = self.kernel_similarity(emb2_gq, emb_all_gk2)
            sim3 = self.kernel_similarity(emb3_gq, emb_all_gk3)
            sim4 = self.kernel_similarity(emb4_gq, emb_all_gk4)

            sim1, sim2, sim3, sim4 = self.softmax(sim1), self.softmax(sim2), self.softmax(sim3), self.softmax(sim4)

            out1 = sim1 * emb_all_gv1
            out2 = sim2 * emb_all_gv2
            out3 = sim3 * emb_all_gv3
            out4 = sim4 * emb_all_gv4

            out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h, w=w)
            out2 = rearrange(out2, 'b c (h w) -> b c h w', h=h, w=w)
            out3 = rearrange(out3, 'b c (h w) -> b c h w', h=h, w=w)
            out4 = rearrange(out4, 'b c (h w) -> b c h w', h=h, w=w)

            att_out1 = self.out_Conv1(out1)
            att_out2 = self.out_Conv2(out2)
            att_out3 = self.out_Conv3(out3)
            att_out4 = self.out_Conv4(out4)

            att_out1 = att_out1 + emb1 if emb1 is not None else None
            att_out2 = att_out2 + emb2 if emb2 is not None else None
            att_out3 = att_out3 + emb3 if emb3 is not None else None
            att_out4 = att_out4 + emb4 if emb4 is not None else None

            res1, res2, res3, res4 = att_out1, att_out2, att_out3, att_out4
            x1 = self.ffn_norm1(res1) if emb1 is not None else None
            x2 = self.ffn_norm2(res2) if emb2 is not None else None
            x3 = self.ffn_norm3(res3) if emb3 is not None else None
            x4 = self.ffn_norm4(res4) if emb4 is not None else None
            x1 = self.ffn1(x1) if emb1 is not None else None
            x2 = self.ffn2(x2) if emb2 is not None else None
            x3 = self.ffn3(x3) if emb3 is not None else None
            x4 = self.ffn4(x4) if emb4 is not None else None
            x1 = x1 + res1 if emb1 is not None else None
            x2 = x2 + res2 if emb2 is not None else None
            x3 = x3 + res3 if emb3 is not None else None
            x4 = x4 + res4 if emb4 is not None else None

            return x1, x2, x3, x4


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.layer_norm1 = LayerNorm3d(hidden_features * 2, LayerNorm_type='WithBias')
        self.act = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.layer_norm1(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


if __name__ == "__main__":
    # Unit test for GSFANet_RCF
    print("Testing GSFANet_RCF...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = GSFANet_RCF(
        size=256,
        input_channels=1,
        K=2,
        decompose_method='low_high',
        n_iters=5,
        rho=10.0,
        guidance_mode='stage',
        uncertainty_mode='diagonal',
    ).to(device)
    
    # Test input
    x = torch.randn(2, 1, 256, 256, device=device)
    
    # Forward pass with deep supervision
    print("\n1. Testing forward pass (tag=True)...")
    masks, output = model(x, tag=True)
    print(f"   Input: {x.shape}")
    print(f"   Masks: {[m.shape for m in masks]}")
    print(f"   Output: {output.shape}")
    
    # Forward pass without deep supervision
    print("\n2. Testing forward pass (tag=False)...")
    _, output = model(x, tag=False)
    print(f"   Output: {output.shape}")
    
    # Forward with weight visualization
    print("\n3. Testing with return_weights=True...")
    masks, output, weights = model(x, tag=True, return_weights=True)
    print(f"   Weights keys: {weights.keys()}")
    for k, v in weights.items():
        print(f"   {k}: {v}")
    
    # Test regularization loss
    print("\n4. Testing regularization loss...")
    reg_loss = model.get_regularization_loss()
    print(f"   Regularization loss: {reg_loss.item():.6f}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    loss = output.sum() + reg_loss
    loss.backward()
    print("   Gradients computed successfully!")
    
    # Print model statistics
    print("\n6. Model statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n[OK] All tests passed!")

