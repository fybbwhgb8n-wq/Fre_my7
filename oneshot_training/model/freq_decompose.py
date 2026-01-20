"""
Frequency Stream Decomposition Module for Robust Capon Fusion

This module provides mechanisms to decompose feature maps into K frequency streams
for use in the Robust Capon Frequency Fusion (RCF) module.

Two main options are provided:
    1. K=2: Low/High frequency decomposition (fast baseline)
    2. K=4: Haar wavelet-like subbands (LL, LH, HL, HH)

The frequency streams serve as the "array elements" in the Capon beamforming
analogy, where each stream represents a different frequency band response.

Author: Implementation for IRSTD frequency fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LowHighDecompose(nn.Module):
    """
    K=2 Low/High frequency decomposition.
    
    Decomposes input feature map into:
        - F_low: Low-frequency component (smoothed)
        - F_high: High-frequency component (F - F_low)
    
    This is a fast baseline that captures the basic frequency separation
    relevant for small target detection.
    
    Args:
        in_channels: Number of input channels
        blur_kernel_size: Kernel size for low-pass filter (default: 3)
        learnable: Whether to learn the low-pass filter weights
    """
    
    def __init__(
        self,
        in_channels: int,
        blur_kernel_size: int = 3,
        learnable: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.K = 2  # Number of frequency streams
        
        # Create Gaussian-like low-pass filter
        if blur_kernel_size == 3:
            kernel = torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.]
            ]) / 16.0
        elif blur_kernel_size == 5:
            kernel = torch.tensor([
                [1., 4., 6., 4., 1.],
                [4., 16., 24., 16., 4.],
                [6., 24., 36., 24., 6.],
                [4., 16., 24., 16., 4.],
                [1., 4., 6., 4., 1.]
            ]) / 256.0
        else:
            # General Gaussian approximation
            sigma = blur_kernel_size / 6.0
            coords = torch.arange(blur_kernel_size).float() - blur_kernel_size // 2
            g1d = torch.exp(-coords**2 / (2 * sigma**2))
            kernel = g1d.unsqueeze(0) * g1d.unsqueeze(1)
            kernel = kernel / kernel.sum()
        
        # Expand kernel for depthwise convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
        kernel = kernel.expand(in_channels, 1, -1, -1)  # [C, 1, k, k]
        
        if learnable:
            self.register_parameter('blur_kernel', nn.Parameter(kernel.clone()))
        else:
            self.register_buffer('blur_kernel', kernel)
        
        self.padding = blur_kernel_size // 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose input into low and high frequency streams.
        
        Args:
            x: [B, C, H, W] input feature map
            
        Returns:
            X_streams: [B, K=2, C, H, W] frequency streams
                       X_streams[:, 0] = low frequency
                       X_streams[:, 1] = high frequency
        """
        B, C, H, W = x.shape
        
        # Low-pass filtering (depthwise)
        x_low = F.conv2d(x, self.blur_kernel, padding=self.padding, groups=C)
        
        # High-pass = residual
        x_high = x - x_low
        
        # Stack as frequency streams [B, K, C, H, W]
        x_streams = torch.stack([x_low, x_high], dim=1)
        
        return x_streams


class HaarWaveletDecompose(nn.Module):
    """
    K=4 Haar wavelet-like subband decomposition.
    
    Decomposes input into 4 subbands:
        - LL: Low-Low (approximation)
        - LH: Low-High (horizontal detail)
        - HL: High-Low (vertical detail)
        - HH: High-High (diagonal detail)
    
    This provides richer frequency information that can capture
    directional features relevant for target detection.
    
    Args:
        in_channels: Number of input channels
        upsample_mode: Interpolation mode for upsampling subbands
        learnable: Whether to learn the wavelet filters
    """
    
    def __init__(
        self,
        in_channels: int,
        upsample_mode: str = 'bilinear',
        learnable: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.K = 4  # Number of frequency streams
        self.upsample_mode = upsample_mode
        
        # Haar wavelet filters (2x2)
        # LL (approximation): [1,1; 1,1] / 2
        # LH (horizontal): [-1,-1; 1,1] / 2
        # HL (vertical): [-1,1; -1,1] / 2  
        # HH (diagonal): [1,-1; -1,1] / 2
        
        haar_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        haar_lh = torch.tensor([[-1., -1.], [1., 1.]]) / 2.0
        haar_hl = torch.tensor([[-1., 1.], [-1., 1.]]) / 2.0
        haar_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        
        # Stack filters: [4, 1, 2, 2]
        filters = torch.stack([haar_ll, haar_lh, haar_hl, haar_hh], dim=0).unsqueeze(1)
        
        # Expand for grouped convolution: [4*C, 1, 2, 2]
        filters = filters.repeat(in_channels, 1, 1, 1)
        
        if learnable:
            self.register_parameter('haar_filters', nn.Parameter(filters.clone()))
        else:
            self.register_buffer('haar_filters', filters)
        
        # Optional: 1x1 conv to compress channels after upsampling
        self.compress = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose input into 4 Haar wavelet subbands.
        
        Args:
            x: [B, C, H, W] input feature map
            
        Returns:
            X_streams: [B, K=4, C, H, W] frequency streams
                       Order: [LL, LH, HL, HH]
        """
        B, C, H, W = x.shape
        
        # Pad if needed for even dimensions
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_pad, W_pad = x.shape[2], x.shape[3]
        
        # Apply Haar filters with stride 2 (grouped conv)
        # Output: [B, 4*C, H/2, W/2]
        subbands = F.conv2d(x, self.haar_filters, stride=2, groups=C)
        
        # Reshape to [B, C, 4, H/2, W/2]
        subbands = subbands.view(B, C, 4, H_pad // 2, W_pad // 2)
        
        # Permute to [B, 4, C, H/2, W/2]
        subbands = subbands.permute(0, 2, 1, 3, 4)
        
        # Upsample each subband to original resolution
        streams = []
        for k in range(4):
            subband_k = subbands[:, k]  # [B, C, H/2, W/2]
            
            # Upsample to original size
            subband_k = F.interpolate(
                subband_k, 
                size=(H, W), 
                mode=self.upsample_mode, 
                align_corners=False if self.upsample_mode != 'nearest' else None
            )
            
            # Optional: apply channel compression
            subband_k = self.compress(subband_k)
            streams.append(subband_k)
        
        # Stack to [B, K=4, C, H, W]
        x_streams = torch.stack(streams, dim=1)
        
        return x_streams


class LearnableFrequencyDecompose(nn.Module):
    """
    Learnable frequency decomposition with K streams.
    
    Uses learnable convolutional filters to decompose the input
    into K frequency-like streams. This is more flexible than
    fixed Haar or low/high decomposition.
    
    Args:
        in_channels: Number of input channels
        K: Number of frequency streams (2 or 4 recommended)
        kernel_size: Kernel size for decomposition filters
    """
    
    def __init__(
        self,
        in_channels: int,
        K: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.K = K
        
        padding = kernel_size // 2
        
        # K separate filter banks, each producing C channels
        self.filter_banks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding, 
                         groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
            )
            for _ in range(K)
        ])
        
        # Initialize first filter as approximate low-pass, rest as band-pass/high-pass
        self._init_filters()
    
    def _init_filters(self):
        """Initialize filters with frequency-inspired patterns."""
        with torch.no_grad():
            for k, bank in enumerate(self.filter_banks):
                conv = bank[0]  # Get the Conv2d layer
                weight = conv.weight.data  # [C, 1, kH, kW]
                
                if k == 0:
                    # Low-pass: Gaussian-like
                    kH, kW = weight.shape[2], weight.shape[3]
                    center = kH // 2
                    for i in range(kH):
                        for j in range(kW):
                            weight[:, 0, i, j] = 0.1 * torch.exp(
                                -((i - center)**2 + (j - center)**2) / 2.0
                            )
                    weight[:, 0, center, center] = 1.0
                    weight /= weight.sum(dim=(-1, -2), keepdim=True)
                elif k == self.K - 1:
                    # High-pass: Laplacian-like (identity - low-pass)
                    kH, kW = weight.shape[2], weight.shape[3]
                    center = kH // 2
                    weight.zero_()
                    weight[:, 0, center, center] = 1.0
                    # Subtract neighbors
                    if center > 0:
                        weight[:, 0, center-1, center] = -0.25
                        weight[:, 0, center+1, center] = -0.25
                        weight[:, 0, center, center-1] = -0.25
                        weight[:, 0, center, center+1] = -0.25
                        weight[:, 0, center, center] = 1.0
                # Middle bands: random initialization is fine (will be learned)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose input into K learnable frequency streams.
        
        Args:
            x: [B, C, H, W] input feature map
            
        Returns:
            X_streams: [B, K, C, H, W] frequency streams
        """
        streams = []
        for k in range(self.K):
            stream_k = self.filter_banks[k](x)
            streams.append(stream_k)
        
        x_streams = torch.stack(streams, dim=1)
        return x_streams


class FrequencyDecomposer(nn.Module):
    """
    Unified frequency decomposition module.
    
    Provides a consistent interface for different decomposition methods:
        - 'low_high' (K=2): Simple low/high frequency split
        - 'haar' (K=4): Haar wavelet subbands
        - 'learnable': Learnable filter banks
    
    Args:
        in_channels: Number of input channels
        method: Decomposition method ('low_high', 'haar', 'learnable')
        K: Number of streams (only for 'learnable' method)
        learnable: Whether filters should be learnable
    """
    
    def __init__(
        self,
        in_channels: int,
        method: str = 'low_high',
        K: int = 2,
        learnable: bool = False,
    ):
        super().__init__()
        self.method = method
        
        if method == 'low_high':
            self.decomposer = LowHighDecompose(in_channels, learnable=learnable)
            self.K = 2
        elif method == 'haar':
            self.decomposer = HaarWaveletDecompose(in_channels, learnable=learnable)
            self.K = 4
        elif method == 'learnable':
            self.decomposer = LearnableFrequencyDecompose(in_channels, K=K)
            self.K = K
        else:
            raise ValueError(f"Unknown decomposition method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose input into K frequency streams.
        
        Args:
            x: [B, C, H, W] input feature map
            
        Returns:
            X_streams: [B, K, C, H, W] frequency streams
        """
        return self.decomposer(x)


if __name__ == "__main__":
    # Unit tests
    print("Testing frequency decomposition modules...")
    
    B, C, H, W = 2, 32, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(B, C, H, W, device=device)
    
    # Test Low/High decomposition
    print("\n1. Testing LowHighDecompose...")
    decomp_lh = LowHighDecompose(C).to(device)
    out_lh = decomp_lh(x)
    print(f"   Input: {x.shape} -> Output: {out_lh.shape}")
    assert out_lh.shape == (B, 2, C, H, W), "LowHighDecompose shape mismatch"
    
    # Test Haar wavelet decomposition
    print("\n2. Testing HaarWaveletDecompose...")
    decomp_haar = HaarWaveletDecompose(C).to(device)
    out_haar = decomp_haar(x)
    print(f"   Input: {x.shape} -> Output: {out_haar.shape}")
    assert out_haar.shape == (B, 4, C, H, W), "HaarWaveletDecompose shape mismatch"
    
    # Test learnable decomposition
    print("\n3. Testing LearnableFrequencyDecompose...")
    decomp_learn = LearnableFrequencyDecompose(C, K=3).to(device)
    out_learn = decomp_learn(x)
    print(f"   Input: {x.shape} -> Output: {out_learn.shape}")
    assert out_learn.shape == (B, 3, C, H, W), "LearnableFrequencyDecompose shape mismatch"
    
    # Test unified interface
    print("\n4. Testing FrequencyDecomposer (unified)...")
    for method in ['low_high', 'haar', 'learnable']:
        decomp = FrequencyDecomposer(C, method=method, K=4).to(device)
        out = decomp(x)
        print(f"   Method '{method}': K={decomp.K}, output shape={out.shape}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    decomp_grad = FrequencyDecomposer(C, method='learnable', K=2).to(device)
    out_grad = decomp_grad(x)
    loss = out_grad.sum()
    loss.backward()
    print("   Gradients computed successfully!")
    
    print("\n[OK] All tests passed!")

