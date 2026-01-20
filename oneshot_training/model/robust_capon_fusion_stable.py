"""
Stable Robust Capon Frequency Fusion Module

This is a simplified, numerically stable version of the Robust Capon Fusion.
It uses a simplified attention mechanism instead of the full one-shot solver
to avoid numerical instability issues.

Key simplifications:
    1. Use simple attention weights instead of solving SOCP
    2. Learnable per-band weights with softmax normalization
    3. Optional local context-aware weighting

Author: Stable implementation for IRSTD frequency fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Literal, Union


class SimpleSensorMap(nn.Module):
    """Generate scalar sensor maps from frequency streams."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.sensor_proj = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, K, C, H, W] frequency streams
        Returns:
            S: [B, K, H, W] scalar sensor maps
        """
        B, K, C, H, W = X.shape
        X_flat = X.view(B * K, C, H, W)
        S_flat = self.sensor_proj(X_flat)
        S = S_flat.view(B, K, H, W)
        return S


class StableFrequencyFusion(nn.Module):
    """
    Stable Frequency Fusion with learnable attention weights.
    
    Instead of solving a complex optimization problem, this module uses:
    1. Learnable per-band base weights
    2. Local context modulation (optional)
    3. Softmax normalization for stable weights
    
    Args:
        K: Number of frequency bands
        in_channels: Channels per band
        use_local_context: Whether to use local context for weight modulation
        temperature: Softmax temperature (lower = sharper)
    """
    
    def __init__(
        self,
        K: int,
        in_channels: int,
        use_local_context: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.use_local_context = use_local_context
        self.temperature = temperature
        
        # Learnable base weights for each frequency band
        self.base_weights = nn.Parameter(torch.zeros(K))
        
        # Sensor map generator
        self.sensor_gen = SimpleSensorMap(in_channels)
        
        # Local context modulation
        if use_local_context:
            self.context_conv = nn.Sequential(
                nn.Conv2d(K, K * 2, 3, 1, 1, groups=K, bias=False),
                nn.BatchNorm2d(K * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(K * 2, K, 1, 1, bias=False),
            )
    
    def forward(
        self,
        X: torch.Tensor,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Fuse frequency streams with learned attention.
        
        Args:
            X: [B, K, C, H, W] frequency streams
            return_debug: Whether to return debug info
            
        Returns:
            Y: [B, C, H, W] fused features
            (optional) debug_dict
        """
        B, K, C, H, W = X.shape
        
        # Get sensor maps for context
        S = self.sensor_gen(X)  # [B, K, H, W]
        
        # Compute attention weights
        # Start with learned base weights
        weights = self.base_weights.view(1, K, 1, 1).expand(B, K, H, W)
        
        # Add local context modulation
        if self.use_local_context:
            context = self.context_conv(S)  # [B, K, H, W]
            weights = weights + context
        
        # Apply temperature and softmax for stable weights
        weights = F.softmax(weights / self.temperature, dim=1)  # [B, K, H, W]
        
        # Weighted fusion
        # X: [B, K, C, H, W], weights: [B, K, H, W]
        Y = (weights.unsqueeze(2) * X).sum(dim=1)  # [B, C, H, W]
        
        if return_debug:
            debug_dict = {
                "w_mean": weights.mean().item(),
                "w_std": weights.std().item(),
                "w_max": weights.max().item(),
                "w_min": weights.min().item(),
            }
            return Y, debug_dict
        
        return Y
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Regularization to prevent weight collapse."""
        # Encourage diverse weights
        weights = F.softmax(self.base_weights, dim=0)
        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        # Maximize entropy (return negative)
        return -0.01 * entropy


class RobustCaponFusionStable(nn.Module):
    """
    Stable Robust Capon Frequency Fusion module.
    
    Uses simplified attention mechanism for maximum numerical stability
    while maintaining the core idea of adaptive frequency fusion.
    
    Args:
        K: Number of frequency bands
        in_channels: Channels per band
        use_local_context: Whether to use local context
        temperature: Softmax temperature
    """
    
    def __init__(
        self,
        K: int,
        in_channels: int,
        kernel_size: int = 3,  # Kept for API compatibility
        use_local_context: bool = True,
        temperature: float = 1.0,
        # Legacy parameters (ignored)
        a0_mode: str = 'stage',
        uncert_mode: str = 'diag',
        sigma_max: float = 1.0,
        gamma: float = 1.0,
        cov_shrink: float = 0.1,
        cov_delta: float = 1e-3,
        use_double: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        
        self.fusion = StableFrequencyFusion(
            K=K,
            in_channels=in_channels,
            use_local_context=use_local_context,
            temperature=temperature,
        )
    
    def forward(
        self,
        X: torch.Tensor,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply stable frequency fusion.
        
        Args:
            X: [B, K, C, H, W] frequency streams
            return_debug: Whether to return debug info
            
        Returns:
            Y: [B, C, H, W] fused features
        """
        return self.fusion(X, return_debug=return_debug)
    
    def get_regularization_loss(self) -> torch.Tensor:
        return self.fusion.get_regularization_loss()


class RobustCaponFusionBlockStable(nn.Module):
    """
    Stable fusion block with frequency decomposition.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,
        decompose_method: str = 'low_high',
        **fusion_kwargs,
    ):
        super().__init__()
        
        from model.freq_decompose import FrequencyDecomposer
        
        self.K = K
        
        self.in_proj = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        
        self.decomposer = FrequencyDecomposer(
            in_channels=out_channels,
            method=decompose_method,
            K=K,
        )
        self.K = self.decomposer.K
        
        self.fusion = RobustCaponFusionStable(
            K=self.K,
            in_channels=out_channels,
            **fusion_kwargs,
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        x = self.in_proj(x)
        x_streams = self.decomposer(x)
        
        if return_debug:
            y_fused, debug_dict = self.fusion(x_streams, return_debug=True)
        else:
            y_fused = self.fusion(x_streams)
        
        y = self.out_proj(y_fused) + x
        
        if return_debug:
            return y, debug_dict
        return y


if __name__ == "__main__":
    print("Testing Stable Robust Capon Fusion...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, K, C, H, W = 2, 2, 32, 64, 64
    
    X = torch.randn(B, K, C, H, W, device=device)
    
    # Test StableFrequencyFusion
    print("\n1. Testing StableFrequencyFusion...")
    fusion = StableFrequencyFusion(K=K, in_channels=C).to(device)
    Y, debug = fusion(X, return_debug=True)
    print(f"   Input: {X.shape}")
    print(f"   Output: {Y.shape}")
    print(f"   Has NaN: {torch.isnan(Y).any()}")
    print(f"   Debug: {debug}")
    
    # Test gradient flow
    print("\n2. Testing gradient flow...")
    loss = Y.sum()
    loss.backward()
    print("   Gradients computed successfully!")
    
    # Test RobustCaponFusionBlockStable
    print("\n3. Testing RobustCaponFusionBlockStable...")
    x_input = torch.randn(B, C, H, W, device=device)
    fusion_block = RobustCaponFusionBlockStable(
        in_channels=C,
        out_channels=C,
        K=2,
        decompose_method='low_high',
    ).to(device)
    y_out, debug_block = fusion_block(x_input, return_debug=True)
    print(f"   Input: {x_input.shape} -> Output: {y_out.shape}")
    print(f"   Debug: {debug_block}")
    
    print("\n[OK] All tests passed!")

