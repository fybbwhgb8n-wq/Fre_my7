"""
Robust Capon Frequency Fusion Module

This module implements the complete Robust Capon Frequency Fusion pipeline:
    1. Frequency Decomposition -> K frequency streams
    2. Sensor Map Generation -> scalar maps for covariance computation
    3. Covariance Estimation -> R(p) from local neighborhood patches
    4. Guidance Vector Generation -> a0(p) nominal steering vector
    5. Uncertainty Generation -> sigma(p) for robustness envelope
    6. Robust SOCP Solver -> compute weights w(p) = argmin w^T R w s.t. ||A^T w|| <= a0^T w - 1
    7. Weighted Fusion -> Y = sum_k w_k * X_k

This is the TRUE implementation of Scheme 3 (Robust MVDR/SOCP) as specified.

Key mathematical formulation:
    For each pixel p, we solve:
        min_w   w^T R(p) w           (minimize output power / variance)
        s.t.    ||A(p)^T w||_2 <= a0(p)^T w - 1    (robust distortionless constraint)
    
    where:
        R(p): Sample covariance matrix estimated from local patch around p
        a0(p): Nominal guidance/steering vector
        A(p) = diag(sigma(p)): Uncertainty matrix (diagonal)

Author: Implementation for IRSTD robust frequency fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union, Literal

try:
    from model.robust_capon_solver import RobustCaponSolver, RobustCaponSolverWithSlack
except ImportError:
    from robust_capon_solver import RobustCaponSolver, RobustCaponSolverWithSlack


class SensorMapGenerator(nn.Module):
    """
    Generate scalar sensor maps from multi-channel frequency streams.
    
    Projects [B, K, C, H, W] frequency features to [B, K, H, W] scalar maps
    for covariance computation.
    
    Args:
        in_channels: Number of channels per frequency stream
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.sensor_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1, bias=False),
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Frequency streams [B, K, C, H, W]
        Returns:
            S: Scalar sensor maps [B, K, H, W]
        """
        B, K, C, H, W = X.shape
        X_flat = X.view(B * K, C, H, W)
        S_flat = self.sensor_proj(X_flat)  # [B*K, 1, H, W]
        S = S_flat.view(B, K, H, W)
        return S


class CovarianceEstimator(nn.Module):
    """
    Estimate sample covariance matrix R(p) from local neighborhoods.
    
    For each pixel p, extracts a patch of sensor values and computes:
        R(p) = (1/N) * Y_patch @ Y_patch^T
    
    where Y_patch: [K, N] contains N neighborhood samples from K streams.
    
    Args:
        K: Number of frequency streams
        patch_size: Size of local neighborhood (odd number)
    """
    
    def __init__(self, K: int, patch_size: int = 3):
        super().__init__()
        self.K = K
        self.patch_size = patch_size
        self.padding = patch_size // 2
        
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel covariance matrices from sensor maps.
        
        Args:
            S: Scalar sensor maps [B, K, H, W]
            
        Returns:
            R: Covariance matrices [B, H, W, K, K]
        """
        B, K, H, W = S.shape
        device = S.device
        dtype = S.dtype
        
        # Use unfold to extract local patches
        # First, pad the input
        S_padded = F.pad(S, [self.padding] * 4, mode='reflect')
        
        # Unfold to get patches: [B, K, H, W, patch_size, patch_size]
        patches = S_padded.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        
        # Reshape to [B, K, H, W, N] where N = patch_size^2
        N = self.patch_size ** 2
        patches = patches.reshape(B, K, H, W, N)
        
        # Transpose to [B, H, W, K, N]
        patches = patches.permute(0, 2, 3, 1, 4)
        
        # Center the patches (zero-mean for covariance)
        patches_centered = patches - patches.mean(dim=-1, keepdim=True)
        
        # Compute covariance: R = (1/N) * Y @ Y^T
        # [B, H, W, K, N] @ [B, H, W, N, K] -> [B, H, W, K, K]
        R = torch.matmul(patches_centered, patches_centered.transpose(-1, -2)) / N
        
        return R


class GuidanceVectorHead(nn.Module):
    """
    Generate guidance (steering) vector a0.
    
    Supports multiple modes:
        - 'fixed': Uniform guidance a0 = [1/K, ..., 1/K]
        - 'stage': Learnable per-stage guidance (shared across all pixels)
        - 'pixel': Pixel-wise adaptive guidance via conv network
    
    Args:
        K: Number of frequency streams
        in_channels: Input channels for pixel mode
        mode: 'fixed', 'stage', or 'pixel'
    """
    
    def __init__(
        self,
        K: int,
        in_channels: int = 0,
        mode: Literal['fixed', 'stage', 'pixel'] = 'stage',
    ):
        super().__init__()
        self.K = K
        self.mode = mode
        
        if mode == 'fixed':
            # Fixed uniform guidance
            self.register_buffer('a0', torch.ones(K) / K)
        elif mode == 'stage':
            # Learnable per-stage guidance
            self.a0_raw = nn.Parameter(torch.zeros(K))
        elif mode == 'pixel':
            # Pixel-wise guidance via convolution
            assert in_channels > 0, "in_channels required for pixel mode"
            self.a0_conv = nn.Sequential(
                nn.Conv2d(in_channels * K, K * 2, 3, 1, 1, groups=K, bias=False),
                nn.BatchNorm2d(K * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(K * 2, K, 1, bias=False),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        B: int = 1,
        H: int = 1,
        W: int = 1,
    ) -> torch.Tensor:
        """
        Generate guidance vector.
        
        Args:
            X: Frequency streams [B, K, C, H, W] (required for pixel mode)
            B, H, W: Batch and spatial dimensions
            
        Returns:
            a0: Guidance vectors, normalized to sum to 1
                - fixed/stage: [K] or [B, K]
                - pixel: [B, H, W, K]
        """
        if self.mode == 'fixed':
            return self.a0
        
        elif self.mode == 'stage':
            # Apply softmax to ensure non-negative and sum to 1
            return F.softmax(self.a0_raw, dim=0)
        
        elif self.mode == 'pixel':
            assert X is not None, "X required for pixel mode"
            B, K, C, H, W = X.shape
            
            # Flatten frequency streams: [B, K*C, H, W]
            X_flat = X.view(B, K * C, H, W)
            
            # Generate raw a0: [B, K, H, W]
            a0_raw = self.a0_conv(X_flat)
            
            # Apply softmax across K dimension: [B, K, H, W]
            a0 = F.softmax(a0_raw, dim=1)
            
            # Reshape to [B, H, W, K]
            a0 = a0.permute(0, 2, 3, 1)
            
            return a0


class UncertaintyHead(nn.Module):
    """
    Generate uncertainty sigma for robustness envelope.
    
    sigma defines the diagonal uncertainty matrix A = diag(sigma).
    Larger sigma means more robustness (but less selectivity).
    
    Supports modes:
        - 'fixed': Fixed uniform uncertainty
        - 'stage': Learnable per-stage uncertainty
        - 'pixel': Pixel-wise adaptive uncertainty
    
    Args:
        K: Number of frequency streams
        in_channels: Input channels for pixel mode
        mode: 'fixed', 'stage', or 'pixel'
        sigma_max: Maximum uncertainty value (default: 0.3)
    """
    
    def __init__(
        self,
        K: int,
        in_channels: int = 0,
        mode: Literal['fixed', 'stage', 'pixel'] = 'stage',
        sigma_max: float = 0.3,
    ):
        super().__init__()
        self.K = K
        self.mode = mode
        self.sigma_max = sigma_max
        
        if mode == 'fixed':
            # Fixed uniform uncertainty
            self.register_buffer('sigma', torch.ones(K) * sigma_max * 0.5)
        elif mode == 'stage':
            # Learnable per-stage uncertainty
            self.sigma_raw = nn.Parameter(torch.zeros(K))
        elif mode == 'pixel':
            # Pixel-wise uncertainty via convolution
            assert in_channels > 0, "in_channels required for pixel mode"
            self.sigma_conv = nn.Sequential(
                nn.Conv2d(in_channels * K, K * 2, 3, 1, 1, groups=K, bias=False),
                nn.BatchNorm2d(K * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(K * 2, K, 1, bias=False),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        B: int = 1,
        H: int = 1,
        W: int = 1,
    ) -> torch.Tensor:
        """
        Generate uncertainty values.
        
        Args:
            X: Frequency streams [B, K, C, H, W] (required for pixel mode)
            B, H, W: Batch and spatial dimensions
            
        Returns:
            sigma: Uncertainty values in [0, sigma_max]
                - fixed/stage: [K] or [B, K]
                - pixel: [B, H, W, K]
        """
        if self.mode == 'fixed':
            return self.sigma
        
        elif self.mode == 'stage':
            # Apply sigmoid and scale by sigma_max
            return self.sigma_max * torch.sigmoid(self.sigma_raw)
        
        elif self.mode == 'pixel':
            assert X is not None, "X required for pixel mode"
            B, K, C, H, W = X.shape
            
            # Flatten frequency streams: [B, K*C, H, W]
            X_flat = X.view(B, K * C, H, W)
            
            # Generate raw sigma: [B, K, H, W]
            sigma_raw = self.sigma_conv(X_flat)
            
            # Apply sigmoid and scale: [B, K, H, W]
            sigma = self.sigma_max * torch.sigmoid(sigma_raw)
            
            # Reshape to [B, H, W, K]
            sigma = sigma.permute(0, 2, 3, 1)
            
            return sigma


class RobustCaponFrequencyFusion(nn.Module):
    """
    Complete Robust Capon Frequency Fusion module.
    
    Pipeline:
        1. Generate sensor maps from frequency streams
        2. Estimate covariance R(p) from local patches
        3. Generate guidance vector a0(p)
        4. Generate uncertainty sigma(p)
        5. Solve robust SOCP: w(p) = argmin w^T R w s.t. ||A^T w|| <= a0^T w - 1
        6. Apply weighted fusion: Y = sum_k w_k * X_k
    
    This implements the TRUE Scheme 3 (Robust MVDR/SOCP) specification.
    
    Args:
        K: Number of frequency streams
        in_channels: Channels per frequency stream
        patch_size: Local neighborhood size for covariance estimation
        a0_mode: Guidance vector mode ('fixed', 'stage', 'pixel')
        sigma_mode: Uncertainty mode ('fixed', 'stage', 'pixel')
        sigma_max: Maximum uncertainty value
        gamma: Robustness loading parameter for SOCP solver
        cov_shrink: Covariance shrinkage coefficient
        cov_delta: Covariance diagonal loading
        use_double: Use float64 for SOCP solver
        use_slack: Use slack-based solver for extra stability
        slack_threshold: Margin threshold for slack solver
    """
    
    def __init__(
        self,
        K: int,
        in_channels: int,
        patch_size: int = 3,
        a0_mode: Literal['fixed', 'stage', 'pixel'] = 'stage',
        sigma_mode: Literal['fixed', 'stage', 'pixel'] = 'stage',
        sigma_max: float = 0.3,
        gamma: float = 1.0,
        cov_shrink: float = 0.1,
        cov_delta: float = 1e-3,
        use_double: bool = True,
        use_slack: bool = True,
        slack_threshold: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        # Sensor map generator: [B, K, C, H, W] -> [B, K, H, W]
        self.sensor_gen = SensorMapGenerator(in_channels)
        
        # Covariance estimator: [B, K, H, W] -> [B, H, W, K, K]
        self.cov_estimator = CovarianceEstimator(K, patch_size)
        
        # Guidance vector head: -> a0 [B, H, W, K] or [K]
        self.a0_head = GuidanceVectorHead(K, in_channels, a0_mode)
        
        # Uncertainty head: -> sigma [B, H, W, K] or [K]
        self.sigma_head = UncertaintyHead(K, in_channels, sigma_mode, sigma_max)
        
        # Robust SOCP solver
        solver_kwargs = dict(
            K=K,
            gamma=gamma,
            sigma_max=sigma_max,
            cov_shrink=cov_shrink,
            cov_delta=cov_delta,
            use_double=use_double,
        )
        if use_slack:
            self.solver = RobustCaponSolverWithSlack(
                slack_threshold=slack_threshold,
                **solver_kwargs
            )
        else:
            self.solver = RobustCaponSolver(**solver_kwargs)
    
    def forward(
        self,
        X: torch.Tensor,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply robust Capon frequency fusion.
        
        Args:
            X: Frequency streams [B, K, C, H, W]
            return_debug: Whether to return debug information
            
        Returns:
            Y: Fused features [B, C, H, W]
            (optional) debug_dict: Debug information including solver stats
        """
        B, K, C, H, W = X.shape
        
        # Step 1: Generate sensor maps
        S = self.sensor_gen(X)  # [B, K, H, W]
        
        # Step 2: Estimate covariance matrices
        R = self.cov_estimator(S)  # [B, H, W, K, K]
        
        # Step 3: Generate guidance vector
        a0 = self.a0_head(X, B, H, W)  # [K] or [B, K] or [B, H, W, K]
        
        # Step 4: Generate uncertainty
        sigma = self.sigma_head(X, B, H, W)  # [K] or [B, K] or [B, H, W, K]
        
        # Step 5: Solve robust SOCP for weights
        w, margin, solver_debug = self.solver(R, a0, sigma)  # [B, H, W, K]
        
        # Step 6: Apply weighted fusion
        # w: [B, H, W, K] -> [B, K, H, W] for broadcasting with X
        w_broadcast = w.permute(0, 3, 1, 2)  # [B, K, H, W]
        w_broadcast = w_broadcast.unsqueeze(2)  # [B, K, 1, H, W]
        
        # X: [B, K, C, H, W], w: [B, K, 1, H, W]
        Y = (w_broadcast * X).sum(dim=1)  # [B, C, H, W]
        
        if return_debug:
            debug_dict = {
                'solver': solver_debug,
                'w_min': w.min().item(),
                'w_max': w.max().item(),
                'w_mean': w.mean().item(),
                'w_std': w.std().item(),
                'R_trace_mean': torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1).mean().item(),
            }
            return Y, debug_dict
        
        return Y
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for training stability.
        
        Encourages:
            - Entropy maximization for diverse weights
            - Moderate uncertainty values
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Entropy regularization for stage-mode guidance
        if hasattr(self.a0_head, 'a0_raw'):
            a0 = F.softmax(self.a0_head.a0_raw, dim=0)
            entropy = -(a0 * torch.log(a0 + 1e-8)).sum()
            reg_loss = reg_loss - 0.01 * entropy  # Maximize entropy
        
        return reg_loss


class RobustCaponFusionBlock(nn.Module):
    """
    Complete Robust Capon Fusion block with frequency decomposition.
    
    Integrates:
        - FrequencyDecomposer: Split input into K frequency streams
        - RobustCaponFrequencyFusion: Fuse streams using robust SOCP
        - Output projection with residual connection
    
    Args:
        in_channels: Input feature channels
        out_channels: Output feature channels
        K: Number of frequency streams
        decompose_method: Frequency decomposition method ('low_high', 'haar', etc.)
        **fusion_kwargs: Arguments for RobustCaponFrequencyFusion
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,
        decompose_method: str = 'low_high',
        **fusion_kwargs
    ):
        super().__init__()
        
        try:
            from model.freq_decompose import FrequencyDecomposer
        except ImportError:
            from freq_decompose import FrequencyDecomposer
        
        self.K = K
        
        # Input projection if channels differ
        self.in_proj = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # Frequency decomposer
        self.decomposer = FrequencyDecomposer(
            in_channels=out_channels,
            method=decompose_method,
            K=K,
        )
        self.K = self.decomposer.K  # May be updated by decomposer
        
        # Robust Capon fusion (TRUE SOCP SOLVER)
        self.fusion = RobustCaponFrequencyFusion(
            K=self.K,
            in_channels=out_channels,
            **fusion_kwargs
        )
        
        # Output projection with residual
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply robust Capon fusion block.
        
        Args:
            x: Input features [B, C, H, W]
            return_debug: Whether to return debug info
            
        Returns:
            y: Output features [B, C, H, W]
            (optional) debug_dict: Debug information
        """
        # Input projection
        x = self.in_proj(x)
        
        # Decompose into frequency streams
        x_streams = self.decomposer(x)  # [B, K, C, H, W]
        
        # Apply robust Capon fusion
        if return_debug:
            y_fused, debug_dict = self.fusion(x_streams, return_debug=True)
        else:
            y_fused = self.fusion(x_streams)
        
        # Output projection with residual
        y = self.out_proj(y_fused) + x
        
        if return_debug:
            return y, debug_dict
        return y
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from fusion module."""
        return self.fusion.get_regularization_loss()


if __name__ == "__main__":
    """Unit tests for robust Capon frequency fusion."""
    print("Testing Robust Capon Frequency Fusion...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    B, K, C, H, W = 2, 2, 32, 64, 64
    
    # Test 1: SensorMapGenerator
    print("\n1. Testing SensorMapGenerator...")
    X = torch.randn(B, K, C, H, W, device=device)
    sensor_gen = SensorMapGenerator(C).to(device)
    S = sensor_gen(X)
    print(f"   Input: {X.shape} -> Sensor maps: {S.shape}")
    
    # Test 2: CovarianceEstimator
    print("\n2. Testing CovarianceEstimator...")
    cov_est = CovarianceEstimator(K, patch_size=3).to(device)
    R = cov_est(S)
    print(f"   Sensor maps: {S.shape} -> Covariance: {R.shape}")
    print(f"   R is symmetric: {torch.allclose(R, R.transpose(-1, -2), atol=1e-5)}")
    
    # Test 3: GuidanceVectorHead
    print("\n3. Testing GuidanceVectorHead...")
    for mode in ['fixed', 'stage', 'pixel']:
        a0_head = GuidanceVectorHead(K, C, mode).to(device)
        a0 = a0_head(X, B, H, W)
        print(f"   Mode '{mode}': a0 shape = {a0.shape}")
    
    # Test 4: UncertaintyHead
    print("\n4. Testing UncertaintyHead...")
    for mode in ['fixed', 'stage', 'pixel']:
        sigma_head = UncertaintyHead(K, C, mode, sigma_max=0.3).to(device)
        sigma = sigma_head(X, B, H, W)
        print(f"   Mode '{mode}': sigma shape = {sigma.shape}")
    
    # Test 5: RobustCaponFrequencyFusion
    print("\n5. Testing RobustCaponFrequencyFusion...")
    fusion = RobustCaponFrequencyFusion(
        K=K,
        in_channels=C,
        patch_size=3,
        a0_mode='stage',
        sigma_mode='stage',
        sigma_max=0.3,
        gamma=1.0,
    ).to(device)
    
    Y, debug = fusion(X, return_debug=True)
    print(f"   Input: {X.shape} -> Fused: {Y.shape}")
    print(f"   Debug: {debug}")
    print(f"   Has NaN: {torch.isnan(Y).any().item()}")
    
    # Test 6: RobustCaponFusionBlock
    print("\n6. Testing RobustCaponFusionBlock...")
    x_input = torch.randn(B, C, H, W, device=device)
    fusion_block = RobustCaponFusionBlock(
        in_channels=C,
        out_channels=C,
        K=2,
        decompose_method='low_high',
        a0_mode='stage',
        sigma_mode='stage',
    ).to(device)
    
    y_out, debug_block = fusion_block(x_input, return_debug=True)
    print(f"   Input: {x_input.shape} -> Output: {y_out.shape}")
    print(f"   Debug: {debug_block}")
    
    # Test 7: Gradient flow
    print("\n7. Testing gradient flow...")
    y_test, _ = fusion_block(x_input.requires_grad_(True), return_debug=True)
    loss = y_test.sum()
    loss.backward()
    print("   Gradients computed successfully!")
    
    # Test 8: Regularization loss
    print("\n8. Testing regularization loss...")
    reg_loss = fusion_block.get_regularization_loss()
    print(f"   Regularization loss: {reg_loss.item():.6f}")
    
    print("\n[OK] All tests passed!")

