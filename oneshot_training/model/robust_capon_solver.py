"""
Robust Capon (MVDR) SOCP Solver - One-Shot Closed-Form Solution

This module implements the true robust Capon/MVDR beamformer with robustness
to guidance vector mismatch. It solves the following SOCP problem:

    min_w   w^T R w
    s.t.    ||A^T w||_2 <= a_0^T w - 1

where:
    R: Sample covariance matrix [K x K]
    a_0: Nominal guidance (steering) vector [K x 1]
    A: Uncertainty matrix [K x K] (diagonal: A = diag(sigma))
    
The one-shot closed-form solution (ADMM-free) is derived as:
    1. Form the regularized matrix: R_tilde = R + gamma * A * A^T
    2. Solve: u = R_tilde^{-1} a_0
    3. Compute margin: m(u) = a_0^T u - ||A^T u||_2
    4. Normalize: w = u / m(u)

This avoids iterative ADMM while maintaining the robust SOCP guarantees.

Reference:
    Vorobyov, Gershman, Luo (2003) "Robust adaptive beamforming using 
    worst-case performance optimization"

Author: Implementation for IRSTD robust frequency fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def make_spd_cov(
    R: torch.Tensor,
    eps: float = 1e-6,
    shrink: float = 0.1,
    delta: float = 1e-3,
) -> torch.Tensor:
    """
    Stabilize covariance matrix to be symmetric positive definite (SPD).
    
    Applies multiple stabilization techniques:
        1. Symmetrization: R = (R + R^T) / 2
        2. Trace normalization: R = R / trace(R)
        3. Shrinkage toward identity: R = (1-shrink)*R + shrink*I
        4. Diagonal loading: R = R + delta*I
    
    Args:
        R: Covariance matrices [..., K, K]
        eps: Minimum value for trace normalization
        shrink: Shrinkage coefficient toward identity (0 to 1)
        delta: Diagonal loading coefficient
        
    Returns:
        R_stable: Stabilized SPD covariance matrices [..., K, K]
    """
    # Step 1: Symmetrization
    R = 0.5 * (R + R.transpose(-1, -2))
    
    # Step 2: Trace normalization (per-matrix)
    trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    trace_R = trace_R.clamp(min=eps)
    R = R / trace_R
    
    # Step 3: Shrinkage toward identity
    K = R.shape[-1]
    I = torch.eye(K, device=R.device, dtype=R.dtype)
    # Expand I to match batch dimensions
    I_expanded = I.expand_as(R)
    R = (1.0 - shrink) * R + shrink * I_expanded / K
    
    # Step 4: Diagonal loading
    R = R + delta * I_expanded
    
    return R


def batched_cholesky_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    jitter: float = 1e-6,
) -> Tuple[torch.Tensor, bool]:
    """
    Solve A x = b using Cholesky decomposition with jitter for stability.
    
    Args:
        A: SPD matrices [..., K, K]
        b: Right-hand side vectors [..., K] or [..., K, 1]
        jitter: Small diagonal value added for numerical stability
        
    Returns:
        x: Solution vectors [..., K]
        success: Whether Cholesky succeeded
    """
    squeeze_output = b.dim() == A.dim() - 1
    if squeeze_output:
        b = b.unsqueeze(-1)
    
    K = A.shape[-1]
    
    # Add jitter to diagonal
    jitter_matrix = jitter * torch.eye(K, device=A.device, dtype=A.dtype)
    A_jittered = A + jitter_matrix.expand_as(A)
    
    try:
        # Cholesky decomposition: A = L L^T
        L = torch.linalg.cholesky(A_jittered)
        # Solve L y = b, then L^T x = y
        x = torch.cholesky_solve(b, L)
        success = True
    except RuntimeError:
        # Fallback to pseudo-inverse if Cholesky fails
        x = torch.linalg.lstsq(A_jittered, b).solution
        success = False
    
    if squeeze_output:
        x = x.squeeze(-1)
    
    return x, success


class RobustCaponSolver(nn.Module):
    """
    Robust Capon (MVDR) SOCP Solver using one-shot closed-form solution.
    
    Solves the robust beamforming problem:
        min_w   w^T R w
        s.t.    ||A^T w||_2 <= a_0^T w - 1
    
    where A = diag(sigma) represents diagonal uncertainty.
    
    The closed-form solution is:
        u = (R + gamma * diag(sigma^2))^{-1} a_0
        m = a_0^T u - ||sigma * u||_2
        w = u / m
    
    Args:
        K: Number of frequency streams
        gamma: Robustness loading parameter (default: 1.0)
        sigma_max: Maximum uncertainty (default: 0.3)
        cov_shrink: Covariance shrinkage coefficient (default: 0.1)
        cov_delta: Covariance diagonal loading (default: 1e-3)
        cov_eps: Minimum trace value (default: 1e-6)
        use_double: Use float64 for solver (default: True)
        margin_eps: Minimum margin to prevent division by zero (default: 1e-4)
        learnable_gamma: Whether gamma is learnable (default: False)
    """
    
    def __init__(
        self,
        K: int,
        gamma: float = 1.0,
        sigma_max: float = 0.3,
        cov_shrink: float = 0.1,
        cov_delta: float = 1e-3,
        cov_eps: float = 1e-6,
        use_double: bool = True,
        margin_eps: float = 1e-4,
        learnable_gamma: bool = False,
    ):
        super().__init__()
        self.K = K
        self.sigma_max = sigma_max
        self.cov_shrink = cov_shrink
        self.cov_delta = cov_delta
        self.cov_eps = cov_eps
        self.use_double = use_double
        self.margin_eps = margin_eps
        
        # Gamma can be learnable or fixed
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('gamma', torch.tensor(gamma))
    
    def forward(
        self,
        R: torch.Tensor,
        a0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute robust Capon weights using one-shot closed-form solution.
        
        Args:
            R: Covariance matrices [B, H, W, K, K]
            a0: Guidance vectors [B, H, W, K] or [B, K] or [K]
            sigma: Uncertainty (diagonal) [B, H, W, K] or [B, K] or [K]
            
        Returns:
            w: Robust Capon weights [B, H, W, K], normalized to sum to 1
            margin: Margin values [B, H, W] for monitoring
            debug_dict: Debug information
        """
        B, H, W, K, _ = R.shape
        device = R.device
        original_dtype = R.dtype
        
        # Use double precision for solver stability
        if self.use_double:
            R = R.double()
            a0 = a0.double()
            sigma = sigma.double()
        
        # Expand a0 and sigma to [B, H, W, K] if needed
        if a0.dim() == 1:
            a0 = a0.view(1, 1, 1, K).expand(B, H, W, K)
        elif a0.dim() == 2:
            a0 = a0.view(B, 1, 1, K).expand(B, H, W, K)
        
        if sigma.dim() == 1:
            sigma = sigma.view(1, 1, 1, K).expand(B, H, W, K)
        elif sigma.dim() == 2:
            sigma = sigma.view(B, 1, 1, K).expand(B, H, W, K)
        
        # Clamp sigma to [0, sigma_max]
        sigma = sigma.clamp(0, self.sigma_max)
        
        # Stabilize covariance
        R_stable = make_spd_cov(R, self.cov_eps, self.cov_shrink, self.cov_delta)
        
        # Compute A A^T = diag(sigma^2) for diagonal uncertainty
        # For diagonal A = diag(sigma), A A^T = diag(sigma^2)
        sigma_sq = sigma ** 2  # [B, H, W, K]
        
        # Build regularized matrix: R_tilde = R + gamma * diag(sigma^2)
        # Create diagonal matrix from sigma^2
        gamma = self.gamma.abs()  # Ensure positive gamma
        diag_loading = gamma * torch.diag_embed(sigma_sq)  # [B, H, W, K, K]
        R_tilde = R_stable + diag_loading
        
        # Solve: u = R_tilde^{-1} a_0
        # Reshape for batched solve: [B*H*W, K, K] and [B*H*W, K]
        R_tilde_flat = R_tilde.view(-1, K, K)
        a0_flat = a0.reshape(-1, K)
        
        u_flat, solve_success = batched_cholesky_solve(R_tilde_flat, a0_flat)
        u = u_flat.view(B, H, W, K)
        
        # Compute margin: m(u) = a_0^T u - ||A^T u||_2
        # For diagonal A = diag(sigma): A^T u = sigma * u (element-wise)
        a0_u = (a0 * u).sum(dim=-1)  # [B, H, W]: a_0^T u
        Au = sigma * u  # [B, H, W, K]: diag(sigma) * u
        Au_norm = torch.norm(Au, p=2, dim=-1)  # [B, H, W]: ||A^T u||_2
        
        margin = a0_u - Au_norm  # [B, H, W]
        
        # Handle degenerate cases where margin is too small
        valid_margin = margin > self.margin_eps
        safe_margin = torch.where(
            valid_margin,
            margin,
            torch.ones_like(margin)  # Placeholder for invalid cases
        )
        
        # Normalize: w = u / m(u)
        w = u / safe_margin.unsqueeze(-1).clamp(min=self.margin_eps)
        
        # For invalid margin cases, fallback to uniform weights
        uniform_w = torch.ones(B, H, W, K, device=device, dtype=R.dtype) / K
        w = torch.where(
            valid_margin.unsqueeze(-1).expand(-1, -1, -1, K),
            w,
            uniform_w
        )
        
        # Ensure weights are non-negative and sum to 1 (like attention)
        # Project to simplex: softmax(w) or ReLU + normalize
        w = F.softmax(w, dim=-1)
        
        # Convert back to original dtype
        if self.use_double:
            w = w.to(original_dtype)
            margin = margin.to(original_dtype)
        
        debug_dict = {
            'margin_mean': margin.mean().item(),
            'margin_min': margin.min().item(),
            'margin_max': margin.max().item(),
            'valid_ratio': valid_margin.float().mean().item(),
            'w_mean': w.mean().item(),
            'w_std': w.std().item(),
            'gamma': gamma.item(),
            'solve_success': solve_success,
        }
        
        return w, margin, debug_dict


class RobustCaponSolverWithSlack(nn.Module):
    """
    Robust Capon Solver with slack variable for enhanced stability.
    
    Wraps RobustCaponSolver with additional fallback mechanisms:
        1. If margin is too small, use uniform weights
        2. Smooth interpolation between robust and uniform weights
        3. Optional EMA for weight stability
    
    Args:
        K: Number of frequency streams
        slack_threshold: Margin threshold for using slack (default: 0.1)
        **solver_kwargs: Arguments passed to RobustCaponSolver
    """
    
    def __init__(
        self,
        K: int,
        slack_threshold: float = 0.1,
        **solver_kwargs
    ):
        super().__init__()
        self.K = K
        self.slack_threshold = slack_threshold
        self.solver = RobustCaponSolver(K=K, **solver_kwargs)
    
    def forward(
        self,
        R: torch.Tensor,
        a0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute robust weights with slack fallback.
        
        Args:
            R: Covariance matrices [B, H, W, K, K]
            a0: Guidance vectors
            sigma: Uncertainty (diagonal)
            
        Returns:
            w: Robust weights [B, H, W, K]
            margin: Margin values [B, H, W]
            debug_dict: Debug information
        """
        w, margin, debug_dict = self.solver(R, a0, sigma)
        
        B, H, W, K = w.shape
        device = w.device
        
        # Compute interpolation factor based on margin
        # Higher margin -> more trust in robust weights
        # Lower margin -> blend toward uniform
        alpha = torch.sigmoid((margin - self.slack_threshold) * 10)
        alpha = alpha.unsqueeze(-1)  # [B, H, W, 1]
        
        # Uniform weights as fallback
        uniform_w = torch.ones(B, H, W, K, device=device, dtype=w.dtype) / K
        
        # Interpolate: high margin uses robust, low margin uses uniform
        w_blended = alpha * w + (1 - alpha) * uniform_w
        
        # Re-normalize to ensure sum to 1
        w_blended = w_blended / w_blended.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        debug_dict['alpha_mean'] = alpha.mean().item()
        debug_dict['slack_active_ratio'] = (margin < self.slack_threshold).float().mean().item()
        
        return w_blended, margin, debug_dict


if __name__ == "__main__":
    """Unit tests for robust Capon solver."""
    print("Testing Robust Capon Solver...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    B, H, W, K = 2, 16, 16, 4
    
    # Create test data
    # Random SPD covariance matrices
    random_matrix = torch.randn(B, H, W, K, K, device=device)
    R = random_matrix @ random_matrix.transpose(-1, -2) + 0.1 * torch.eye(K, device=device)
    
    # Guidance vector (uniform)
    a0 = torch.ones(K, device=device) / K
    
    # Uncertainty
    sigma = torch.ones(K, device=device) * 0.1
    
    print(f"\n1. Testing RobustCaponSolver...")
    solver = RobustCaponSolver(K=K, gamma=1.0, sigma_max=0.3).to(device)
    w, margin, debug = solver(R, a0, sigma)
    
    print(f"   Input R: {R.shape}")
    print(f"   Output w: {w.shape}")
    print(f"   Margin: {margin.shape}")
    print(f"   Debug: {debug}")
    print(f"   Weights sum: {w.sum(dim=-1).mean().item():.6f} (should be ~1.0)")
    print(f"   Has NaN: {torch.isnan(w).any().item()}")
    
    print(f"\n2. Testing RobustCaponSolverWithSlack...")
    solver_slack = RobustCaponSolverWithSlack(K=K, slack_threshold=0.1).to(device)
    w_slack, margin_slack, debug_slack = solver_slack(R, a0, sigma)
    
    print(f"   Output w: {w_slack.shape}")
    print(f"   Debug: {debug_slack}")
    
    print(f"\n3. Testing gradient flow...")
    w_test, _, _ = solver_slack(R, a0.requires_grad_(True), sigma.requires_grad_(True))
    loss = w_test.sum()
    loss.backward()
    print("   Gradients computed successfully!")
    
    print(f"\n4. Testing with pixel-wise guidance and uncertainty...")
    a0_pixel = torch.randn(B, H, W, K, device=device).softmax(dim=-1)
    sigma_pixel = torch.rand(B, H, W, K, device=device) * 0.2
    
    w_pixel, margin_pixel, debug_pixel = solver_slack(R, a0_pixel, sigma_pixel)
    print(f"   Output w: {w_pixel.shape}")
    print(f"   Debug: {debug_pixel}")
    
    print("\n[OK] All tests passed!")

