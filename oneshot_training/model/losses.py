"""
Comprehensive Loss Functions for Robust Capon Frequency Fusion IRSTD

This module implements the complete training objective as per GPT specification:

1. Base Segmentation Loss (for IoU/F1):
    L_seg = L_BCE + lambda_D * L_Dice

2. Size-Adaptive Augmented Lagrangian Sparsity Constraint:
    Per image:
        mu_b = mean(sigmoid(logits))  (predicted foreground ratio)
        g_b = mean(mask)              (GT foreground ratio)
        tau_b = alpha * g_b + beta/(H*W)       (size-adaptive upper bound)
        c_b = mu_b - tau_b               (violation)
    
    AL penalty:
        L_AL = mean(lambda_b * c_b + 0.5 * rho * relu(c_b)^2)
    
    Dual update (outside gradients):
        lambda_b = clamp(lambda_b + rho * c_b, min=0)

3. Isolation Loss (anti-scatter regularizer for Fa reduction):
    q = AvgPool2d(p, k=3)
    L_iso = mean(p^gamma * (1 - q))

Total Loss:
    L = L_seg + L_AL + lambda_iso * L_iso

Author: Implementation for IRSTD with robust frequency fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss for segmentation.
    L_seg = BCE + lambda_D * DiceLoss
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] logits (before sigmoid)
            target: [B, 1, H, W] binary ground truth
        """
        # Clamp predictions to prevent overflow
        pred = pred.clamp(-20.0, 20.0)
        
        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        
        # Dice loss
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        union = pred_sig.sum() + target.sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth + 1e-8)
        
        # Handle NaN
        if torch.isnan(bce) or torch.isinf(bce):
            bce = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
        if torch.isnan(dice) or torch.isinf(dice):
            dice = torch.tensor(0.5, device=pred.device, dtype=pred.dtype)
        
        return self.bce_weight * bce + self.dice_weight * dice


class AugLagSparsityConstraint(nn.Module):
    """
    Size-Adaptive Augmented Lagrangian Sparsity Constraint.
    
    This implements the proper AL formulation with dual variable updates:
    
    Per image b:
        mu_b = mean(p) over H,W  (predicted foreground ratio)
        g_b = mean(mask) over H,W  (GT ratio, always >0)
        tau_b = alpha * g_b + beta/(H*W)  (size-adaptive upper bound)
        c_b = mu_b - tau_b  (violation)
    
    AL penalty:
        L_AL = mean(lambda_b * c_b + 0.5 * rho * relu(c_b)^2)
    
    Dual update (call separately, outside backward):
        lambda_b = clamp(lambda_b + rho * c_b, min=0)
    
    Args:
        alpha: Multiplier for GT ratio in upper bound (default 2.5)
        beta_pixels: Baseline pixels in upper bound numerator (default 16)
        rho: AL penalty weight (default 10.0)
        warm_up_epochs: Epochs before enabling AL (default 20)
    """
    
    def __init__(
        self,
        alpha: float = 2.5,
        beta_pixels: float = 16.0,
        rho: float = 10.0,
        warm_up_epochs: int = 20,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta_pixels = beta_pixels
        self.rho = rho
        self.warm_up_epochs = warm_up_epochs
        
        # Dual variables (managed externally or as buffer)
        # Initialize as None, will be created on first forward
        self._lambda_dual = None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lambda_dual: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute AL sparsity loss.
        
        Args:
            pred: [B, 1, H, W] logits or probabilities
            target: [B, 1, H, W] binary ground truth
            lambda_dual: [B] dual variables (optional, uses internal if None)
            current_epoch: Current training epoch
            
        Returns:
            loss: Scalar AL loss
            violations: [B] violation per image (c_b, for dual update)
        """
        B, _, H, W = pred.shape
        HW = H * W
        device = pred.device
        dtype = pred.dtype
        
        # Convert logits to probabilities if needed
        if pred.min() < 0 or pred.max() > 1:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        
        # Predicted foreground ratio mu_b
        mu = pred_prob.view(B, -1).mean(dim=1)  # [B]
        
        # GT foreground ratio g_b
        g = target.view(B, -1).mean(dim=1)  # [B]
        
        # Size-adaptive upper bound: tau_b = alpha * g_b + beta/(H*W)
        tau = self.alpha * g + self.beta_pixels / HW  # [B]
        
        # Violation: c_b = mu_b - tau_b
        violation = mu - tau  # [B]
        
        # During warm-up, return zero loss
        if current_epoch < self.warm_up_epochs:
            loss = torch.tensor(0.0, device=device, dtype=dtype)
            return loss, violation.detach()
        
        # Get or create dual variables
        if lambda_dual is None:
            if self._lambda_dual is None or self._lambda_dual.shape[0] != B:
                self._lambda_dual = torch.zeros(B, device=device, dtype=dtype)
            lambda_dual = self._lambda_dual
        
        # Ensure lambda is on correct device
        lambda_dual = lambda_dual.to(device=device, dtype=dtype)
        
        # AL penalty: L_AL = mean(lambda * c + 0.5 * rho * relu(c)^2)
        # Note: We include both the linear and quadratic terms for proper AL
        linear_term = lambda_dual * violation
        quadratic_term = 0.5 * self.rho * F.relu(violation) ** 2
        
        loss = (linear_term + quadratic_term).mean()
        
        return loss, violation.detach()
    
    def update_dual(
        self,
        violations: torch.Tensor,
        lambda_dual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform dual variable update (call outside backward).
        
        lambda_b <- clamp(lambda_b + rho * c_b, min=0)
        
        Args:
            violations: [B] violation values from forward
            lambda_dual: [B] current dual variables (optional)
            
        Returns:
            lambda_dual: [B] updated dual variables
        """
        if lambda_dual is None:
            lambda_dual = self._lambda_dual
        
        if lambda_dual is None:
            return torch.zeros_like(violations)
        
        # Ensure same device
        lambda_dual = lambda_dual.to(violations.device)
        
        # Dual update with clamp
        lambda_dual = (lambda_dual + self.rho * violations).clamp(min=0.0)
        
        # Store updated dual
        self._lambda_dual = lambda_dual.detach()
        
        return lambda_dual
    
    def reset_dual(self):
        """Reset dual variables to zero."""
        self._lambda_dual = None


class IsolationLoss(nn.Module):
    """
    Isolation Loss (Anti-Scatter) for Fa Reduction.
    
    Penalizes isolated high predictions (high p where neighborhood average is low):
        q = AvgPool2d(p, k, stride=1, padding=k//2)
        L_iso = mean(p^gamma * (1 - q))
    
    This encourages predictions to be spatially coherent, reducing false alarms.
    
    Args:
        pool_kernel: Kernel size for average pooling (default 3)
        gamma: Exponent for prediction weighting (default 2.0)
    """
    
    def __init__(
        self,
        pool_kernel: int = 3,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.gamma = gamma
        padding = pool_kernel // 2
        self.avg_pool = nn.AvgPool2d(pool_kernel, stride=1, padding=padding)
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] logits or probabilities
            
        Returns:
            loss: Scalar isolation loss
        """
        # Convert to probabilities if needed
        if pred.min() < 0 or pred.max() > 1:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        
        # Local average
        q = self.avg_pool(pred_prob)
        
        # Isolation score: p^gamma * (1 - q)
        isolation = pred_prob.pow(self.gamma) * (1.0 - q)
        
        return isolation.mean()


class DualVariableManager:
    """
    Manages dual variables for Augmented Lagrangian training.
    
    Features:
        - Per-sample dual variables
        - Exponential moving average tracking
        - Warm-up scheduling
    
    Args:
        rho: AL penalty weight
        decay: EMA decay for tracking (default 0.9)
        max_dual: Maximum dual variable value (default 100.0)
    """
    
    def __init__(
        self,
        rho: float = 10.0,
        decay: float = 0.9,
        max_dual: float = 100.0,
    ):
        self.rho = rho
        self.decay = decay
        self.max_dual = max_dual
        
        # Running statistics
        self._lambda_ema = 0.0
        self._violation_ema = 0.0
        
        # Per-sample storage (optional)
        self._sample_duals = {}
    
    def get_dual(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get dual variables for current batch."""
        # Use EMA value as initialization
        return torch.full(
            (batch_size,),
            self._lambda_ema,
            device=device,
            dtype=torch.float32
        )
    
    def update(
        self,
        violations: torch.Tensor,
        lambda_dual: torch.Tensor
    ) -> torch.Tensor:
        """
        Update dual variables and return new values.
        
        Args:
            violations: [B] violation values
            lambda_dual: [B] current dual variables
            
        Returns:
            new_lambda: [B] updated dual variables
        """
        # Dual update: lambda <- clamp(lambda + rho * c, min=0, max=max_dual)
        new_lambda = (lambda_dual + self.rho * violations).clamp(
            min=0.0, max=self.max_dual
        )
        
        # Update EMAs
        mean_lambda = new_lambda.mean().item()
        mean_violation = violations.mean().item()
        
        self._lambda_ema = self.decay * self._lambda_ema + (1 - self.decay) * mean_lambda
        self._violation_ema = self.decay * self._violation_ema + (1 - self.decay) * mean_violation
        
        return new_lambda.detach()
    
    @property
    def current_dual(self) -> float:
        """Current EMA of dual variable."""
        return self._lambda_ema
    
    @property
    def current_violation(self) -> float:
        """Current EMA of violation."""
        return self._violation_ema
    
    def reset(self):
        """Reset all state."""
        self._lambda_ema = 0.0
        self._violation_ema = 0.0
        self._sample_duals = {}


class RCFTrainingLoss(nn.Module):
    """
    Complete Training Loss for Robust Capon Frequency Fusion IRSTD.
    
    L = L_seg + L_AL + lambda_iso * L_iso
    
    Components:
        - L_seg: BCE + Dice segmentation loss
        - L_AL: Augmented Lagrangian sparsity constraint
        - L_iso: Isolation loss for false alarm reduction
    
    Args:
        bce_weight: Weight for BCE loss (default 1.0)
        dice_weight: Weight for Dice loss (default 1.0)
        alpha: AL size-adaptive alpha parameter (default 2.5)
        beta_pixels: AL baseline pixels beta (default 16)
        rho_al: AL penalty weight rho (default 10.0)
        iso_weight: Weight for isolation loss (default 1e-3)
        iso_pool_kernel: Isolation pooling kernel (default 3)
        iso_gamma: Isolation exponent (default 2.0)
        warm_up_epochs: Epochs before enabling AL (default 20)
        use_al: Whether to use AL loss (default True)
        use_iso: Whether to use isolation loss (default True)
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: float = 2.5,
        beta_pixels: float = 16.0,
        rho_al: float = 10.0,
        iso_weight: float = 1e-3,
        iso_pool_kernel: int = 3,
        iso_gamma: float = 2.0,
        warm_up_epochs: int = 20,
        use_al: bool = True,
        use_iso: bool = True,
    ):
        super().__init__()
        
        self.seg_loss = BCEDiceLoss(bce_weight, dice_weight)
        self.sparsity_loss = AugLagSparsityConstraint(
            alpha=alpha,
            beta_pixels=beta_pixels,
            rho=rho_al,
            warm_up_epochs=warm_up_epochs,
        )
        self.iso_loss = IsolationLoss(pool_kernel=iso_pool_kernel, gamma=iso_gamma)
        self.dual_manager = DualVariableManager(rho=rho_al)
        
        self.iso_weight = iso_weight
        self.rho_al = rho_al
        self.warm_up_epochs = warm_up_epochs
        self.use_al = use_al
        self.use_iso = use_iso
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        current_epoch: int = 0,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute total training loss.
        
        Args:
            pred: [B, 1, H, W] prediction logits
            target: [B, 1, H, W] binary ground truth
            current_epoch: Current training epoch
            return_components: Whether to return loss components dict
            
        Returns:
            loss: Total scalar loss
            (optional) components: Dict with individual loss values
        """
        B = pred.shape[0]
        device = pred.device
        
        # 1. Segmentation loss
        l_seg = self.seg_loss(pred, target)
        
        # 2. AL sparsity loss
        if self.use_al and current_epoch >= self.warm_up_epochs:
            lambda_dual = self.dual_manager.get_dual(B, device)
            l_al, violations = self.sparsity_loss(
                pred, target, lambda_dual, current_epoch
            )
            
            # Update dual variables (detached, no gradients)
            with torch.no_grad():
                self.dual_manager.update(violations, lambda_dual)
        else:
            l_al = torch.tensor(0.0, device=device, dtype=pred.dtype)
            violations = torch.zeros(B, device=device, dtype=pred.dtype)
        
        # 3. Isolation loss
        if self.use_iso:
            l_iso = self.iso_loss(pred)
        else:
            l_iso = torch.tensor(0.0, device=device, dtype=pred.dtype)
        
        # Ensure all losses are non-negative
        l_seg = torch.clamp(l_seg, min=0.0)
        l_al = torch.clamp(l_al, min=0.0)
        l_iso = torch.clamp(l_iso, min=0.0)
        
        # Total loss
        total_loss = l_seg + l_al + self.iso_weight * l_iso
        
        if return_components:
            # Compute monitoring metrics
            pred_prob = torch.sigmoid(pred)
            mu = pred_prob.view(B, -1).mean(dim=1)
            g = target.view(B, -1).mean(dim=1)
            _, _, H, W = pred.shape
            tau = self.sparsity_loss.alpha * g + self.sparsity_loss.beta_pixels / (H * W)
            
            components = {
                'loss_seg': l_seg.item(),
                'loss_al': l_al.item(),
                'loss_iso': l_iso.item(),
                'violation_mean': violations.mean().item(),
                'violation_max': violations.max().item() if B > 0 else 0.0,
                'dual_lambda': self.dual_manager.current_dual,
                'mu_mean': mu.mean().item(),
                'tau_mean': tau.mean().item(),
                'relu_violation': F.relu(violations).mean().item(),
            }
            return total_loss, components
        
        return total_loss
    
    def reset_dual(self):
        """Reset dual variables."""
        self.dual_manager.reset()
        self.sparsity_loss.reset_dual()


class MultiScaleRCFLoss(nn.Module):
    """
    Multi-Scale Training Loss with Deep Supervision.
    
    Applies main loss at full resolution and auxiliary BCE+Dice
    at intermediate scales.
    
    Args:
        num_scales: Number of auxiliary scales
        scale_weights: Weights for each scale (including main)
        **rcf_loss_kwargs: Arguments for RCFTrainingLoss
    """
    
    def __init__(
        self,
        num_scales: int = 4,
        scale_weights: Optional[list] = None,
        **rcf_loss_kwargs,
    ):
        super().__init__()
        self.num_scales = num_scales
        
        if scale_weights is None:
            self.scale_weights = [1.0 / num_scales] * num_scales
        else:
            self.scale_weights = scale_weights
        
        self.main_loss = RCFTrainingLoss(**rcf_loss_kwargs)
        self.aux_loss = BCEDiceLoss()
    
    def forward(
        self,
        preds: list,
        final_pred: torch.Tensor,
        target: torch.Tensor,
        current_epoch: int = 0,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            preds: List of intermediate predictions
            final_pred: Final prediction at full resolution
            target: Ground truth mask
            current_epoch: Current epoch
            return_components: Whether to return components dict
            
        Returns:
            loss: Total loss
            (optional) components: Loss components dict
        """
        if return_components:
            main_loss, components = self.main_loss(
                final_pred, target, current_epoch, return_components=True
            )
        else:
            main_loss = self.main_loss(final_pred, target, current_epoch)
            components = {}
        
        total_loss = self.scale_weights[-1] * main_loss
        
        # Auxiliary losses at intermediate scales
        target_ds = target
        for i, pred in enumerate(preds):
            if i > 0:
                target_ds = F.interpolate(target_ds, scale_factor=0.5, mode='nearest')
            
            if pred.shape[2:] != target_ds.shape[2:]:
                target_ds_resized = F.interpolate(target, size=pred.shape[2:], mode='nearest')
            else:
                target_ds_resized = target_ds
            
            aux_loss = self.aux_loss(pred, target_ds_resized)
            total_loss = total_loss + self.scale_weights[i] * aux_loss
        
        if return_components:
            return total_loss, components
        return total_loss
    
    def reset_dual(self):
        """Reset dual variables."""
        self.main_loss.reset_dual()


# ============== Legacy Compatibility ==============

class AdaFocalLoss(nn.Module):
    """Adaptive Focal Loss (from original implementation)."""
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        area_weight = self._get_area_weight(target)
        smooth = 1

        intersection = pred.sigmoid() * target
        iou = (intersection.sum() + smooth) / (
            pred.sigmoid().sum() + target.sum() - intersection.sum() + smooth
        )
        iou = torch.clamp(iou, min=1e-6, max=1 - 1e-6).detach()
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        target_long = target.type(torch.long)
        at = target_long * area_weight + (1 - target_long) * (1 - area_weight)
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        F_loss = (1 - pt) ** (1 - iou + 1e-6) * BCE_loss

        F_loss = at * F_loss
        return F_loss.sum()

    def _get_area_weight(self, target):
        area = target.sum(dim=(1, 2, 3))
        return torch.sigmoid(1 - area / (area.max() + 1)).view(-1, 1, 1, 1)


class SoftIoULoss(nn.Module):
    """Soft IoU Loss (from original implementation)."""
    
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def IOU(self, pred, mask):
        smooth = 1
        intersection = pred * mask
        loss = (intersection.sum() + smooth) / (
            pred.sum() + mask.sum() - intersection.sum() + smooth
        )
        loss = 1 - torch.mean(loss)
        return loss

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        loss_iou = self.IOU(pred, mask)
        return loss_iou


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    print("Testing loss functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 4, 256, 256
    
    pred = torch.randn(B, 1, H, W, device=device)
    target = (torch.rand(B, 1, H, W, device=device) > 0.99).float()
    
    # Test RCFTrainingLoss
    rcf_fn = RCFTrainingLoss(
        alpha=2.5,
        beta_pixels=16,
        rho_al=10.0,
        warm_up_epochs=20,
        use_al=True,
        use_iso=True,
    )
    
    print("\nTesting RCFTrainingLoss over epochs:")
    for epoch in [10, 25, 50, 100]:
        loss, comps = rcf_fn(pred, target, current_epoch=epoch, return_components=True)
        print(f"\nEpoch {epoch}:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  L_seg: {comps['loss_seg']:.4f}")
        print(f"  L_AL: {comps['loss_al']:.4f}")
        print(f"  L_iso: {comps['loss_iso']:.4f}")
        print(f"  Violation: {comps['violation_mean']:.4f}")
        print(f"  Dual lambda: {comps['dual_lambda']:.4f}")
        print(f"  mu/tau: {comps['mu_mean']:.4f}/{comps['tau_mean']:.4f}")
        
        assert loss.item() >= 0, "Loss should be non-negative!"
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = rcf_fn(pred, target, current_epoch=50)
    loss.backward()
    print("Gradients computed successfully!")
    
    # Test dual variable update
    print("\nTesting dual variable dynamics...")
    rcf_fn.reset_dual()
    for i in range(10):
        loss, comps = rcf_fn(pred, target, current_epoch=30, return_components=True)
        print(f"  Step {i}: lambda={comps['dual_lambda']:.4f}, violation={comps['violation_mean']:.4f}")
    
    print("\n[OK] All tests passed!")

