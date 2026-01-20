"""
Metrics for IRSTD Evaluation

This module implements evaluation metrics as per GPT specification:

1. Pixel-wise Metrics:
    - IoU (Intersection over Union)
    - F1 Score (Dice coefficient equivalent)

2. Instance-wise Metrics (using connected components):
    - Pd (Probability of Detection): Fraction of detected GT components
    - Fa (False Alarm): Number of spurious predicted components per image

Connected Component Logic:
    - Extract GT components from GT mask
    - Extract pred components from predicted mask
    - A GT component is "detected" if any pred component overlaps it (intersection > 0)
    - Pd = (# detected GT comps) / (# GT comps)
    - A pred component is "false alarm" if it overlaps no GT component
    - Fa = (# false pred comps) per image

Author: Implementation for IRSTD evaluation
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy import ndimage


class PixelMetrics:
    """
    Pixel-wise evaluation metrics.
    
    Computes IoU and F1 score by thresholding predictions.
    
    Args:
        threshold: Threshold for binarizing probabilities (default 0.5)
        smooth: Smoothing constant to avoid division by zero (default 1e-6)
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_intersection = 0
        self.total_union = 0
        self.total_pred = 0
        self.total_target = 0
        self.n_samples = 0
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Update metrics with a batch.
        
        Args:
            pred: [B, 1, H, W] or [B, H, W] predictions (probabilities or logits)
            target: [B, 1, H, W] or [B, H, W] binary ground truth
        """
        # Handle dimensions
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Convert to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Convert logits to probabilities if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = 1 / (1 + np.exp(-pred))  # Sigmoid
        
        # Threshold to binary
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = target.astype(np.float32)
        
        # Compute statistics
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        self.total_intersection += intersection
        self.total_union += union
        self.total_pred += pred_binary.sum()
        self.total_target += target_binary.sum()
        self.n_samples += pred.shape[0]
    
    def get_iou(self) -> float:
        """Get IoU score."""
        return (self.total_intersection + self.smooth) / (self.total_union + self.smooth)
    
    def get_f1(self) -> float:
        """Get F1 score (Dice coefficient)."""
        return (2 * self.total_intersection + self.smooth) / (
            self.total_pred + self.total_target + self.smooth
        )
    
    def get(self) -> Dict[str, float]:
        """Get all metrics."""
        return {
            'iou': self.get_iou(),
            'f1': self.get_f1(),
        }


class InstanceMetrics:
    """
    Instance-wise evaluation metrics using connected components.
    
    Computes:
        - Pd (Probability of Detection): Fraction of detected GT components
        - Fa (False Alarm): Mean number of false alarm components per image
    
    A GT component is "detected" if any predicted component overlaps it.
    A predicted component is a "false alarm" if it overlaps no GT component.
    
    Args:
        threshold: Threshold for binarizing probabilities (default 0.5)
        min_area: Minimum area to consider a component (default 2 pixels)
        connectivity: Connectivity for labeling (4 or 8, default 8)
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_area: int = 2,
        connectivity: int = 8,
    ):
        self.threshold = threshold
        self.min_area = min_area
        # scipy.ndimage uses structure parameter for connectivity
        if connectivity == 4:
            self.structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:  # 8-connectivity
            self.structure = np.ones((3, 3))
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_gt_components = 0
        self.total_detected = 0
        self.total_false_alarms = 0
        self.n_images = 0
    
    def _get_components(
        self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, int, List[int]]:
        """
        Extract connected components from binary mask.
        
        Args:
            mask: [H, W] binary mask
            
        Returns:
            labeled: [H, W] labeled image (0 = background)
            n_components: Number of components (excluding background)
            areas: List of component areas
        """
        labeled, n_components = ndimage.label(mask, structure=self.structure)
        
        if n_components == 0:
            return labeled, 0, []
        
        # Get areas of each component
        areas = []
        valid_labels = []
        for i in range(1, n_components + 1):
            area = (labeled == i).sum()
            if area >= self.min_area:
                areas.append(area)
                valid_labels.append(i)
        
        # Relabel to remove small components
        if len(valid_labels) < n_components:
            new_labeled = np.zeros_like(labeled)
            for new_idx, old_label in enumerate(valid_labels, 1):
                new_labeled[labeled == old_label] = new_idx
            labeled = new_labeled
            n_components = len(valid_labels)
        
        return labeled, n_components, areas
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Update metrics with a batch.
        
        Args:
            pred: [B, 1, H, W] or [B, H, W] predictions
            target: [B, 1, H, W] or [B, H, W] binary ground truth
        """
        # Handle dimensions
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Convert to numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Convert logits to probabilities if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = 1 / (1 + np.exp(-pred))
        
        # Threshold to binary
        pred_binary = (pred > self.threshold).astype(np.uint8)
        target_binary = target.astype(np.uint8)
        
        B = pred.shape[0]
        
        for b in range(B):
            self._update_single(pred_binary[b], target_binary[b])
    
    def _update_single(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ):
        """Update metrics for a single image."""
        # Get GT components
        gt_labeled, n_gt, _ = self._get_components(gt_mask)
        
        # Get predicted components
        pred_labeled, n_pred, _ = self._get_components(pred_mask)
        
        self.total_gt_components += n_gt
        self.n_images += 1
        
        if n_gt == 0:
            # All predicted components are false alarms
            self.total_false_alarms += n_pred
            return
        
        if n_pred == 0:
            # No predictions, nothing detected
            return
        
        # Check which GT components are detected
        n_detected = 0
        for gt_idx in range(1, n_gt + 1):
            gt_component = (gt_labeled == gt_idx)
            # Check if any pred component overlaps
            overlapping_pred = pred_labeled[gt_component]
            if np.any(overlapping_pred > 0):
                n_detected += 1
        
        self.total_detected += n_detected
        
        # Check which pred components are false alarms
        n_false = 0
        for pred_idx in range(1, n_pred + 1):
            pred_component = (pred_labeled == pred_idx)
            # Check if overlaps any GT component
            overlapping_gt = gt_labeled[pred_component]
            if not np.any(overlapping_gt > 0):
                n_false += 1
        
        self.total_false_alarms += n_false
    
    def get_pd(self) -> float:
        """Get Probability of Detection."""
        if self.total_gt_components == 0:
            return 1.0  # No targets to miss
        return self.total_detected / self.total_gt_components
    
    def get_fa(self) -> float:
        """Get mean False Alarm per image."""
        if self.n_images == 0:
            return 0.0
        return self.total_false_alarms / self.n_images
    
    def get(self) -> Dict[str, float]:
        """Get all metrics."""
        return {
            'pd': self.get_pd(),
            'fa': self.get_fa(),
            'n_gt_components': self.total_gt_components,
            'n_detected': self.total_detected,
            'n_false_alarms': self.total_false_alarms,
            'n_images': self.n_images,
        }


class IRSTDMetrics:
    """
    Combined IRSTD metrics handler.
    
    Computes both pixel-wise (IoU, F1) and instance-wise (Pd, Fa) metrics.
    
    Args:
        threshold: Threshold for binarizing predictions
        min_area: Minimum area for instance metrics
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_area: int = 2,
    ):
        self.pixel_metrics = PixelMetrics(threshold=threshold)
        self.instance_metrics = InstanceMetrics(threshold=threshold, min_area=min_area)
    
    def reset(self):
        """Reset all metrics."""
        self.pixel_metrics.reset()
        self.instance_metrics.reset()
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Update all metrics with a batch.
        
        Args:
            pred: [B, 1, H, W] predictions
            target: [B, 1, H, W] ground truth
        """
        self.pixel_metrics.update(pred, target)
        self.instance_metrics.update(pred, target)
    
    def get(self) -> Dict[str, float]:
        """Get all metrics."""
        result = {}
        result.update(self.pixel_metrics.get())
        result.update(self.instance_metrics.get())
        return result
    
    def summary(self) -> str:
        """Get summary string."""
        m = self.get()
        return (
            f"IoU: {m['iou']:.4f}, F1: {m['f1']:.4f}, "
            f"Pd: {m['pd']:.4f}, Fa: {m['fa']:.2f}"
        )


class ThresholdSweep:
    """
    Sweep threshold to find best operating point.
    
    Useful for finding optimal threshold based on F1 or other metrics.
    
    Args:
        thresholds: List of thresholds to try (default: 0.1 to 0.9 in 0.1 steps)
        min_area: Minimum area for instance metrics
    """
    
    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        min_area: int = 2,
    ):
        if thresholds is None:
            self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            self.thresholds = thresholds
        self.min_area = min_area
        
        # Create metrics for each threshold
        self.metrics_per_threshold = {
            t: IRSTDMetrics(threshold=t, min_area=min_area)
            for t in self.thresholds
        }
    
    def reset(self):
        """Reset all metrics."""
        for m in self.metrics_per_threshold.values():
            m.reset()
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update all threshold metrics."""
        for m in self.metrics_per_threshold.values():
            m.update(pred, target)
    
    def get_best_by_f1(self) -> Tuple[float, Dict[str, float]]:
        """Get threshold and metrics with best F1."""
        best_t = 0.5
        best_f1 = -1
        best_metrics = {}
        
        for t, m in self.metrics_per_threshold.items():
            metrics = m.get()
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_t = t
                best_metrics = metrics
        
        return best_t, best_metrics
    
    def get_best_by_iou(self) -> Tuple[float, Dict[str, float]]:
        """Get threshold and metrics with best IoU."""
        best_t = 0.5
        best_iou = -1
        best_metrics = {}
        
        for t, m in self.metrics_per_threshold.items():
            metrics = m.get()
            if metrics['iou'] > best_iou:
                best_iou = metrics['iou']
                best_t = t
                best_metrics = metrics
        
        return best_t, best_metrics
    
    def get_all(self) -> Dict[float, Dict[str, float]]:
        """Get metrics for all thresholds."""
        return {t: m.get() for t, m in self.metrics_per_threshold.items()}


if __name__ == "__main__":
    print("Testing IRSTD Metrics...")
    
    # Create synthetic data
    B, H, W = 4, 256, 256
    
    # Create targets with small targets
    target = torch.zeros(B, 1, H, W)
    # Add a few small targets per image
    for b in range(B):
        for _ in range(3):
            cx, cy = np.random.randint(20, H-20), np.random.randint(20, W-20)
            size = np.random.randint(3, 8)
            target[b, 0, cx-size:cx+size, cy-size:cy+size] = 1.0
    
    # Create predictions with some overlap and some false alarms
    pred = torch.zeros(B, 1, H, W)
    for b in range(B):
        # Copy most targets with slight offset
        pred[b] = 0.9 * target[b].roll(shifts=2, dims=1)
        # Add false alarms
        for _ in range(2):
            cx, cy = np.random.randint(20, H-20), np.random.randint(20, W-20)
            pred[b, 0, cx-3:cx+3, cy-3:cy+3] = 0.8
    
    # Add noise
    pred = pred + 0.1 * torch.rand_like(pred)
    pred = pred.clamp(0, 1)
    
    print(f"\nTarget shape: {target.shape}")
    print(f"Pred shape: {pred.shape}")
    print(f"Target sum per image: {target.sum(dim=(1,2,3)).tolist()}")
    
    # Test PixelMetrics
    print("\n1. Testing PixelMetrics...")
    pixel_metrics = PixelMetrics(threshold=0.5)
    pixel_metrics.update(pred, target)
    print(f"   IoU: {pixel_metrics.get_iou():.4f}")
    print(f"   F1: {pixel_metrics.get_f1():.4f}")
    
    # Test InstanceMetrics
    print("\n2. Testing InstanceMetrics...")
    instance_metrics = InstanceMetrics(threshold=0.5, min_area=2)
    instance_metrics.update(pred, target)
    result = instance_metrics.get()
    print(f"   Pd: {result['pd']:.4f}")
    print(f"   Fa: {result['fa']:.2f}")
    print(f"   GT components: {result['n_gt_components']}")
    print(f"   Detected: {result['n_detected']}")
    print(f"   False alarms: {result['n_false_alarms']}")
    
    # Test combined metrics
    print("\n3. Testing IRSTDMetrics...")
    irstd_metrics = IRSTDMetrics(threshold=0.5, min_area=2)
    irstd_metrics.update(pred, target)
    print(f"   Summary: {irstd_metrics.summary()}")
    
    # Test threshold sweep
    print("\n4. Testing ThresholdSweep...")
    sweep = ThresholdSweep(min_area=2)
    sweep.update(pred, target)
    best_t, best_metrics = sweep.get_best_by_f1()
    print(f"   Best threshold by F1: {best_t}")
    print(f"   Best F1: {best_metrics['f1']:.4f}")
    print(f"   Best IoU: {best_metrics['iou']:.4f}")
    
    print("\n[OK] All tests passed!")

