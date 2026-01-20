"""
Training Script for One-Shot Robust Capon Frequency Fusion IRSTD

This script implements the complete training pipeline:

1. Load configuration from YAML
2. Build model with frequency fusion
3. Train with:
   - Pixel-wise losses (BCE + Dice)
   - Size-adaptive Augmented Lagrangian sparsity constraint
   - Optional isolation loss for Fa reduction
4. Validate with IoU/F1 and Pd/Fa metrics
5. NaN/Inf safety with gradient clipping

Usage:
    python train_oneshot.py

Author: Implementation for IRSTD with frequency fusion
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

# Local imports
from utils.data import TrainDataset
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.metrics import IRSTDMetrics

from model.GSFANet_RCF import GSFANet_RCF
from model.GSFANet import GSFANet
from model.losses import RCFTrainingLoss, MultiScaleRCFLoss, AverageMeter


# ============================================================================
# Configuration Section - Modify parameters here without command line args
# ============================================================================

# Dataset configuration
# Modify this path to your dataset location
DATASET_ROOT = './dataset'  # Dataset root directory (symbolic link to parent dataset)
DATASET_NAME = 'NUAA-SIRST'  # Dataset name: NUDT-SIRST, NUAA-SIRST, IRSTD-1k

# Training configuration
CONFIG_FILE = 'configs/oneshot_k2.yaml'  # Config file path
EPOCHS = 400                              # Number of epochs (None to use config value)
BATCH_SIZE = 8                            # Batch size (None to use config value)
LEARNING_RATE = 0.0002                    # Learning rate (None to use config value)

# Hardware configuration
GPU_INDEX = 0                             # GPU index

# Mode configuration
MODE = 'train'                            # Mode: 'train' or 'test'
RESUME_PATH = None                        # Resume checkpoint path (None for fresh start)
WEIGHT_PATH = None                        # Weight path for test mode

# Robust Fusion configuration
# TRUE SCHEME 3 IMPLEMENTATION: SOCP solver with covariance estimation
USE_ROBUST_FUSION = True                  # Enable TRUE Robust Capon SOCP Fusion
DEBUG_MODE = False                        # Print debug information (set True for troubleshooting)

# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description='One-Shot RCF Training for IRSTD')
    
    # Config file (use configuration section defaults)
    parser.add_argument('--config', type=str, default=CONFIG_FILE,
                        help='Path to config YAML file')
    
    # Override options (use configuration section defaults)
    parser.add_argument('--dataset', type=str, default=DATASET_NAME,
                        help='Override dataset name')
    parser.add_argument('--dataset-root', type=str, default=DATASET_ROOT,
                        help='Dataset root directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Override learning rate')
    parser.add_argument('--ngpu', type=int, default=GPU_INDEX,
                        help='GPU index to use')
    parser.add_argument('--resume', type=str, default=RESUME_PATH,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, default=MODE,
                        choices=['train', 'test'],
                        help='Train or test mode')
    parser.add_argument('--weight-path', type=str, default=WEIGHT_PATH,
                        help='Path to weights for test mode')
    
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OneShotTrainer:
    """
    Trainer for One-Shot Robust Capon Frequency Fusion IRSTD.
    
    Implements:
    - Training with BCE+Dice, AL sparsity, and isolation loss
    - Dual variable updates for AL
    - Validation with IoU/F1 and Pd/Fa metrics
    - Gradient clipping and NaN recovery
    - Checkpoint saving
    """
    
    def __init__(self, args, config: dict):
        self.args = args
        self.config = config
        self.mode = args.mode
        
        # Override config with args
        if args.dataset:
            config['dataset']['name'] = args.dataset
        if args.epochs:
            config['train']['epochs'] = args.epochs
        if args.batch_size:
            config['train']['batch_size'] = args.batch_size
        if args.lr:
            config['train']['lr'] = args.lr
        
        # Set dataset-specific sizes
        dataset_name = config['dataset']['name']
        if dataset_name == 'IRSTD-1k':
            config['dataset']['base_size'] = config['dataset'].get('base_size', 512)
            config['dataset']['crop_size'] = config['dataset'].get('crop_size', 512)
        else:
            config['dataset']['base_size'] = config['dataset'].get('base_size', 256)
            config['dataset']['crop_size'] = config['dataset'].get('crop_size', 256)
        
        # Device
        self.device = torch.device(
            f'cuda:{args.ngpu}' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Build components
        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()
        self._build_loss()
        self._build_metrics()
        
        # Training state
        self.start_epoch = 0
        self.best_iou = 0
        self.best_f1 = 0
        self.best_pd = 0
        self.best_fa = float('inf')
        
        # AMP scaler
        self.use_amp = config['train'].get('amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Save folder
        if self.mode == 'train':
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            self.save_folder = Path(f"weight/{dataset_name}/OneShot-{timestamp}")
            self.save_folder.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(self.save_folder / 'config.yaml', 'w') as f:
                yaml.dump(config, f)
        
        # Resume from checkpoint
        if args.resume:
            self._load_checkpoint(args.resume)
        
        # Load weights for test mode
        if self.mode == 'test' and args.weight_path:
            self._load_weights(args.weight_path)
    
    def _build_dataloaders(self):
        """Build train and validation dataloaders."""
        config = self.config
        
        # Create mock args for TrainDataset
        class DataArgs:
            pass
        data_args = DataArgs()
        data_args.dataset_root = self.args.dataset_root
        data_args.dataset = config['dataset']['name']
        data_args.base_size = config['dataset']['base_size']
        data_args.crop_size = config['dataset']['crop_size']
        
        trainset = TrainDataset(data_args, mode='train')
        valset = TrainDataset(data_args, mode='val')
        
        batch_size = config['train']['batch_size']
        num_workers = config['hardware'].get('num_workers', 8)
        pin_memory = config['hardware'].get('pin_memory', True)
        
        self.train_loader = Data.DataLoader(
            trainset, batch_size, shuffle=True, 
            drop_last=False, num_workers=num_workers, pin_memory=pin_memory
        )
        self.val_loader = Data.DataLoader(
            valset, 1, shuffle=False, 
            drop_last=False, num_workers=0, pin_memory=False
        )
        
        print(f"Train samples: {len(trainset)}, Val samples: {len(valset)}")
    
    def _build_model(self):
        """Build model with frequency fusion."""
        config = self.config
        crop_size = config['dataset']['crop_size']
        
        # Choose model based on configuration
        if USE_ROBUST_FUSION:
            print("Using GSFANet_RCF (Robust Capon Fusion)")
            model = GSFANet_RCF(
                size=crop_size,
                input_channels=1,
                K=config['model'].get('K', 2),
                decompose_method=config['model'].get('freq_mode', 'low_high'),
                gamma=config['solver'].get('gamma', 1.0),
                cov_shrink=config['solver'].get('cov_shrink', 0.1),
                sigma_max=config['fusion'].get('sigma_max', 1.0),
                guidance_mode=config['fusion'].get('a0_mode', 'stage'),
            )
        else:
            print("Using GSFANet (Original, no Robust Fusion)")
            model = GSFANet(
                size=crop_size,
                input_channels=1,
            )
        
        # Multi-GPU support
        if config['hardware'].get('multi_gpus', False) and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        model.to(self.device)
        self.model = model
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
    
    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        config = self.config['train']
        
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Scheduler
        epochs = config['epochs']
        warmup_epochs = config.get('warmup_epochs', 10)
        min_lr = config.get('min_lr', 1e-5)
        
        scheduler_cosine = CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs - warmup_epochs,
            eta_min=min_lr
        )
        self.scheduler = GradualWarmupScheduler(
            self.optimizer, 
            multiplier=1, 
            total_epoch=warmup_epochs,
            after_scheduler=scheduler_cosine
        )
        
        self.warmup_epochs = warmup_epochs
        self.grad_clip = config.get('grad_clip', 1.0)
    
    def _build_loss(self):
        """Build loss functions."""
        loss_config = self.config['loss']
        
        self.loss_fn = RCFTrainingLoss(
            bce_weight=1.0,
            dice_weight=loss_config.get('lambda_dice', 1.0),
            alpha=loss_config.get('al_alpha', 2.5),
            beta_pixels=loss_config.get('al_beta_pixels', 16),
            rho_al=loss_config.get('al_rho', 10.0),
            iso_weight=loss_config.get('lambda_iso', 1e-3),
            iso_pool_kernel=loss_config.get('iso_kernel', 3),
            iso_gamma=loss_config.get('iso_gamma', 2.0),
            warm_up_epochs=loss_config.get('al_warmup_epochs', 20),
            use_al=loss_config.get('use_al', True),
            use_iso=loss_config.get('use_iso', True),
        )
    
    def _build_metrics(self):
        """Build evaluation metrics."""
        threshold = self.config['train'].get('threshold', 0.5)
        min_area = self.config['metrics'].get('min_area', 2)
        self.metrics = IRSTDMetrics(threshold=threshold, min_area=min_area)
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_iou = checkpoint.get('best_iou', 0)
        self.best_f1 = checkpoint.get('best_f1', 0)
        print(f"Resumed from epoch {self.start_epoch}")
    
    def _load_weights(self, path: str):
        """Load model weights only."""
        print(f"Loading weights: {path}")
        weights = torch.load(path, map_location=self.device)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        self.model.load_state_dict(weights, strict=False)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
            'best_f1': self.best_f1,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.save_folder / 'checkpoint.pkl')
        
        # Save best
        if is_best:
            torch.save(
                self.model.state_dict(),
                self.save_folder / f'weight-{self.config["dataset"]["name"]}.pkl'
            )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        loss_components = {}
        
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for i, (data, mask) in enumerate(tbar):
            data = data.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass with optional AMP
            self.optimizer.zero_grad()
            
            # Debug: print first batch info
            if DEBUG_MODE and i == 0 and epoch == 0:
                print(f"\n[DEBUG] Input: min={data.min():.4f}, max={data.max():.4f}")
                print(f"[DEBUG] Mask: sum={mask.sum():.0f}, ratio={mask.mean():.6f}")
            
            if self.use_amp:
                with autocast():
                    masks_aux, pred = self.model(data, tag=True)
                    
                    # Debug: check prediction output
                    if DEBUG_MODE and i == 0 and epoch % 50 == 0:
                        pred_prob = torch.sigmoid(pred)
                        print(f"\n[DEBUG Epoch {epoch}] Pred logits: min={pred.min():.4f}, max={pred.max():.4f}")
                        print(f"[DEBUG Epoch {epoch}] Pred prob: min={pred_prob.min():.4f}, max={pred_prob.max():.4f}, mean={pred_prob.mean():.6f}")
                        print(f"[DEBUG Epoch {epoch}] Has NaN: {torch.isnan(pred).any()}, Has Inf: {torch.isinf(pred).any()}")
                    
                    loss, comps = self.loss_fn(
                        pred, mask, 
                        current_epoch=epoch, 
                        return_components=True
                    )
                    
                    # Add auxiliary losses
                    aux_loss = 0
                    target_ds = mask
                    for j, aux_pred in enumerate(masks_aux):
                        if j > 0:
                            target_ds = F.interpolate(target_ds, scale_factor=0.5, mode='nearest')
                        if aux_pred.shape[2:] != target_ds.shape[2:]:
                            target_ds_resized = F.interpolate(mask, size=aux_pred.shape[2:], mode='nearest')
                        else:
                            target_ds_resized = target_ds
                        aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                            aux_pred, target_ds_resized
                        )
                    
                    if len(masks_aux) > 0:
                        loss = loss + aux_loss / len(masks_aux)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at step {i}, skipping...")
                    continue
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                masks_aux, pred = self.model(data, tag=True)
                loss, comps = self.loss_fn(
                    pred, mask, 
                    current_epoch=epoch, 
                    return_components=True
                )
                
                # Add auxiliary losses
                aux_loss = 0
                target_ds = mask
                for j, aux_pred in enumerate(masks_aux):
                    if j > 0:
                        target_ds = F.interpolate(target_ds, scale_factor=0.5, mode='nearest')
                    if aux_pred.shape[2:] != target_ds.shape[2:]:
                        target_ds_resized = F.interpolate(mask, size=aux_pred.shape[2:], mode='nearest')
                    else:
                        target_ds_resized = target_ds
                    aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                        aux_pred, target_ds_resized
                    )
                
                if len(masks_aux) > 0:
                    loss = loss + aux_loss / len(masks_aux)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at step {i}, skipping...")
                    continue
                
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                
                self.optimizer.step()
            
            # Update statistics
            losses.update(loss.item(), data.size(0))
            
            # Accumulate components
            for key, value in comps.items():
                if key not in loss_components:
                    loss_components[key] = AverageMeter()
                loss_components[key].update(value, data.size(0))
            
            # Update progress bar
            tbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'seg': f'{comps["loss_seg"]:.4f}',
                'al': f'{comps["loss_al"]:.4f}',
                'lambda': f'{comps["dual_lambda"]:.3f}',
            })
        
        # Update scheduler
        self.scheduler.step(epoch)
        
        # Log epoch summary
        lr = self.optimizer.param_groups[0]['lr']
        log_str = (
            f'Epoch {epoch}: loss={losses.avg:.4f}, lr={lr:.6f}, '
            f'seg={loss_components.get("loss_seg", AverageMeter()).avg:.4f}, '
            f'al={loss_components.get("loss_al", AverageMeter()).avg:.4f}, '
            f'iso={loss_components.get("loss_iso", AverageMeter()).avg:.4f}, '
            f'lambda={loss_components.get("dual_lambda", AverageMeter()).avg:.3f}'
        )
        print(log_str)
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        self.metrics.reset()
        
        tbar = tqdm(self.val_loader, desc=f'Val {epoch}')
        
        for data, mask in tbar:
            data = data.to(self.device)
            mask = mask.to(self.device)
            
            _, pred = self.model(data, tag=False)
            pred_prob = torch.sigmoid(pred)
            
            # Update metrics
            self.metrics.update(pred_prob, mask)
            
            # Show current IoU
            results = self.metrics.get()
            tbar.set_postfix({
                'IoU': f'{results["iou"]:.4f}',
                'F1': f'{results["f1"]:.4f}',
            })
        
        # Get final metrics
        results = self.metrics.get()
        
        # Print results
        print(
            f'Val {epoch}: IoU={results["iou"]:.4f}, F1={results["f1"]:.4f}, '
            f'Pd={results["pd"]:.4f}, Fa={results["fa"]:.2f}'
        )
        
        # Check for best
        is_best = False
        if results['iou'] > self.best_iou:
            self.best_iou = results['iou']
            is_best = True
        if results['f1'] > self.best_f1:
            self.best_f1 = results['f1']
        
        # Save checkpoint
        if self.mode == 'train':
            self._save_checkpoint(epoch, is_best=is_best)
            
            # Log to file
            with open(self.save_folder / 'metric.log', 'a') as f:
                f.write(
                    f'{time.strftime("%Y-%m-%d-%H-%M-%S")}\t'
                    f'Epoch {epoch}\t'
                    f'IoU {results["iou"]:.5f}\t'
                    f'F1 {results["f1"]:.5f}\t'
                    f'Pd {results["pd"]:.5f}\t'
                    f'Fa {results["fa"]:.2f}\n'
                )
        
        return results
    
    def run(self):
        """Run training or evaluation."""
        config = self.config['train']
        
        if self.mode == 'test':
            print("Running evaluation...")
            results = self.validate(0)
            print(f"\nFinal Results:")
            print(f"  IoU: {results['iou']:.4f}")
            print(f"  F1:  {results['f1']:.4f}")
            print(f"  Pd:  {results['pd']:.4f}")
            print(f"  Fa:  {results['fa']:.2f}")
            return
        
        # Training loop
        epochs = config['epochs']
        val_interval = config.get('val_interval', 5)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Config: K={self.config['model'].get('K', 2)}, "
              f"freq={self.config['model'].get('freq_mode', 'low_high')}")
        
        for epoch in range(self.start_epoch, epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate periodically
            if (epoch + 1) % val_interval == 0 or epoch >= epochs - 5:
                results = self.validate(epoch)
        
        print(f"\nTraining complete!")
        print(f"Best IoU: {self.best_iou:.4f}")
        print(f"Best F1: {self.best_f1:.4f}")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    seed = config.get('seed', 42)
    seed_everything(seed)
    print(f"Random seed: {seed}")
    
    # Create trainer
    trainer = OneShotTrainer(args, config)
    
    # Run
    trainer.run()


if __name__ == '__main__':
    main()
