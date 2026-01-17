#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mix_cre Training Script"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import create_dataloaders, load_and_prepare_csv, reverse_complement_tokens
from model import Evo2MixModel
from losses import MultiTaskLoss
from metrics import macro_average, per_species_metrics, classification_metrics


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device, class_weight_tensor, grad_clip):
    model.train()
    total_loss = total_reg = total_cls = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        species = batch["species"].to(device)

        optimizer.zero_grad()
        y_hat, logits = model(tokens)
        loss, l_reg, l_cls = loss_fn(
            y_hat, y, logits, species, class_weight=class_weight_tensor, use_focal=False
        )
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_reg += float(l_reg.item())
        total_cls += float(l_cls.item())

    n = len(loader)
    return total_loss / n, total_reg / n, total_cls / n


@torch.no_grad()
def evaluate(model, loader, device, rc_pool=True):
    model.eval()
    preds, trues, species_ids, cls_preds, cls_scores = [], [], [], [], []
    
    for batch in tqdm(loader, desc="eval", leave=False):
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        species = batch["species"].to(device)
        
        y_hat, logits = model(tokens)
        
        if rc_pool:
            toks_np = tokens.detach().cpu().numpy()
            rc_np = [reverse_complement_tokens(t) for t in toks_np]
            rc_tokens = torch.from_numpy(np.stack(rc_np, axis=0)).to(device)
            y_hat_rc, logits_rc = model(rc_tokens)
            y_hat = 0.5 * (y_hat + y_hat_rc)
            logits = 0.5 * (logits + logits_rc)
            
        preds.append(y_hat.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
        species_ids.append(species.detach().cpu().numpy())
        cls_preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
        cls_scores.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
        
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(species_ids, axis=0),
        np.concatenate(cls_preds, axis=0),
        np.concatenate(cls_scores, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Mix_cre Training")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--max_len", type=int, default=3001, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--model_name", type=str, default="Mix_cre", help="Model name")
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--tail_layers", type=int, default=3, help="Number of SSM tail layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    print(f"Loading data from {args.csv_path}")
    split = load_and_prepare_csv(
        csv_path=args.csv_path,
        seq_len=args.max_len,
        seed=args.seed,
    )
    
    num_species = len(split.species_to_id)
    print(f"Number of species: {num_species}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        split,
        seq_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    # Create model
    model = Evo2MixModel(
        num_species=num_species,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        tail="mamba",
        tail_layers=args.tail_layers,
        dropout=args.dropout,
    ).to(device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = MultiTaskLoss()
    
    # Class weights
    class_weight_tensor = None
    if len(class_weights) > 1:
        max_c = max(class_weights.keys()) + 1
        cw = torch.ones(max_c, dtype=torch.float32)
        for k, v in class_weights.items():
            cw[int(k)] = float(v)
        class_weight_tensor = cw.to(device)
    
    # Training loop
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_spearman = -1.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_reg, train_cls = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, class_weight_tensor, args.grad_clip
        )
        
        # Validate
        pred_v, true_v, sp_v, cls_v, cls_prob_v = evaluate(model, val_loader, device)
        per_sp_metrics = per_species_metrics(true_v, pred_v, sp_v)
        macro_metrics = macro_average(per_sp_metrics)
        cls_metrics = classification_metrics(sp_v, cls_v, num_species)
        
        val_spearman = macro_metrics.get("spearman", 0.0)
        val_pearson = macro_metrics.get("pearson", 0.0)
        val_mse = macro_metrics.get("mse", float("inf"))
        val_acc = cls_metrics.get("accuracy", 0.0)
        
        print(f"Train - Loss: {train_loss:.4f}, Reg: {train_reg:.4f}, Cls: {train_cls:.4f}")
        print(f"Val - Spearman: {val_spearman:.4f}, Pearson: {val_pearson:.4f}, MSE: {val_mse:.4f}, Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_spearman': val_spearman,
                'val_pearson': val_pearson,
                'val_mse': val_mse,
                'val_acc': val_acc,
            }, os.path.join(args.out_dir, 'best.pt'))
            print(f"  -> New best model saved (Spearman: {val_spearman:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
    
    print(f"\nTraining completed. Best validation Spearman: {best_val_spearman:.4f}")


if __name__ == "__main__":
    main()