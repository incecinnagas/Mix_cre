#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mix_cre Inference utilities"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from model import Evo2MixModel
from dataset import tokenize_sequence, reverse_complement_tokens


def load_model(checkpoint_path: str, device: Optional[torch.device] = None) -> Evo2MixModel:
    """Load trained model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to infer model parameters from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer num_species from classification head
    num_species = 4  # default
    for key in state_dict.keys():
        if 'cls_head' in key and 'weight' in key:
            if state_dict[key].dim() == 2:
                num_species = state_dict[key].shape[0]
                break
    
    # Infer d_model from embedding or first linear layer
    d_model = 1024  # default
    for key in state_dict.keys():
        if 'token_emb.weight' in key:
            d_model = state_dict[key].shape[1]
            break
        elif 'backbone.token_emb.weight' in key:
            d_model = state_dict[key].shape[1]
            break
    
    # Create model with inferred parameters
    model = Evo2MixModel(
        num_species=num_species,
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        tail="mamba",
        tail_layers=3,
        pool="attn",
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def predict_sequence(
    model: Evo2MixModel, 
    sequence: str, 
    seq_len: int = 3001,
    rc_pool: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Predict promoter activity for a DNA sequence"""
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize sequence
    tokens = tokenize_sequence(sequence, seq_len)
    tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)  # [1, L]
    
    with torch.no_grad():
        y_hat, logits = model(tokens_tensor)
        
        if rc_pool:
            # Reverse complement pooling
            rc_tokens = reverse_complement_tokens(tokens)
            rc_tensor = torch.from_numpy(rc_tokens).unsqueeze(0).to(device)
            y_hat_rc, logits_rc = model(rc_tensor)
            
            # Average predictions
            y_hat = 0.5 * (y_hat + y_hat_rc)
            logits = 0.5 * (logits + logits_rc)
        
        # Convert to numpy
        y_pred = float(y_hat.squeeze().cpu().item())
        species_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        predicted_species = int(torch.argmax(logits, dim=-1).squeeze().cpu().item())
    
    return {
        'y_pred': y_pred,
        'species_probs': species_probs.tolist(),
        'predicted_species': predicted_species,
        'input_length': len(sequence.replace('\n', '').replace(' ', '')),
    }


def batch_predict(
    model: Evo2MixModel,
    sequences: list,
    seq_len: int = 3001,
    batch_size: int = 32,
    rc_pool: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Predict promoter activity for multiple sequences"""
    if device is None:
        device = next(model.parameters()).device
    
    all_preds = []
    all_species_probs = []
    all_species_preds = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            # Tokenize batch
            batch_tokens = []
            for seq in batch_seqs:
                tokens = tokenize_sequence(seq, seq_len)
                batch_tokens.append(tokens)
            
            batch_tensor = torch.from_numpy(np.stack(batch_tokens)).to(device)
            
            # Forward pass
            y_hat, logits = model(batch_tensor)
            
            if rc_pool:
                # Reverse complement pooling
                rc_batch = []
                for tokens in batch_tokens:
                    rc_tokens = reverse_complement_tokens(tokens)
                    rc_batch.append(rc_tokens)
                
                rc_tensor = torch.from_numpy(np.stack(rc_batch)).to(device)
                y_hat_rc, logits_rc = model(rc_tensor)
                
                # Average predictions
                y_hat = 0.5 * (y_hat + y_hat_rc)
                logits = 0.5 * (logits + logits_rc)
            
            # Collect results
            all_preds.extend(y_hat.cpu().numpy().tolist())
            species_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_species_probs.extend(species_probs.tolist())
            species_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_species_preds.extend(species_preds.tolist())
    
    return {
        'y_pred': all_preds,
        'species_probs': all_species_probs,
        'predicted_species': all_species_preds,
    }