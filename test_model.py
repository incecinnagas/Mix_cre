#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for Mix_cre model"""

import torch
import numpy as np
from model import Evo2MixModel
from dataset import tokenize_sequence

def test_model():
    """Test model forward pass"""
    print("Testing Mix_cre model...")
    
    # Model parameters
    num_species = 4
    d_model = 512  # Smaller for testing
    seq_len = 1000  # Shorter for testing
    batch_size = 2
    
    # Create model
    model = Evo2MixModel(
        num_species=num_species,
        d_model=d_model,
        n_layers=4,  # Fewer layers for testing
        n_heads=8,
        tail="mamba",
        tail_layers=2,
        pool="attn",
        dropout=0.1,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    # Generate random DNA sequence
    bases = ['A', 'T', 'C', 'G', 'N']
    sequences = []
    for _ in range(batch_size):
        seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)
    
    # Tokenize sequences
    tokens_list = []
    for seq in sequences:
        tokens = tokenize_sequence(seq, seq_len)
        tokens_list.append(tokens)
    
    # Convert to tensor
    tokens_tensor = torch.from_numpy(np.stack(tokens_list, axis=0))
    print(f"Input shape: {tokens_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        y_hat, logits = model(tokens_tensor)
    
    print(f"Regression output shape: {y_hat.shape}")
    print(f"Classification output shape: {logits.shape}")
    print(f"Sample regression predictions: {y_hat.numpy()}")
    print(f"Sample classification logits: {logits.numpy()}")
    
    # Test species probabilities
    species_probs = torch.softmax(logits, dim=-1)
    print(f"Species probabilities: {species_probs.numpy()}")
    
    print("✅ Model test passed!")
    return True

def test_tokenization():
    """Test DNA sequence tokenization"""
    print("\nTesting tokenization...")
    
    test_seq = "ATCGATCGNNNN"
    seq_len = 20
    
    tokens = tokenize_sequence(test_seq, seq_len)
    print(f"Input sequence: {test_seq}")
    print(f"Tokenized: {tokens}")
    print(f"Expected mapping: A=0, T=3, C=1, G=2, N=4")
    
    # Verify mapping
    expected = [0, 3, 1, 2, 0, 3, 1, 2, 4, 4, 4, 4] + [4] * 8  # Padded with N
    assert len(tokens) == seq_len
    assert np.array_equal(tokens[:12], expected[:12])
    
    print("✅ Tokenization test passed!")
    return True

if __name__ == "__main__":
    try:
        test_tokenization()
        test_model()
        print("\n🎉 All tests passed! The model is ready to use.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()