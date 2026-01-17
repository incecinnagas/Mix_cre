#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mix_cre Example Usage"""

import torch
from inference import load_model, predict_sequence

def main():
    # Example DNA sequence (3000bp promoter region)
    example_sequence = "ATCGATCG" * 375  # Simple repeat pattern for demo
    
    # Load trained model
    try:
        model = load_model('checkpoints/best.pt')
        print("Model loaded successfully!")
        
        # Predict promoter activity
        result = predict_sequence(model, example_sequence)
        
        print(f"\nPrediction Results:")
        print(f"Activity Score (Z-score): {result['y_pred']:.4f}")
        print(f"Species Probabilities: {[f'{p:.3f}' for p in result['species_probs']]}")
        print(f"Predicted Species ID: {result['predicted_species']}")
        print(f"Input Length: {result['input_length']} bp")
        
        # Interpret activity level
        activity_score = result['y_pred']
        if activity_score > 1.0:
            level = "High Activity"
        elif activity_score > 0:
            level = "Medium Activity"
        elif activity_score > -1.0:
            level = "Low Activity"
        else:
            level = "Very Low Activity"
        
        print(f"Activity Level: {level}")
        
    except FileNotFoundError:
        print("Model checkpoint not found. Please train the model first or download pre-trained weights.")
        print("Run: python train.py --csv_path your_data.csv")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()