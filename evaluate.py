"""
Evaluation script for climate forecasting models.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from models.resnet import ClimateResNet
from utils.helpers import (
    compute_metrics,
    plot_confusion_matrix,
    plot_prediction_vs_actual,
    plot_feature_correlations,
    save_results
)
from config import (
    FEATURE_SET_1,
    FEATURE_SET_2,
    LOSS_FUNCTIONS,
    MODEL_SAVE_DIR,
    PLOT_SAVE_DIR,
    RESULTS_DIR,
    BATCH_SIZE
)

def load_model(model_path, input_channels, output_channels):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the saved model
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        
    Returns:
        ClimateResNet: Loaded model
    """
    model = ClimateResNet(input_channels, output_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, test_data, feature_set):
    """
    Evaluate model on test data.
    
    Args:
        model (ClimateResNet): Trained model
        test_data (dict): Test data
        feature_set (list): List of features used
        
    Returns:
        tuple: (y_true, y_pred, metrics)
    """
    # Prepare test data
    X_test = np.stack([test_data[var] for var in feature_set], axis=1)
    X_test = torch.FloatTensor(X_test)
    y_test = test_data['precipitation']
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    return y_test, y_pred, metrics

def main():
    """
    Main evaluation pipeline.
    """
    # Create output directories
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load test data
    test_data = np.load(
        os.path.join('data', 'processed', 'test_data.npy'),
        allow_pickle=True
    ).item()
    
    # Evaluate models
    results = {}
    
    for feature_set in [FEATURE_SET_1, FEATURE_SET_2]:
        for loss_fn in LOSS_FUNCTIONS:
            print(f"\nEvaluating model with features {feature_set} and loss function {loss_fn}")
            
            # Load model
            model_path = os.path.join(
                MODEL_SAVE_DIR,
                f'model_{"_".join(feature_set)}_{loss_fn}.pt'
            )
            model = load_model(model_path, len(feature_set), 1)
            
            # Evaluate model
            y_true, y_pred, metrics = evaluate_model(model, test_data, feature_set)
            
            # Save results
            model_name = f"{'_'.join(feature_set)}_{loss_fn}"
            results[model_name] = metrics
            
            # Generate plots
            plot_confusion_matrix(
                y_true,
                y_pred,
                os.path.join(PLOT_SAVE_DIR, f"{model_name}_confusion_matrix.png")
            )
            
            plot_prediction_vs_actual(
                y_true,
                y_pred,
                os.path.join(PLOT_SAVE_DIR, f"{model_name}_prediction_vs_actual.png")
            )
            
            plot_feature_correlations(
                test_data,
                feature_set,
                os.path.join(PLOT_SAVE_DIR, f"{model_name}_feature_correlations.png")
            )
    
    # Save all results
    save_results(
        results,
        os.path.join(RESULTS_DIR, 'evaluation_results.txt')
    )
    
    # Print best model
    best_model = max(
        results.items(),
        key=lambda x: x[1]['f1']
    )
    print(f"\nBest model: {best_model[0]}")
    print(f"F1 Score: {best_model[1]['f1']:.3f}")
    print(f"Precision: {best_model[1]['precision']:.3f}")
    print(f"Recall: {best_model[1]['recall']:.3f}")

if __name__ == "__main__":
    main() 