"""
Interpretability analysis using SHAP values.
"""

import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.resnet import ClimateResNet
from config import (
    SHAP_BACKGROUND_SIZE,
    SHAP_NUM_SAMPLES,
    FEATURE_SET_1,
    FEATURE_SET_2,
    PLOT_SAVE_DIR,
    RESULTS_DIR
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

def prepare_background_data(data, feature_set, n_samples=SHAP_BACKGROUND_SIZE):
    """
    Prepare background data for SHAP analysis.
    
    Args:
        data (dict): Dictionary of data
        feature_set (list): List of features to use
        n_samples (int): Number of background samples
        
    Returns:
        torch.Tensor: Background data
    """
    # Stack features
    X = np.stack([data[var] for var in feature_set], axis=1)
    
    # Randomly sample background data
    indices = np.random.choice(len(X), n_samples, replace=False)
    background = torch.FloatTensor(X[indices])
    
    return background

def compute_deep_shap(model, background, test_data, feature_set):
    """
    Compute Deep SHAP values.
    
    Args:
        model (ClimateResNet): Trained model
        background (torch.Tensor): Background data
        test_data (dict): Test data
        feature_set (list): List of features used
        
    Returns:
        tuple: (shap_values, feature_names)
    """
    # Create DeepExplainer
    explainer = shap.DeepExplainer(model, background)
    
    # Prepare test data
    X_test = np.stack([test_data[var] for var in feature_set], axis=1)
    X_test = torch.FloatTensor(X_test)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)
    
    return shap_values, feature_set

def compute_kernel_shap(model, background, test_data, feature_set):
    """
    Compute Kernel SHAP values.
    
    Args:
        model (ClimateResNet): Trained model
        background (torch.Tensor): Background data
        test_data (dict): Test data
        feature_set (list): List of features used
        
    Returns:
        tuple: (shap_values, feature_names)
    """
    # Create KernelExplainer
    explainer = shap.KernelExplainer(
        lambda x: model(torch.FloatTensor(x)).detach().numpy(),
        background.numpy()
    )
    
    # Prepare test data
    X_test = np.stack([test_data[var] for var in feature_set], axis=1)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(
        X_test,
        nsamples=SHAP_NUM_SAMPLES
    )
    
    return shap_values, feature_set

def compute_weighted_shap(model, background, test_data, feature_set):
    """
    Compute Weighted SHAP values.
    
    Args:
        model (ClimateResNet): Trained model
        background (torch.Tensor): Background data
        test_data (dict): Test data
        feature_set (list): List of features used
        
    Returns:
        tuple: (shap_values, feature_names)
    """
    # Compute standard SHAP values
    shap_values, feature_names = compute_kernel_shap(
        model, background, test_data, feature_set
    )
    
    # Compute feature weights based on model performance
    X_test = np.stack([test_data[var] for var in feature_set], axis=1)
    X_test = torch.FloatTensor(X_test)
    y_test = test_data['precipitation']
    
    # Compute feature importance using permutation importance
    feature_weights = []
    for i, feature in enumerate(feature_set):
        X_permuted = X_test.clone()
        X_permuted[:, i] = torch.randn_like(X_permuted[:, i])
        
        with torch.no_grad():
            original_pred = model(X_test)
            permuted_pred = model(X_permuted)
        
        # Compute weight as inverse of performance degradation
        weight = 1 / (torch.mean((original_pred - permuted_pred) ** 2) + 1e-6)
        feature_weights.append(weight.item())
    
    # Normalize weights
    feature_weights = np.array(feature_weights)
    feature_weights = feature_weights / np.sum(feature_weights)
    
    # Apply weights to SHAP values
    weighted_shap_values = shap_values * feature_weights[:, np.newaxis]
    
    return weighted_shap_values, feature_names

def plot_shap_summary(shap_values, feature_names, title, save_path):
    """
    Plot SHAP summary.
    
    Args:
        shap_values (np.ndarray): SHAP values
        feature_names (list): Feature names
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        show=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(shap_values, feature_names, title, save_path):
    """
    Plot feature importance.
    
    Args:
        shap_values (np.ndarray): SHAP values
        feature_names (list): Feature names
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    # Compute mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_shap
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Main interpretability analysis pipeline.
    """
    # Create output directories
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load test data
    test_data = np.load(
        os.path.join('data', 'processed', 'test_data.npy'),
        allow_pickle=True
    ).item()
    
    # Load best model (using FEATURE_SET_1 and cross_entropy loss)
    model_path = os.path.join(
        'models', 'saved',
        f'model_{"_".join(FEATURE_SET_1)}_cross_entropy.pt'
    )
    model = load_model(model_path, len(FEATURE_SET_1), 1)
    
    # Prepare background data
    background = prepare_background_data(test_data, FEATURE_SET_1)
    
    # Compute SHAP values using different methods
    print("Computing Deep SHAP values...")
    deep_shap_values, feature_names = compute_deep_shap(
        model, background, test_data, FEATURE_SET_1
    )
    
    print("Computing Kernel SHAP values...")
    kernel_shap_values, _ = compute_kernel_shap(
        model, background, test_data, FEATURE_SET_1
    )
    
    print("Computing Weighted SHAP values...")
    weighted_shap_values, _ = compute_weighted_shap(
        model, background, test_data, FEATURE_SET_1
    )
    
    # Plot results
    print("Generating plots...")
    plot_shap_summary(
        deep_shap_values,
        feature_names,
        "Deep SHAP Summary",
        os.path.join(PLOT_SAVE_DIR, "deep_shap_summary.png")
    )
    
    plot_shap_summary(
        kernel_shap_values,
        feature_names,
        "Kernel SHAP Summary",
        os.path.join(PLOT_SAVE_DIR, "kernel_shap_summary.png")
    )
    
    plot_shap_summary(
        weighted_shap_values,
        feature_names,
        "Weighted SHAP Summary",
        os.path.join(PLOT_SAVE_DIR, "weighted_shap_summary.png")
    )
    
    plot_feature_importance(
        deep_shap_values,
        feature_names,
        "Feature Importance (Deep SHAP)",
        os.path.join(PLOT_SAVE_DIR, "deep_shap_importance.png")
    )
    
    plot_feature_importance(
        kernel_shap_values,
        feature_names,
        "Feature Importance (Kernel SHAP)",
        os.path.join(PLOT_SAVE_DIR, "kernel_shap_importance.png")
    )
    
    plot_feature_importance(
        weighted_shap_values,
        feature_names,
        "Feature Importance (Weighted SHAP)",
        os.path.join(PLOT_SAVE_DIR, "weighted_shap_importance.png")
    )
    
    print("Analysis complete! Results saved to", PLOT_SAVE_DIR)

if __name__ == "__main__":
    main() 