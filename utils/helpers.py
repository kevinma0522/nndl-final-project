"""
Helper functions for the climate forecasting project.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }

def plot_training_history(history, save_path):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results(results, save_path):
    """
    Save results to file.
    
    Args:
        results (dict): Results dictionary
        save_path (str): Path to save the results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

def load_results(load_path):
    """
    Load results from file.
    
    Args:
        load_path (str): Path to load the results from
        
    Returns:
        dict: Results dictionary
    """
    results = {}
    with open(load_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            results[key] = float(value)
    return results

def plot_feature_correlations(data, feature_set, save_path):
    """
    Plot feature correlations.
    
    Args:
        data (dict): Dictionary of data
        feature_set (list): List of features to plot
        save_path (str): Path to save the plot
    """
    # Create correlation matrix
    corr_matrix = np.corrcoef(
        np.stack([data[var] for var in feature_set], axis=1).T
    )
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        xticklabels=feature_set,
        yticklabels=feature_set
    )
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, save_path):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 