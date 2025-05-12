"""
Preprocessing script for the ClimSim dataset.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import (
    DATA_DIR,
    NUM_LEVELS,
    MIDDLE_LEVELS,
    FEATURE_SET_1,
    FEATURE_SET_2,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED
)

def load_climsim_data(file_path):
    """
    Load ClimSim dataset from netCDF file.
    
    Args:
        file_path (str): Path to the netCDF file
        
    Returns:
        xarray.Dataset: Loaded dataset
    """
    return xr.open_dataset(file_path)

def preprocess_variables(dataset):
    """
    Preprocess climate variables.
    
    Args:
        dataset (xarray.Dataset): Raw dataset
        
    Returns:
        dict: Preprocessed variables
    """
    # Calculate daily means
    daily_means = dataset.resample(time='D').mean()
    
    # Select middle levels for vertically dependent variables
    middle_levels = slice(NUM_LEVELS//2 - MIDDLE_LEVELS//2, 
                         NUM_LEVELS//2 + MIDDLE_LEVELS//2)
    
    # Process each variable
    processed_data = {}
    
    for var in FEATURE_SET_1 + FEATURE_SET_2:
        if 'level' in dataset[var].dims:
            processed_data[var] = daily_means[var].sel(level=middle_levels)
        else:
            processed_data[var] = daily_means[var]
    
    return processed_data

def normalize_data(data_dict):
    """
    Normalize the data using StandardScaler.
    
    Args:
        data_dict (dict): Dictionary of preprocessed variables
        
    Returns:
        dict: Normalized data and scalers
    """
    scalers = {}
    normalized_data = {}
    
    for var, data in data_dict.items():
        scaler = StandardScaler()
        normalized_data[var] = scaler.fit_transform(data.values.reshape(-1, 1))
        scalers[var] = scaler
    
    return normalized_data, scalers

def split_data(data_dict):
    """
    Split data into train, validation, and test sets.
    
    Args:
        data_dict (dict): Dictionary of normalized variables
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    np.random.seed(RANDOM_SEED)
    
    # Get the length of the data
    n_samples = len(next(iter(data_dict.values())))
    
    # Create indices for splitting
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * TRAIN_RATIO)
    val_size = int(n_samples * VAL_RATIO)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split the data
    train_data = {var: data[train_indices] for var, data in data_dict.items()}
    val_data = {var: data[val_indices] for var, data in data_dict.items()}
    test_data = {var: data[test_indices] for var, data in data_dict.items()}
    
    return train_data, val_data, test_data

def save_processed_data(train_data, val_data, test_data, scalers, output_dir):
    """
    Save processed data and scalers.
    
    Args:
        train_data (dict): Training data
        val_data (dict): Validation data
        test_data (dict): Test data
        scalers (dict): Fitted scalers
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    
    # Save scalers
    import pickle
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

def main():
    """
    Main preprocessing pipeline.
    """
    # Create output directory
    processed_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load data
    print("Loading ClimSim dataset...")
    dataset = load_climsim_data(os.path.join(DATA_DIR, 'climsim.nc'))
    
    # Preprocess variables
    print("Preprocessing variables...")
    processed_data = preprocess_variables(dataset)
    
    # Normalize data
    print("Normalizing data...")
    normalized_data, scalers = normalize_data(processed_data)
    
    # Split data
    print("Splitting data into train/val/test sets...")
    train_data, val_data, test_data = split_data(normalized_data)
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(train_data, val_data, test_data, scalers, processed_dir)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 