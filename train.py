"""
Training script for climate forecasting models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.resnet import ClimateResNet
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    NUM_EPOCHS,
    WARM_RESTART_PERIOD,
    L1_REG,
    L2_REG,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    FEATURE_SET_1,
    FEATURE_SET_2,
    LOSS_FUNCTIONS,
    MODEL_SAVE_DIR,
    RANDOM_SEED
)

class ClimateModel(pl.LightningModule):
    def __init__(self, input_channels, output_channels, loss_fn='mse'):
        """
        Initialize the climate model.
        
        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            loss_fn (str): Loss function to use ('mse' or 'cross_entropy')
        """
        super().__init__()
        self.model = ClimateResNet(input_channels, output_channels)
        self.loss_fn = loss_fn
        
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:  # cross_entropy
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Add regularization
        l1_reg = torch.tensor(0., device=self.device)
        l2_reg = torch.tensor(0., device=self.device)
        
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)
        
        loss += L1_REG * l1_reg + L2_REG * l2_reg
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=WARM_RESTART_PERIOD,
            T_mult=1,
            eta_min=MIN_LEARNING_RATE
        )
        return [optimizer], [scheduler]

def prepare_data(data_dict, feature_set):
    """
    Prepare data for training.
    
    Args:
        data_dict (dict): Dictionary of data
        feature_set (list): List of features to use
        
    Returns:
        tuple: (X, y) where X is input data and y is target data
    """
    X = np.stack([data_dict[var] for var in feature_set], axis=1)
    y = data_dict['precipitation']
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_model(feature_set, loss_fn, train_data, val_data):
    """
    Train a model with specified features and loss function.
    
    Args:
        feature_set (list): List of features to use
        loss_fn (str): Loss function to use
        train_data (dict): Training data
        val_data (dict): Validation data
        
    Returns:
        ClimateModel: Trained model
    """
    # Prepare data
    X_train, y_train = prepare_data(train_data, feature_set)
    X_val, y_val = prepare_data(val_data, feature_set)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model and training components
    model = ClimateModel(INPUT_CHANNELS, OUTPUT_CHANNELS, loss_fn)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR,
        filename=f'model_{"_".join(feature_set)}_{loss_fn}',
        save_top_k=1,
        monitor='val_loss'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model

def main():
    """
    Main training pipeline.
    """
    # Set random seed
    pl.seed_everything(RANDOM_SEED)
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Load processed data
    processed_dir = os.path.join('data', 'processed')
    train_data = np.load(os.path.join(processed_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(processed_dir, 'val_data.npy'), allow_pickle=True).item()
    
    # Train models for each feature set and loss function
    for feature_set in [FEATURE_SET_1, FEATURE_SET_2]:
        for loss_fn in LOSS_FUNCTIONS:
            print(f"\nTraining model with features {feature_set} and loss function {loss_fn}")
            model = train_model(feature_set, loss_fn, train_data, val_data)
            print(f"Model saved to {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main() 