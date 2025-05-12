# Transparent Climate Forecasting with SHAP-Driven Interpretability

This repository implements a deep learning approach for climate forecasting with a focus on interpretability using SHAP (SHapley Additive exPlanations) values. The project uses the ClimSim dataset to train ResNet50 models for precipitation prediction and analyzes their interpretability using various SHAP techniques.

## Project Structure

```
.
├── data/                  # Data directory
├── models/               # Model implementations
├── preprocessing/        # Data preprocessing scripts
├── interpretability/     # SHAP and interpretability analysis
├── utils/               # Utility functions
├── config.py            # Configuration parameters
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── requirements.txt     # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the ClimSim dataset and place it in the `data/` directory.

## Usage

1. Preprocess the data:
```bash
python preprocessing/preprocess.py
```

2. Train the model:
```bash
python train.py
```

3. Run interpretability analysis:
```bash
python interpretability/analyze.py
```

## Model Architecture

The project implements four ResNet50 models:
- Two models using cloud liquid mixing ratio, surface pressure, and specific humidity
- Two models using air temperature, ozone mixing ratio, and solar insolation

Each pair of models is trained with either mean-squared error or cross-entropy loss.

## Interpretability Analysis

The project implements three interpretability techniques:
1. DeepExplainer (Deep SHAP)
2. Kernel SHAP
3. WeightedSHAP

These methods are used to analyze feature importance and model predictions for precipitation forecasting.

## Results

The best performing model achieves an F1-score of 0.865 using cross-entropy loss and the first set of input variables (cloud liquid mixing ratio, surface pressure, and specific humidity).

