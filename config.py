"""
Configuration parameters for the climate forecasting project.
"""

# Data parameters
DATA_DIR = "data"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_LEVELS = 60
MIDDLE_LEVELS = 20  # Number of middle levels to use

# Model parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
WARM_RESTART_PERIOD = 10
L1_REG = 0.001
L2_REG = 0.01

# Model architecture
INPUT_CHANNELS = 6  # Number of input variables
OUTPUT_CHANNELS = 10  # Number of output variables
RESNET_DEPTH = 50

# Feature sets
FEATURE_SET_1 = [
    "cloud_liquid_mixing_ratio",
    "surface_pressure",
    "specific_humidity"
]

FEATURE_SET_2 = [
    "air_temperature",
    "ozone_volume_mixing_ratio",
    "solar_insolation"
]

# Loss functions
LOSS_FUNCTIONS = ["mse", "cross_entropy"]

# SHAP parameters
SHAP_BACKGROUND_SIZE = 100
SHAP_NUM_SAMPLES = 1000

# Paths
MODEL_SAVE_DIR = "models/saved"
PLOT_SAVE_DIR = "plots"
RESULTS_DIR = "results"

# Random seed
RANDOM_SEED = 42 