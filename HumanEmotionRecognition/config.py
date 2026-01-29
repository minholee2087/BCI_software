import os
from pathlib import Path

# Get the absolute path of the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Define standard data directories relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DATA_DIR = os.path.join(DATA_DIR, "data_processing")
PRETRAINED_MODELS_DIR = os.path.join(DATA_DIR, "pretrained_models")
FINETUNED_MODELS_DIR = os.path.join(DATA_DIR, "finetuned_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Function to check if data exists
def check_data_exists():
    return os.path.exists(INPUT_DATA_DIR) and os.path.exists(PRETRAINED_MODELS_DIR)