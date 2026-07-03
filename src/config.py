"""
Configuration file for Product Return Analysis System
Contains all project constants, paths, and configuration parameters
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data file paths
DATASET_PATH = DATA_DIR / "returns.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"

# Dataset parameters
N_SAMPLES = 1200
RANDOM_STATE = 42

# Categories and regions
CATEGORIES = ["Electronics", "Fashion", "Home", "Sports"]
REGIONS = ["North", "South", "East", "West"]
RETURN_REASONS = ["Size Issue", "Damaged", "Wrong Item", "Quality Issue", "None"]

# Price ranges by category (min, max)
PRICE_RANGES = {
    "Electronics": (500, 5000),
    "Fashion": (300, 2000),
    "Home": (200, 3000),
    "Sports": (150, 1500)
}

# Return probability by category (base rate)
RETURN_PROB_BY_CATEGORY = {
    "Electronics": 0.15,
    "Fashion": 0.30,
    "Home": 0.18,
    "Sports": 0.12
}

# ML model parameters
TEST_SIZE = 0.2
ML_RANDOM_STATE = 42

# Streamlit configuration
APP_TITLE = "📦 Product Return Analysis & Prediction System"
APP_ICON = "📦"
