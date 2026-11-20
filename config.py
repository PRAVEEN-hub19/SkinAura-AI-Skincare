"""
config.py
----------
Central configuration for the SkinAura project.
Edit the paths here once you have downloaded the dataset from Kaggle.
"""

import os

# ---------------------------------------------------------------------------
# DATASET SETTINGS
# ---------------------------------------------------------------------------
# Download the dataset from:
# https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset
#
# After downloading and extracting, arrange the folder like this
# (the Kaggle download already comes close to this structure -
#  just make sure it matches, or edit DATASET_DIR below):
#
# dataset/
#   train/
#       oily/
#       dry/
#       normal/
#   test/
#       oily/
#       dry/
#       normal/
#
# If your extracted folder uses different casing (Oily/Dry/Normal), that's
# fine too - flow_from_directory reads folder names automatically.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EXTRACTED_DATASET_DIR = os.path.join(BASE_DIR, "Oily-Dry-Skin-Types")
DATASET_DIR = (
	EXTRACTED_DATASET_DIR
	if os.path.isdir(EXTRACTED_DATASET_DIR)
	else DEFAULT_DATASET_DIR
)  # root dataset folder
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
PREPARED_DATASET_DIR = os.path.join(MODEL_DIR, "prepared_stratified_data")
PREPARED_TRAIN_DIR = os.path.join(PREPARED_DATASET_DIR, "train")
PREPARED_VALIDATION_DIR = os.path.join(PREPARED_DATASET_DIR, "validation")
PREPARED_TEST_DIR = os.path.join(PREPARED_DATASET_DIR, "test")
MODEL_PATH = os.path.join(MODEL_DIR, "skinaura_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")

# ---------------------------------------------------------------------------
# IMAGE / TRAINING SETTINGS
# Kept deliberately small so the model trains on a Windows machine with
# an i5 CPU and 4 GB RAM, with no GPU required.
# ---------------------------------------------------------------------------
IMG_SIZE = (160, 160)    # MobileNetV2's compact transfer-learning input size
BATCH_SIZE = 16          # small batch size to keep RAM usage low
EPOCHS = 30
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
DATA_SPLIT_SEED = 42
LEARNING_RATE = 1e-3

# The three skin type classes handled by this project.
CLASS_NAMES = ["dry", "normal", "oily"]  # alphabetical - matches Keras default sort
