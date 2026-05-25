import os
import random
import numpy as np
import tensorflow as tf

class MLConfig:
    # ==========================================
    # SINR Prediction Module (ML Model Hyperparameters)
    # ==========================================
    RANDOM_SEED = 42
    # WINDOW_SIZE = 16          # Timesteps for CNN/DNN sliding windows
    BATCH_SIZE = 16
    EPOCHS = 150
    LEARNING_RATE = 0.0005
    L2_REGULARIZATION = 0.00001
    DROPOUT_RATE = 0.2
    EARLY_STOPPING_PATIENCE = 25

    # ==========================================
    # Error Analysis Module (Statistical Parameters)
    # ==========================================
    KDE_BANDWIDTH = 1.0       # 'h' parameter for Gaussian smoothing
    CI_LOWER_PCT = 2.5        # Lower bound percentile (for 95% Confidence Interval)
    CI_UPPER_PCT = 97.5       # Upper bound percentile (for 95% Confidence Interval)

    # ==========================================
    # Thresholding Module (System Parameters & Constraints)
    # ==========================================
    BANDWIDTH_HZ = 100e6      # System bandwidth (100 MHz)
    
    # Average Reliability (AR) Constraint
    AR_EPSILON = 0.7          # Maximum allowed switching probability 
    
    # Probabilistic Constraint Reliability (PCR)
    PCR_XI = 0.05             # Outage probability limit
    PCR_CONFIDENCE = 0.95     # Confidence level for the Beta distribution

    # ==========================================
    # System Evaluation & Online Selector
    # ==========================================
    TARGET_THROUGHPUT_MBPS = {
        'preprocessed_paper': 50.0,   # High threshold to force handovers in clean environment
        'preprocessed_proposal': 1.0  # Lower threshold for heavy interference environment
    }
    # CONSTRAINT_TYPE = 'AR'        # Default constraint applied: 'AR' or 'PCR'
    MODELS_TO_EVALUATE = ['gru', 'lstm', 'cnn', 'dnn']
    BASELINE_SINR_THRESHOLD_DB = 0  # SINR threshold in dB for the baseline policy

    # ==========================================
    # Ablation Study Parameters
    # ==========================================

    WINDOW_SIZE = int(os.environ.get('ML_WINDOW_SIZE', 16))         # Timesteps for CNN/DNN sliding windows
    CONSTRAINT_TYPE = os.environ.get('ML_CONSTRAINT_TYPE', 'AR')    # 'AR' or 'PCR'
    USE_KDE = os.environ.get('ML_USE_KDE', 'True') == 'True'        # Whether to apply KDE smoothing in error analysis (True/False)

    # Grab the seed from the pipeline, default to 42 if running manually
    RANDOM_SEED = int(os.environ.get('ML_SEED', 42))
    
    # Master Folder Routing (if there are no ablation experiments, it will default to a standard name based on the seed)
    EXPERIMENT_NAME = os.environ.get('ML_EXPERIMENT_NAME', f'run_seed_{RANDOM_SEED}')

# 1. Enforce Random Seeds Globally
os.environ['PYTHONHASHSEED'] = str(MLConfig.RANDOM_SEED)
random.seed(MLConfig.RANDOM_SEED)
np.random.seed(MLConfig.RANDOM_SEED)
tf.random.set_seed(MLConfig.RANDOM_SEED)

# 2. Force Colab to use the GPU & prevent memory crashes
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✅ GPU is active! (Running Seed: {MLConfig.RANDOM_SEED})")
    except RuntimeError as e:
        print(f"⚠️ GPU Memory Error: {e}")
else:
    print(f"⚠️ No GPU found. Training on CPU. (Running Seed: {MLConfig.RANDOM_SEED})")