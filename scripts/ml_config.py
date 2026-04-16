class MLConfig:
    # ==========================================
    # SINR Prediction Module (ML Model Hyperparameters)
    # ==========================================
    RANDOM_SEED = 42
    WINDOW_SIZE = 16          # Timesteps for CNN/DNN sliding windows
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    L2_REGULARIZATION = 0.00001
    DROPOUT_RATE = 0.2
    EARLY_STOPPING_PATIENCE = 8

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
    TARGET_THROUGHPUT_MBPS = 1.0  # Ablation variable: e.g., 1.0 Mbps or 10.0 Mbps
    CONSTRAINT_TYPE = 'AR'        # Default constraint applied: 'AR' or 'PCR'
    MODELS_TO_EVALUATE = ['gru', 'lstm', 'cnn', 'dnn']