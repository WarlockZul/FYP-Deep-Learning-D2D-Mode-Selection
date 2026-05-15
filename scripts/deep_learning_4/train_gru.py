import os
import random
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tensorflow.keras.regularizers import l2 # pyright: ignore[reportMissingModuleSource, reportMissingImports]

# Force TensorFlow to allocate GPU memory dynamically
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ GPU is active and ready for training!")
    except:
        print("⚠️ Failed to initialize GPU memory.")
else:
    print("❌ No GPU found. Defaulting to CPU.")
    
current_dir = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(current_dir, ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))

from ml_config import MLConfig

# Function to ensure deterministic/constant outputs for every run
def set_seeds(seed_value=MLConfig.RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    # Force TensorFlow to use deterministic operations where possible
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Call the function immediately
set_seeds(MLConfig.RANDOM_SEED)

# Load the processed data (train and validation sets) from data/model_ready/
def load_processed_data(dataset_folder, mode):
    base_path = os.path.join(PROJECT_ROOT, "data", dataset_folder, mode)
    
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    X_val   = np.load(os.path.join(base_path, "X_val.npy"))
    y_val   = np.load(os.path.join(base_path, "y_val.npy"))
    
    print(f"Loaded Data from {dataset_folder}/{mode}:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape}") 
    
    return X_train, y_train, X_val, y_val

# Build the GRU model
def build_gru_model(input_shape):
    """
    Constructs the GRU model based on these specifications:
    - 2 GRU Layers
    - 64 Hidden Units
    - Dropout 0.2
    """
    # L2 Regularization value
    reg_val = MLConfig.L2_REGULARIZATION
    
    model = Sequential([
        # Layer 1: GRU
        GRU(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(), 
        Dropout(MLConfig.DROPOUT_RATE),
        
        # Layer 2: GRU
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(MLConfig.DROPOUT_RATE),
        
        # Output Layer
        Dense(1, activation='linear', kernel_regularizer=l2(reg_val))
    ])
    
    # Compile the model by defining optimizer, loss function, and metrics
    model.compile(
        optimizer=Adam(learning_rate=MLConfig.LEARNING_RATE),
        loss='mse',  
        metrics=['mae'] 
    )
    
    return model

# Plot accuracy and loss over time (epochs)
def plot_training_history(history, dataset_folder, mode):
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(mae) + 1)

    plt.figure(figsize=(14, 5))
    
    # Plot MAE of Loss (dB) vs Epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mae, 'bo-', label='Training MAE')
    plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
    plt.title(f'{mode.upper()} Model: Mean Absolute Error')
    plt.ylabel('Error (dB)')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot MSE of Loss (dB) vs Epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'{mode.upper()} Model: Mean Squared Error (Loss)')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    # Save graph dynamically based on dataset (Proposed or Research Paper) and mode (D2D or Cellular)
    save_dir = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, mode, "gru")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "gru_training_curve.png"), dpi=300)
    plt.close()

# Main training function
def main():
    datasets = ['preprocessed_paper', 'preprocessed_proposal']
    modes = ['d2d', 'cellular']

    for dataset_folder in datasets:
        for mode in modes:
            print(f"\n{'='*50}")
            print(f"🚀 TRAINING GRU | Dataset: {dataset_folder} | Mode: {mode.upper()}")
            print(f"{'='*50}")

            # Load data from processed files
            X_train, y_train, X_val, y_val = load_processed_data(dataset_folder, mode)
            
            # Check input shape
            # shape[1]: Time steps
            # shape[2]: Features
            input_shape = (X_train.shape[1], X_train.shape[2]) 
            
            # Build the GRU Model
            model = build_gru_model(input_shape)
            model.summary() 
            
            # Set up Callbacks for Training 
            save_dir = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, mode, "gru")
            os.makedirs(save_dir, exist_ok=True)
            callbacks = [   
                EarlyStopping(monitor='val_loss', patience=MLConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
                ModelCheckpoint(os.path.join(save_dir, "gru_model.keras"), monitor='val_mae', save_best_only=True),
                CSVLogger(os.path.join(save_dir, "gru_training_log.csv"), separator=',', append=False)
            ]
            
            # Train the GRU model
            print("\nStarting GRU Training...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=MLConfig.EPOCHS,
                batch_size=MLConfig.BATCH_SIZE,   
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            plot_training_history(history, dataset_folder, mode)
            print(f"\nTraining Complete. Best model saved to '{save_dir}/gru_model.keras'")

if __name__ == "__main__":
    main()