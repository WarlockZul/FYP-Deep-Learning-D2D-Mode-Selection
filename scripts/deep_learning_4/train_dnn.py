import os
import random
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.regularizers import l2 # pyright: ignore[reportMissingModuleSource]

current_dir = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(current_dir, ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))

from ml_config import MLConfig

# Load the processed data (train and validation sets) from data/model_ready/
def load_processed_data(dataset_folder, mode):
    base_path = os.path.join(PROJECT_ROOT, "data", dataset_folder, mode)
    
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    X_val   = np.load(os.path.join(base_path, "X_val.npy"))
    y_val   = np.load(os.path.join(base_path, "y_val.npy"))
    
    print(f"Loaded Data:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape}") 
    
    return X_train, y_train, X_val, y_val

# Create sliding windows for CNN/DNN input to predict one step at a time from the timestep(second) sequences
# Converts (Episodes, Timesteps, Features) into (Total_Windows, Window_Size, Features)
def create_sliding_windows(X, y, window_size=MLConfig.WINDOW_SIZE):
    X_win, y_win = [], []
    for i in range(X.shape[0]):

        # Slide a window across the 300 timesteps
        for t in range(X.shape[1] - window_size + 1):
            X_win.append(X[i, t:t+window_size, :])

            # The target is the label of the last timestep in the window
            y_win.append(y[i, t+window_size-1, :])
            
    return np.array(X_win), np.array(y_win)

# Build the DNN model
def build_dnn_model(input_shape):
    """
    Constructs the DNN model based on these specifications:
    - Flatten Layer
    - 3 Fully Connected Layers (256 -> 128 -> 64)
    - ReLU Activation
    """
    reg_val = MLConfig.L2_REGULARIZATION
    
    model = Sequential([
        # RESTORED: Flatten the 2D window into a 1D vector
        Flatten(input_shape=input_shape),
        
        # Layer 1: Dense Feed-Forward (256 Neurons)
        Dense(256, activation='relu'),
        BatchNormalization(), 
        Dropout(MLConfig.DROPOUT_RATE),        
        
        # Layer 2: Dense Feed-Forward (128 Neurons)
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(MLConfig.DROPOUT_RATE),
        
        # Layer 3: Dense Feed-Forward (64 Neurons)
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(MLConfig.DROPOUT_RATE),
        
        # Output Layer
        Dense(1, activation='linear', kernel_regularizer=l2(reg_val))
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=MLConfig.LEARNING_RATE),
        loss='mse', 
        metrics=['mae']
    )
    return model

def plot_training_history(history, dataset_folder, mode):
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(mae) + 1)

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mae, 'bo-', label='Training MAE')
    plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
    plt.title(f'{mode.upper()} Model: Mean Absolute Error')
    plt.ylabel('Error (dB)')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'{mode.upper()} Model: Mean Squared Error (Loss)')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    # Save graph dynamically based on dataset (Proposed or Research Paper) and mode (D2D or Cellular)
    save_dir = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, mode, "dnn")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "dnn_training_curve.png"), dpi=300)
    plt.close()

# Main training function
def main():
    dataset_folder = "preprocessed_proposal" 
    modes = ['d2d', 'cellular'] 

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"🚀 TRAINING DNN | Dataset: {dataset_folder} | Mode: {mode.upper()}")
        print(f"{'='*50}")

        X_train_raw, y_train_raw, X_val_raw, y_val_raw = load_processed_data(dataset_folder, mode)
        
        print("\nReformatting data into sliding windows...")
        X_train, y_train = create_sliding_windows(X_train_raw, y_train_raw, window_size=MLConfig.WINDOW_SIZE)
        X_val, y_val = create_sliding_windows(X_val_raw, y_val_raw, window_size=MLConfig.WINDOW_SIZE)
        
        print(f"Windowed Train Data: X={X_train.shape}, y={y_train.shape}")
        
        input_shape = (X_train.shape[1], X_train.shape[2]) 
        model = build_dnn_model(input_shape)
        model.summary()
        
        # Set up Callbacks for Training 
        save_dir = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, mode, "dnn")
        os.makedirs(save_dir, exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=MLConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ModelCheckpoint(os.path.join(save_dir, "dnn_model.keras"), monitor='val_mae', save_best_only=True),
            CSVLogger(os.path.join(save_dir, "dnn_training_log.csv"), separator=',', append=False)
        ]
        
        print("\nStarting DNN Training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MLConfig.EPOCHS,
            batch_size=MLConfig.BATCH_SIZE,    
            callbacks=callbacks,
            verbose=1
        )
        
        plot_training_history(history, dataset_folder, mode)
        print(f"\nTraining Complete. Best model saved to '{save_dir}/dnn_model.keras'")

if __name__ == "__main__":
    main()