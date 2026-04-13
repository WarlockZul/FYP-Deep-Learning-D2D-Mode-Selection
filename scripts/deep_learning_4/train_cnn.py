import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.regularizers import l2 # pyright: ignore[reportMissingModuleSource]
from ml_config import MLConfig

TARGET_MODE = 'cellular'  # Options: 'd2d' or 'cellular'

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
def load_processed_data(mode):
    base_path = f"data/model_ready/{mode}/"
    
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    X_val   = np.load(os.path.join(base_path, "X_val.npy"))
    y_val   = np.load(os.path.join(base_path, "y_val.npy"))
    
    print(f"Loaded Data:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape}") 
    
    return X_train, y_train, X_val, y_val

# Create sliding windows for CNN/DNN input to predict one step at a time from the 300-second sequences
# Converts (Episodes, 300, Features) into (Total_Windows, Window_Size, Features)
def create_sliding_windows(X, y, window_size=MLConfig.WINDOW_SIZE):
    X_win, y_win = [], []
    for i in range(X.shape[0]):

        # Slide a window across the 300 timesteps
        for t in range(X.shape[1] - window_size + 1):
            X_win.append(X[i, t:t+window_size, :])

            # The target is the label of the last timestep in the window
            y_win.append(y[i, t+window_size-1, :])
            
    return np.array(X_win), np.array(y_win)

# Build the CNN model
def build_cnn_model(input_shape):
    """
    Constructs the CNN model based on these specifications:
    - 3 Temporal Conv1D Layers (kernel sizes 7, 5, 3)
    - Global Average Pooling
    - Dense Output
    """
    reg_val = MLConfig.L2_REGULARIZATION
    
    model = Sequential([
        # Layer 1: 1D Convolution (Kernel Size 7)
        Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), 
        Dropout(MLConfig.DROPOUT_RATE),
        
        # Layer 2: 1D Convolution (Kernel Size 5)
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(MLConfig.DROPOUT_RATE),
        
        # Layer 3: 1D Convolution (Kernel Size 3)
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(MLConfig.DROPOUT_RATE),
        
        # RESTORED: Global Average Pooling Layer
        GlobalAveragePooling1D(), 
        
        # Output Layer
        Dense(1, activation='linear', kernel_regularizer=l2(reg_val))
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=MLConfig.LEARNING_RATE),
        loss='mse', 
        metrics=['mae']
    )
    return model

# Main training function
def main():
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = load_processed_data(TARGET_MODE)
    
    # Apply Sliding Window (Window Size = 16 seconds)
    print("\nReformatting data into sliding windows...")
    X_train, y_train = create_sliding_windows(X_train_raw, y_train_raw, window_size=MLConfig.WINDOW_SIZE)
    X_val, y_val = create_sliding_windows(X_val_raw, y_val_raw, window_size=MLConfig.WINDOW_SIZE)
    
    print(f"Windowed Train Data: X={X_train.shape}, y={y_train.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    model = build_cnn_model(input_shape)
    model.summary() 
    
    # Set up Callbacks for Training 
    save_dir = f"models/{TARGET_MODE}/cnn"
    os.makedirs(save_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MLConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ModelCheckpoint(f"{save_dir}/cnn_model.keras", monitor='val_mae', save_best_only=True),
        CSVLogger(f"{save_dir}/cnn_training_log.csv", separator=',', append=False)
    ]
    
    print("\nStarting CNN Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MLConfig.EPOCHS,
        batch_size=MLConfig.BATCH_SIZE,   
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history, TARGET_MODE)
    print(f"\nTraining Complete. Best model saved to '{save_dir}/cnn_model.keras'")

def plot_training_history(history, mode):
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
    plt.legend()
    
    plt.tight_layout()
    
    # Save graph dynamically
    save_dir = f"models/{mode}/cnn"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/cnn_training_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()