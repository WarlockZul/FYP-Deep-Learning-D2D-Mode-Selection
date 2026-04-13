import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingModuleSource]
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

# Build the LSTM model
def build_lstm_model(input_shape):
    """
    Constructs the LSTM model based on these specifications:
    - 2 LSTM Layers
    - 128 Hidden Units
    - Dropout 0.2
    """
    # L2 Regularization value
    reg_val = MLConfig.L2_REGULARIZATION
    
    model = Sequential([
        # Layer 1: LSTM
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(), 
        Dropout(MLConfig.DROPOUT_RATE),        
        
        # Layer 2: LSTM
        LSTM(128, return_sequences=True),
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

# Main training function
def main():
    X_train, y_train, X_val, y_val = load_processed_data(TARGET_MODE)
    
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    
    # Build the LSTM Model
    model = build_lstm_model(input_shape)
    model.summary() 
    
    # Set up Callbacks for Training 
    save_dir = f"models/{TARGET_MODE}/lstm"
    os.makedirs(save_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=MLConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ModelCheckpoint(f"{save_dir}/lstm_model.keras", monitor='val_mae', save_best_only=True),
        CSVLogger(f"{save_dir}/lstm_training_log.csv", separator=',', append=False)
    ]
    
    print("\nStarting LSTM Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MLConfig.EPOCHS,
        batch_size=MLConfig.BATCH_SIZE,     
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history, TARGET_MODE)
    print(f"\nTraining Complete. Best model saved to '{save_dir}/lstm_model.keras'")

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
    save_dir = f"models/{mode}/lstm"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/lstm_training_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()