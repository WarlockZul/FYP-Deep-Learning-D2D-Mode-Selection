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

# Function to ensure deterministic/constant outputs for every run
def set_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    # Force TensorFlow to use deterministic operations where possible
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Call the function immediately
set_seeds(42)

# Load the processed data (train and validation sets) from data/model_ready/
def load_processed_data():
    base_path = "data/model_ready"
    
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
    Constructs the LSTM model based on FYP Proposal:
    - 2 LSTM Layers
    - 128 Hidden Units
    - Dropout 0.2
    """
    # L2 Regularization value
    reg_val = 0.00001
    
    model = Sequential([
        # Layer 1: LSTM
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(), 
        Dropout(0.2),        
        
        # Layer 2: LSTM
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output Layer
        Dense(1, activation='linear', kernel_regularizer=l2(reg_val))
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  
        metrics=['mae'] 
    )
    
    return model

# Main training function
def main():
    X_train, y_train, X_val, y_val = load_processed_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    
    # Build the LSTM Model
    model = build_lstm_model(input_shape)
    model.summary() 
    
    # Set up Callbacks for Training (Updated to lstm folder)
    os.makedirs("models/lstm", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint("models/lstm/lstm_model.keras", monitor='val_mae', save_best_only=True),
        CSVLogger("models/lstm/lstm_training_log.csv", separator=',', append=False)
    ]
    
    print("\nStarting LSTM Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,          
        batch_size=64,      
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history)
    print("\nTraining Complete. Best model saved to 'models/lstm/lstm_model.keras'")

def plot_training_history(history):
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(mae) + 1)

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mae, 'bo-', label='Training MAE')
    plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
    plt.title('LSTM: Mean Absolute Error')
    plt.ylabel('Error (dB)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('LSTM: Mean Squared Error (Loss)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()