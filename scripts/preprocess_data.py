import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import the configuration file from Simulator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.config import SimulationConfig

def preprocess_data():
    # Load simulation data from data/ folder
    input_path = SimulationConfig.OUTPUT_FILE
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} not found.")
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Sort to ensure time order is correct for rolling calculations
    df = df.sort_values(by=['episode_id', 'timestamp'])

    # Perform Feature Engineering
    # - For rolling means & std devs: Window size = 5 timesteps (5 seconds)
    # - Group by episode_id to avoid leakage between episodes
    print("Performing Feature Engineering (Lags, Rolling Means)...")
    df['sinr_d2d_mean_5s'] = df.groupby('episode_id')['sinr_d2d_db'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['sinr_d2d_std_5s'] = df.groupby('episode_id')['sinr_d2d_db'].rolling(5, min_periods=1).std().fillna(0).reset_index(0, drop=True)
    df['throughput_rolling_mean'] = df.groupby('episode_id')['throughput_d2d_mbps'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # - For lagged Features: Create lags for SINR and Interference
    # - Lags: 1, 2, 4, 8, 16 seconds
    # - Fill NaNs with 0 (indicating no prior data)
    lags = [1, 2, 4, 8, 16]
    for lag in lags:
        df[f'sinr_lag_{lag}'] = df.groupby('episode_id')['sinr_d2d_db'].shift(lag).fillna(0)
        df[f'interf_lag_{lag}'] = df.groupby('episode_id')['interference_dbm'].shift(lag).fillna(0)
    
    # Define the full list of features (X)
    # Add the generated lag names to the list dynamically
    features = [
        'sinr_d2d_db', 'sinr_cell_db', 
        'throughput_d2d_mbps', 'throughput_cell_mbps', 
        'distance_tx_rx', 'distance_bs_rx',
        'interference_dbm',
        'sinr_d2d_mean_5s', 'sinr_d2d_std_5s', 'throughput_rolling_mean'
    ]
    for lag in lags:
        features.append(f'sinr_lag_{lag}')
        features.append(f'interf_lag_{lag}')
    
    print(f"Total Features Selected: {len(features)}")
    print(f"Feature List: {features}")
    
    # Prepare labels (Y)
    target = 'optimal_mode'
    df['label'] = df[target].apply(lambda x: 1 if x == 'D2D' else 0)
    
    # NOTE: Save intermediate CSV to check it in Excel
    debug_path = "data/processed_debug.csv"
    df.to_csv(debug_path, index=False)
    print(f"Debug file saved to {debug_path} (Open in Excel to verify features)")

    # Normalize features using Min-Max Scaling
    print("Normalizing features...")
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save Scaler for future use during DL model inference
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Reshape data into sequences for DL model
    print("Reshaping data into sequences...")
    num_episodes = SimulationConfig.NUM_EPISODES
    steps_per_episode = SimulationConfig.STEPS_PER_EPISODE
    num_features = len(features)
    
    # Convert to numpy arrays to facilitate reshaping 
    X_data = np.asarray(df[features].to_numpy(dtype=float))
    y_data = np.asarray(df['label'].to_numpy(dtype=int))
    
    try:
        X_sequenced = X_data.reshape(num_episodes, steps_per_episode, num_features)
        y_sequenced = y_data.reshape(num_episodes, steps_per_episode, 1)
    except ValueError as e:
        print(f"Reshape Error: {e}")
        return

    print(f"Data Reshaped: X={X_sequenced.shape}, y={y_sequenced.shape}")
    
    # Split data into Training (70%), Validation (15%), and Testing (15%)
    print("Splitting data into Training, Validation, and Testing sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sequenced, y_sequenced, test_size=0.30, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )
    
    print(f"Training Sets (70%):   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Sets (15%): X={X_val.shape},   y={y_val.shape}")
    print(f"Testing Sets (15%):    X={X_test.shape},  y={y_test.shape}")
    
    # Save .npy files
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy", X_val)   # <--- New Validation File
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)   # <--- New Validation File
    np.save("data/processed/y_test.npy", y_test)
    
    print("Processed data saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()