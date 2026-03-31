import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import the configuration file from Simulator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from simulator.config import SimulationConfig

def clean_dataset(df):
    print("Cleaning Dataset: Handling Infs, NaNs, and Outliers...")
    
    # 1. Handle Infinite Values
    # Logarithmic formulas (like SINR in dB) can produce -inf if the linear value is 0.
    # Replace inf and -inf with NaN to handle them safely.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Handle Missing Values (NaNs)
    # Forward fill (ffill) uses the previous second's value to patch the hole. 
    # Backfill (bfill) catches any NaNs that might occur at the very first row.
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 3. Handle Extreme Outliers (Clipping)
    # Fast Fading can cause unrealistic massive spikes or deep fades.
    # Clip the top 1% and bottom 1% of values to keep the dataset stable for the neural network.
    cols_to_clip = [
        'sinr_d2d_db', 'sinr_cell_db', 
        'throughput_d2d_mbps', 'throughput_cell_mbps', 
        'interference_dbm'
    ]
    for col in cols_to_clip:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    print("Dataset cleaned: Handled Infs, NaNs, and Outliers.")
    return df

# Generate separate datasets for D2D and Cellular modes with engineered features and labels
def generate_dataset_for_mode(df_raw, mode):
    df = df_raw.copy()
    print(f"\n--- Generating Dataset for {mode.upper()} Mode ---")
    
    if mode == 'D2D':
        target_sinr = 'sinr_d2d_db'
        target_tput = 'throughput_d2d_mbps'
        save_dir = "data/model_ready/d2d"
    else:
        target_sinr = 'sinr_cell_db'
        target_tput = 'throughput_cell_mbps'
        save_dir = "data/model_ready/cellular"

    # Perform Feature Engineering based on mode (D2D or Cellular)
    # - For rolling means & std devs: Window size = 5 timesteps (5 seconds)
    # - Group by episode_id to avoid leakage between episodes
    df['sinr_mean_5s'] = df.groupby('episode_id')[target_sinr].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['sinr_std_5s'] = df.groupby('episode_id')[target_sinr].rolling(5, min_periods=1).std().fillna(0).reset_index(0, drop=True)
    df['tput_mean_5s'] = df.groupby('episode_id')[target_tput].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # - For lagged Features: Create lags for SINR and Interference
    # - Lags: 1, 2, 4, 8, 16 seconds
    # - Fill NaNs with 0 (indicating no prior data)
    lags = [1, 2, 4, 8, 16]
    for lag in lags:
        df[f'sinr_lag_{lag}'] = df.groupby('episode_id')[target_sinr].shift(lag).fillna(0)
        df[f'interf_lag_{lag}'] = df.groupby('episode_id')['interference_dbm'].shift(lag).fillna(0)
    
    # Universal features + Mode-specific engineered features
    features = [
        target_sinr, target_tput, 
        'distance_tx_rx', 'distance_bs_rx', 'interference_dbm',
        'sinr_mean_5s', 'sinr_std_5s', 'tput_mean_5s'
    ]
    for lag in lags:
        features.append(f'sinr_lag_{lag}')
        features.append(f'interf_lag_{lag}')

    print(f"Total Features Selected: {len(features)}")
    print(f"Feature List: {features}")
        
    # Prepare labels (Y) (Predicting the next timestep of the target mode)
    # Shift(-1): Pulls the next timestep's SINR to the current row.
    # dropna: Drop the last timestep of every episode because it has no "next" step to predict.
    print("Creating Targets (Next Step SINR)...")
    df['label'] = df.groupby('episode_id')[target_sinr].shift(-1)
    df = df.dropna(subset=['label'])
    
    # NOTE: Save intermediate CSV file containing the new features to check it in Excel
    debug_path = "data/model_ready/feature_engineering_debug.csv"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    df.to_csv(debug_path, index=False)
    print(f"Debug file saved to {debug_path} (Open in Excel to verify features)")

    # Normalize features using Min-Max Scaling
    print("Normalizing features...")
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save Scaler for future use during DL model inference
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # Reshape data into sequences for DL model
    print("Reshaping data into sequences...")
    num_episodes = SimulationConfig.NUM_EPISODES
    steps_per_episode = SimulationConfig.STEPS_PER_EPISODE
    
    # Convert to numpy arrays to facilitate reshaping 
    X_data = np.asarray(df[features].to_numpy(dtype=float))
    y_data = np.asarray(df['label'].to_numpy(dtype=float))
    
    try:
        X_seq = X_data.reshape(num_episodes, steps_per_episode, len(features))
        y_seq = y_data.reshape(num_episodes, steps_per_episode, 1)
    except ValueError as e:
        print(f"Reshape Error: {e}. Check if total rows ({len(df)}) matches {num_episodes} * {steps_per_episode}.")
        return

    print(f"Data Reshaped: X={X_seq.shape}, y={y_seq.shape}")

    # Split data into Training (70%), Validation (15%), and Testing (15%)
    print("Splitting data into Training, Validation, and Testing sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"Training Sets (70%):   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Sets (15%): X={X_val.shape},   y={y_val.shape}")
    print(f"Testing Sets (15%):    X={X_test.shape},  y={y_test.shape}")

    # Save .npy files for DL model training
    np.save(f"{save_dir}/X_train.npy", X_train)
    np.save(f"{save_dir}/X_val.npy", X_val)   
    np.save(f"{save_dir}/X_test.npy", X_test)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/y_val.npy", y_val)   
    np.save(f"{save_dir}/y_test.npy", y_test)
    
    print(f"✓ {mode} preprocessed dataset saved to {save_dir}/")

def preprocess_data():
    input_path = SimulationConfig.OUTPUT_FILE
    df = pd.read_csv(input_path).sort_values(by=['episode_id', 'timestamp'])
    df = clean_dataset(df)
    
    # Generate both datasets independently!
    generate_dataset_for_mode(df, 'D2D')
    generate_dataset_for_mode(df, 'CELLULAR')

if __name__ == "__main__":
    preprocess_data()