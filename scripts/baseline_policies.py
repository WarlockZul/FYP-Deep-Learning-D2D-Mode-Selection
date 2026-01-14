import pandas as pd
import numpy as np
import os
import sys

# Read in project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the dataset based on the provided configuration
# Use 'config' argument to choose between different configurations
def load_data(config):
    file_path = config.OUTPUT_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run the appropriate simulation first.")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
    return df

# Policy to always choose D2D mode
def policy_always_d2d(df):
    return pd.Series(["D2D"] * len(df), index=df.index)

# Policy to always choose Cellular mode
def policy_always_cellular(df):
    return pd.Series(["Cellular"] * len(df), index=df.index)

# Policy to randomly choose between D2D and Cellular with equal probability
def policy_random(df):
    # Set seed to produce same results for each run
    np.random.seed(42) 
    choices = np.random.choice(["D2D", "Cellular"], size=len(df))
    return pd.Series(choices, index=df.index)

# Policy to choose mode based on SINR Threshold
def policy_sinr_threshold(df, threshold_db=0):
    # If sinr_d2d_db < threshold, return 'Cellular', else 'D2D'
    decisions = np.where(df['sinr_d2d_db'] < threshold_db, "Cellular", "D2D")
    return pd.Series(decisions, index=df.index)

# Policy to choose the optimal mode based on actual throughput (from optimal_mode column)
def policy_ground_truth(df):
    return df['optimal_mode']

# Function to calculate evaluation metrics
def calculate_metrics(df, mode_decisions, policy_name, config):
    # Create a temp dataframe to handle logic safely
    df_temp = df.copy()
    df_temp['decision'] = mode_decisions
    
    # 1. Detect Switching
    # We must group by episode_id so we don't penalize the first step of a new episode
    df_temp['prev_decision'] = df_temp.groupby('episode_id')['decision'].shift(1)
    
    # A switch happens if Current != Prev AND Prev is not NaN (Start of episode)
    df_temp['is_switch'] = (df_temp['decision'] != df_temp['prev_decision']) & (df_temp['prev_decision'].notna())
    
    # 2. Get Raw Throughput
    df_temp['raw_throughput'] = np.where(
        df_temp['decision'] == "D2D", 
        df_temp['throughput_d2d_mbps'], 
        df_temp['throughput_cell_mbps']
    )
    
    # 3. Apply Handover Latency Penalty
    # Get latency from config (default to 0.05 if missing)
    latency = getattr(config, 'HANDOVER_LATENCY_S', 0.05)
    
    # Efficiency Factor: 1.0 (100%) if no switch, (1.0 - Latency) if switch
    # Example: 1.0 - 0.05 = 0.95 (95% efficiency)
    df_temp['efficiency'] = np.where(df_temp['is_switch'], 1.0 - latency, 1.0)
    
    # Calculate Effective Throughput
    df_temp['effective_throughput'] = df_temp['raw_throughput'] * df_temp['efficiency']
    
    avg_throughput = df_temp['effective_throughput'].mean()
    
    # 4. Calculate Switching Rate
    total_switches = df_temp['is_switch'].sum()
    total_time_seconds = len(df) * config.TIME_STEP_S
    switch_rate = (total_switches / total_time_seconds) * 100 

    # 5. Calculate Spectral Efficiency
    avg_spectral_efficiency = (avg_throughput * 1e6) / config.BANDWIDTH_HZ

    # 6. Calculate Average D2D Residence Time
    d2d_blocks = (df_temp['decision'] == "D2D").astype(int)
    if d2d_blocks.sum() == 0:
        avg_residence_time = 0.0
    else:
        # Create block IDs
        df_temp['block_id'] = (
            (df_temp['decision'] != df_temp['prev_decision']) | 
            (df_temp['episode_id'] != df_temp['episode_id'].shift(1))
        ).cumsum()
        
        d2d_only = df_temp[df_temp['decision'] == "D2D"]
        block_lengths = d2d_only.groupby('block_id').size()
        block_durations = block_lengths * config.TIME_STEP_S
        avg_residence_time = block_durations.mean()
    
    return {
        "Policy": policy_name,
        "Avg Throughput (Mbps)": round(avg_throughput, 2),
        "Spectral Eff (bps/Hz)": round(avg_spectral_efficiency, 2),
        "Switching Rate (per 100s)": round(switch_rate, 2),
        "Avg D2D Residence (s)": round(avg_residence_time, 2)
    }