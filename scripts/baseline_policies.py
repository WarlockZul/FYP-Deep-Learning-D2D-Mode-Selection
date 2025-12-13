import pandas as pd
import numpy as np
import os
import sys

# Read in project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.config import SimulationConfig

# Load the dataset generated in Step 3
def load_data():
    """Loads the dataset generated in Step 3."""
    file_path = SimulationConfig.OUTPUT_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run 'generate_data.py' first.")
    
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
def calculate_metrics(df, mode_decisions, policy_name):
    # 1. Calculate Average Throughput
    # Select throughput based on mode selected
    final_throughput = np.where(
        mode_decisions == "D2D", 
        df['throughput_d2d_mbps'], 
        df['throughput_cell_mbps']
    )

    avg_throughput = np.mean(final_throughput)
    
    # 2. Calculate Switching Rate
    # NOTE: Switching only counts within the same episode
    # A switch happens if Mode(t) != Mode(t-1)
    # - Step 1: Create a temporary DataFrame to track previous decisions
    # - Step 2: Use groupby on episode_id to ensure switches are only counted within episodes
    # - Step 3: Calculate total switches and total time to get switching rate
    df_temp = df.copy()
    df_temp['decision'] = mode_decisions
    df_temp['prev_decision'] = df_temp.groupby('episode_id')['decision'].shift(1).fillna(df_temp['decision'])

    switches = (df_temp['decision'] != df_temp['prev_decision'])
    total_switches = switches.sum()
    total_time_seconds = len(df) * SimulationConfig.TIME_STEP_S
    switch_rate = (total_switches / total_time_seconds) * 100 

    # 3. Calculate Average Spectral Efficiency (bps/Hz) [bps, not Mbps]
    avg_spectral_efficiency = (avg_throughput * 1e6) / SimulationConfig.BANDWIDTH_HZ

    # 4. Calculate Average D2D Residence Time
    # Logic: Identify blocks of "D2D", calculate their lengths, and average them.
    # - Filter only rows where decision is D2D
    # - If not D2D blocks, avg residence time is 0
    # - Else, find contiguous blocks of D2D and calculate their lengths
    # - Step 1: Store blocks by detecting changes in decision & new episodes
    # - Step 2: Exclude non-D2D blocks (cellular)
    # - Step 3: Calculate lengths of D2D blocks
    # - Step 4: Multiply lengths by time step to get durations in seconds
    # - Step 5: Average the durations to get final avg residence time
    d2d_blocks = (mode_decisions == "D2D").astype(int)
    if d2d_blocks.sum() == 0:
        avg_residence_time = 0.0
    else:
        df_temp['block_id'] = (
            (df_temp['decision'] != df_temp['prev_decision']) | 
            (df_temp['episode_id'] != df_temp['episode_id'].shift(1))
        ).cumsum()
        
        d2d_only = df_temp[df_temp['decision'] == "D2D"]
        block_lengths = d2d_only.groupby('block_id').size()
        block_durations = block_lengths * SimulationConfig.TIME_STEP_S
        avg_residence_time = block_durations.mean()
    
    return {
        "Policy": policy_name,
        "Avg Throughput (Mbps)": round(avg_throughput, 2),
        "Spectral Eff (bps/Hz)": round(avg_spectral_efficiency, 2),
        "Switching Rate (per 100s)": round(switch_rate, 2),
        "Avg D2D Residence (s)": round(avg_residence_time, 2)
    }