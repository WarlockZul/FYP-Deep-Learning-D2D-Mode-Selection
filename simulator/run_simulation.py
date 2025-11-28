import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.environment import D2DEnvironment
from simulator.config import SimulationConfig

def generate_dataset():
    # Set random seed for reproducibility (meaning same outputs on each run, unless changed)
    np.random.seed(SimulationConfig.SEED)
    print(f"Random Seed set to: {SimulationConfig.SEED}")
    print(f"Starting Simulation: {SimulationConfig.NUM_EPISODES} episodes, {SimulationConfig.STEPS_PER_EPISODE} steps each.")
    
    env = D2DEnvironment()
    all_records = []
    
    # Loop through episodes
    for episode in tqdm(range(SimulationConfig.NUM_EPISODES), desc="Simulating Episodes"):
        # Reset environment for new episode (new positions, new shadowing)
        env.reset()
        
        # Loop through time steps in the episode
        for step in range(SimulationConfig.STEPS_PER_EPISODE):
            state = env.step()
            state['episode_id'] = episode
            all_records.append(state)
            
    # Save results to CSV
    print("Converting to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Update column order for better readability
    cols = [
        'episode_id', 'timestamp', 'optimal_mode',      # Episode, Time, and Current Mode 
        'tx_pos_x', 'tx_pos_y',                         # Positions of Tx
        'rx_pos_x', 'rx_pos_y',                         # Positions of Rx
        'distance_tx_rx', 'distance_bs_rx',             # Distances (D2D: Tx to Rx, Cellular: BS to Rx)
        'tx_speed_mps', 'rx_speed_mps',                 # Speeds of Tx and Rx
        'tx_power_d2d_dbm', 'rx_power_d2d_dbm',         # Transmit and Received Powers for D2D
        'tx_power_bs_dbm', 'rx_power_cell_dbm',         # Transmit and Received Powers for Cellular
        'sinr_d2d_db', 'sinr_cell_db',                  # SINR values (D2D and Cellular)
        'interference_dbm', 'noise_dbm',                # Interference and Noise Levels
        'throughput_d2d_mbps', 'throughput_cell_mbps',  # Throughputs (D2D and Cellular)
    ]
    
    # FAILSAFE: Add any remaining columns that aren't in the priority list
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]
    
    # FAILSAFE: Ensure directory exists
    os.makedirs(os.path.dirname(SimulationConfig.OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving {len(df)} samples to {SimulationConfig.OUTPUT_FILE}...")
    df.to_csv(SimulationConfig.OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_dataset()