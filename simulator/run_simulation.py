import sys
import os
import pandas as pd
from tqdm import tqdm  # Progress bar (optional, install via 'pip install tqdm' if needed)

# Add the project root to the python path so we can import 'simulator'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.environment import D2DEnvironment
from simulator.config import SimulationConfig

# --- Configuration ---
NUM_EPISODES = 50          # How many separate 100-second runs to do
STEPS_PER_EPISODE = 100    # Duration of each run (seconds)
OUTPUT_FILE = "data/d2d_simulation_data.csv"

def generate_dataset():
    print(f"Starting Simulation: {NUM_EPISODES} episodes, {STEPS_PER_EPISODE} steps each.")
    
    env = D2DEnvironment()
    all_records = []
    
    # Loop through episodes (independent runs with different start positions)
    for episode in tqdm(range(NUM_EPISODES), desc="Simulating Episodes"):
        env.reset()
        
        for step in range(STEPS_PER_EPISODE):
            # 1. Run one second of simulation
            state = env.step()
            
            # 2. Add episode ID for tracking
            state['episode_id'] = episode
            
            # 3. Collect data
            all_records.append(state)
            
    # --- Save to CSV ---
    print("Converting to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Reorder columns for readability (optional)
    cols = ['episode_id', 'timestamp', 'optimal_mode', 'sinr_d2d_db', 'sinr_cell_db', 
            'throughput_d2d_mbps', 'throughput_cell_mbps', 'distance_tx_rx']
    # Add remaining columns that aren't in the priority list
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]
    
    # Create data folder if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving {len(df)} samples to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_dataset()