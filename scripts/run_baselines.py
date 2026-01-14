import pandas as pd
import os
from baseline_policies import (
    load_data, policy_always_d2d, policy_always_cellular, 
    policy_random, policy_sinr_threshold, policy_ground_truth, calculate_metrics
)

from simulator.config import SimulationConfig
from simulator_paper.config import PaperConfig

# NOTE: Change this variable to switch between different configurations (PAPER or PROPOSED)
ENV_MODE = "PAPER"

def main():
    # Select configuration based on ENV_MODE
    if ENV_MODE == "PAPER":
        print(">>> RUNNING BASELINES FOR: PAPER REPLICATION ENVIRONMENT")
        selected_config = PaperConfig
        output_csv_name = "data/baseline_results_paper.csv"
    else:
        print(">>> RUNNING BASELINES FOR: PROPOSED SIMULATION ENVIRONMENT")
        selected_config = SimulationConfig
        output_csv_name = "data/baseline_results_proposed.csv"
    
    # Load Data
    try:
        df = load_data(selected_config)
    except FileNotFoundError as e:
        print(e)
        return

    results = []
    print("\n--- Evaluating Baselines ---")
    
    # 1. Always D2D
    print("Evaluating: Always D2D...")
    decisions = policy_always_d2d(df)
    results.append(calculate_metrics(df, decisions, "Always D2D", selected_config))
    
    # 2. Always Cellular
    print("Evaluating: Always Cellular...")
    decisions = policy_always_cellular(df)
    results.append(calculate_metrics(df, decisions, "Always Cellular", selected_config))
    
    # 3. Random
    print("Evaluating: Random...")
    decisions = policy_random(df)
    results.append(calculate_metrics(df, decisions, "Random", selected_config))
    
    # 4. SINR Threshold (0 dB)
    print("Evaluating: SINR Threshold (0 dB)...")
    decisions = policy_sinr_threshold(df, threshold_db=0)
    results.append(calculate_metrics(df, decisions, "SINR Threshold (0dB)", selected_config))

    # 5. Ground Truth
    print("Evaluating: Ground Truth...")
    decisions = policy_ground_truth(df)
    results.append(calculate_metrics(df, decisions, "Ground Truth (Max Tput)", selected_config))
    
    # Display and Save Results
    results_df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(results_df)
    
    os.makedirs("data", exist_ok=True)
    output_path = output_csv_name
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()