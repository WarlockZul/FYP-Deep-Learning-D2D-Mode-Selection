import pandas as pd
import os
from baseline_policies import (
    load_data, policy_always_d2d, policy_always_cellular, 
    policy_random, policy_sinr_threshold, policy_ground_truth, calculate_metrics
)

def main():
    # Load Data
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return

    results = []
    print("\n--- Evaluating Baselines ---")
    
    # 1. Always D2D
    print("Evaluating: Always D2D...")
    decisions = policy_always_d2d(df)
    results.append(calculate_metrics(df, decisions, "Always D2D"))
    
    # 2. Always Cellular
    print("Evaluating: Always Cellular...")
    decisions = policy_always_cellular(df)
    results.append(calculate_metrics(df, decisions, "Always Cellular"))
    
    # 3. Random
    print("Evaluating: Random...")
    decisions = policy_random(df)
    results.append(calculate_metrics(df, decisions, "Random"))
    
    # 4. SINR Threshold (0 dB)
    print("Evaluating: SINR Threshold (0 dB)...")
    decisions = policy_sinr_threshold(df, threshold_db=0)
    results.append(calculate_metrics(df, decisions, "SINR Threshold (0dB)"))

    # 5. Ground Truth
    print("Evaluating: Ground Truth...")
    decisions = policy_ground_truth(df)
    results.append(calculate_metrics(df, decisions, "Ground Truth (Max Tput)"))
    
    # Display and Save Results
    results_df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(results_df)
    
    os.makedirs("data", exist_ok=True)
    output_path = "data/baseline_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()