import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd
from threshold_selection import ThresholdSelector
from online_selector import OnlineModeSelector 

# Function to load the unseen testing dataset for a specific mode (D2D or Cellular).
# The final 15% of testing data from the 70:15:15 split.
def load_test_data(mode):
    base_path = f"data/model_ready/{mode}"
    X_test = np.load(os.path.join(base_path, "X_test.npy"))
    y_test = np.load(os.path.join(base_path, "y_test.npy"))
    return X_test, y_test

# Function to format data for CNN/DNN by creating localized timesteps while keeping them grouped by episode
def create_sliding_windows_per_episode(X, y, window_size=16):
    X_episodes, y_episodes = [], []
    for i in range(X.shape[0]): # Loop through episodes
        X_win, y_win = [], []
        for t in range(X.shape[1] - window_size + 1): # Loop through timesteps
            X_win.append(X[i, t:t+window_size, :])
            y_win.append(y[i, t+window_size-1, :])
        X_episodes.append(X_win)
        y_episodes.append(y_win)
    return np.array(X_episodes), np.array(y_episodes)

# Function to calculate PICP (% inside bounds) and MPIW (width of bounds).
def calculate_uncertainty_metrics(y_true, y_pred, lower_margin, upper_margin):
    lower_bounds = y_pred + lower_margin
    upper_bounds = y_pred + upper_margin
    within_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    picp = np.mean(within_bounds) * 100.0 
    mpiw = upper_margin - lower_margin    
    return picp, mpiw

# Main function to evaluate all models across computational, prediction, uncertainty, and system-level metrics. 
# Results are printed and saved to a CSV file.
def evaluate_all_models():
    # Load Test Data for both modes
    print("\nLoading Test Data for both modes...")
    X_test_d2d_raw, y_test_d2d_raw = load_test_data('d2d')
    X_test_cell_raw, y_test_cell_raw = load_test_data('cellular')
    
    # Generate sliding windows for CNN/DNN models (which require localized temporal context)
    print("Generating Sliding Windows for CNN/DNN...")
    X_d2d_win, y_d2d_win = create_sliding_windows_per_episode(X_test_d2d_raw, y_test_d2d_raw)
    X_cell_win, y_cell_win = create_sliding_windows_per_episode(X_test_cell_raw, y_test_cell_raw)
    
    models = ['gru', 'lstm', 'cnn', 'dnn']
    results = {}
    
    # Target parameter for system-level evaluation (the Mbps threshold to achieve in the real system).
    TEST_TARGET_MBPS = 2.0 
    
    # Loop through each model
    for model_name in models:
        print(f"\n" + "="*40)
        print(f" EVALUATING MODEL: {model_name.upper()}")
        print("="*40)
        
        # 1. Prepare correct data shapes
        if model_name in ['gru', 'lstm']:
            X_d2d, y_d2d = X_test_d2d_raw, y_test_d2d_raw
            X_cell, y_cell = X_test_cell_raw, y_test_cell_raw
        else:
            X_d2d, y_d2d = X_d2d_win, y_d2d_win
            X_cell, y_cell = X_cell_win, y_cell_win
            
        y_d2d_flat = y_d2d.flatten()
        y_cell_flat = y_cell.flatten()

        # 2. Load the D2D and Cellular Models for Prediction/Computational Metrics (SINR Prediction Module)
        path_d2d = f"models/d2d/{model_name}/{model_name}_model.keras"
        path_cell = f"models/cellular/{model_name}/{model_name}_model.keras"
        if not os.path.exists(path_d2d):
            print(f"⚠️ Skipping {model_name.upper()} - D2D model not found.")
            continue
        if not os.path.exists(path_cell):
            print(f"⚠️ Skipping {model_name.upper()} - Cellular model not found.")
            continue
            
        model = tf.keras.models.load_model(path_d2d)
        model_cell = tf.keras.models.load_model(path_cell)
        
        # 3. Load Error Params for Uncertainty Metrics (Error Analysis Module)
        with open(f"models/d2d/{model_name}/{model_name}_error_params_kde.pkl", "rb") as f:
            error_params = pickle.load(f)

        # Calculate the computational metrics:
        # - Model Size (number of parameters)
        # - Inference Time (ms per sample)
        # - Memory Usage and FLOPs estimate
        start_time = time.time()
        
        if model_name in ['gru', 'lstm']:
            # GRU/LSTM are already in the correct 3D shape: (Episodes, Timesteps, Features)
            preds_d2d = model.predict(X_d2d, batch_size=64, verbose=0)
            preds_cell = model_cell.predict(X_cell, batch_size=64, verbose=0)
            
            # Flatten to 1D array for error calculations
            preds_d2d_flat = preds_d2d.flatten()
            
        else:
            # CNN/DNN are in 4D shape: (Episodes, Windows, WindowSize, Features)
            # Flatten to 3D for Keras: (Total_Windows, WindowSize, Features)
            X_d2d_batch = X_d2d.reshape(-1, X_d2d.shape[2], X_d2d.shape[3])
            X_cell_batch = X_cell.reshape(-1, X_cell.shape[2], X_cell.shape[3])
            
            # Predict
            preds_d2d_raw = model.predict(X_d2d_batch, batch_size=64, verbose=0)
            preds_cell_raw = model_cell.predict(X_cell_batch, batch_size=64, verbose=0)
            
            # Reshape back to episodic format: (Episodes, Windows, 1)
            preds_d2d = preds_d2d_raw.reshape(X_d2d.shape[0], X_d2d.shape[1], 1)
            preds_cell = preds_cell_raw.reshape(X_cell.shape[0], X_cell.shape[1], 1)
            
            # Flatten to 1D array for error calculations
            preds_d2d_flat = preds_d2d_raw.flatten()

        inference_time_ms = ((time.time() - start_time) / len(preds_d2d_flat)) * 1000

        # Float32 uses 4 bytes per parameter, convert to KB for easier interpretation
        # Basic proxy for Keras FLOPs (1 Multiply + 1 Add per parameter)
        param_count = model.count_params() 
        memory_usage_kb = (param_count * 4) / 1024.0
        flops_estimate = param_count * 2

        # Calculate the predictions metrics:
        # - MAE (Mean Absolute Error)
        # - RMSE (Root Mean Squared Error)
        # - R2 Score (Coefficient of Determination)
        mae = mean_absolute_error(y_d2d_flat, preds_d2d_flat)
        rmse = np.sqrt(mean_squared_error(y_d2d_flat, preds_d2d_flat))
        r2 = r2_score(y_d2d_flat, preds_d2d_flat) 
        
        # Calculate the uncertainty metrics:
        # - PICP (Prediction Interval Coverage Probability)
        # - MPIW (Mean Prediction Interval Width)
        picp, mpiw = calculate_uncertainty_metrics(
            y_d2d_flat, preds_d2d_flat, error_params['lower_bound'], error_params['upper_bound']
        )

        # Calculate the system-level metrics:
        # - Average Throughput (Mbps)
        # - Spectral Efficiency (bps/Hz)
        # - Mode switching Rate (switches per 100 seconds or timesteps) (Need to check)
        # - Average D2D Residence Time (% of time in D2D mode in seconds or timesteps)
        selector = OnlineModeSelector(
            model_name=model_name, 
            constraint_type='AR', 
            target_tput_mbps=TEST_TARGET_MBPS
        )
        
        switches = 0
        d2d_time = 0
        total_throughput_mbps = 0.0
        d2d_sessions = 0
        total_steps = X_d2d.shape[0] * X_d2d.shape[1]

        # Loop through each episode
        for e in range(X_d2d.shape[0]):
            current_mode = 'D2D' # Reset to D2D at the start of every new episode
            d2d_sessions += 1
            
            # Loop through each timestep within the episode
            for t in range(X_d2d.shape[1]):
                true_sinr_d2d = y_d2d[e, t, 0]
                true_sinr_cell = y_cell[e, t, 0]
                
                # Fetch the pre-calculated predictions
                pred_d2d = preds_d2d[e, t, 0]
                pred_cell = preds_cell[e, t, 0]
                
                # Pass predictions instead of features
                new_mode, logs = selector.make_decision(pred_d2d, pred_cell, current_mode)
                
                if new_mode != current_mode:
                    switches += 1
                    if new_mode == 'D2D':
                        d2d_sessions += 1
                current_mode = new_mode
                
                if current_mode == 'D2D':
                    d2d_time += 1
                    actual_tput = selector.ts.shannon_throughput(true_sinr_d2d)
                else:
                    actual_tput = selector.ts.shannon_throughput(true_sinr_cell)
                    
                total_throughput_mbps += actual_tput
                
        avg_throughput = total_throughput_mbps / total_steps
        switch_rate = (switches / total_steps) * 100 
        spectral_efficiency = avg_throughput / (100e6 / 1e6) 
        avg_d2d_residence_s = (d2d_time / d2d_sessions) if d2d_sessions > 0 else 0.0
        
        # Save all metrics in a structured format for final comparison across models.
        results[model_name] = {
            'MAE (dB)': f"{mae:.2f}", 
            'RMSE': f"{rmse:.2f}", 
            'R2 Score': f"{r2:.3f}",           
            'PICP (%)': f"{picp:.1f}", 
            'MPIW (dB)': f"{mpiw:.2f}",
            'Params': param_count,
            'Mem (KB)': f"{memory_usage_kb:.1f}",
            'FLOPs': flops_estimate,
            'Inference (ms)': f"{inference_time_ms:.2f}",
            'Avg Tput (Mbps)': f"{avg_throughput:.2f}",
            'Spectral Eff (bps/Hz)': f"{spectral_efficiency:.4f}",
            'Switch Rate (%)': f"{switch_rate:.2f}",
            'Avg D2D Stay (s)': f"{avg_d2d_residence_s:.1f}"
        }
        
    print("\n" + "="*125)
    print(f" FINAL EVALUATION RESULTS (Target: {TEST_TARGET_MBPS} Mbps)")
    print("="*125)
    df = pd.DataFrame(results).T
    print(df.to_string())
    
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/final_evaluation_metrics.csv")
    print("\n✓ Results saved to data/results/final_evaluation_metrics.csv")

if __name__ == "__main__":
    evaluate_all_models()