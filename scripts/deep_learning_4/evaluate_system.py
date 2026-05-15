import os
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd
from threshold_selection import ThresholdSelector
from online_selector import OnlineModeSelector 

current_dir = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(current_dir, ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))

from ml_config import MLConfig

# Function to load the unseen testing dataset for a specific mode (D2D or Cellular).
# The final 15% of testing data from the 70:15:15 split.
def load_test_data(dataset_folder, mode):
    base_path = os.path.join(PROJECT_ROOT, "data", dataset_folder, mode)
    X_test = np.load(os.path.join(base_path, "X_test.npy"))
    y_test = np.load(os.path.join(base_path, "y_test.npy"))
    return X_test, y_test

# Function to format data for CNN/DNN by creating localized timesteps while keeping them grouped by episode
def create_sliding_windows_per_episode(X, y, window_size=MLConfig.WINDOW_SIZE):
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
def evaluate_all_models(dataset_folder):
    # Load Test Data for both modes
    print(f"\nLoading Test Data from {dataset_folder}...")
    X_test_d2d_raw, y_test_d2d_raw = load_test_data(dataset_folder, 'd2d')
    X_test_cell_raw, y_test_cell_raw = load_test_data(dataset_folder, 'cellular')
    
    # Generate sliding windows for CNN/DNN models (which require localized temporal context)
    print("Generating Sliding Windows for CNN/DNN...")
    X_d2d_win, y_d2d_win = create_sliding_windows_per_episode(X_test_d2d_raw, y_test_d2d_raw)
    X_cell_win, y_cell_win = create_sliding_windows_per_episode(X_test_cell_raw, y_test_cell_raw)
    
    # Initialize models to evaluate and a dictionary to store results for final comparison
    models = MLConfig.MODELS_TO_EVALUATE
    results = {}
    
    # Target parameter for system-level evaluation (the Mbps threshold to achieve in the real system).
    TEST_TARGET_MBPS = MLConfig.TARGET_THROUGHPUT_MBPS

    # Load Target Scalers to convert Z-scores back to raw dB
    with open(os.path.join(PROJECT_ROOT, "data", dataset_folder, "d2d", "target_scaler.pkl"), "rb") as f:
        t_scaler_d2d = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, "data", dataset_folder, "cellular", "target_scaler.pkl"), "rb") as f:
        t_scaler_cell = pickle.load(f)
    
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
        path_d2d = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, "d2d", model_name, f"{model_name}_model.keras")
        path_cell = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, "cellular", model_name, f"{model_name}_model.keras")
        
        if not os.path.exists(path_d2d):
            print(f"⚠️ Skipping {model_name.upper()} - D2D model not found.")
            continue
        if not os.path.exists(path_cell):
            print(f"⚠️ Skipping {model_name.upper()} - Cellular model not found.")
            continue
            
        model_d2d = tf.keras.models.load_model(path_d2d)
        model_cell = tf.keras.models.load_model(path_cell)
        
        # 3. Load Error Params for Uncertainty Metrics (Error Analysis Module)
        pkl_path_d2d = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, "d2d", model_name, f"{model_name}_error_params_kde.pkl")
        with open(pkl_path_d2d, "rb") as f:
            error_params_d2d = pickle.load(f)
            
        pkl_path_cell = os.path.join(PROJECT_ROOT, "models", dataset_folder, MLConfig.EXPERIMENT_NAME, "cellular", model_name, f"{model_name}_error_params_kde.pkl")
        with open(pkl_path_cell, "rb") as f:
            error_params_cell = pickle.load(f)

        # A) Calculate the computational metrics:
        # - Model Size (number of parameters)
        # - Inference Time (ms per sample)
        # - Memory Usage and FLOPs estimate
        start_time = time.time()
        
        if model_name in ['gru', 'lstm']:
            # GRU/LSTM are already in the correct 3D shape: (Episodes, Timesteps, Features)
            preds_d2d = model_d2d.predict(X_d2d, batch_size=64, verbose=0)
            preds_cell = model_cell.predict(X_cell, batch_size=64, verbose=0)
            
            # Flatten to 1D array for error calculations
            preds_d2d_flat = preds_d2d.flatten()
            preds_cell_flat = preds_cell.flatten()
            
        else:
            # CNN/DNN are in 4D shape: (Episodes, Windows, WindowSize, Features)
            # Flatten to 3D for Keras: (Total_Windows, WindowSize, Features)
            X_d2d_batch = X_d2d.reshape(-1, X_d2d.shape[2], X_d2d.shape[3])
            X_cell_batch = X_cell.reshape(-1, X_cell.shape[2], X_cell.shape[3])
            
            # Predict the SINR values for each window
            preds_d2d_raw = model_d2d.predict(X_d2d_batch, batch_size=64, verbose=0)
            preds_cell_raw = model_cell.predict(X_cell_batch, batch_size=64, verbose=0)
            
            # Reshape back to episodic format: (Episodes, Windows, 1)
            preds_d2d = preds_d2d_raw.reshape(X_d2d.shape[0], X_d2d.shape[1], 1)
            preds_cell = preds_cell_raw.reshape(X_cell.shape[0], X_cell.shape[1], 1)
            
            # Flatten to 1D array for error calculations
            preds_d2d_flat = preds_d2d_raw.flatten()
            preds_cell_flat = preds_cell_raw.flatten()

        # Inverse transform the 3D arrays back to raw dB before any metrics are calculated
        old_shape_d2d = y_d2d.shape
        y_d2d = t_scaler_d2d.inverse_transform(y_d2d.reshape(-1, 1)).reshape(old_shape_d2d)
        preds_d2d = t_scaler_d2d.inverse_transform(preds_d2d.reshape(-1, 1)).reshape(old_shape_d2d)
        
        old_shape_cell = y_cell.shape
        y_cell = t_scaler_cell.inverse_transform(y_cell.reshape(-1, 1)).reshape(old_shape_cell)
        preds_cell = t_scaler_cell.inverse_transform(preds_cell.reshape(-1, 1)).reshape(old_shape_cell)

        # Re-flatten for Step B and Step C calculations
        preds_d2d_flat = preds_d2d.flatten()
        preds_cell_flat = preds_cell.flatten()
        y_d2d_flat = y_d2d.flatten()
        y_cell_flat = y_cell.flatten()
        
        inference_time_ms = ((time.time() - start_time) / len(preds_d2d_flat)) * 1000

        # Float32 uses 4 bytes per parameter, convert to KB for easier interpretation
        # Basic proxy for Keras FLOPs (1 Multiply + 1 Add per parameter)
        param_count = model_d2d.count_params() 
        memory_usage_kb = (param_count * 4) / 1024.0
        flops_estimate = param_count * 2

        # B) Calculate the predictions metrics:
        # - MAE (Mean Absolute Error)
        # - RMSE (Root Mean Squared Error)
        # - R2 Score (Coefficient of Determination)
        mae_d2d = mean_absolute_error(y_d2d_flat, preds_d2d_flat)
        rmse_d2d = np.sqrt(mean_squared_error(y_d2d_flat, preds_d2d_flat))
        r2_d2d = r2_score(y_d2d_flat, preds_d2d_flat) 
        
        mae_cell = mean_absolute_error(y_cell_flat, preds_cell_flat)
        rmse_cell = np.sqrt(mean_squared_error(y_cell_flat, preds_cell_flat))
        r2_cell = r2_score(y_cell_flat, preds_cell_flat)
        
        # C) Calculate the uncertainty metrics:
        # - PICP (Prediction Interval Coverage Probability)
        # - MPIW (Mean Prediction Interval Width)
        picp_d2d, mpiw_d2d = calculate_uncertainty_metrics(
            y_d2d_flat, preds_d2d_flat, error_params_d2d['lower_bound'], error_params_d2d['upper_bound']
        )
        picp_cell, mpiw_cell = calculate_uncertainty_metrics(
            y_cell_flat, preds_cell_flat, error_params_cell['lower_bound'], error_params_cell['upper_bound']
        )

        # D) Calculate the system-level metrics:
        # - Average Throughput (Mbps)
        # - Spectral Efficiency (bps/Hz)
        # - Mode switching Rate (switches per 100 seconds or timesteps) (Need to check)
        # - Average D2D Residence Time (% of time in D2D mode in seconds or timesteps)
        selector = OnlineModeSelector(
            model_name=model_name, 
            dataset_folder=dataset_folder,
            constraint_type=MLConfig.CONSTRAINT_TYPE, 
            target_tput_mbps=TEST_TARGET_MBPS
        )
        
        # Initialize counters for system-level metrics
        switches = 0
        d2d_time = 0
        total_throughput_mbps = 0.0
        d2d_sessions = 0
        total_steps = X_d2d.shape[0] * X_d2d.shape[1]

        # Loop through each episode
        for e in range(X_d2d.shape[0]):
            current_mode = 'D2D' # Reset to D2D at the start of every new episode (based on research paper)
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
                
                # Update switching rate and D2D residence time based on the new mode decision
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
                
        # Finalize calculations for system-level metrics
        avg_throughput = total_throughput_mbps / total_steps
        switch_rate = (switches / total_steps) * 100 
        spectral_efficiency = avg_throughput / (100e6 / 1e6) 
        avg_d2d_residence_s = (d2d_time / d2d_sessions) if d2d_sessions > 0 else 0.0
        
        # Save all metrics in a structured format for final comparison across models.
        results[model_name] = {
            'D2D MAE': f"{mae_d2d:.2f}", 
            'Cell MAE': f"{mae_cell:.2f}",
            'D2D RMSE': f"{rmse_d2d:.2f}", 
            'Cell RMSE': f"{rmse_cell:.2f}",
            'D2D PICP%': f"{picp_d2d:.1f}", 
            'Cell PICP%': f"{picp_cell:.1f}",
            'D2D MPIW (dB)': f"{mpiw_d2d:.2f}",
            'Cell MPIW (dB)': f"{mpiw_cell:.2f}",
            'R2 D2D': f"{r2_d2d:.3f}",
            'R2 Cell': f"{r2_cell:.3f}",
            'Params': param_count,
            'Mem (KB)': f"{memory_usage_kb:.1f}",
            'FLOPs': flops_estimate,
            'Inference (ms)': f"{inference_time_ms:.2f}",
            'Avg Tput (Mbps)': f"{avg_throughput:.2f}",
            'Spectral Eff (bps/Hz)': f"{spectral_efficiency:.4f}",
            'Switch Rate (%)': f"{switch_rate:.2f}",
            'Avg D2D Stay (s)': f"{avg_d2d_residence_s:.1f}"
        }
        
    if results:
        print("\n" + "="*100)
        print(f" FINAL EVALUATION RESULTS: {dataset_folder.upper()} (Target: {TEST_TARGET_MBPS} Mbps)")
        print("="*100)
        df = pd.DataFrame(results).T
        print(df.to_string())
        
        # FIX: Save results into the correct dataset-specific results folder
        save_dir = os.path.join(PROJECT_ROOT, "results", dataset_folder, MLConfig.EXPERIMENT_NAME)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "final_evaluation_metrics.csv")
        df.to_csv(csv_path)
        print(f"\n✓ Results saved to {csv_path}")

# Main execution loop
def main():
    datasets = ['preprocessed_paper', 'preprocessed_proposal']
    for dataset in datasets:
        print(f"\n\n{'*'*80}")
        print(f"🚀 INITIATING SYSTEM EVALUATION FOR: {dataset.upper()}")
        print(f"{'*'*80}")
        evaluate_all_models(dataset)

if __name__ == "__main__":
    main()