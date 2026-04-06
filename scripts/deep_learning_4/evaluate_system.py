import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd
from threshold_selection import ThresholdSelector
from online_selector import OnlineModeSelector 

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_test_data(mode):
    """Loads the unseen testing dataset for a specific mode."""
    base_path = f"data/model_ready/{mode}"
    X_test = np.load(os.path.join(base_path, "X_test.npy"))
    y_test = np.load(os.path.join(base_path, "y_test.npy"))
    return X_test, y_test

def create_sliding_windows(X, y, window_size=16):
    """Formats data for CNN/DNN by creating localized timesteps."""
    X_win, y_win = [], []
    for i in range(X.shape[0]):
        for t in range(X.shape[1] - window_size + 1):
            X_win.append(X[i, t:t+window_size, :])
            y_win.append(y[i, t+window_size-1, :])
    return np.array(X_win), np.array(y_win)

def calculate_uncertainty_metrics(y_true, y_pred, lower_margin, upper_margin):
    """Calculates PICP (% inside bounds) and MPIW (width of bounds)."""
    lower_bounds = y_pred + lower_margin
    upper_bounds = y_pred + upper_margin
    within_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    picp = np.mean(within_bounds) * 100.0 
    mpiw = upper_margin - lower_margin    
    return picp, mpiw

# ==========================================
# MAIN EVALUATION FUNCTION
# ==========================================
def evaluate_all_models():
    print("Loading Test Data for both modes...")
    X_test_d2d_raw, y_test_d2d_raw = load_test_data('d2d')
    X_test_cell_raw, y_test_cell_raw = load_test_data('cellular')
    
    print("Generating Sliding Windows for CNN/DNN...")
    X_d2d_win, y_d2d_win = create_sliding_windows(X_test_d2d_raw, y_test_d2d_raw)
    X_cell_win, y_cell_win = create_sliding_windows(X_test_cell_raw, y_test_cell_raw)
    
    models = ['gru', 'lstm', 'cnn', 'dnn']
    results = {}
    
    # Target parameter based on your supervisor meeting
    TEST_TARGET_MBPS = 1.0 
    
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

        # Load the D2D Model for Prediction/Computational Metrics
        path_d2d = f"models/d2d/{model_name}/{model_name}_model.keras"
        if not os.path.exists(path_d2d):
            print(f"⚠️ Skipping {model_name.upper()} - Model not found.")
            continue
            
        model = tf.keras.models.load_model(path_d2d)
        
        # Load Error Params for Uncertainty Metrics
        with open(f"models/d2d/{model_name}/{model_name}_error_params_kde.pkl", "rb") as f:
            error_params = pickle.load(f)

        # ==========================================
        # A. COMPUTATIONAL METRICS (Restored from old file)
        # ==========================================
        start_time = time.time()
        y_pred = model.predict(X_d2d, batch_size=64, verbose=0).flatten()
        inference_time_ms = ((time.time() - start_time) / len(y_d2d_flat)) * 1000
        
        param_count = model.count_params() # RESTORED

        # ==========================================
        # B. PREDICTION & UNCERTAINTY METRICS (Restored R2)
        # ==========================================
        mae = mean_absolute_error(y_d2d_flat, y_pred)
        rmse = np.sqrt(mean_squared_error(y_d2d_flat, y_pred))
        r2 = r2_score(y_d2d_flat, y_pred) # RESTORED
        
        picp, mpiw = calculate_uncertainty_metrics(
            y_d2d_flat, y_pred, error_params['lower_bound'], error_params['upper_bound']
        )

        # ==========================================
        # C. SYSTEM-LEVEL METRICS (The Decision Loop)
        # ==========================================
        selector = OnlineModeSelector(model_name=model_name, constraint_type='AR', target_tput_mbps=TEST_TARGET_MBPS)
        
        current_mode = 'D2D'
        switches = 0
        d2d_time = 0
        total_throughput_mbps = 0.0
        
        # We run it across all available timesteps
        test_steps = len(X_d2d)
        
        for t in range(test_steps):
            feat_d2d = X_d2d[t:t+1]
            feat_cell = X_cell[t:t+1]
            
            true_sinr_d2d = y_d2d_flat[t]
            true_sinr_cell = y_cell_flat[t]
            
            new_mode, logs = selector.make_decision(feat_d2d, feat_cell, current_mode)
            
            if new_mode != current_mode:
                switches += 1
            current_mode = new_mode
            
            if current_mode == 'D2D':
                d2d_time += 1
                actual_tput = selector.ts.shannon_throughput(true_sinr_d2d)
            else:
                actual_tput = selector.ts.shannon_throughput(true_sinr_cell)
                
            total_throughput_mbps += actual_tput
            
        avg_throughput = total_throughput_mbps / test_steps
        switch_rate = (switches / test_steps) * 100 
        spectral_efficiency = avg_throughput / (100e6 / 1e6) 
        d2d_ratio = (d2d_time / test_steps) * 100
        
        # ==========================================
        # D. SAVE RESULTS
        # ==========================================
        results[model_name] = {
            'MAE (dB)': f"{mae:.2f}", 
            'RMSE': f"{rmse:.2f}", 
            'R2 Score': f"{r2:.3f}",           # RESTORED
            'PICP (%)': f"{picp:.1f}", 
            'MPIW (dB)': f"{mpiw:.2f}",
            'Params': param_count,             # RESTORED
            'Inference (ms)': f"{inference_time_ms:.2f}",
            'Avg Tput (Mbps)': f"{avg_throughput:.2f}",
            'Spectral Eff (bps/Hz)': f"{spectral_efficiency:.4f}",
            'Switch Rate (%)': f"{switch_rate:.2f}",
            'Time in D2D (%)': f"{d2d_ratio:.1f}"
        }
        
    print("\n" + "="*110)
    print(f" FINAL EVALUATION RESULTS (Target: {TEST_TARGET_MBPS} Mbps)")
    print("="*110)
    df = pd.DataFrame(results).T
    print(df.to_string())
    
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/final_evaluation_metrics.csv")
    print("\n✓ Results saved to data/results/final_evaluation_metrics.csv")

if __name__ == "__main__":
    evaluate_all_models()