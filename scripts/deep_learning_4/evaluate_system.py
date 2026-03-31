import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from threshold_selection import ThresholdSelector
from online_selector import OnlineModeSelector

# Load the unseen testing dataset (the final 15% of the simulation data)
def load_test_data():
    base_path = "data/model_ready"
    X_test = np.load(os.path.join(base_path, "X_test.npy"))
    y_test = np.load(os.path.join(base_path, "y_test.npy"))
    return X_test, y_test

# Calculates PICP and MPIW for the given true values, predictions, and margins.
# PICP: % of true values that fell safely inside the confidence bounds.
# MPIW: The average width of those bounds.
def calculate_uncertainty_metrics(y_true, y_pred, lower_margin, upper_margin):
    # The predicted boundaries
    lower_bounds = y_pred + lower_margin
    upper_bounds = y_pred + upper_margin
    
    # Check if true value is within bounds
    within_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    
    picp = np.mean(within_bounds) * 100.0 # Percentage
    mpiw = upper_margin - lower_margin    # Width in dB
    
    return picp, mpiw

def evaluate_all_models():
    print("Loading Test Data...")
    X_test_raw, y_test_raw = load_test_data()
    
    # [PLACEHOLDER 1: SCENARIO SPLITTING]
    # The proposal requires testing under "different mobility and interference scenarios".
    # We will need to discuss if your dataset has a column indicating "High Mobility" vs "Low Mobility"
    # so we can filter X_test here before running the evaluation.
    
    models = ['gru', 'lstm', 'cnn', 'dnn']
    results = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name.upper()}...")
        
        # 1. Load Model & Parameters
        model = tf.keras.models.load_model(f"models/{model_name}/{model_name}_model.keras")
        with open(f"models/{model_name}/{model_name}_error_params_kde.pkl", "rb") as f:
            error_params = pickle.load(f)
            
        # [PLACEHOLDER 2: DATA FORMATTING]
        # We need to format X_test into sequences for GRU/LSTM or sliding windows for CNN/DNN
        # For this prototype, we assume X_test is properly formatted.
        X_test = X_test_raw 
        y_test = y_test_raw.flatten()
        
        # ==========================================
        # 1. COMPUTATIONAL METRICS
        # ==========================================
        # Measure Inference Time
        start_time = time.time()
        y_pred = model.predict(X_test, batch_size=64, verbose=0).flatten()
        end_time = time.time()
        
        inference_time_ms = ((end_time - start_time) / len(X_test)) * 1000
        param_count = model.count_params()
        
        # [PLACEHOLDER 3: FLOPS ESTIMATE]
        # Calculating exact FLOPs requires the TensorFlow Profiler. 
        # We can discuss whether you want to implement the TF Profiler or just use Param Count.

        # ==========================================
        # 2. PREDICTION METRICS
        # ==========================================
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # ==========================================
        # 3. UNCERTAINTY METRICS
        # ==========================================
        lower_m = error_params['lower_bound']
        upper_m = error_params['upper_bound']
        picp, mpiw = calculate_uncertainty_metrics(y_test, y_pred, lower_m, upper_m)
        
        # ==========================================
        # 4. SYSTEM-LEVEL METRICS (The Loop)
        # ==========================================
        # We use the OnlineModeSelector from Step 10 to simulate the network
        selector = OnlineModeSelector(model_name=model_name, constraint_type='AR', target_tput_mbps=15.0)
        
        current_mode = 'D2D'
        switches = 0
        d2d_time = 0
        total_throughput_mbps = 0.0
        
        # Simulate a 100-second episode (assuming 1 step = 1 second)
        test_steps = min(100, len(X_test)) 
        
        for t in range(test_steps):
            feature_t = X_test[t:t+1] # Take one timestep
            true_sinr_t = y_test[t]   # The actual physical SINR
            
            # Ask the AI to make a decision
            new_mode, logs = selector.make_decision(feature_t, current_mode)
            
            # Count switches
            if new_mode != current_mode:
                switches += 1
            current_mode = new_mode
            
            # Track Residence Time
            if current_mode == 'D2D':
                d2d_time += 1
                # [PLACEHOLDER 4: ACTUAL THROUGHPUT]
                # If in D2D, actual throughput is calculated using the TRUE D2D SINR.
                actual_tput = selector.ts.shannon_throughput(true_sinr_t)
            else:
                # If in Cellular, throughput is calculated using the TRUE Cellular SINR.
                # Right now, we assume a stable Cellular SINR (e.g., 20 dB).
                actual_tput = selector.ts.shannon_throughput(20.0) 
                
            total_throughput_mbps += actual_tput
            
        avg_throughput = total_throughput_mbps / test_steps
        switch_rate = (switches / test_steps) * 100 # Switches per 100s
        spectral_efficiency = avg_throughput / (100e6 / 1e6) # bps/Hz
        
        # Save Results
        results[model_name] = {
            'MAE (dB)': mae, 'RMSE': rmse, 'R2': r2,
            'PICP (%)': picp, 'MPIW (dB)': mpiw,
            'Params': param_count, 'Inference (ms)': inference_time_ms,
            'Avg Throughput (Mbps)': avg_throughput,
            'Spectral Eff (bps/Hz)': spectral_efficiency,
            'Switch Rate (/100s)': switch_rate,
            'D2D Time (s)': d2d_time
        }
        
    # Print the final comparative table
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    import pandas as pd
    df = pd.DataFrame(results).T
    print(df.to_string())

if __name__ == "__main__":
    evaluate_all_models()