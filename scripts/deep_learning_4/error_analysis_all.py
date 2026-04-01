import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle

TARGET_MODE = 'cellular'  # Options: 'd2d' or 'cellular'

# Load the processed validation data (X_val and y_val) from data/model_ready/
def load_validation_data(mode):
    base_path = f"data/model_ready/{mode}"

    X_val = np.load(os.path.join(base_path, "X_val.npy"))
    y_val = np.load(os.path.join(base_path, "y_val.npy"))

    return X_val, y_val

# Create sliding windows for CNN/DNN validation input to predict one step at a time from the 300-second sequences
# Converts (Episodes, 300, Features) into (Total_Windows, Window_Size, Features)
def create_sliding_windows(X, y, window_size=16):
    X_win, y_win = [], []
    for i in range(X.shape[0]):

        # Slide a window across the 300 timesteps
        for t in range(X.shape[1] - window_size + 1):
            X_win.append(X[i, t:t+window_size, :])

            # The target is the label of the last timestep in the window
            y_win.append(y[i, t+window_size-1, :])

    return np.array(X_win), np.array(y_win)

# Main Function to Perform Error Analysis for All Models
def perform_error_analysis():
    # Step 1: Prepare Validation Data, create sliding windows for CNN/DNN, and load all models
    print(f"Loading Validation Data for {TARGET_MODE.upper()} mode...")
    X_val_raw, y_val_raw = load_validation_data(TARGET_MODE)
    print("Generating Sliding Windows for CNN/DNN...")
    X_val_win, y_val_win = create_sliding_windows(X_val_raw, y_val_raw, window_size=16)
    models_to_evaluate = ['gru', 'lstm', 'cnn', 'dnn']
    
    # Dictionary to store results for the final comparison plot
    results_dict = {}

    # Step 2: Loop through each model, compute residuals, derive confidence intervals, and save parameters
    for model_name in models_to_evaluate:
        print(f"\n" + "="*40)
        print(f" ANALYZING MODEL: {model_name.upper()} ({TARGET_MODE.upper()})")
        print("="*40)
        
        model_path = f"models/{TARGET_MODE}/{model_name}/{model_name}_model.keras"
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model {model_name.upper()} not found at {model_path}. Skipping.")
            continue
            
        # Load Model
        model = tf.keras.models.load_model(model_path)
        
        # Select correct data format (Raw: GRU/LSTM, Windows: CNN/DNN)
        if model_name in ['gru', 'lstm']:
            X_val, y_val = X_val_raw, y_val_raw
        else:
            X_val, y_val = X_val_win, y_val_win

        # Step 2.1: Compute Residuals (Error = True SINR - Predicted SINR)
        y_pred = model.predict(X_val, verbose=0)
        residuals = (y_val.flatten() - y_pred.flatten())
        
        # Step 2.2: Derive 95% Confidence Intervals
        lower_bound = np.percentile(residuals, 2.5)
        upper_bound = np.percentile(residuals, 97.5)
        
        print(f"Mean Residual: {np.mean(residuals):.4f} dB")
        print(f"95% CI: [{lower_bound:.4f} dB, {upper_bound:.4f} dB]")
        print(f"CI Width: {upper_bound - lower_bound:.4f} dB")
        
        # Step 2.3: Save Parameters for Threshold Module 
        # NOTE: Refer to proposal
        bw_method = 1.0  
        error_params = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'residuals_data': residuals,
            'bandwidth': bw_method
        }

        save_dir = f"models/{TARGET_MODE}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{model_name}_error_params_kde.pkl"
        
        with open(save_path, "wb") as f:
            pickle.dump(error_params, f)
        print(f"✓ Saved parameters to '{save_path}'")
        
        # Store for plotting
        results_dict[model_name] = {
            'residuals': residuals,
            'lower': lower_bound,
            'upper': upper_bound
        }
        
        # Clear memory to prevent slowdowns
        tf.keras.backend.clear_session()

    # Step 3: Generate Final Comparison Plot with KDEs and Confidence Intervals for All Models
    if results_dict:
        print("\nGenerating Final Comparison Plot...")
        plot_model_comparisons(results_dict, TARGET_MODE)

# Function to plot Gaussian KDEs and confidence intervals for all models on the same graph
def plot_model_comparisons(results_dict, mode):
    plt.figure(figsize=(12, 7))
    colors = {'gru': 'blue', 'lstm': 'green', 'cnn': 'red', 'dnn': 'purple'}
    
    for model_name, data in results_dict.items():
        residuals = data['residuals']
        
        # Fit KDE
        # NOTE: Edit bandwith selection (h = 1.0) here
        kde = gaussian_kde(residuals, bw_method=1.0) 
        x_grid = np.linspace(-40, 40, 1000) # Fixed x-axis for fair visual comparison
        pdf_values = kde(x_grid)
        
        # Plot KDE Line
        plt.plot(x_grid, pdf_values, color=colors[model_name], lw=2, 
                 label=f"{model_name.upper()} (CI Width: {data['upper'] - data['lower']:.2f} dB)")
        
        # Shade the 95% Confidence Interval
        plt.fill_between(x_grid, 0, pdf_values, 
                         where=(x_grid >= data['lower']) & (x_grid <= data['upper']), 
                         color=colors[model_name], alpha=0.1)

    plt.title(f'Error Analysis: 95% Confidence Intervals ({mode.upper()} Mode)', fontsize=14)
    plt.xlabel('Prediction Error (True SINR - Predicted SINR) [dB]')
    plt.ylabel('Probability Density')
    plt.xlim(-30, 30) # Zoom in on the relevant area
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Save the plot dynamically
    save_dir = f"data/results/{mode}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/error_comparison_all_models.png", dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    perform_error_analysis()