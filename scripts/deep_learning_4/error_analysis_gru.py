import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle

def load_validation_data():
    base_path = "data/model_ready"
    X_val = np.load(os.path.join(base_path, "X_val.npy"))
    y_val = np.load(os.path.join(base_path, "y_val.npy"))
    return X_val, y_val

def perform_error_analysis():
    print("Loading Validation Data & Model...")
    X_val, y_val = load_validation_data()
    
    model_path = "models/gru/gru_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run 'train_gru.py' first.")
    model = tf.keras.models.load_model(model_path) # type: ignore
    
    # --- Step 8.1: Compute Residuals ---
    print("\n--- Step 8.1: Compute Residuals ---")
    y_pred = model.predict(X_val, verbose=0)
    residuals = (y_val.flatten() - y_pred.flatten())
    
    print(f"Total Residuals: {len(residuals)}")
    print(f"Mean Residual: {np.mean(residuals):.4f} dB")
    print(f"Std Dev: {np.std(residuals):.4f} dB")

    # --- Step 8.2: Fit Gaussian KDE (For Visualization) ---
    print("\n--- Step 8.2: Fit Gaussian KDE ---")
    # We use the KDE for the SHAPE (the blue curve), but not for the BOUNDS.
    bw_method = 0.5  
    kde = gaussian_kde(residuals, bw_method=bw_method) 
    
    x_grid = np.linspace(min(residuals) - 1.0, max(residuals) + 1.0, 2000)
    pdf_values = kde(x_grid)

    # --- Step 8.3: Derive Confidence Intervals (Using Raw Data) ---
    print("\n--- Step 8.3: Derive 95% Confidence Intervals ---")
    
    # NOTE: REVERTED TO EMPIRICAL PERCENTILE (More Accurate/Tighter)
    # We want the actual 2.5% and 97.5% cutoffs of the real data.
    lower_bound = np.percentile(residuals, 2.5)
    upper_bound = np.percentile(residuals, 97.5)
    
    print(f"95% Confidence Interval: [{lower_bound:.4f} dB, {upper_bound:.4f} dB]")

    # --- Step 8.4: Save Parameters ---
    error_params = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'residuals_data': residuals,
        'bandwidth': bw_method
    }
    os.makedirs("models/gru", exist_ok=True)
    save_path = "models/gru/gru_error_params_kde.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(error_params, f)
    print(f"Error parameters saved to '{save_path}'")

    # ==========================================
    # FIGURE 1: THE ORIGINAL HISTOGRAM (Result)
    # ==========================================
    print("\nGenerating Figure 1: Original Error Distribution...")
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(residuals, bins=50, density=True, alpha=0.5, color='gray', label='Residual Histogram')
    
    # KDE Line
    plt.plot(x_grid, pdf_values, color='blue', linewidth=2, label='KDE (Gaussian Fit)')
    
    # Confidence Lines
    plt.axvline(lower_bound, color='red', linestyle='--', label='95% CI Lower')
    plt.axvline(upper_bound, color='red', linestyle='--', label='95% CI Upper')
    
    plt.title(f'Error Analysis: Residual Distribution (KDE)\n95% CI: [{lower_bound:.2f}, {upper_bound:.2f}]')
    plt.xlabel('Prediction Error (True SINR - Predicted SINR) [dB]')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # FIGURE 2: PDF vs CDF (Analysis/Math)
    # ==========================================
    # We re-calculate the CDF from the empirical data to match the bounds
    print("Generating Figure 2: PDF & CDF Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PLOT 2A: PDF (Visualizing the Trust Zone)
    axes[0].plot(x_grid, pdf_values, color='blue', lw=2, label='PDF (Eq. 7)')
    axes[0].fill_between(x_grid, 0, pdf_values, where=(x_grid >= lower_bound) & (x_grid <= upper_bound), 
                         color='blue', alpha=0.1, label='Trusted Region (95%)')
    axes[0].axvline(lower_bound, color='red', linestyle='--')
    axes[0].axvline(upper_bound, color='red', linestyle='--')
    axes[0].set_title("PDF: Error Distribution", fontsize=12)
    axes[0].set_xlabel("Error (dB)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PLOT 2B: CDF (Deriving the Bounds)
    # Use Empirical CDF (sorted data) for exact match
    sorted_residuals = np.sort(residuals)
    y_vals = np.arange(len(sorted_residuals)) / float(len(sorted_residuals) - 1)
    
    axes[1].plot(sorted_residuals, y_vals, color='green', lw=2, label='Empirical CDF')
    axes[1].axhline(0.025, color='red', linestyle=':', label='2.5% Cutoff')
    axes[1].axhline(0.975, color='red', linestyle=':', label='97.5% Cutoff')
    axes[1].axvline(lower_bound, color='red', linestyle='--')
    axes[1].axvline(upper_bound, color='red', linestyle='--')
    axes[1].scatter([lower_bound, upper_bound], [0.025, 0.975], color='red', zorder=5)
    
    axes[1].set_title("CDF: Deriving Confidence Interval (Eq. 8)", fontsize=12)
    axes[1].set_xlabel("Error (dB)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    perform_error_analysis()