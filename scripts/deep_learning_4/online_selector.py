import os
import numpy as np
import pickle
import tensorflow as tf
from threshold_selection import ThresholdSelector

class OnlineModeSelector:
    # Initialize the selector with the chosen model and constraint type.
    # NOTE: This workflow only uses models that are trained in D2D mode only. 
    def __init__(self, model_name='cnn', constraint_type='AR', target_tput_mbps=10.0):
        self.model_name = model_name
        self.constraint_type = constraint_type
        
        # Initialize parameters
        print(f"Initializing Online Mode Selector ({model_name.upper()} model | {constraint_type} constraint)")
        
        # Load the trained DL Model (specifically the weights and architecture)
        model_path = f"models/{model_name}/{model_name}_model.keras"
        self.model = tf.keras.models.load_model(model_path)
        
        # Load Error Parameters (specifically the residuals for margin calculation)
        pkl_path = f"models/{model_name}/{model_name}_error_params_kde.pkl"
        with open(pkl_path, "rb") as f:
            self.error_params = pickle.load(f)
            
        self.residuals = self.error_params['residuals_data']
        
        # Initialize Threshold Selector
        self.ts = ThresholdSelector(bandwidth_hz=100e6)
        
        # Calculate Base Threshold (Algorithm 1, Lines 7 & 16)
        self.base_threshold_db = self.ts.required_sinr_for_throughput(target_tput_mbps)
        
        # Calculate the specific margin (buffer) once based on the chosen constraint
        if self.constraint_type == 'AR':
            self.margin_db = self.ts.get_ar_margin(self.residuals, epsilon=0.7)
        elif self.constraint_type == 'PCR':
            self.margin_db = self.ts.get_pcr_margin(self.residuals, xi=0.05, confidence=0.95)
        else:
            self.margin_db = 0.0 # No correction baseline
            
        print(f"Target Throughput: {target_tput_mbps} Mbps requires base SINR of {self.base_threshold_db:.2f} dB")
        print(f"Applied {constraint_type} Safety Margin: {self.margin_db:.2f} dB\n")

    # Executes the flowchart logic (from the research paper) to decide whether to switch modes based on the current features and mode.
    def make_decision(self, current_features, current_mode):
        # Step 1: Predict SINR in D2D Mode
        raw_pred = self.model.predict(current_features, verbose=0)[0][0]
        
        # Step 2: Confidence Bound Correction
        corrected_pred = self.ts.get_corrected_sinr(raw_pred, self.margin_db)
        
        # Step 3: Flowchart Decision Logic
        # If its D2D mode and predicted value below the threshold, then switch to Cellular mode.
        # If its Cellular mode and predicted value above the threshold, then switch to D2D mode.
        new_mode = current_mode
        if current_mode == 'D2D':
            if corrected_pred < self.base_threshold_db:
                new_mode = 'Cellular' # Switch to Cellular Mode
            else:
                new_mode = 'D2D'      # Stay in D2D Mode
        elif current_mode == 'Cellular':
            if corrected_pred >= self.base_threshold_db:
                new_mode = 'D2D'      # Switch to D2D Mode
            else:
                new_mode = 'Cellular' # Stay in Cellular Mode

        # Return the logs required by Step 10
        log_data = {
            'predicted_sinr': raw_pred,
            'corrected_sinr': corrected_pred,
            'threshold_used': self.base_threshold_db,
            'old_mode': current_mode,
            'new_mode': new_mode,
            'switched': current_mode != new_mode
        }
        
        return new_mode, log_data

# ==========================================
# QUICK TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- USER CONTROLS ---
    SELECTED_MODEL = 'cnn'       # Options: 'gru', 'lstm', 'cnn', 'dnn'
    CONSTRAINT_TYPE = 'AR'       # Options: 'AR' or 'PCR'
    STARTING_MODE = 'D2D'        # Options: 'D2D' or 'Cellular'
    TARGET_THROUGHPUT = 15.0     # Target speed in Mbps
    # ---------------------
    
    # 1. Initialize the master controller
    selector = OnlineModeSelector(
        model_name=SELECTED_MODEL, 
        constraint_type=CONSTRAINT_TYPE, 
        target_tput_mbps=TARGET_THROUGHPUT
    )
    
    # 2. Create dummy input data matching the model's expected shape
    if SELECTED_MODEL in ['cnn', 'dnn']:
        dummy_input = np.zeros((1, 16, 20))  # 1 sample, 16 window steps, 20 features
    else:
        dummy_input = np.zeros((1, 300, 20)) # 1 sample, 300 timesteps, 20 features
    
    # 3. Ask the system to make a decision
    print(f"--- Simulating 1 Timestep ---")
    print(f"Current Mode: {STARTING_MODE}")
    new_mode, logs = selector.make_decision(dummy_input, current_mode=STARTING_MODE)
    
    print("\nDecision Output Logs:")
    for key, value in logs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")