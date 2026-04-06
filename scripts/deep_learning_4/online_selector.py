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
        path_d2d = f"models/d2d/{model_name}/{model_name}_model.keras"
        path_cell = f"models/cellular/{model_name}/{model_name}_model.keras"
        
        self.model_d2d = tf.keras.models.load_model(path_d2d)
        self.model_cell = tf.keras.models.load_model(path_cell)
        
        # Load Error Parameters (specifically the residuals for margin calculation)
        with open(f"models/d2d/{model_name}/{model_name}_error_params_kde.pkl", "rb") as f:
            res_d2d = pickle.load(f)['residuals_data']
            
        with open(f"models/cellular/{model_name}/{model_name}_error_params_kde.pkl", "rb") as f:
            res_cell = pickle.load(f)['residuals_data']
        
        # Initialize Threshold Selector
        self.ts = ThresholdSelector(bandwidth_hz=100e6)
        
        # Calculate base threshold needed to achieve the target Mbps
        self.base_threshold_db = self.ts.required_sinr_for_throughput(target_tput_mbps)
        
        # Calculate the specific margin (buffer) once based on the chosen constraint type and the residuals from the error analysis module.
        if self.constraint_type == 'AR':
            self.margin_d2d = self.ts.get_ar_margin(res_d2d, epsilon=0.7)
            self.margin_cell = self.ts.get_ar_margin(res_cell, epsilon=0.7)
        elif self.constraint_type == 'PCR':
            self.margin_d2d = self.ts.get_pcr_margin(res_d2d, xi=0.05, confidence=0.95)
            self.margin_cell = self.ts.get_pcr_margin(res_cell, xi=0.05, confidence=0.95)
        else:
            self.margin_d2d, self.margin_cell = 0.0, 0.0

        print(f"Target Throughput: {target_tput_mbps} Mbps requires base SINR of {self.base_threshold_db:.2f} dB")
        print(f"Safety Margins Applied -> D2D: {self.margin_d2d:.2f} dB | Cellular: {self.margin_cell:.2f} dB\n")

    # Executes the flowchart logic (from the research paper) to decide whether to switch modes based on the current features and mode.
    def make_decision(self, feature_d2d, feature_cell, current_mode):
        new_mode = current_mode

        if current_mode == 'D2D':
            # Step 1: Predict SINR in D2D Mode
            raw_pred = self.model_d2d.predict(feature_d2d, verbose=0)[0][0]

            # Step 2: Confidence Bound Correction
            corrected_pred = self.ts.get_corrected_sinr(raw_pred, self.margin_d2d)
            
            # Step 3: Flowchart Logic: If D2D prediction falls below threshold, switch to Cellular.
            if corrected_pred < self.base_threshold_db:
                new_mode = 'Cellular'
            else:
                new_mode = 'D2D'
        elif current_mode == 'Cellular':
            # Step 1: Predict SINR in Cellular Mode
            raw_pred = self.model_cell.predict(feature_cell, verbose=0)[0][0]

            # Step 2: Confidence Bound Correction
            corrected_pred = self.ts.get_corrected_sinr(raw_pred, self.margin_cell)
            
            # Step 3: Flowchart Logic: If Cellular prediction falls below threshold, switch to D2D.
            if corrected_pred < self.base_threshold_db:
                new_mode = 'D2D'
            else:
                new_mode = 'Cellular'
        
        # Return the logs of simulation parameters (predicted & corrected SINR, threshold used, old & new mode)
        log_data = {
            'predicted_sinr': raw_pred,
            'corrected_sinr': corrected_pred,
            'threshold_used': self.base_threshold_db,
            'old_mode': current_mode,
            'new_mode': new_mode,
            'switched': current_mode != new_mode
        }
        
        return new_mode, log_data

# Function to test run the online selector with dummy data
if __name__ == "__main__":
    SELECTED_MODEL = 'cnn'       # Options: 'gru', 'lstm', 'cnn', 'dnn'
    CONSTRAINT_TYPE = 'AR'       # Options: 'AR' or 'PCR'
    STARTING_MODE = 'D2D'        # Options: 'D2D' or 'Cellular'
    TARGET_THROUGHPUT = 10.0     # Target speed in Mbps
    
    # Initialize the master controller
    selector = OnlineModeSelector(
        model_name=SELECTED_MODEL, 
        constraint_type=CONSTRAINT_TYPE, 
        target_tput_mbps=TARGET_THROUGHPUT
    )
    
    # Dummy data arrays (Shape depends on model: (1, 16, 20) for CNN/DNN, (1, 300, 20) for GRU/LSTM)
    # CNN/DNN: (1, 16, 20) -> 1 sample, 16 window steps, 20 features
    # GRU/LSTM: (1, 300, 20) -> 1 sample, 300 timesteps, 20 features
    if SELECTED_MODEL in ['cnn', 'dnn']:
        dummy_d2d = np.zeros((1, 16, 20))
        dummy_cell = np.zeros((1, 16, 20))
    else:
        dummy_d2d = np.zeros((1, 300, 20))
        dummy_cell = np.zeros((1, 300, 20))
        
    print(f"--- Simulating 1 Timestep ---")
    print(f"Starting Mode: {STARTING_MODE}")
    
    # Pass both feature sets into the decision engine
    final_mode, logs = selector.make_decision(dummy_d2d, dummy_cell, current_mode=STARTING_MODE)
    
    print("\nDecision Output Logs:")
    for key, value in logs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")