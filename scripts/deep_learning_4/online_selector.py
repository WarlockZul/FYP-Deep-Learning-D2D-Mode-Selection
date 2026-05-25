import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from threshold_selection import ThresholdSelector

current_dir = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(current_dir, ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))

from ml_config import MLConfig

class OnlineModeSelector:
    # Initialize the selector with the chosen model and constraint type.
    # NOTE: This workflow only uses models that are trained in D2D mode only. 
    def __init__(self, model_name='cnn', dataset_folder='preprocessed_paper', constraint_type=MLConfig.CONSTRAINT_TYPE, target_tput_mbps=MLConfig.TARGET_THROUGHPUT_MBPS):
        self.model_name = model_name
        self.dataset_folder = dataset_folder
        self.constraint_type = constraint_type

        # Fetch the target throughput from the config, or use the default provided value. 
        if target_tput_mbps is None:
            target_tput_mbps = MLConfig.TARGET_THROUGHPUT_MBPS.get(self.dataset_folder, 1.0)
        
        # Initialize parameters
        print(f"Initializing Online Mode Selector ({model_name.upper()} model | {constraint_type} constraint)")
        d2d_pkl_path = os.path.join(PROJECT_ROOT, "models", self.dataset_folder, MLConfig.EXPERIMENT_NAME, "d2d", model_name, f"{model_name}_error_params_kde.pkl")
        cell_pkl_path = os.path.join(PROJECT_ROOT, "models", self.dataset_folder, MLConfig.EXPERIMENT_NAME, "cellular", model_name, f"{model_name}_error_params_kde.pkl")
        
        # Load Error Parameters (specifically the residuals for margin calculation)
        with open(d2d_pkl_path, "rb") as f:
            res_d2d = pickle.load(f)['residuals_data']
            
        with open(cell_pkl_path, "rb") as f:
            res_cell = pickle.load(f)['residuals_data']
        
        # Initialize Threshold Selector
        self.ts = ThresholdSelector(bandwidth_hz=100e6)
        
        # Calculate base threshold needed to achieve the target Mbps
        self.base_threshold_db = self.ts.required_sinr_for_throughput(target_tput_mbps)
        
        # Calculate the specific margin (buffer) once based on the chosen constraint type and the residuals from the error analysis module.
        # Set margins to 0.0 dB if KDE is disabled (ablation test)
        if not MLConfig.USE_KDE:
            self.margin_d2d, self.margin_cell = 0.0, 0.0
            print("⚠️ KDE Correction is DISABLED (Ablation Test). Safety Margins forced to 0.0 dB.")
        elif self.constraint_type == 'AR':
            self.margin_d2d = self.ts.get_ar_margin(res_d2d, epsilon=MLConfig.AR_EPSILON)
            self.margin_cell = self.ts.get_ar_margin(res_cell, epsilon=MLConfig.AR_EPSILON)
        elif self.constraint_type == 'PCR':
            self.margin_d2d = self.ts.get_pcr_margin(res_d2d, xi=MLConfig.PCR_XI, confidence=MLConfig.PCR_CONFIDENCE)
            self.margin_cell = self.ts.get_pcr_margin(res_cell, xi=MLConfig.PCR_XI, confidence=MLConfig.PCR_CONFIDENCE)
        else:
            self.margin_d2d, self.margin_cell = 0.0, 0.0

        print(f"Target Throughput: {target_tput_mbps} Mbps requires base SINR of {self.base_threshold_db:.2f} dB")
        print(f"Safety Margins Applied -> D2D: {self.margin_d2d:.2f} dB | Cellular: {self.margin_cell:.2f} dB\n")
    
    # Executes the flowchart logic using PRE-CALCULATED predictions.
    def make_decision(self, pred_d2d, pred_cell, current_mode):
        new_mode = current_mode

        if current_mode == 'D2D':
            # Step 2: Confidence Bound Correction (Step 1 is now done outside)
            corrected_pred = self.ts.get_corrected_sinr(pred_d2d, self.margin_d2d)
            
            # Step 3: Flowchart Logic: If D2D prediction falls below threshold, switch to Cellular.
            if corrected_pred < self.base_threshold_db:
                new_mode = 'Cellular'
            else:
                new_mode = 'D2D'
                
        elif current_mode == 'Cellular':
            # Step 2: Confidence Bound Correction
            corrected_pred = self.ts.get_corrected_sinr(pred_cell, self.margin_cell)
            
            # Step 3: Flowchart Logic: If Cellular prediction falls below threshold, switch to D2D.
            if corrected_pred < self.base_threshold_db:
                new_mode = 'D2D'
            else:
                new_mode = 'Cellular'
        
        # Return the logs
        log_data = {
            'predicted_sinr': pred_d2d if current_mode == 'D2D' else pred_cell,
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
    STARTING_MODE = 'D2D'        # Options: 'D2D' or 'Cellular'
    DATASET = 'preprocessed_proposal'
    
    # Initialize the master controller
    selector = OnlineModeSelector(model_name=SELECTED_MODEL, dataset_folder=DATASET)

    # The make_decision function expects SINR floats (predictions), not massive NumPy arrays
    dummy_pred_d2d = 5.0
    dummy_pred_cell = 10.0
        
    print(f"--- Simulating 1 Timestep ---")
    print(f"Starting Mode: {STARTING_MODE}")
    
    # Pass both feature sets into the decision engine
    final_mode, logs = selector.make_decision(dummy_pred_d2d, dummy_pred_cell, current_mode=STARTING_MODE)
    
    print("\nDecision Output Logs:")
    for key, value in logs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")