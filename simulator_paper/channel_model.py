import numpy as np
from simulator_paper.config import PaperConfig

'''
SINR Formula:

SINR = (P_i * G_ij) / Sum of [(p_k * P_k * G_kj) + Sigma^2]
* Sum of interference exclude Tx-Rx pair (i,j) and include all active interferers k

Where:
- i: Transmitter index
- j: Receiver index
- k: Interferer index (excluding the Tx-Rx pair)
- P_i:      Transmission power of transmitter i (Watts)
- G_ij:     Channel gain from transmitter i to receiver j (Unitless, linear scale)
- p_k:      Expected load from the interferer k    
- P_k:      Transmission power of interferer k (Watts)
- G_kj:     Channel gain from interferer k to receiver j (Unitless, linear scale)
- Sigma^2:  Noise power at receiver j (Watts)

Provided: P_i, P_k
Manipulated: p_k, Sigma^2
Calculated: G_ij, G_kj
'''

class ChannelModelPaper:
    # Helper to calculate Path Loss for cellular mode
    @staticmethod
    def calculate_path_loss_cellular(distance_m):
        # Prevent log(0) errors by setting a small min distance
        distance_m = max(distance_m, 1.0) 
        distance_km = distance_m / 1000.0
        
        # Cellular Path Loss Equation: 128.1 + 37.6 * log10(d_km)
        # NOTE: [Refer to research paper]
        pl = PaperConfig.PATH_LOSS_CELLULAR_A + \
             (PaperConfig.PATH_LOSS_CELLULAR_B * np.log10(distance_km))
        return pl

    # Helper to calculate Path Loss for D2D mode
    @staticmethod
    def calculate_path_loss_d2d(distance_m):
        # Prevent log(0) errors by setting a small min distance
        distance_m = max(distance_m, 0.1)
        distance_km = distance_m / 1000.0
        
        # Initialize carrier frequency
        freq_mhz = PaperConfig.CARRIER_FREQ_MHZ
        
        # D2D Path Loss Equation: 32.45 + 20 * log10(f in MHz) + 20 * log10(d in km)
        # NOTE: [Refer to research paper]
        pl = PaperConfig.PATH_LOSS_D2D_A + \
             (PaperConfig.PATH_LOSS_D2D_B_FREQ * np.log10(freq_mhz)) + \
             (PaperConfig.PATH_LOSS_D2D_C_DIST * np.log10(distance_km))
        return pl

    # Helper to compute final received power in Watts considering all channel effects.
    # NOTE: Computes G_ij and multiple with P_i (not including interference or noise)
    # NOTE: [Refer to research paper]
    @staticmethod
    def compute_received_power(tx_power_dbm, distance_m, is_d2d=False):
        # 1. Calculate Path Loss (dB)
        if is_d2d:
            path_loss_db = ChannelModelPaper.calculate_path_loss_d2d(distance_m)
        else:
            path_loss_db = ChannelModelPaper.calculate_path_loss_cellular(distance_m)
        
        # 2. Calculate Received Power (dBm)
        # Rx = Tx - PL (dB)
        rx_power_dbm = tx_power_dbm - path_loss_db
        
        # 3. Convert to Linear (Watts)
        rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
        
        return rx_power_watts