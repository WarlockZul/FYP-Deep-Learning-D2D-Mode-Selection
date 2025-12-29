import numpy as np
from simulator_paper.config import PaperConfig

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
        
        # D2D Path Loss: 32.45 + 20 * log10(f_MHz) + 20 * log10(d_km)
        # NOTE: [Refer to research paper]
        pl = PaperConfig.PATH_LOSS_D2D_A + \
             (PaperConfig.PATH_LOSS_D2D_B_FREQ * np.log10(freq_mhz)) + \
             (PaperConfig.PATH_LOSS_D2D_C_DIST * np.log10(distance_km))
        return pl

    # Helper to compute final received power in Watts considering all channel effects.
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