import numpy as np
from simulator.config import SimulationConfig

class ChannelModel:
    # Helper to calculate Path Loss for cellular mode
    @staticmethod
    def calculate_path_loss_cellular(distance_m):
        # Prevent log(0) errors by setting a small min distance
        distance_m = max(distance_m, 1.0) 
        distance_km = distance_m / 1000.0
        
        # Cellular Path Loss Equation: 128.1 + 37.6 * log10(d_km)
        pl = SimulationConfig.PATH_LOSS_CELLULAR_A + \
             (SimulationConfig.PATH_LOSS_CELLULAR_B * np.log10(distance_km))
        return pl

    # Helper to calculate Path Loss for D2D mode
    @staticmethod
    def calculate_path_loss_d2d(distance_m):
        # Prevent log(0) errors by setting a small min distance
        distance_m = max(distance_m, 0.1)
        distance_km = distance_m / 1000.0
        
        # Initialize carrier frequency 
        freq_mhz = SimulationConfig.CARRIER_FREQ_MHZ
        
        # Based on standard Free Space Path Loss structure for d in km and f in MHz:
        # D2D Path Loss Equation: 32.45 + 20 * log10(f in MHz) + 20 * log10(d in km)
        pl = SimulationConfig.PATH_LOSS_D2D_A + \
             (SimulationConfig.PATH_LOSS_D2D_B_FREQ * np.log10(freq_mhz)) + \
             (SimulationConfig.PATH_LOSS_D2D_C_DIST * np.log10(distance_km))
        return pl

    # Helper to return shadowing value in dB
    @staticmethod
    def get_shadowing():
        if SimulationConfig.USE_RANDOM_SHADOWING:
            # Pick a random Sigma uniformly between 4 and 8
            sigma = np.random.uniform(
                SimulationConfig.SHADOWING_SIGMA_MIN, 
                SimulationConfig.SHADOWING_SIGMA_MAX
            )
        else:
            # Use the fixed Sigma (e.g., 6)
            sigma = SimulationConfig.SHADOWING_SIGMA_DB

        # np.random.randn() gives standard normal (mean 0, var 1)
        return sigma * np.random.randn()

    # Helper to return the power gain from Rayleigh Fading (Linear scale, not dB).
    @staticmethod
    def get_rayleigh_fading_gain():
        # Rayleigh fading implies the magnitude is Rayleigh distributed.
        # The power (magnitude^2) is Exponentially distributed with mean 1.
        if not SimulationConfig.USE_RAYLEIGH_FADING:
            return 1.0
        
        # Exponential distribution with scale=1.0 models the power gain |h|^2
        return np.random.exponential(scale=1.0)

    # Helper to compute final received power in Watts considering all channel effects.
    @staticmethod
    def compute_received_power(tx_power_dbm, distance_m, is_d2d=False):
        # Calculate Path Loss (dB)
        if is_d2d:
            path_loss_db = ChannelModel.calculate_path_loss_d2d(distance_m)
        else:
            path_loss_db = ChannelModel.calculate_path_loss_cellular(distance_m)
            
        # Calculate Shadowing (dB)
        shadowing_db = ChannelModel.get_shadowing()
        
        # Combine Large-scale fading (dB)
        # Received_dB = Transmitted_dB - PL_dB + Shadowing_dB
        rx_power_dbm = tx_power_dbm - path_loss_db + shadowing_db
        
        # Convert to Linear (Watts)
        rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
        
        # Apply Small-scale Rayleigh Fading (Linear multiplication)
        fading_gain = ChannelModel.get_rayleigh_fading_gain()
        final_rx_power_watts = rx_power_watts * fading_gain
        
        return final_rx_power_watts