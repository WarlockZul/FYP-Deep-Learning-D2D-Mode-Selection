import numpy as np
from simulator.config import SimulationConfig

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

class ChannelModel:
    # Helper to calculate Path Loss for cellular mode
    @staticmethod
    def calculate_path_loss_cellular(distance_m):
        # Prevent log(0) errors by setting a small min distance
        distance_m = max(distance_m, 1.0) 
        distance_km = distance_m / 1000.0
        
        # Cellular Path Loss Equation: 128.1 + 37.6 * log10(d_km)
        # NOTE: [Refer to proposal document]
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
        
        # D2D Path Loss Equation: 32.45 + 20 * log10(f in MHz) + 20 * log10(d in km)
        # NOTE: [Refer to proposal document]
        pl = SimulationConfig.PATH_LOSS_D2D_A + \
             (SimulationConfig.PATH_LOSS_D2D_B_FREQ * np.log10(freq_mhz)) + \
             (SimulationConfig.PATH_LOSS_D2D_C_DIST * np.log10(distance_km))
        return pl

    # Helper to return shadowing value in dB
    @staticmethod
    def get_shadowing():
        sigma = SimulationConfig.SHADOWING_SIGMA_DB
        return sigma * np.random.randn()

    # Helper to return the power gain from Rayleigh Fading (Linear scale, not dB).
    @staticmethod
    def get_rayleigh_fading_gain():
        # Rayleigh fading implies the magnitude is Rayleigh distributed.
        # The power (magnitude^2) is exponentially distributed with mean 1.
        if not SimulationConfig.USE_RAYLEIGH_FADING:
            return 1.0
        
        # Exponential distribution with scale=1.0 models the power gain |h|^2
        return np.random.exponential(scale=1.0)

    # Helper to compute final received power in Watts considering all channel effects.
    # NOTE: Computes G_ij and multiple with P_i (not including interference or noise)
    # NOTE: [Refer to research paper]
    @staticmethod
    def compute_received_power(tx_power_dbm, distance_m, is_d2d=False):
        # 1. Calculate Path Loss (dB)
        if is_d2d:
            path_loss_db = ChannelModel.calculate_path_loss_d2d(distance_m)
        else:
            path_loss_db = ChannelModel.calculate_path_loss_cellular(distance_m)
            
        # 2. Calculate Shadowing (dB)
        shadowing_db = ChannelModel.get_shadowing()
        
        # 3. Combine Large-scale fading (dB)
        # Received_dB = Transmitted_dB - PathLoss_dB + Shadowing_dB
        # Rx = Tx - PL (dB) + Shadowing (dB)
        rx_power_dbm = tx_power_dbm - path_loss_db + shadowing_db
        
        # 4. Convert to Linear (Watts)
        rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
        
        # 5. Apply Small-scale Rayleigh Fading (Linear multiplication)
        # Transmission Power x Channel Gain (P_i x G_ij)
        channel_gain = ChannelModel.get_rayleigh_fading_gain()
        final_rx_power_watts = rx_power_watts * channel_gain
        
        return final_rx_power_watts