import numpy as np

class PaperConfig:
    # Simulation Settings 
    NUM_EPISODES = 100              # How many separate 100-second runs to do
    STEPS_PER_EPISODE = 300         # Duration of each run (seconds)
    D2D_MAX_DIST_M = 500            # Maximum distance to consider D2D pairing feasible
    SEED = 42                       # For reproducibility
    OUTPUT_FILE = "data/test_simulation_data.csv"
    
    # Mobility Settings
    PROBABILITY_START_MOVING = 1.00 # Probability that a UE starts moving in a time step

    # Channel Model Settings
    INTERFERENCE_LOAD_FACTOR = 0.1  # Default to 1.0 (Full load/Worst Case). Lower values reduce interference.
    BANDWIDTH_HZ = 20e6             # System Bandwidth (10 MHz is standard for LTE)

    #####################################################################################################

    # General Network Settings
    CARRIER_FREQ_MHZ = 700          # Carrier frequency in MHz
    CELL_RADIUS_M = 500             # Radius of the single cell
    NOISE_POWER_DBM = -174 + 10 * np.log10(BANDWIDTH_HZ) # Thermal Noise in dBm
    
    # Transmitter Power Settings (in dBm) 
    TX_POWER_BS_DBM = 46            # Base Station Transmit Power
    TX_POWER_D2D_DBM = 23           # D2D Device Transmit Power
    
    # Channel Modelling Coefficients 
    # Equation: PL = A + B * log10(d) + C * log10(f)
    
    # Cellular Link 
    # 128.1 + 37.6 * log10(d in km) 
    PATH_LOSS_CELLULAR_A = 128.1
    PATH_LOSS_CELLULAR_B = 37.6
    
    # D2D Link 
    # 32.45 + 20 * log10(f in MHz) + 20 * log10(d in km)
    PATH_LOSS_D2D_A = 32.45
    PATH_LOSS_D2D_B_FREQ = 20       # Factor for frequency log
    PATH_LOSS_D2D_C_DIST = 20       # Factor for distance log
    
    # Mobility & Environment Settings
    TIME_STEP_S = 1                 # Delta t = 1s
    NUM_INTERFERER = 20             # Max number of interferers per time step
    
    # Speed ranges (m/s)
    SPEED = 3                       # Constant mobility

    # Helper to convert Noise dBm to Watts
    @staticmethod
    def get_noise_power_watts():
        return 10 ** ((PaperConfig.NOISE_POWER_DBM - 30) / 10)

    # Helper to convert BS dBm to Watts
    @staticmethod
    def get_bs_power_watts():
        return 10 ** ((PaperConfig.TX_POWER_BS_DBM - 30) / 10)

    # Helper to convert D2D Tx dBm to Watts
    @staticmethod
    def get_d2d_power_watts():
        return 10 ** ((PaperConfig.TX_POWER_D2D_DBM - 30) / 10)