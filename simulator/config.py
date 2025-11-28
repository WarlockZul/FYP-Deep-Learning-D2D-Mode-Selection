import numpy as np

class SimulationConfig:
    # --- Simulation Settings ---
    NUM_EPISODES = 50               # How many separate 100-second runs to do
    STEPS_PER_EPISODE = 100         # Duration of each run (seconds)
    D2D_MAX_DIST_M = 50             # Maximum distance to consider D2D pairing feasible
    SEED = 42                       # For reproducibility
    OUTPUT_FILE = "data/test_simulation_data.csv"
    
    # --- Mobility Settings ---
    PROBABILITY_START_MOVING = 0.10 # Probability that a UE starts moving in a time step

    # --- Channel Model Settings ---
    USE_RANDOM_SHADOWING = False    # Set to True to use range (4-8), False for fixed (6)
    USE_RAYLEIGH_FADING = True      # Toggle fast fading True: on, False: off
    INTERFERENCE_LOAD_FACTOR = 1.0  # Default to 1.0 (Full load/Worst Case). Lower values reduce interference.
    BANDWIDTH_HZ = 10e6             # System Bandwidth (10 MHz is standard for LTE)

    #####################################################################################################

    # --- General Network Settings ---
    CARRIER_FREQ_MHZ = 700          # Carrier frequency in MHz
    BANDWIDTH_HZ = 10e6             # System Bandwidth (10 MHz is standard for LTE)
    CELL_RADIUS_M = 500             # Radius of the single cell
    NOISE_POWER_DBM = -174 + 10 * np.log10(BANDWIDTH_HZ) # Thermal Noise in dBm
    
    # --- Transmit Power Settings (in dBm) ---
    TX_POWER_BS_DBM = 46            # Base Station Transmit Power
    TX_POWER_D2D_DBM = 23           # D2D Device Transmit Power
    
    # --- Channel Modeling Coefficients ---
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
    
    # Shadowing and Fading Parameters
    SHADOWING_SIGMA_DB = 6          # Fixed value (Average Urban)
    SHADOWING_SIGMA_MIN = 4         # Minimum spread (Cleaner env)
    SHADOWING_SIGMA_MAX = 8         # Maximum spread (Dense Urban)
    
    # --- Mobility & Environment ---
    TIME_STEP_S = 1                 # Delta t = 1s
    NUM_INTERFERING_DEVICES = 20    # Number of interferers per time step
    
    # Speed ranges (m/s)
    SPEED_MIN = 1                   # Pedestrian
    SPEED_MAX = 10                  # Moderate mobility

    # Helper to convert Noise dBm to Watts
    @staticmethod
    def get_noise_power_watts():
        return 10 ** ((SimulationConfig.NOISE_POWER_DBM - 30) / 10)

    # Helper to convert BS dBm to Watts
    @staticmethod
    def get_bs_power_watts():
        return 10 ** ((SimulationConfig.TX_POWER_BS_DBM - 30) / 10)

    # Helper to convert D2D dBm to Watts
    @staticmethod
    def get_d2d_power_watts():
        return 10 ** ((SimulationConfig.TX_POWER_D2D_DBM - 30) / 10)