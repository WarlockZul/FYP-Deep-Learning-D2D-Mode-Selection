import numpy as np
from typing import Optional, List
from simulator_paper.config import PaperConfig
from simulator_paper.channel_model import ChannelModelPaper
from simulator_paper.entities import BaseStation, UserEquipment

class D2DEnvironmentPaper:
    def __init__(self):
        self.bs = BaseStation()
        self.d2d_tx: Optional[UserEquipment] = None 
        self.d2d_rx: Optional[UserEquipment] = None
        self.interferers: list[UserEquipment] = []
        self.time_step = 0
        self.episode_id = 0
        self.last_optimal_mode = None
        
        # Initialize the environment immediately
        self.reset()
        
    def reset(self):
        # Increment episode ID for tracking
        self.episode_id += 1
        
        # Reset time step for new episode
        self.time_step = 0

        # Reset last optimal mode
        self.last_optimal_mode = None
        
        # Create the D2D Pair (Tx and Rx)
        self.d2d_tx = UserEquipment(device_id="Target_Tx")
        self.d2d_rx = UserEquipment(device_id="Target_Rx")
        
        # Override Rx position to be within max D2D distance from Tx
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(10, PaperConfig.D2D_MAX_DIST_M)
        rx_pos = self.d2d_tx.position + np.array([radius * np.cos(angle), radius * np.sin(angle)])
        self.d2d_rx.position = rx_pos 
        
        # Create Interfering Devices with moderate speed (20 interferers)
        num_interferers = PaperConfig.NUM_INTERFERER

        # Create interferers list and set their speed type same as D2D pair for consistency
        self.interferers = [
            UserEquipment(f"Int_{i}") 
            for i in range(num_interferers)
        ]
        
        return self._compute_physics_state()

    def step(self):
        assert self.d2d_tx is not None
        assert self.d2d_rx is not None
        
        # Increment time step
        self.time_step += 1
        
        # Move for next time step (or 1 second) 
        self.d2d_tx.move()
        self.d2d_rx.move()
        for device in self.interferers:
            device.move()
            
        # Return the new state
        return self._compute_physics_state()

    def get_state(self):
        # Return the current state (used for t=0 reset)
        return self._compute_physics_state()

    def _compute_physics_state(self):
        assert self.d2d_tx is not None
        assert self.d2d_rx is not None

        # Calculate distances between entities (D2D: Tx to Rx, Cellular: BS to Rx)
        dist_d2d = self.d2d_tx.get_distance_to(self.d2d_rx)
        dist_cellular = self.bs.get_distance_to(self.d2d_rx)
        
        # Calculate Received Signal Powers (Watts) 
        s_d2d_watts = ChannelModelPaper.compute_received_power(
            self.d2d_tx.tx_power_dbm, dist_d2d, is_d2d=True
        )
        s_cellular_watts = ChannelModelPaper.compute_received_power(
            self.bs.tx_power_dbm, dist_cellular, is_d2d=False
        )
        
        # Calculate total interference power from all interferers (Watts)
        total_interference_watts = 0.0
        rho = PaperConfig.INTERFERENCE_LOAD_FACTOR
        for interferer in self.interferers:
            dist_int_to_rx = interferer.get_distance_to(self.d2d_rx)
            
            # Raw received power from interferer (P_k * G_kj)
            raw_interference_watts = ChannelModelPaper.compute_received_power(
                interferer.tx_power_dbm, dist_int_to_rx, is_d2d=True
            )
            
            # Apply Load Factor (Rho)
            total_interference_watts += (rho * raw_interference_watts)
            
        # Calculate Noise Power (Watts)
        noise_watts = PaperConfig.get_noise_power_watts()
        
        # Calculate SINR
        # SINR = S / (I + N)
        sinr_d2d_linear = s_d2d_watts / (total_interference_watts + noise_watts)
        sinr_d2d_db = 10 * np.log10(sinr_d2d_linear)
        sinr_cell_linear = s_cellular_watts / (total_interference_watts + noise_watts)
        sinr_cell_db = 10 * np.log10(sinr_cell_linear)
        
        # Calculate Throughputs (Mbps)
        tput_d2d_mbps = (PaperConfig.BANDWIDTH_HZ * np.log2(1 + sinr_d2d_linear)) / 1e6
        tput_cell_mbps = (PaperConfig.BANDWIDTH_HZ * np.log2(1 + sinr_cell_linear)) / 1e6

        # Get Latency (Fallback to 0.05s if not in config)
        # NOTE: Not in proposal or research paper, but added for realism in switching cost
        latency = PaperConfig.HANDOVER_LATENCY_S
        penalty_factor = 1.0 - latency
        
        # Determine Optimal Mode with Handover Cost Consideration
        # T=0: No switching cost
        if self.last_optimal_mode is None:
            optimal_mode = "D2D" if tput_d2d_mbps >= tput_cell_mbps else "Cellular"
        # T>0: Consider switching cost
        else:
            # Staying in D2D costs nothing. Switching to Cell costs penalty.
            if self.last_optimal_mode == "D2D":
                payoff_stay = tput_d2d_mbps
                payoff_switch = tput_cell_mbps * penalty_factor
                optimal_mode = "D2D" if payoff_stay >= payoff_switch else "Cellular"
            # Staying in Cell costs nothing. Switching to D2D costs penalty.
            else:
                payoff_stay = tput_cell_mbps
                payoff_switch = tput_d2d_mbps * penalty_factor
                optimal_mode = "Cellular" if payoff_stay >= payoff_switch else "D2D"

        # Update history
        self.last_optimal_mode = optimal_mode
        
        # Create state dictionary
        state = {
            "timestamp": self.time_step,
            "episode_id": self.episode_id,
            
            # Positions
            "tx_pos_x": self.d2d_tx.position[0],
            "tx_pos_y": self.d2d_tx.position[1],
            "rx_pos_x": self.d2d_rx.position[0],
            "rx_pos_y": self.d2d_rx.position[1],
            
            # Distances
            "distance_tx_rx": dist_d2d,
            "distance_bs_rx": dist_cellular,

            # Speeds (m/s)
            "tx_speed_mps": self.d2d_tx.speed,
            "rx_speed_mps": self.d2d_rx.speed,
            
            # Transmit Powers (dBm)
            "tx_power_d2d_dbm": self.d2d_tx.tx_power_dbm,
            "tx_power_bs_dbm": self.bs.tx_power_dbm,
            
            # Received Signal Powers (dBm)
            "rx_power_d2d_dbm": 10 * np.log10(s_d2d_watts * 1000),
            "rx_power_cell_dbm": 10 * np.log10(s_cellular_watts * 1000),
            
            # Interference & Noise 
            "interference_dbm": 10 * np.log10(total_interference_watts * 1000) if total_interference_watts > 0 else -174,
            "noise_dbm": 10 * np.log10(noise_watts * 1000),
            
            # SINR
            "sinr_d2d_db": sinr_d2d_db,
            "sinr_cell_db": sinr_cell_db,
            
            # Throughput
            "throughput_d2d_mbps": tput_d2d_mbps,
            "throughput_cell_mbps": tput_cell_mbps,
            
            # Label
            "optimal_mode": optimal_mode
        }
        
        return state