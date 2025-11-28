import numpy as np
from typing import Optional, List
from simulator.config import SimulationConfig
from simulator.channel_model import ChannelModel
from simulator.entities import BaseStation, UserEquipment

class D2DEnvironment:
    def __init__(self):
        self.bs = BaseStation()
        self.d2d_tx: Optional[UserEquipment] = None 
        self.d2d_rx: Optional[UserEquipment] = None
        self.interferers: list[UserEquipment] = []
        self.time_step = 0
        
        # Initialize the environment immediately
        self.reset()
        
    def reset(self):
        # Reset time step for new episode
        self.time_step = 0
        
        # Create the D2D Pair (Tx and Rx)
        self.d2d_tx = UserEquipment(device_id="Target_Tx", speed_type='mixed')
        self.d2d_rx = UserEquipment(device_id="Target_Rx", speed_type='mixed')
        
        # Override Rx position to be within max D2D distance from Tx
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(10, SimulationConfig.D2D_MAX_DIST_M)
        rx_pos = self.d2d_tx.position + np.array([radius * np.cos(angle), radius * np.sin(angle)])
        self.d2d_rx.position = rx_pos 
        
        # Create Interfering Devices
        num_interferers = np.random.randint(10, 21)
        self.interferers = [UserEquipment(f"Int_{i}", speed_type='vehicle') for i in range(num_interferers)]
        
        return self.get_state()

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
            
        # Calculate distances between entities (D2D: Tx to Rx, Cellular: BS to Rx)
        dist_d2d = self.d2d_tx.get_distance_to(self.d2d_rx)
        dist_cellular = self.bs.get_distance_to(self.d2d_rx)
        
        # Calculate Received Signal Powers (Watts) 
        s_d2d_watts = ChannelModel.compute_received_power(
            self.d2d_tx.tx_power_dbm, dist_d2d, is_d2d=True
        )
        s_cellular_watts = ChannelModel.compute_received_power(
            self.bs.tx_power_dbm, dist_cellular, is_d2d=False
        )
        
        # Calculate Interference Power from all interferers (Watts)
        total_interference_watts = 0.0
        rho = SimulationConfig.INTERFERENCE_LOAD_FACTOR
        for interferer in self.interferers:
            dist_int_to_rx = interferer.get_distance_to(self.d2d_rx)
            
            # Raw received power from interferer (P_k * G_kj)
            raw_interference_watts = ChannelModel.compute_received_power(
                interferer.tx_power_dbm, dist_int_to_rx, is_d2d=True
            )
            
            # Apply Load Factor (Rho): sum(rho * P * G)
            total_interference_watts += (rho * raw_interference_watts)
            
        # Calculate Noise Power (Watts) 
        noise_watts = SimulationConfig.get_noise_power_watts()
        
        # Calculate SINR in linear scale and then convert to dB
        sinr_d2d_linear = s_d2d_watts / (total_interference_watts + noise_watts)
        sinr_d2d_db = 10 * np.log10(sinr_d2d_linear)
        sinr_cell_linear = s_cellular_watts / (total_interference_watts + noise_watts)
        sinr_cell_db = 10 * np.log10(sinr_cell_linear)
        
        # Calculate Throughputs (Mbps) using Shannon Capacity (Shannon-Hartley Theorem)
        tput_d2d_mbps = (SimulationConfig.BANDWIDTH_HZ * np.log2(1 + sinr_d2d_linear)) / 1e6
        tput_cell_mbps = (SimulationConfig.BANDWIDTH_HZ * np.log2(1 + sinr_cell_linear)) / 1e6
        
        # Choose Optimal Mode based on Higher Throughput for this time step
        optimal_mode = "D2D" if tput_d2d_mbps >= tput_cell_mbps else "Cellular"
        
        # --- Pack Data ---
        state = {
            "timestamp": self.time_step,
            "episode_id": getattr(self, 'episode_id', 0), # Safe fallback if not set
            
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
            # Convert Watts to dBm: 10*log10(P*1000)
            "rx_power_d2d_dbm": 10 * np.log10(s_d2d_watts * 1000),
            "rx_power_cell_dbm": 10 * np.log10(s_cellular_watts * 1000),
            
            # Interference & Noise 
            "interference_dbm": 10 * np.log10(total_interference_watts * 1000),
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

    def get_state(self):
        pass