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
        
        # Initialize the environment immediately
        self.reset()
        
    def reset(self):
        # Reset time step for new episode
        self.time_step = 0
 
        # Create the D2D Pair (Tx and Rx)
        self.d2d_tx = UserEquipment(device_id="Target_Tx")
        self.d2d_rx = UserEquipment(device_id="Target_Rx")

        # Spawn Rx within D2D_MAX_DIST_M from Tx
        self.d2d_rx.position = self.d2d_rx._get_random_point_in_cell()
        
        # To ensure Rx is within D2D_MAX_DIST_M from Tx
        dist = self.d2d_tx.get_distance_to(self.d2d_rx)
        if dist > PaperConfig.D2D_MAX_DIST_M:
            pass

        # Create interferers (number fixed as per research paper)
        num_interferers = PaperConfig.NUM_INTERFERER
        
        # Create interferers list
        self.interferers = [
            UserEquipment(f"Int_{i}") 
            for i in range(num_interferers)
        ]
        
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
        
        # Calculate received signal powers (Watts)
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
            
            # Apply Load Factor (Rho): sum(rho_k * P_k * G_kj) 
            # NOTE: [Refer to research paper]
            total_interference_watts += (rho * raw_interference_watts)
            
        # Calculate Noise Power (Watts) (Sigma^2)
        # NOTE: [Refer to research paper]
        noise_watts = PaperConfig.get_noise_power_watts()
        
        # Calculate SINR in linear scale and then convert to dB
        # SINR = S / (I + N)
        sinr_d2d_linear = s_d2d_watts / (total_interference_watts + noise_watts)
        sinr_d2d_db = 10 * np.log10(sinr_d2d_linear)
        sinr_cell_linear = s_cellular_watts / (total_interference_watts + noise_watts)
        sinr_cell_db = 10 * np.log10(sinr_cell_linear)
        
        # Calculate Throughputs (Mbps) using Shannon Capacity (shannon-Hartley Theorem)
        # C = B * log2(1 + SINR)
        # C: Throughput (Mbps), B: Bandwidth (Hz)
        # NOTE: Throughput calculated in Mbps, hence division by 1e6
        tput_d2d_mbps = (PaperConfig.BANDWIDTH_HZ * np.log2(1 + sinr_d2d_linear)) / 1e6
        tput_cell_mbps = (PaperConfig.BANDWIDTH_HZ * np.log2(1 + sinr_cell_linear)) / 1e6
        
        # Choose Optimal Mode based on Higher Throughput for this time step
        optimal_mode = "D2D" if tput_d2d_mbps >= tput_cell_mbps else "Cellular"
        
        # Create state list/dictionary to return
        state = {
            "timestamp": self.time_step,
            "episode_id": getattr(self, 'episode_id', 0),
            
            "tx_pos_x": self.d2d_tx.position[0],
            "tx_pos_y": self.d2d_tx.position[1],
            "rx_pos_x": self.d2d_rx.position[0],
            "rx_pos_y": self.d2d_rx.position[1],
            
            "distance_tx_rx": dist_d2d,
            "distance_bs_rx": dist_cellular,

            "tx_speed_mps": self.d2d_tx.speed,
            "rx_speed_mps": self.d2d_rx.speed,
            
            "tx_power_d2d_dbm": self.d2d_tx.tx_power_dbm,
            "tx_power_bs_dbm": self.bs.tx_power_dbm,
            
            "rx_power_d2d_dbm": 10 * np.log10(s_d2d_watts * 1000),
            "rx_power_cell_dbm": 10 * np.log10(s_cellular_watts * 1000),
            
            "interference_dbm": 10 * np.log10(total_interference_watts * 1000),
            "noise_dbm": 10 * np.log10(noise_watts * 1000),
            
            "sinr_d2d_db": sinr_d2d_db,
            "sinr_cell_db": sinr_cell_db,
            
            "throughput_d2d_mbps": tput_d2d_mbps,
            "throughput_cell_mbps": tput_cell_mbps,
            
            "optimal_mode": optimal_mode
        }
        
        return state

    def get_state(self):
        pass