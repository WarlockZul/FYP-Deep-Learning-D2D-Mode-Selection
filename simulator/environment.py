import numpy as np
from typing import Optional, List
from simulator.config import SimulationConfig
from simulator.channel_model import ChannelModel
from simulator.entities import BaseStation, UserEquipment

class D2DEnvironment:
    def __init__(self):
        self.bs = BaseStation()
        # TYPE HINTING: We tell Python these will be UserEquipment objects
        # This fixes the "attribute of None" red squiggles
        self.d2d_tx: Optional[UserEquipment] = None 
        self.d2d_rx: Optional[UserEquipment] = None
        self.interferers: list[UserEquipment] = []
        
        self.time_step = 0
        
        # Initialize the environment immediately
        self.reset()
        
    def reset(self):
        """
        Resets the simulation for a new 'episode'.
        """
        self.time_step = 0
        
        # 1. Create the D2D Pair (Tx and Rx)
        self.d2d_tx = UserEquipment(device_id="Target_Tx", speed_type='mixed')
        
        # Create Rx close to Tx
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(10, SimulationConfig.D2D_MAX_DIST_M)
        
        rx_pos = self.d2d_tx.position + np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        self.d2d_rx = UserEquipment(device_id="Target_Rx", speed_type='mixed')
        self.d2d_rx.position = rx_pos 
        
        # 2. Create Interfering Devices
        num_interferers = np.random.randint(10, 21)
        self.interferers = [UserEquipment(f"Int_{i}", speed_type='vehicle') for i in range(num_interferers)]
        
        return self.get_state()

    def step(self):
        assert self.d2d_tx is not None
        assert self.d2d_rx is not None
        
        self.time_step += 1
        
        # --- 1. Mobility ---
        self.d2d_tx.move()
        self.d2d_rx.move()
        for device in self.interferers:
            device.move()
            
        # --- 2. Calculate Distances ---
        dist_d2d = self.d2d_tx.get_distance_to(self.d2d_rx)
        dist_cellular = self.bs.get_distance_to(self.d2d_rx)
        
        # --- 3. Signal Power ---
        s_d2d_watts = ChannelModel.compute_received_power(
            self.d2d_tx.tx_power_dbm, dist_d2d, is_d2d=True
        )
        
        s_cell_watts = ChannelModel.compute_received_power(
            self.bs.tx_power_dbm, dist_cellular, is_d2d=False
        )
        
        # --- 4. Interference ---
        total_interference_watts = 0.0
        for interferer in self.interferers:
            dist_int_to_rx = interferer.get_distance_to(self.d2d_rx)
            i_watts = ChannelModel.compute_received_power(
                interferer.tx_power_dbm, dist_int_to_rx, is_d2d=True
            )
            total_interference_watts += i_watts
            
        # --- 5. Noise ---
        noise_watts = SimulationConfig.get_noise_power_watts()
        
        # --- 6. SINR ---
        sinr_d2d_linear = s_d2d_watts / (total_interference_watts + noise_watts)
        sinr_d2d_db = 10 * np.log10(sinr_d2d_linear)
        
        sinr_cell_linear = s_cell_watts / (total_interference_watts + noise_watts)
        sinr_cell_db = 10 * np.log10(sinr_cell_linear)
        
        # --- 7. Throughput ---
        tput_d2d_mbps = (SimulationConfig.BANDWIDTH_HZ * np.log2(1 + sinr_d2d_linear)) / 1e6
        tput_cell_mbps = (SimulationConfig.BANDWIDTH_HZ * np.log2(1 + sinr_cell_linear)) / 1e6
        
        # --- 8. Optimal Mode ---
        optimal_mode = "D2D" if tput_d2d_mbps >= tput_cell_mbps else "Cellular"
        
        # --- 9. Pack Data ---
        state = {
            "timestamp": self.time_step,
            "tx_pos_x": self.d2d_tx.position[0],
            "tx_pos_y": self.d2d_tx.position[1],
            "rx_pos_x": self.d2d_rx.position[0],
            "rx_pos_y": self.d2d_rx.position[1],
            "distance_tx_rx": dist_d2d,
            "distance_bs_rx": dist_cellular,
            "sinr_d2d_db": sinr_d2d_db,
            "sinr_cell_db": sinr_cell_db,
            "interference_dbm": 10 * np.log10(total_interference_watts * 1000),
            "throughput_d2d_mbps": tput_d2d_mbps,
            "throughput_cell_mbps": tput_cell_mbps,
            "optimal_mode": optimal_mode
        }
        
        return state

    def get_state(self):
        pass