import numpy as np
from simulator.config import SimulationConfig

class BaseStation:
    """
    Represents the cellular Base Station (BS).
    Located at the center (0,0) of the cell.
    """
    def __init__(self):
        # BS is always at coordinate (0,0)
        self.position = np.array([0.0, 0.0])
        self.tx_power_dbm = SimulationConfig.TX_POWER_BS_DBM

    def get_distance_to(self, other_entity):
        """Calculates Euclidean distance to another entity (UE)"""
        return np.linalg.norm(self.position - other_entity.position)

class UserEquipment:
    """
    Represents a mobile device (UE).
    Can function as a cellular user or a D2D transmitter/receiver.
    """
    def __init__(self, device_id, speed_type='mixed'):
        self.device_id = device_id
        
        # 1. Initialize Position (Uniformly distributed in circle)
        # r = R * sqrt(random) accounts for area density
        radius = SimulationConfig.CELL_RADIUS_M * np.sqrt(np.random.rand())
        angle = 2 * np.pi * np.random.rand()
        
        self.position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])
        
        # 2. Initialize Speed and Direction
        self.direction = np.random.uniform(0, 2 * np.pi) # Radians
        
        # Assign speed based on type [cite: 54]
        if speed_type == 'pedestrian':
            self.speed = np.random.uniform(1, 3)
        elif speed_type == 'vehicle':
            self.speed = np.random.uniform(3, 10)
        else:
            # Mixed scenario
            self.speed = np.random.uniform(SimulationConfig.SPEED_MIN, SimulationConfig.SPEED_MAX)
            
        self.tx_power_dbm = SimulationConfig.TX_POWER_D2D_DBM
        
    def move(self):
        """
        Updates position based on speed and direction (1 time step).
        Implements simple boundary checking.
        """
        dt = SimulationConfig.TIME_STEP_S
        
        # Calculate displacement
        dx = self.speed * np.cos(self.direction) * dt
        dy = self.speed * np.sin(self.direction) * dt
        
        new_position = self.position + np.array([dx, dy])
        
        # Boundary Check: Ensure device stays within Cell Radius
        dist_from_center = np.linalg.norm(new_position)
        
        if dist_from_center > SimulationConfig.CELL_RADIUS_M:
            # If out of bounds, bounce back (turn 180 degrees + random noise)
            self.direction = self.direction + np.pi + np.random.uniform(-0.5, 0.5)
            
            # Clamp position to boundary to prevent escaping
            # Normalize vector and multiply by radius
            new_position = (new_position / dist_from_center) * SimulationConfig.CELL_RADIUS_M
            
        self.position = new_position
        
        # Randomly adjust direction slightly to simulate realistic wandering
        # 20% chance to change direction by up to +/- 30 degrees (approx 0.5 rad)
        if np.random.rand() < 0.2:
            self.direction += np.random.uniform(-0.5, 0.5)

    def get_distance_to(self, other_entity):
        """Calculates Euclidean distance to another entity (BS or UE)"""
        return np.linalg.norm(self.position - other_entity.position)