import numpy as np
from simulator.config import SimulationConfig

# Represents the cellular Base Station (BS).
class BaseStation:
    # Located at the center (0,0) of the cell.
    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.tx_power_dbm = SimulationConfig.TX_POWER_BS_DBM

    # Calculates Euclidean distance to another entity (UE)
    def get_distance_to(self, other_entity):
        return np.linalg.norm(self.position - other_entity.position)

# Represents a mobile device, or user equipement (UE).
class UserEquipment:
    def __init__(self, device_id, speed_type='pedestrian'):
        # Initialize Speed: Pedestrian (1-3 ms) or Moderate (3-10 ms)
        if speed_type == 'pedestrian':
            speed_range = SimulationConfig.SPEED_MODE_PEDESTRIAN 
        elif speed_type == 'moderate':
            speed_range = SimulationConfig.SPEED_MODE_MODERATE 
        else:
            raise ValueError(f"Invalid speed_type '{speed_type}'. Must be 'pedestrian' or 'moderate'.")
        
        # Initialize the following:
        # - Device ID
        # - Start Position
        # - Random Waypoint Destination
        # - State Flag: True = Paused, False = Moving
        # - Speed (m/s)
        # - Transmit Power (dBm)
        self.device_id = device_id
        self.position = self._get_random_point_in_cell()
        self.destination = self._get_random_point_in_cell()
        self.is_paused = False
        self.speed = np.random.uniform(speed_range[0], speed_range[1])
        self.tx_power_dbm = SimulationConfig.TX_POWER_D2D_DBM
        
    # Picks a new random point within the cell
    def _get_random_point_in_cell(self):
        # r = R * sqrt(random) ensures uniform distribution in a circle
        radius = SimulationConfig.CELL_RADIUS_M * np.sqrt(np.random.rand())
        angle = 2 * np.pi * np.random.rand()
        return np.array([radius * np.cos(angle), radius * np.sin(angle)])

    # Updates position based on Random Waypoint Mobility Model.
    def move(self):
        # --- CASE 1: PAUSED ---
        if self.is_paused:
            # Check probability to start moving
            if np.random.rand() < SimulationConfig.PROBABILITY_START_MOVING:
                # Start moving to a new random destination
                self.is_paused = False
                self.destination = self._get_random_point_in_cell()
            else:
                # Stay paused at current position
                return 

        # --- CASE 2: MOVING ---
        dt = SimulationConfig.TIME_STEP_S

        # Calculate the following:
        # - Direction vector to destination
        # - Distance to destination
        # - Step distance for this time step
        direction_vector = self.destination - self.position
        distance_to_dest = np.linalg.norm(direction_vector)
        step_distance = self.speed * dt
        
        # Check if we reach the destination or overshoot
        if step_distance >= distance_to_dest:
            # Correct overshoot position to exact destination
            self.position = self.destination

            # Transition to PAUSE state/stop moving
            self.is_paused = True
        else:
            # Continue moving towards destination
            unit_vector = direction_vector / distance_to_dest
            self.position = self.position + (unit_vector * step_distance)

    # Calculates Euclidean distance to another entity (BS or UE)
    def get_distance_to(self, other_entity):
        return np.linalg.norm(self.position - other_entity.position)