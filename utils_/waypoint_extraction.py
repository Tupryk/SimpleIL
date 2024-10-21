import numpy as np
from utils_.utils import find_switch_indices_with_delta


def pickplace_task_waypoints(gripper_positions: np.ndarray, gripper_widths: np.ndarray) -> np.ndarray:
    ways = []
    switch_indices = find_switch_indices_with_delta(gripper_widths)

    # Gripper fingers state change
    for i, si in enumerate(switch_indices):    
        ways.append(gripper_positions[si])
        
        # Highest points during gripper closed
        if i%2 == 0:
            closed_state = gripper_positions[switch_indices[i]:switch_indices[i+1]]
            max_index = max(range(len(closed_state)), key=lambda j: closed_state[j][-1])
            ways.append(closed_state[max_index])

    ways = np.array(ways)
    return ways
