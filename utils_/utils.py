import os
import numpy as np
import robotic as ry


def waypoints_from_bridge_build(path: str):

    joints = np.load(path)
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    pos = []
    quat = []
    for j in joints:
        C.setJointState(j[-1])
        p = C.getFrame("l_gripper").getPosition()
        q = C.getFrame("l_gripper").getQuaternion()
        pos.append(p)
        quat.append(q)

    pos = np.array(pos)
    quat = np.array(quat)

    return pos, quat

def joint_state_to_pose(C: ry.Config, q):
    C.setJointState(q)
    frame = C.getFrame("l_gripper")
    pos = frame.getPosition()
    quat = frame.getQuaternion()
    return pos, quat

def get_gripper_poses(file_path):
    pos = []
    quat = []

    with open(file_path, 'r') as file:
        for line in file:
            left_str, right_str = line.split('] [')

            left_list = [float(x) for x in left_str.strip('[] \n').split(',')]
            right_list = [float(x) for x in right_str.strip('[] \n').split(',')]

            left_array = np.array(left_list)
            right_array = np.array(right_list)

            pos.append(left_array)
            quat.append(right_array)

    pos = np.array(pos)
    quat = np.array(quat)
    return pos, quat


def from_sim_get_poses_n_gripper(file_path):
    try:
        pos = []
        quat = []
        gripper_widths = []
        C = ry.Config()
        C.addFile(ry.raiPath("scenarios/pandaSingle.g"))

        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        if not lines:
            print(f"No data in {file_path}. Skipping.")
            return

        for l in lines:
            last_line = l.strip()
            numbers = last_line.split()

            if len(numbers) != 8:
                print(f"Incorrect format in {file_path}. Expected 8 numbers, found {len(numbers)}. Skipping.")
                return
            
            q = [float(numbers[i]) for i in range(7)]
            p, rot = joint_state_to_pose(C, q)
            pos.append(p)
            quat.append(rot)
            gripper_width = float(numbers[-1])
            gripper_widths.append(gripper_width)

        gripper_widths[-1] = 0
        pos = np.array(pos)
        quat = np.array(quat)
        gripper_widths = np.array(gripper_widths)
        return pos, quat, gripper_widths
    
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return [], [], []
    


def get_joint_states(file_path):
    qs = []
    taus = []
    gripper_widths = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('] [')

            if len(parts) == 2:
                right_str, float_str = parts[1].rsplit('] ', 1)

                qs_list = [float(x) for x in parts[0].strip('[]').split(',')]
                taus_list = [float(x) for x in right_str.strip('[]').split(',')]
                width_value = float(float_str.strip())

                left_array = np.array(qs_list)
                right_array = np.array(taus_list)

                qs.append(left_array)
                taus.append(right_array)
                gripper_widths.append(width_value)
            else:
                print(f"Unexpected line format: {line}")

    return qs, taus, gripper_widths


def find_switch_indices_with_delta(values, delta=.02):
    switch_indices = []
    
    for i in range(1, len(values)):
        if abs(values[i] - values[i - 1]) > delta:
            switch_indices.append(i)
    
    return switch_indices


def get_first_pickplace(path: np.ndarray, gripper_width: np.ndarray):
    switch_indices = find_switch_indices_with_delta(gripper_width, .02)
    cut_path = path[:switch_indices[1]]
    return cut_path

def q_to_endeffcoor(episode):
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
    gripper_frame = C.getFrame("l_gripper")
    new_ep = []
    for q in episode:
        C.setJointState(q)
        pos = gripper_frame.getPosition()
        new_ep.append(pos)
    return new_ep

def get_alpha_betas(N: int):
  """Schedule from the original paper.
  """
  beta_min = 0.1
  beta_max = 20.
  betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
  alpha_bars = np.cumprod(1 - betas)
  return alpha_bars, betas
