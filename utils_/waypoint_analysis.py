"""
We want to analyse the nature of the waypoints extracted for
a given set of demonstrations of a task. For this first script
we assume that we extract the same amount of waypoints for each
demostration.

For this first pick and place task we generate waypoints at
each gripper open and close instance and at the highest point
of the endeffector while the gripper is closed.
"""

import os
import random
import numpy as np
import robotic as ry
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plotting import plot_path
from utils import get_joint_states, get_first_pickplace
from waypoint_extraction import pickplace_task_waypoints


DATA_DIR = "./pickplace_dataset"
C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
gripper_frame = C.getFrame("l_gripper")


### Without waypoints ###
episode_joint_states = []
for episode_path in tqdm(os.listdir(DATA_DIR)):
    qs, taus, gws = get_joint_states(f"{DATA_DIR}/{episode_path}/proprioceptive.txt")
    qs = get_first_pickplace(qs, gws)
    episode_joint_states.append(qs)

# Normalize lengths
min_ep_len = min(len(ep) for ep in episode_joint_states)
max_ep_len = max(len(ep) for ep in episode_joint_states)
print("Min episode length: ", min_ep_len)
print("Max episode length: ", max_ep_len)

new_episode_joint_states = []
for ep in episode_joint_states:
    new_ep = [ep[int(np.round(i*len(ep)/min_ep_len))] for i in range(min_ep_len)]
    new_episode_joint_states.append(new_ep)
episode_joint_states = new_episode_joint_states
episode_joint_states = np.array(episode_joint_states)

# Visualice gripper in single episode on the xz plane
episode = random.choice(episode_joint_states)
gripper_xz = []
for q in episode:
    C.setJointState(q)
    xz = gripper_frame.getPosition()[::2]
    gripper_xz.append(xz)

gripper_xz = np.array(gripper_xz)
# plot_path(gripper_xz[:, 0], gripper_xz[:, 1])

# Visualice gripper in all episodes on the xz plane
for episode in episode_joint_states:
    gripper_xz = []
    for q in episode:
        C.setJointState(q)
        xz = gripper_frame.getPosition()[::2]
        gripper_xz.append(xz)

    gripper_xz = np.array(gripper_xz)
    # plt.plot(gripper_xz[:, 1])
# plt.show()

# Visualize the mean of all paths and their std. deviation
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

for i, v in enumerate(['x', 'y', 'z']):
    mean = []
    std_dev = []
    for timestep in episode_joint_states.transpose(1, 0, 2):

        # Mean
        q = np.mean(timestep, axis=0)
        C.setJointState(q)
        xz = gripper_frame.getPosition()[i]
        mean.append(xz)

        # Std. dev.
        q = np.std(timestep, axis=0)
        C.setJointState(q)
        xz = gripper_frame.getPosition()[i]
        std_dev.append(xz)

    mean = np.array(mean)
    std_dev = np.array(std_dev)

    x = list(range(min_ep_len))
    axs[i].plot(x, mean, color='blue', label='Mean')
    # axs[i].fill_between(x, mean - std_dev, mean + std_dev, color='blue', alpha=0.2, label='Std Deviation')
    axs[i].set_xlabel('timestamp')
    axs[i].set_ylabel(f'{v}-axis')
    axs[i].set_title(f'Mean and Standard Deviation ({v})')
    axs[i].legend()

plt.show()

# Difference between the mean in joint state and mean in gripper position

episodes_gripper_pos = []
for episode in episode_joint_states:
    gripper_pos = []
    for q in episode:
        C.setJointState(q)
        xyz = gripper_frame.getPosition()
        gripper_pos.append(xyz)
    episodes_gripper_pos.append(gripper_pos)
episodes_gripper_pos = np.array(episodes_gripper_pos)
mean_pos = np.mean(episodes_gripper_pos, axis=0)
# mean_pos = random.choice(episodes_gripper_pos)
print(mean_pos.shape)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
for i, v in enumerate(['x', 'y', 'z']):
    axs[i].plot(mean, color='blue', label='Mean')
    axs[i].plot(mean_pos[:, 2], color='red', label='Mean')
    axs[i].set_xlabel('timestamp')
    axs[i].set_ylabel(f'{v}-axis')
    axs[i].set_title(f'Mean and Standard Deviation ({v})')
    axs[i].legend()
plt.show()

exit()
### With waypoints ###
episode_waypoints = []
for episode_path in tqdm(os.listdir(DATA_DIR)):
    ways = pickplace_task_waypoints()
    episode_waypoints.append(ways)