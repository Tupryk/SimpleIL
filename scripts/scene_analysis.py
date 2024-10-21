import os
import numpy as np
import robotic as ry
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_joint_states, find_switch_indices_with_delta


# Distribution of the objects on the scene
DATA_DIR = "./pickplace_dataset"
C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
gripper_frame = C.getFrame("l_gripper")


### Get object positions ###
obj1_poss = []
obj2_poss = []
for episode_path in tqdm(os.listdir(DATA_DIR)):
    qs, taus, gws = get_joint_states(f"{DATA_DIR}/{episode_path}/proprioceptive.txt")
    switches = find_switch_indices_with_delta(gws)

    # First to be picked
    C.setJointState(qs[switches[0]])
    pos = gripper_frame.getPosition()[:2]
    obj1_poss.append(pos)

    # Second to be picked
    C.setJointState(qs[switches[2]])
    pos = gripper_frame.getPosition()[:2]
    obj2_poss.append(pos)

obj1_poss = np.array(obj1_poss)
obj2_poss = np.array(obj2_poss)
plt.scatter(obj1_poss[:, 0], obj1_poss[:, 1], c="r", alpha=.5, label="First pick")
plt.scatter(obj2_poss[:, 0], obj2_poss[:, 1], c="b", alpha=.5, label="Second pick")
plt.legend()
plt.show()

### Get object positions and placing positions ###
positions = []
for episode_path in tqdm(os.listdir(DATA_DIR)):
    qs, taus, gws = get_joint_states(f"{DATA_DIR}/{episode_path}/proprioceptive.txt")
    switches = find_switch_indices_with_delta(gws)

    cats = []
    for i in range(len(switches)):
        C.setJointState(qs[switches[i]])
        pos = gripper_frame.getPosition()[:2]
        cats.append(pos)
    positions.append(cats)

positions = np.array(positions)

plt.scatter(positions[:, 0, 0], positions[:, 0, 1], c="r", alpha=.5, label="First pick")
plt.scatter(positions[:, 1, 0], positions[:, 1, 1], c="g", alpha=.5, label="First place")
plt.scatter(positions[:, 2, 0], positions[:, 2, 1], c="b", alpha=.5, label="Second pick")
plt.scatter(positions[:, 3, 0], positions[:, 3, 1], c="y", alpha=.5, label="Second place")
plt.legend()
plt.show()
