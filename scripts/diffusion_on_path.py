import os
import torch
import random
import numpy as np
import robotic as ry
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from plotting_utils import plot_path, vis_denoising
from utils import get_joint_states, get_first_pickplace, q_to_endeffcoor
from waypoint_extraction import pickplace_task_waypoints
from models import Diffuser


DATA_DIR = "./pickplace_dataset"
device = "cpu"


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

# Get endeffector coordinates
episodes_endeff_coor = []
for ep in episode_joint_states:
    new_ep = q_to_endeffcoor(ep)
    episodes_endeff_coor.append(new_ep)
episodes_endeff_coor = np.array(episodes_endeff_coor)

# Visualize data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for ep in episodes_endeff_coor:
#     ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2], alpha=.1)
# ax.set_title("Training data")
# plt.show()

# Visualize noising
episode = random.choice(episode_joint_states)
# denoising_steps = 128
# scheduler = DDPMScheduler(num_train_timesteps=denoising_steps)
# paths = []
# eps = torch.randn(episode.shape, device=device)
# for i in tqdm(range(denoising_steps)):
#     i = denoising_steps-1-i
#     noisy_episode = scheduler.add_noise(episode, eps, torch.tensor([i], dtype=torch.int))
#     vis_episode = q_to_endeffcoor(noisy_episode)
#     paths.append(vis_episode)
# paths = np.array(paths)
# vis_denoising(paths)

# Train model for denoising
episode = np.concatenate(episode, axis=-1)
model = Diffuser()
model(torch.Tensor(episode), torch.tensor([120], dtype=torch.int))

# Visualize denoising
