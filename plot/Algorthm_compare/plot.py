import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))
v21_SAC_epoch_plot = np.load(os.path.join(file_path, "array/v21_SAC_epoch_plot.npy"))
# v21_A2C_epoch_plot = np.load(os.path.join(file_path, "array/v21_A2C_epoch_plot.npy"))
v21_PPO_epoch_plot = np.load(os.path.join(file_path, "array/v21_PPO_epoch_plot.npy"))
v21_TD3_epoch_plot = np.load(os.path.join(file_path, "array/v21_TD3_epoch_plot.npy"))
v26_SAC_step_reward_plot = np.load(os.path.join(file_path, "array/v26_SAC_step_reward_plot.npy"))/1.8
# v21_A2C_step_reward_plot = np.load(os.path.join(file_path, "array/v21_A2C_step_reward_plot.npy"))
v21_PPO_step_reward_plot = np.load(os.path.join(file_path, "array/v21_PPO_step_reward_plot.npy"))/1.8
v21_TD3_step_reward_plot = np.load(os.path.join(file_path, "array/v21_TD3_step_reward_plot.npy"))/1.8
smoothed_v21_SAC_step_reward_plot = gaussian_filter1d(v26_SAC_step_reward_plot, sigma=40.0)
# smoothed_v21_A2C_step_reward_plot = gaussian_filter1d(v21_A2C_step_reward_plot, sigma=40.0)
smoothed_v21_PPO_step_reward_plot = gaussian_filter1d(v21_PPO_step_reward_plot, sigma=40.0)
smoothed_v21_TD3_step_reward_plot = gaussian_filter1d(v21_TD3_step_reward_plot, sigma=40.0)
length = len(v21_TD3_epoch_plot)

fig = plt.figure(figsize=(9, 6))
plt.plot(v21_TD3_epoch_plot, v26_SAC_step_reward_plot[:length], color='red', alpha=0.1)
plt.plot(v21_TD3_epoch_plot, smoothed_v21_SAC_step_reward_plot[:length], label="SAC", color='red', alpha=1)
# plt.plot(v21_TD3_epoch_plot, v21_A2C_step_reward_plot[:length], color='blue', alpha=0.1)
# plt.plot(v21_TD3_epoch_plot, smoothed_v21_A2C_step_reward_plot[:length], label="A2C", color='blue', alpha=1)
plt.plot(v21_TD3_epoch_plot, v21_PPO_step_reward_plot[:length], color='green', alpha=0.1)
plt.plot(v21_TD3_epoch_plot, smoothed_v21_PPO_step_reward_plot[:length], label="PPO", color='green', alpha=1)
plt.plot(v21_TD3_epoch_plot, v21_TD3_step_reward_plot, color='black', alpha=0.1)
plt.plot(v21_TD3_epoch_plot, smoothed_v21_TD3_step_reward_plot, label="TD3", color='black', alpha=1)
plt.title("Training with different algorithm.")
plt.xlabel("Epoch")
plt.ylabel("Average Step Reward")
plt.legend()
plt.savefig(os.path.join(file_path, "Algorithm Camparasion.png"))
plt.show()