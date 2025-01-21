import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))
v19_epoch_plot       = np.load(os.path.join(file_path, "array/v19_epoch_plot.npy"))
v19_step_reward_plot = np.load(os.path.join(file_path, "array/v19_step_reward_plot.npy"))
v21_epoch_plot       = np.load(os.path.join(file_path, "array/v21_epoch_plot.npy"))
v21_step_reward_plot = np.load(os.path.join(file_path, "array/v21_step_reward_plot.npy"))
smoothed_v19_step_reward_plot = gaussian_filter1d(v19_step_reward_plot, sigma=40.0)
smoothed_v21_step_reward_plot = gaussian_filter1d(v21_step_reward_plot, sigma=40.0)
lenth = min(len(v19_epoch_plot), len(v21_epoch_plot))

fig = plt.figure(figsize=(7.5, 5))
plt.plot(v21_epoch_plot[:lenth], v21_step_reward_plot[:lenth], color='red', alpha=0.1)
plt.plot(v21_epoch_plot[:lenth], smoothed_v21_step_reward_plot[:lenth], label="WITH future state prediction", color='red', alpha=1)
plt.plot(v19_epoch_plot, v19_step_reward_plot, color='black', alpha=0.1)
plt.plot(v19_epoch_plot, smoothed_v19_step_reward_plot, label="WITHOUT future state prediction", color='black', alpha=1)
plt.title("Training with/without future state prediction.")
plt.xlabel("Epoch")
plt.ylabel("Average Step Reward")
plt.legend()
plt.savefig(os.path.join(file_path, "Training with&without future state prediction.png"))
plt.show()