import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))
epoch_plot_0       = np.load(os.path.join(file_path, "array/epoch_plot_0.npy"))
epoch_plot_5       = np.load(os.path.join(file_path, "array/epoch_plot_5.npy"))
step_reward_plot_0 = np.load(os.path.join(file_path, "array/step_reward_plot_0.npy"))
step_reward_plot_5 = np.load(os.path.join(file_path, "array/step_reward_plot_5.npy"))
smoothed_step_reward_plot_0 = gaussian_filter1d(step_reward_plot_0, sigma=40.0)
smoothed_step_reward_plot_5 = gaussian_filter1d(step_reward_plot_5, sigma=40.0)
lenth = min(len(epoch_plot_0), len(epoch_plot_5))
lenth = 1300

fig = plt.figure(figsize=(7.5, 5))
plt.plot(epoch_plot_0[:lenth], step_reward_plot_0[:lenth], color='black', alpha=0.1)
plt.plot(epoch_plot_0[:lenth], smoothed_step_reward_plot_0[:lenth], label="WITHOUT future state prediction", color='black', alpha=1)
plt.plot(epoch_plot_0[:lenth], step_reward_plot_5[:lenth], color='red', alpha=0.1)
plt.plot(epoch_plot_0[:lenth], smoothed_step_reward_plot_5[:lenth], label="WITH 5 future state prediction", color='red', alpha=1)
plt.title("Training with/without future state prediction.")
plt.xlabel("Epoch")
plt.ylabel("Average Step Reward")
plt.legend()
plt.savefig(os.path.join(file_path, "Training with&without future state prediction.png"))
plt.show()