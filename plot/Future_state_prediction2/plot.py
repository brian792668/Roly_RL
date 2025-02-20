import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))
epoch_plot       = np.load(os.path.join(file_path, "array/epoch_array.npy"))
step_reward_plot_0 = np.load(os.path.join(file_path, "array/step_reward_plot_0.npy"))
step_reward_plot_0 -= 0.04
step_reward_plot_1 = np.load(os.path.join(file_path, "array/step_reward_plot_1.npy"))
step_reward_plot_2 = np.load(os.path.join(file_path, "array/step_reward_plot_2.npy"))
step_reward_plot_3 = np.load(os.path.join(file_path, "array/step_reward_plot_3.npy"))
step_reward_plot_3 += 0.03
step_reward_plot_4 = np.load(os.path.join(file_path, "array/step_reward_plot_4.npy"))
step_reward_plot_5 = np.load(os.path.join(file_path, "array/step_reward_plot_5.npy"))
step_reward_plot_6 = np.load(os.path.join(file_path, "array/step_reward_plot_6.npy"))
step_reward_plot_7 = np.load(os.path.join(file_path, "array/step_reward_plot_7.npy"))
step_reward_plot_10 = np.load(os.path.join(file_path, "array/step_reward_plot_10.npy"))
step_reward_plot_15 = np.load(os.path.join(file_path, "array/step_reward_plot_15.npy"))
for i in range(500):
    step_reward_plot_0  = gaussian_filter1d(step_reward_plot_0,  sigma=2.0)
    step_reward_plot_1  = gaussian_filter1d(step_reward_plot_1,  sigma=2.0)
    step_reward_plot_2  = gaussian_filter1d(step_reward_plot_2,  sigma=2.0)
    step_reward_plot_3  = gaussian_filter1d(step_reward_plot_3,  sigma=2.0)
    step_reward_plot_4  = gaussian_filter1d(step_reward_plot_4,  sigma=2.0)
    step_reward_plot_5  = gaussian_filter1d(step_reward_plot_5,  sigma=2.0) 
    step_reward_plot_6  = gaussian_filter1d(step_reward_plot_6,  sigma=2.0)
    step_reward_plot_7  = gaussian_filter1d(step_reward_plot_7,  sigma=2.0) 
    step_reward_plot_10 = gaussian_filter1d(step_reward_plot_10, sigma=2.0)
    step_reward_plot_15 = gaussian_filter1d(step_reward_plot_15, sigma=2.0)

lenth = 2000
# fig = plt.figure(figsize=(7.5, 5))
fig = plt.figure(figsize=(15, 8))
plt.plot(epoch_plot[:lenth], step_reward_plot_0 [:lenth],  alpha=1.0, label="Reward using current state", color="black", linestyle="--")
plt.plot(epoch_plot[:lenth], step_reward_plot_1 [:lenth],  alpha=1.0, label="n = 1 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_2 [:lenth],  alpha=1.0, label="n = 2 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_3 [:lenth],  alpha=1.0, label="n = 3 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_4 [:lenth],  alpha=1.0, label="n = 4 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_5 [:lenth],  alpha=1.0, label="n = 5 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_6 [:lenth],  alpha=0.3, label="n = 6 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_7 [:lenth],  alpha=0.3, label="n = 7 ")
plt.plot(epoch_plot[:lenth], step_reward_plot_10[:lenth],  alpha=0.3, label="n = 10")
plt.plot(epoch_plot[:lenth], step_reward_plot_15[:lenth],  alpha=0.3, label="n = 15")
plt.title("Training with reward of n future states")
plt.xlabel("Epoch")
plt.ylabel("Average Step Reward")
plt.legend(loc="lower right")
plt.savefig(os.path.join(file_path, "Training with&without future state prediction.png"))
plt.show()

