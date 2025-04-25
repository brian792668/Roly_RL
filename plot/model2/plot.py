import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))

epoch_plot       = np.load(os.path.join(file_path, "array/epoch_plot.npy"))
step_reward_plot = np.load(os.path.join(file_path, "array/step_reward_plot.npy"))
smoothed_reward_plot = gaussian_filter1d(step_reward_plot, sigma=3)
error = (1 - step_reward_plot) * 0.1
smoothed_error = gaussian_filter1d(error, sigma=3)

fig, ax1 = plt.subplots(figsize=(9, 6))

# 左邊的主 y 軸：畫 reward
ax1.plot(epoch_plot, step_reward_plot, color='red', alpha=0.1)
ax1.plot(epoch_plot, smoothed_reward_plot, label="Average Step Reward", color='red', alpha=1)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Average Step Reward")
ax1.set_ylim(0, 1.0)
ax1.legend(loc="upper right")

# 右邊的次 y 軸：畫 error
ax2 = ax1.twinx()
ax2.plot(epoch_plot, error, color='black', alpha=0.1)
ax2.plot(epoch_plot, smoothed_error, label="Distance Error", color='black', alpha=0.7)
ax2.set_ylabel("Distance Error (m)")
ax2.set_ylim(0, 0.1)
ax2.legend(loc="lower right")

# ax1.set_title("Model2 step reward and error.")
plt.savefig(os.path.join(file_path, "Model2 step reward and error.png"))
plt.show()
