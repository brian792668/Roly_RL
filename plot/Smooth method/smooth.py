import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

def moving_average(data, window_size=10):
    """計算移動平均"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def exponential_moving_average(data, alpha=0.2):
    """計算指數加權平均"""
    ema = [data[0]]  # 初始值設定為第一筆數據
    for point in data[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return np.array(ema)

def bidirectional_exponential_moving_average(data, alpha):
    forward_ema = [data[0]]  # 初始值設定為第一筆數據
    for point in data[1:]:
        forward_ema.append(alpha * point + (1 - alpha) * forward_ema[-1])

    # 反向 EMA
    backward_ema = [data[-1]]  # 初始值設定為最後一筆數據
    for point in reversed(data[:-1]):
        backward_ema.append(alpha * point + (1 - alpha) * backward_ema[-1])

    # 反轉 backward EMA 並計算平均
    backward_ema.reverse()
    bema = [(f + b) / 2 for f, b in zip(forward_ema, backward_ema)]

    return np.array(bema)

def run():
    # 載入資料
    file_path = os.path.dirname(os.path.abspath(__file__))
    x_path = "epoch_plot.npy"
    y_path = "step_reward_plot.npy"
    x_plot = np.load(os.path.join(file_path, x_path))
    y_plot = np.load(os.path.join(file_path, y_path))

    # 設定繪圖參數
    window_size = 50
    alpha = 0.05
    sigma = 20

    # 計算平滑數據
    smoothed_ma = moving_average(y_plot, window_size=window_size)
    # smoothed_ema = exponential_moving_average(y_plot, alpha=alpha)
    smoothed_ema = bidirectional_exponential_moving_average(y_plot, alpha=alpha)
    smoothed_gau = gaussian_filter1d(y_plot, sigma=sigma)

    # 創建子圖
    plt.figure(figsize=(10, 15))

    # 移動平均圖
    plt.subplot(3, 1, 1)
    plt.plot(x_plot, y_plot, label="Original Data", color='black', alpha=0.2)
    plt.plot(x_plot[:len(smoothed_ma)], smoothed_ma, label="Moving Average", color='blue', alpha=1)
    plt.title("Moving Average")
    plt.xlabel("Epoch")
    plt.ylabel("Step Reward")
    plt.legend()

    # 指數移動平均圖
    plt.subplot(3, 1, 2)
    plt.plot(x_plot, y_plot, label="Original Data", color='black', alpha=0.2)
    plt.plot(x_plot[:len(smoothed_ema)], smoothed_ema, label="Exponential Moving Average", color='red', alpha=1)
    plt.title("Bidirectional Exponential Moving Average")
    plt.xlabel("Epoch")
    plt.ylabel("Step Reward")
    plt.legend()

    # 高斯濾波圖
    plt.subplot(3, 1, 3)
    plt.plot(x_plot, y_plot, label="Original Data", color='black', alpha=0.2)
    plt.plot(x_plot[:len(smoothed_gau)], smoothed_gau, label="Gaussian Moving Average", color='green', alpha=1)
    plt.title("Gaussian Moving Average")
    plt.xlabel("Epoch")
    plt.ylabel("Step Reward")
    plt.legend()

    # 調整子圖間距
    plt.tight_layout()

    # 儲存圖表
    plt.savefig(os.path.join(file_path, "smoothing_methods_comparison.png"))
    plt.show()

if __name__ == '__main__':
    run()