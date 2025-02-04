import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RL_arm import *
from scipy.ndimage import gaussian_filter1d

def train(model, env, file_path, train_num):
    print(f"Start Training Iteration {train_num}")
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])

    epoch = 0
    timer0 = time.time()
    best_total_reward = float('-inf')
    best_total_reward_model_path = None


    while epoch < 2000:
        model.learn(total_timesteps = 2048)
        epoch += 1

        avg_step_reward = env.inf.total_reward / env.inf.totaltimestep

        # Save model with best total_reward
        if env.inf.total_reward > best_total_reward:
            best_total_reward = env.inf.total_reward
            if best_total_reward_model_path and os.path.exists(best_total_reward_model_path):
                os.remove(best_total_reward_model_path)
            # best_total_reward_model_path = os.path.join(file_path, f"best_total_reward_model_iter{train_num}.zip")
            best_total_reward_model_path = os.path.join(file_path, f"models/iter{train_num}_best.zip")
            model.save(best_total_reward_model_path)

        env.reset()
        env.inf.totaltimestep = 0
        env.inf.total_reward = 0

        epoch_plot = np.append(epoch_plot, epoch)
        step_reward_plot = np.append(step_reward_plot, avg_step_reward)
        smoothed_gau = gaussian_filter1d(step_reward_plot, sigma=epoch/50.0)

        print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")

        # Save epoch and step reward plots
        fig = plt.figure(figsize=(15, 10))
        plt.plot(epoch_plot, step_reward_plot, label="Original Data", color='black', alpha=0.2)
        plt.plot(epoch_plot[:len(smoothed_gau)], smoothed_gau, label="gaussian filtered", color='red', alpha=1)
        plt.title(f"Average Step Reward (Gaussian Filtered) - Iteration {train_num}")
        plt.xlabel("Epoch")
        plt.ylabel("Step Reward")
        plt.legend()

        # Create directories if they don't exist
        os.makedirs(os.path.join(file_path, "array"), exist_ok=True)
        os.makedirs(os.path.join(file_path, "plots"), exist_ok=True)

        # Save numpy arrays
        np.save(os.path.join(file_path, f"array/epoch_plot_{train_num}.npy"), epoch_plot)
        np.save(os.path.join(file_path, f"array/step_reward_plot_{train_num}.npy"), step_reward_plot)

        # Save plot
        plt.savefig(os.path.join(file_path, f"plots/epoch_plot_{train_num}.png"))
        plt.close()

    # Save model at epoch 2000
    model.save(os.path.join(file_path, f"models/iter{train_num}_final.zip"))
    env.close()

def main():
    file_path = os.path.dirname(os.path.abspath(__file__))

    for train_num in range(21, 41):  # Train 20 times
        my_env = RL_arm()
        current_model_path = os.path.join(file_path, f"current_model_{train_num}.zip")

        # Initialize or load model
        if os.path.exists(current_model_path):
            print(f"Loading existing model: {current_model_path}")
            RL_model = stable_baselines3.SAC.load(current_model_path, my_env)
        else:
            print(f"Creating new model for iteration {train_num}")
            RL_model = stable_baselines3.SAC('MlpPolicy', my_env, learning_rate=0.0005)

        # Train the model
        train(RL_model, my_env, file_path, train_num)

if __name__ == '__main__':
    main()