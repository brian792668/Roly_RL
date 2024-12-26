import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RL_arm import *
from scipy.ndimage import gaussian_filter1d

def train(model, env, file_path):
    print(f"Start Training ...")
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])
    best_avg_step_reward = np.array([0.0])

    if os.path.exists(os.path.join(file_path, "array/epoch_plot.npy")):
        epoch_plot              = np.load(os.path.join(file_path, "array/epoch_plot.npy"))
        step_reward_plot        = np.load(os.path.join(file_path, "array/step_reward_plot.npy"))
        best_avg_step_reward    = np.load(os.path.join(file_path, "array/best_avg_step_reward.npy"))
    else:
        np.save(os.path.join(file_path, "array/epoch_plot.npy"), epoch_plot)
        np.save(os.path.join(file_path, "array/step_reward_plot.npy"), step_reward_plot)
        np.save(os.path.join(file_path, "array/best_avg_step_reward.npy"), best_avg_step_reward)

    epoch = epoch_plot[-1]
    timer0 = time.time()

    while True:
        model.learn(total_timesteps = 2048)
        model.save(os.path.join(file_path, "current_model.zip"))
        epoch += 1

        avg_step_reward = env.inf.total_reward / env.inf.totaltimestep
        env.reset()
        env.inf.totaltimestep = 0
        env.inf.total_reward = 0

        if avg_step_reward >= best_avg_step_reward[0]:
            best_avg_step_reward[0] = avg_step_reward
            model.save(os.path.join(file_path, f"best_step/best_step_model_epoch{epoch}.zip"))
            print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr  ||  best avg step reward = {round(avg_step_reward,3)}")
        else:
            print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")
        epoch_plot = np.append(epoch_plot, epoch)
        step_reward_plot = np.append(step_reward_plot, avg_step_reward)
        smoothed_gau = gaussian_filter1d(step_reward_plot, sigma=epoch/50.0)
        np.save(os.path.join(file_path, "array/epoch_plot.npy"), epoch_plot)
        np.save(os.path.join(file_path, "array/step_reward_plot.npy"), step_reward_plot)
        np.save(os.path.join(file_path, "array/best_avg_step_reward.npy"), best_avg_step_reward)

        fig = plt.figure(figsize=(15, 10))
        plt.plot(epoch_plot, step_reward_plot, label="Original Data", color='black', alpha=0.2)
        plt.plot(epoch_plot[:len(smoothed_gau)], smoothed_gau, label="gaussian filtered", color='red', alpha=1)
        plt.title("Average Step Reward (Gaussian Filtered)")
        plt.xlabel("Epoch")
        plt.ylabel("Step Reward")
        plt.legend()

        plt.savefig(os.path.join(file_path, "epoch_vs_reward.png"))
        plt.close()
       
def test(model, env, model_path):
    model = stable_baselines3.SAC.load(model_path, env)
    obs, _ = env.reset()
    env.inf.totaltimestep = 0
    env.inf.total_reward = 0
    while env.viewer.is_running() == True:
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    avg_step_reward = env.inf.total_reward / env.inf.totaltimestep
    # print(f"\ntotal reward = {round(env.inf.total_reward,2)}")
    print(f"\navg step reward = {round(avg_step_reward,3)}")
    env.close()

if __name__ == '__main__':
    my_env = RL_arm()
    file_path = os.path.dirname(os.path.abspath(__file__))
    current_model_path = os.path.join(file_path, "current_model.zip")
    best_model_path = os.path.join(file_path, "best_step/best_step_model_epoch2192.zip")
    if os.path.exists(current_model_path):
        print(f"model file: {current_model_path}")
        RL_model = stable_baselines3.SAC.load(current_model_path, my_env)
    else:
        RL_model = stable_baselines3.SAC('MlpPolicy', my_env, learning_rate=0.0005)
        RL_model.save(current_model_path)

    # train(RL_model, my_env, file_path)
    # test(RL_model, my_env, current_model_path)
    test(RL_model, my_env, best_model_path)
