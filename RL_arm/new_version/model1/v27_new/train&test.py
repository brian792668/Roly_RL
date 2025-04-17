import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RL_arm import *
from scipy.ndimage import gaussian_filter1d

def train(model, env, file_path, render_speed=1):
    print(f"Start Training ...")
    epoch_array = np.array([0])
    step_reward_array_standard = np.array([0.0])
    step_reward_array_future_state = np.array([0.0])

    if os.path.exists(os.path.join(file_path, "array/epoch_array.npy")):
        epoch_array       = np.load(os.path.join(file_path, "array/epoch_array.npy"))
        step_reward_array_standard = np.load(os.path.join(file_path, "array/step_reward_array_standard.npy"))
        step_reward_array_future_state  = np.load(os.path.join(file_path, "array/step_reward_array_future_state.npy"))
    else:
        np.save(os.path.join(file_path, "array/epoch_array.npy"), epoch_array)
        np.save(os.path.join(file_path, "array/step_reward_array_standard.npy"), step_reward_array_standard)
        np.save(os.path.join(file_path, "array/step_reward_array_future_state.npy"), step_reward_array_future_state)

    epoch = epoch_array[-1]
    timer0 = time.time()
    env.render_speed = render_speed

    while epoch <= 2000:
        model.learn(total_timesteps = 2048)
        model.save(os.path.join(file_path, "current_model.zip"))
        epoch += 1

        avg_step_reward_standard = env.inf.total_reward_standard / env.inf.totaltimestep
        avg_step_reward_future_state  = env.inf.total_reward_future_state / env.inf.totaltimestep
        env.reset()
        env.inf.totaltimestep = 0
        env.inf.total_reward_standard = 0
        env.inf.total_reward_future_state = 0

        epoch_array = np.append(epoch_array, epoch)
        step_reward_array_standard = np.append(step_reward_array_standard, avg_step_reward_standard)
        step_reward_array_future_state  = np.append(step_reward_array_future_state,  avg_step_reward_future_state)
        smoothed_standard = gaussian_filter1d(step_reward_array_standard, sigma=epoch/50.0)
        smoothed_future_state  = gaussian_filter1d(step_reward_array_future_state,  sigma=epoch/50.0)
        if step_reward_array_standard[-1] == np.max(step_reward_array_standard):
            model.save(os.path.join(file_path, f"best_step_model.zip"))
            print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr  ||  best avg step reward = {round(avg_step_reward_standard,3)}")
        else:
            print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")
        np.save(os.path.join(file_path, "array/epoch_array.npy"), epoch_array)
        np.save(os.path.join(file_path, "array/step_reward_array_standard.npy"), step_reward_array_standard)
        np.save(os.path.join(file_path, "array/step_reward_array_future_state.npy"), step_reward_array_future_state)

        fig = plt.figure(figsize=(15, 10))
        plt.plot(epoch_array, step_reward_array_standard, label="Step Reward (Standard Original Data)", color='black', alpha=0.2)
        plt.plot(epoch_array, smoothed_standard, label="Step Reward (Standard)", color='red', alpha=1)
        plt.plot(epoch_array, smoothed_future_state, label="Step Reward (Training, Future State)", color='red', alpha=0.2)
        plt.title("Average Step Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Step Reward")
        plt.ylim(0, 1.5)
        plt.legend()

        plt.savefig(os.path.join(file_path, "epoch_vs_reward.png"))
        plt.close()
       
def test(model, env, model_path, render_speed=0.0):
    env.render_speed = render_speed
    model = stable_baselines3.SAC.load(model_path, env)
    obs, _ = env.reset()
    env.inf.totaltimestep = 0
    env.inf.total_reward_standard = 0
    while env.viewer.is_running() == True:
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    avg_step_reward_standard = env.inf.total_reward_standard / env.inf.totaltimestep
    # print(f"\ntotal reward = {round(env.inf.total_reward_standard,2)}")
    print(f"\navg step reward = {round(avg_step_reward_standard,3)}")
    env.close()

if __name__ == '__main__':
    my_env = RL_arm()
    file_path = os.path.dirname(os.path.abspath(__file__))
    current_model_path = os.path.join(file_path, "current_model.zip")
    best_model_path = os.path.join(file_path, "best_step_model.zip")
    if os.path.exists(current_model_path):
        print(f"model file: {current_model_path}")
        RL_model = stable_baselines3.SAC.load(current_model_path, my_env)
    else:
        RL_model = stable_baselines3.SAC('MlpPolicy', my_env, learning_rate=0.0005)
        RL_model.save(current_model_path)

    # train(RL_model, my_env, file_path, render_speed = 1.0)
    test(RL_model, my_env, current_model_path, render_speed = 0.1)
    # test(RL_model, my_env, best_model_path, render_speed = 0.1)
