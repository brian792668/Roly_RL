import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RL_arm import *

def train(model, env, file_path):
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])
    total_reward_plot = np.array([0.0])
    best_avg_total_reward = np.array([0.0])
    best_avg_step_reward = np.array([0.0])

    if os.path.exists(os.path.join(file_path, "array/epoch_plot.npy")):
        epoch_plot              = np.load(os.path.join(file_path, "array/epoch_plot.npy"))
        step_reward_plot        = np.load(os.path.join(file_path, "array/step_reward_plot.npy"))
        total_reward_plot       = np.load(os.path.join(file_path, "array/total_reward_plot.npy"))
        best_avg_total_reward   = np.load(os.path.join(file_path, "array/best_avg_total_reward.npy"))
        best_avg_step_reward    = np.load(os.path.join(file_path, "array/best_avg_step_reward.npy"))
    else:
        np.save(os.path.join(file_path, "array/epoch_plot.npy"), epoch_plot)
        np.save(os.path.join(file_path, "array/step_reward_plot.npy"), step_reward_plot)
        np.save(os.path.join(file_path, "array/total_reward_plot.npy"), total_reward_plot)
        np.save(os.path.join(file_path, "array/best_avg_total_reward.npy"), best_avg_total_reward)
        np.save(os.path.join(file_path, "array/best_avg_step_reward.npy"), best_avg_step_reward)

    epoch = epoch_plot[-1]
    timer0 = time.time()

    while True:
        epoch += 1
        print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")
        model.learn(total_timesteps = 2048)
        model.save(os.path.join(file_path, "current_model.zip"))

        avg_step_reward = env.inf.total_reward / env.inf.totaltimestep
        avg_total_reward = env.inf.total_reward
        env.reset()
        env.inf.totaltimestep = 0
        env.inf.total_reward = 0

        if avg_step_reward >= best_avg_step_reward[0]:
            best_avg_step_reward[0] = avg_step_reward
            model.save(os.path.join(file_path, f"best_step/best_step_model_epoch{epoch}.zip"))
            print(f"best avg step reward = {round(avg_step_reward,3)}")
        if avg_total_reward >= best_avg_total_reward[0]:
            best_avg_total_reward[0] = avg_total_reward
            model.save(os.path.join(file_path, f"best_total/best_total_model_epoch{epoch}.zip"))
            print(f"best avg total reward = {round(best_avg_total_reward[0],2)}  avg step reward = {round(avg_step_reward,3)}")
        epoch_plot = np.append(epoch_plot, epoch)
        step_reward_plot = np.append(step_reward_plot, avg_step_reward)
        total_reward_plot = np.append(total_reward_plot, avg_total_reward)
        np.save(os.path.join(file_path, "array/epoch_plot.npy"), epoch_plot)
        np.save(os.path.join(file_path, "array/step_reward_plot.npy"), step_reward_plot)
        np.save(os.path.join(file_path, "array/total_reward_plot.npy"), total_reward_plot)
        np.save(os.path.join(file_path, "array/best_avg_total_reward.npy"), best_avg_total_reward)
        np.save(os.path.join(file_path, "array/best_avg_step_reward.npy"), best_avg_step_reward)

        fig = plt.figure(figsize=(14, 14))
        plt.subplot(2,1,1)
        plt.plot(epoch_plot, total_reward_plot, label='Total Reward')
        plt.title('Epoch vs. Total Reward')
        plt.xlabel('Epoch')
        plt.ylabel('Total Reward')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(epoch_plot, step_reward_plot, label='Step Reward')
        plt.title('Epoch vs. Step reward (average)')
        plt.xlabel('Epoch')
        plt.ylabel('Step reward (average)')
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
    best_model_path = os.path.join(file_path, "best_total/best_total_model_epoch1732.zip")
    if os.path.exists(current_model_path):
        print(f"model file: {current_model_path}")
        my_model = stable_baselines3.SAC.load(current_model_path, my_env)
    else:
        my_model = stable_baselines3.SAC('MlpPolicy', my_env, learning_rate=0.0002)
        my_model.save(current_model_path)

    # train(my_model, my_env, file_path)
    # test(my_model, my_env, current_model_path)
    test(my_model, my_env, best_model_path)
