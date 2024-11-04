import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from RL_arm import *

def train(model, env, current_model_path):
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])
    total_reward_plot = np.array([0.0])
    best_avg_total_reward = np.array([0.0])
    best_avg_step_reward = np.array([0.0])

    if os.path.exists("Roly/RL_arm/v14/model/SAC/array/epoch_plot.npy"):
        epoch_plot = np.load("Roly/RL_arm/v14/model/SAC/array/epoch_plot.npy")
        step_reward_plot = np.load("Roly/RL_arm/v14/model/SAC/array/step_reward_plot.npy")
        total_reward_plot = np.load("Roly/RL_arm/v14/model/SAC/array/total_reward_plot.npy")
        best_avg_total_reward = np.load("Roly/RL_arm/v14/model/SAC/array/best_avg_total_reward.npy")
        best_avg_step_reward = np.load("Roly/RL_arm/v14/model/SAC/array/best_avg_step_reward.npy")

    else:
        np.save("Roly/RL_arm/v14/model/SAC/array/epoch_plot.npy", epoch_plot)
        np.save("Roly/RL_arm/v14/model/SAC/array/step_reward_plot.npy", step_reward_plot)
        np.save("Roly/RL_arm/v14/model/SAC/array/total_reward_plot.npy", total_reward_plot)
        np.save("Roly/RL_arm/v14/model/SAC/array/best_avg_total_reward.npy", best_avg_total_reward)
        np.save("Roly/RL_arm/v14/model/SAC/array/best_avg_step_reward.npy", best_avg_step_reward)

    epoch = epoch_plot[-1]
    timer0 = time.time()

    while True:
        epoch += 1
        print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")
        model.learn(total_timesteps = 2048)
        model.save(current_model_path)

        avg_step_reward = env.inf.total_reward / env.inf.totaltimestep
        avg_total_reward = env.inf.total_reward
        env.reset()
        env.inf.totaltimestep = 0
        env.inf.total_reward = 0

        if avg_step_reward >= best_avg_step_reward[0]:
            best_avg_step_reward[0] = avg_step_reward
            model.save(f"Roly/RL_arm/v14/model/SAC/best_step/best_step_model_epoch{epoch}.zip")
            print(f"best avg step reward = {round(avg_step_reward,3)}")
            # print(f"reward of case = {round(reward_of_case[0],2)} {round(reward_of_case[1],2)} {round(reward_of_case[2],2)} {round(reward_of_case[3],2)} {round(reward_of_case[4],2)} {round(reward_of_case[5],2)}")
        if avg_total_reward >= best_avg_total_reward[0]:
            best_avg_total_reward[0] = avg_total_reward
            model.save(f"Roly/RL_arm/v14/model/SAC/best_total/best_total_model_epoch{epoch}.zip")
            print(f"best avg total reward = {round(best_avg_total_reward[0],2)}  avg step reward = {round(avg_step_reward,3)}")
            # print(f"reward of case = {round(reward_of_case[0],2)} {round(reward_of_case[1],2)} {round(reward_of_case[2],2)} {round(reward_of_case[3],2)} {round(reward_of_case[4],2)} {round(reward_of_case[5],2)}")
        epoch_plot = np.append(epoch_plot, epoch)
        step_reward_plot = np.append(step_reward_plot, avg_step_reward)
        total_reward_plot = np.append(total_reward_plot, avg_total_reward)
        np.save("Roly/RL_arm/v14/model/SAC/array/epoch_plot.npy",epoch_plot)
        np.save("Roly/RL_arm/v14/model/SAC/array/step_reward_plot.npy",step_reward_plot)
        np.save("Roly/RL_arm/v14/model/SAC/array/total_reward_plot.npy",total_reward_plot)
        np.save("Roly/RL_arm/v14/model/SAC/array/best_avg_total_reward.npy",best_avg_total_reward)
        np.save("Roly/RL_arm/v14/model/SAC/array/best_avg_step_reward.npy",best_avg_step_reward)

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

        plt.savefig("Roly/RL_arm/v14/model/SAC/epoch_vs_reward.png")
        plt.close()
       
def test(model, env, model_path):
    model = stable_baselines3.SAC.load(model_path, env)
    sum_of_total_reward = 0.0
    sum_of_total_step = 0

    env.reset()
    env.inf.totaltimestep = 0
    env.inf.total_reward = 0

    reward_of_case = np.array([0.0])
    for i in range(len(reward_of_case)):
        obs, _ = env.reset()
        while env.viewer.is_running() == True:
            action, _ = model.predict(obs)
            obs, _, _, _, _ = env.step(action)
        # while env.inf.truncated == False:
        #     action, _ = model.predict(obs)
        #     obs, _, _, _, _ = env.step(action)
        sum_of_total_reward += env.inf.total_reward
        sum_of_total_step += env.inf.timestep
        reward_of_case[i] = env.inf.total_reward/env.inf.timestep
    avg_step_reward = sum_of_total_reward / sum_of_total_step
    avg_total_reward = sum_of_total_reward / len(reward_of_case)
    print(f"\navg total reward = {round(avg_total_reward,2)}  avg step reward = {round(avg_step_reward,3)}")
    # print(f"reward of case = {round(reward_of_case[0],2)} {round(reward_of_case[1],2)} {round(reward_of_case[2],2)} {round(reward_of_case[3],2)} {round(reward_of_case[4],2)} {round(reward_of_case[5],2)}")
    env.close()

if __name__ == '__main__':
    my_env = RL_arm()
    best_model_path = "Roly/RL_arm/v14/model/SAC/best_total/best_total_model_epoch1333.zip"
    current_model_path = "Roly/RL_arm/v14/model/SAC/current_model.zip"
    if os.path.exists(current_model_path):
        print(f"model file: {current_model_path}")
        my_model = stable_baselines3.SAC.load(current_model_path, my_env)
    else:
        my_model = stable_baselines3.SAC('MlpPolicy', my_env, verbose=0)
        my_model.save(current_model_path)

    # train(my_model, my_env, current_model_path)
    # test(my_model, my_env, current_model_path)
    test(my_model, my_env, best_model_path)
