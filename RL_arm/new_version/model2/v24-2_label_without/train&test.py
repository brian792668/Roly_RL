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
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])

    if os.path.exists(os.path.join(file_path, "array/epoch_plot.npy")):
        epoch_plot       = np.load(os.path.join(file_path, "array/epoch_plot.npy"))
        step_reward_plot = np.load(os.path.join(file_path, "array/step_reward_plot.npy"))
    else:
        np.save(os.path.join(file_path, "array/epoch_plot.npy"), epoch_plot)
        np.save(os.path.join(file_path, "array/step_reward_plot.npy"), step_reward_plot)

    epoch = epoch_plot[-1]
    timer0 = time.time()
    env.render_speed = render_speed

    # while True:
    for i in range(500):
        env.reset()
        env.inf.totaltimestep = 0
        env.inf.total_reward = 0
        model.learn(total_timesteps = 4096)
        model.save(os.path.join(file_path, "model_last.zip"))
        
        epoch += 1
        avg_step_reward = env.inf.total_reward / env.inf.totaltimestep
        epoch_plot = np.append(epoch_plot, epoch)
        step_reward_plot = np.append(step_reward_plot, avg_step_reward)
        smoothed_gau = gaussian_filter1d(step_reward_plot, sigma=epoch/100.0)
        if step_reward_plot[-1] == np.max(step_reward_plot):
            model.save(os.path.join(file_path, f"model_best.zip"))
            print(f"\repoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr  ||  best avg step reward = {round(avg_step_reward,3)}                                                   ")
        else:
            print(f"\repoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr                                                              ")
        np.save(os.path.join(file_path, "array/epoch_plot.npy"), epoch_plot)
        np.save(os.path.join(file_path, "array/step_reward_plot.npy"), step_reward_plot)

        fig = plt.figure(figsize=(15, 10))
        plt.plot(epoch_plot, step_reward_plot, label="Original Data", color='black', alpha=0.2)
        plt.plot(epoch_plot[:len(smoothed_gau)], smoothed_gau, label="gaussian filtered", color='red', alpha=1)
        plt.title("Average Step Reward (Gaussian Filtered)")
        plt.xlabel("Epoch")
        plt.ylabel("Step Reward")
        plt.legend()

        plt.savefig(os.path.join(file_path, "epoch_vs_reward.png"))
        plt.close()

        # save label
        folder_path = os.path.join(file_path, f"datasets/new/500_epoch_points")
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, f"EE_xyz_label_epoch{i}.npy"), env.EE_xyz_label)
        np.save(os.path.join(folder_path, f"collision_label_epoch{i}.npy"), env.collision_label)
        del env.EE_xyz_label
        del env.collision_label
        env.EE_xyz_label = np.array([])
        env.collision_label = np.array([])
    print("Done labeling\n")
          
def test(model, env, model_path, render_speed=0):
    env.render_speed = render_speed
    model = stable_baselines3.PPO.load(model_path, env)
    obs, _ = env.reset()
    env.inf.totaltimestep = 0
    env.inf.total_reward = 0
    while env.viewer.is_running() == True:
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
        if env.inf.truncated == True:
            obs, _ = env.reset()
    avg_step_reward = env.inf.total_reward / env.inf.totaltimestep
    # print(f"\ntotal reward = {round(env.inf.total_reward,2)}")
    print(f"\navg step reward = {round(avg_step_reward,3)}")
    env.close()

if __name__ == '__main__':
    my_env = RL_arm()
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_last_path = os.path.join(file_path, "model_last.zip")
    model_best_path = os.path.join(file_path, "model_best.zip")
    if os.path.exists(model_last_path):
        RL_model = stable_baselines3.PPO.load(model_last_path, my_env)
    else:
        RL_model = stable_baselines3.PPO('MlpPolicy', my_env, learning_rate=0.0003)
        RL_model.save(model_last_path)
        print("Create new MLP RL model.")
    
    # if torch.cuda.is_available():
    #     RL_model.policy.to("cuda")
    #     print("Model 2 : CUDA")
    # else:
    #     RL_model.policy.to("cpu")
    #     print("Model 2 : CPU")

    # train(RL_model, my_env, file_path, render_speed = 1.0)
    test(RL_model, my_env, model_last_path, render_speed = 0.3)
    # test(RL_model, my_env, model_best_path, render_speed = 0)
