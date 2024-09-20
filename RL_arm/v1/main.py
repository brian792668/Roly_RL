import mujoco
import mujoco.viewer
import cv2
import gymnasium as gym
import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os

from Camera import *
from state_action import *
from RL_info import *

class RL_arm(gym.Env):
    def __init__(self):
        self.done = False
        self.robot = mujoco.MjModel.from_xml_path('RL/RolyURDF2/Roly.xml')
        self.data = mujoco.MjData(self.robot)
        self.action_space = gym.spaces.box.Box( low  = act_low,      # action (rad)
                                                high = act_high,
                                                dtype = np.float32)
        self.observation_space = gym.spaces.box.Box(low  = obs_low,
                                                    high = obs_high,
                                                    dtype = np.float32 )
        
        self.renderer = mujoco.Renderer(self.robot)
        self.inf = RL_inf()
        self.sys = RL_sys()
        self.obs = RL_obs()

        self.head_camera = Camera(renderer=self.renderer, camID=0)
        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data)
        self.viewer.cam.distance = 2.5
        self.viewer.cam.lookat = [0.0, 0.0, 0.8]
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 160
        
    def step(self, action): 
        if self.viewer.is_running() == False:
            self.close()
        elif self.inf.timestep == 2048:
            self.done = False
            truncated = True
            info = {}
            return self.observation_space, self.inf.reward, self.done, truncated, info
        else:
            self.inf.timestep += 1
            # print(self.data.time)
            
            while self.head_camera.track_done != True:
                self.sys.ctrlpos[1:3] = self.head_camera.track(self.sys.ctrlpos[1:3], self.data, speed=0.5 )
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)
                # self.head_camera.show(rgb=True)
            
            self.sys.ctrlpos[1:3] = self.head_camera.track(self.sys.ctrlpos[1:3], self.data, speed=0.5 )
            for i in range(20):
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)

            self.inf.reward = self.get_reward()
            self.get_state()

            # self.viewer.sync()
            if self.inf.timestep%5 == 0:
                self.head_camera.show(rgb=True)
            self.observation_space = np.concatenate([self.obs.pos_camera, [self.obs.dis_target], self.obs.pos_arm]).astype(np.float32)
            info = {}
            truncated = False
            return self.observation_space, self.inf.reward, self.done, truncated, info
    
    def reset(self, seed=None, **kwargs):
        if self.viewer.is_running() == False:
            self.close()
        else:
            mujoco.mj_resetData(self.robot, self.data)
            self.inf.reset()
            self.sys.reset()
            self.obs.reset()
            self.head_camera.track_done = False

            while self.head_camera.track_done != True:
                self.sys.ctrlpos[1:3] = self.head_camera.track(self.sys.ctrlpos[1:3], self.data, speed=0.5 )
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)
                self.head_camera.show(rgb=True)

            self.get_state()
            self.observation_space = np.concatenate([self.obs.pos_camera, [self.obs.dis_target], self.obs.pos_arm]).astype(np.float32)
            self.done = False
            return self.observation_space, {}

    def get_reward(self):
        self.inf.reward = random.uniform(-0.1, 0.3)
        self.inf.total_reward += self.inf.reward
        return self.inf.reward
 
    def get_state(self):
        if self.inf.timestep%150 == 0:
            self.data.qpos[36] = random.uniform(-0.1, 0.3)
            self.data.qpos[37] = random.uniform(-0.4, 0.0)
            self.head_camera.track_done = False

        self.obs.pos_camera = self.data.qpos[8:10].copy()
        self.obs.dis_target = self.head_camera.target_depth
        self.obs.pos_arm = self.data.qpos[10:13].copy()

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

def train(model, env, current_model_path, best_model_path, best_step_model_path):
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])
    total_reward_plot = np.array([0.0])
    best_avg_total_reward = np.array([0.0])
    best_avg_step_reward = np.array([0.0])

    if os.path.exists("RL/RL_arm/v1/model/model_1/array/epoch_plot.npy"):
        epoch_plot = np.load("RL/RL_arm/v1/model/model_1/array/epoch_plot.npy")
        step_reward_plot = np.load("RL/RL_arm/v1/model/model_1/array/step_reward_plot.npy")
        total_reward_plot = np.load("RL/RL_arm/v1/model/model_1/array/total_reward_plot.npy")
        best_avg_total_reward = np.load("RL/RL_arm/v1/model/model_1/array/best_avg_total_reward.npy")
        best_avg_step_reward = np.load("RL/RL_arm/v1/model/model_1/array/best_avg_step_reward.npy")

    else:
        np.save("RL/RL_arm/v1/model/model_1/array/epoch_plot.npy", epoch_plot)
        np.save("RL/RL_arm/v1/model/model_1/array/step_reward_plot.npy", step_reward_plot)
        np.save("RL/RL_arm/v1/model/model_1/array/total_reward_plot.npy", total_reward_plot)
        np.save("RL/RL_arm/v1/model/model_1/array/best_avg_total_reward.npy", best_avg_total_reward)
        np.save("RL/RL_arm/v1/model/model_1/array/best_avg_step_reward.npy", best_avg_step_reward)

    epoch = epoch_plot[-1]
    timer0 = time.time()

    while True:
        epoch += 1
        print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")
        model.learn(total_timesteps = 2048)
        model.save(current_model_path)
        
        sum_of_total_reward = 0.0
        sum_of_total_step = 0

        reward_of_case = np.array([0.0])
        for i in range(len(reward_of_case)):
            obs, _ = env.reset()
            while env.done == False:
                action, _ = model.predict(obs)
                obs, _, _, _, _ = env.step(action)
            sum_of_total_reward += env.inf.total_reward
            sum_of_total_step += env.inf.timestep
            reward_of_case[i] = env.inf.total_reward/env.inf.timestep

        avg_step_reward = sum_of_total_reward / sum_of_total_step
        avg_total_reward = sum_of_total_reward / len(reward_of_case)
        if avg_step_reward >= best_avg_step_reward[0]:
            best_avg_step_reward[0] = avg_step_reward
            model.save(best_step_model_path)
            print(f"best avg step reward = {round(avg_step_reward,3)}")
            # print(f"reward of case = {round(reward_of_case[0],2)} {round(reward_of_case[1],2)} {round(reward_of_case[2],2)} {round(reward_of_case[3],2)} {round(reward_of_case[4],2)} {round(reward_of_case[5],2)}")
        if avg_total_reward >= best_avg_total_reward[0]:
            best_avg_total_reward[0] = avg_total_reward
            model.save(best_model_path)
            print(f"best avg total reward = {round(best_avg_total_reward[0],2)}  avg step reward = {round(avg_step_reward,3)}")
            # print(f"reward of case = {round(reward_of_case[0],2)} {round(reward_of_case[1],2)} {round(reward_of_case[2],2)} {round(reward_of_case[3],2)} {round(reward_of_case[4],2)} {round(reward_of_case[5],2)}")
        epoch_plot = np.append(epoch_plot, epoch)
        step_reward_plot = np.append(step_reward_plot, avg_step_reward)
        total_reward_plot = np.append(total_reward_plot, avg_total_reward)
        np.save("RL/RL_arm/v1/model/model_1/array/epoch_plot.npy",epoch_plot)
        np.save("RL/RL_arm/v1/model/model_1/array/step_reward_plot.npy",step_reward_plot)
        np.save("RL/RL_arm/v1/model/model_1/array/total_reward_plot.npy",total_reward_plot)
        np.save("RL/RL_arm/v1/model/model_1/array/best_avg_total_reward.npy",best_avg_total_reward)
        np.save("RL/RL_arm/v1/model/model_1/array/best_avg_step_reward.npy",best_avg_step_reward)

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

        plt.savefig("RL/RL_arm/v1/model/model_1/epoch_vs_reward.png")
        plt.close()
       
def test(model, env, model_path):
    model = stable_baselines3.PPO.load(model_path, env)
    sum_of_total_reward = 0.0
    sum_of_total_step = 0

    reward_of_case = np.array([0.0])
    for i in range(len(reward_of_case)):
        obs, _ = env.reset()
        while env.done == False:
            action, _ = model.predict(obs)
            obs, _, _, _, _ = env.step(action)
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
    best_model_path = "RL/RL_arm/v1/model/model_1/best_model.zip"
    best_step_model_path = "RL/RL_arm/v1/model/model_1/best_step_model.zip"
    current_model_path = "RL/RL_arm/v1/model/current_model.zip"
    if os.path.exists(current_model_path):
        print(f"model file: {current_model_path}")
        my_model = stable_baselines3.PPO.load(current_model_path, my_env)
    else:
        my_model = stable_baselines3.PPO('MlpPolicy', my_env, verbose=1)
        my_model.save(current_model_path)

    # my_model.learn(total_timesteps = 20)
    # my_model.save(current_model_path)

    train(my_model, my_env, current_model_path, best_model_path, best_step_model_path)
    # test(my_model, my_env, best_model_path)
    # test(my_model, my_env, best_step_model_path)
    # test(my_model, my_env, current_model_path)
