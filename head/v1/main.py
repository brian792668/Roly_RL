import mujoco
import mujoco.viewer
import cv2

# import gym
# from gym import spaces 
import gymnasium as gym
import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import state_action
import time
import random
import os

from Settings import *
from Controller import *
from Draw_joint_info import *
from Show_camera_view import *
from Camera import *
from state_action import *
# gym.logger.set_level(40)

class BipedelWalkingEnv(gym.Env):
    def __init__(self):
        self.done = False
        self.robot = mujoco.MjModel.from_xml_path('RL/RolyURDF2/Roly.xml')
        self.data = mujoco.MjData(self.robot)
        self.renderer = mujoco.Renderer(self.robot)
        self.action_space = gym.spaces.box.Box( low  = act_low_flat,      # action (rad)
                                                high = act_high_flat,
                                                dtype = np.float32)
        self.observation_space = gym.spaces.box.Box(low  = obs_low_flat,
                                                    high = obs_high_flat,
                                                    dtype = np.float32 )
        
        self.pos = initPos
        self.vel = initPos
        self.ctrlpos = initTarget
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)
        self.data.qpos[:] = self.pos[:]

        self.head_camera = Camera(renderer=self.renderer, camID=0)
        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data)
        self.viewer.cam.distance = 2.5
        self.viewer.cam.lookat = [0.0, 0.0, 0.8]
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 160
        self.timestep = 0
        self.reward = 0.0
        self.total_reward = 0.0

        self.jointpos = [0.0, 0.0]
        self.cameraxy = [0.0, 0.0]
        
    def step(self, action): 
        if self.viewer.is_running() == False:
            self.close()
        elif self.timestep >= 20:
            self.done = True
            info = {}
            truncated = True
            return self.observation_space, self.reward, self.done, truncated, info
        else:
            self.timestep += 1
            for i in range(50):
                self.ctrlpos[1] = 0.95*self.ctrlpos[1] + 0.05*np.pi/180*action[0]
                self.ctrlpos[2] = 0.95*self.ctrlpos[2] + 0.05*np.pi/180*(-45+action[1])
                self.pos = [self.data.qpos[i] for i in controlList]
                self.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.PIDctrl.getSignal(self.pos, self.vel, self.ctrlpos)
                self.data.ctrl[3:17] = [0]*14
                self.data.qpos[38] = 0.0
                mujoco.mj_step(self.robot, self.data)
                if i%25 == 0:
                    self.viewer.sync()
                    self.head_camera.get_img(self.data, rgb=True, depth=True)
                    self.head_camera.get_target()
                    self.head_camera.show(rgb=True, depth=False)

            self.get_state()
            self.calculate_step_reward()
            self.data.qpos[36] = random.uniform(-0.55, 0.55)
            self.data.qpos[37] = random.uniform(-0.5, 0.5)
            # print(self.jointpos, self.cameraxy)
            self.observation_space = np.concatenate([self.jointpos, self.cameraxy]).astype(np.float32)

            info = {}
            truncated = False
            return self.observation_space, self.reward, self.done, truncated, info
    
    def reset(self, seed=None, **kwargs):
        if self.viewer.is_running() == False:
            self.close()
        else:
            mujoco.mj_resetData(self.robot, self.data)
            self.timestep = 0
            self.reward = 0.0
            self.total_reward = 0.0

            self.ctrlpos = initTarget
            for i in range(100):
                self.ctrlpos[1] = 0.95*self.ctrlpos[1] + 0.05*0
                self.ctrlpos[2] = 0.95*self.ctrlpos[2] + 0.05*np.pi/180*(-45)
                self.pos = [self.data.qpos[i] for i in controlList]
                self.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.PIDctrl.getSignal(self.pos, self.vel, self.ctrlpos)
                self.data.ctrl[3:17] = [0]*14
                mujoco.mj_step(self.robot, self.data)
                if i%50==0:
                    self.viewer.sync()
                    self.head_camera.get_img(self.data, rgb=True, depth=True)
                    self.head_camera.get_target()
                    self.head_camera.show(rgb=True, depth=False)

            self.get_state()
            self.observation_space = np.concatenate([self.jointpos, self.cameraxy]).astype(np.float32)
            self.done = False
            return self.observation_space, {}

    def calculate_step_reward(self):
        distance_to_target = (self.cameraxy[0]**2 + self.cameraxy[1]**2) ** 0.5
        self.reward = np.exp(-5*distance_to_target)
        self.total_reward += self.reward
        # print(self.reward)
        return self.reward
 
    def get_state(self):
        self.pos = [self.data.qpos[i] for i in controlList]
        self.vel = [self.data.qvel[i-1] for i in controlList]
        self.head_camera.get_img(self.data, rgb=True, depth=True)
        self.head_camera.get_target()
        self.jointpos = [180/np.pi*self.pos[1], 180/np.pi*self.pos[2]]
        self.cameraxy = self.head_camera.target

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

def train(model, env, current_model_path, best_model_path, best_step_model_path):
    epoch_plot = np.array([0])
    step_reward_plot = np.array([0.0])
    total_reward_plot = np.array([0.0])
    best_avg_total_reward = np.array([0.0])
    best_avg_step_reward = np.array([0.0])

    if os.path.exists("RL/head/v1/model/model_1/array/epoch_plot.npy"):
        epoch_plot = np.load("RL/head/v1/model/model_1/array/epoch_plot.npy")
        step_reward_plot = np.load("RL/head/v1/model/model_1/array/step_reward_plot.npy")
        total_reward_plot = np.load("RL/head/v1/model/model_1/array/total_reward_plot.npy")
        best_avg_total_reward = np.load("RL/head/v1/model/model_1/array/best_avg_total_reward.npy")
        best_avg_step_reward = np.load("RL/head/v1/model/model_1/array/best_avg_step_reward.npy")

    else:
        np.save("RL/head/v1/model/model_1/array/epoch_plot.npy", epoch_plot)
        np.save("RL/head/v1/model/model_1/array/step_reward_plot.npy", step_reward_plot)
        np.save("RL/head/v1/model/model_1/array/total_reward_plot.npy", total_reward_plot)
        np.save("RL/head/v1/model/model_1/array/best_avg_total_reward.npy", best_avg_total_reward)
        np.save("RL/head/v1/model/model_1/array/best_avg_step_reward.npy", best_avg_step_reward)

    epoch = epoch_plot[-1]
    timer0 = time.time()

    while True:
        epoch += 1
        print(f"epoch = {epoch}   { round((time.time()-timer0)/3600, 2) } hr")
        model.learn(total_timesteps = 1000)
        model.save(current_model_path)
        
        sum_of_total_reward = 0.0
        sum_of_total_step = 0

        reward_of_case = np.array([0.0])
        for i in range(len(reward_of_case)):
            obs, _ = env.reset()
            while env.done == False:
                action, _ = model.predict(obs)
                obs, _, _, _, _ = env.step(action)
            sum_of_total_reward += env.total_reward
            sum_of_total_step += env.timestep
            reward_of_case[i] = env.total_reward/env.timestep

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
        np.save("RL/head/v1/model/model_1/array/epoch_plot.npy",epoch_plot)
        np.save("RL/head/v1/model/model_1/array/step_reward_plot.npy",step_reward_plot)
        np.save("RL/head/v1/model/model_1/array/total_reward_plot.npy",total_reward_plot)
        np.save("RL/head/v1/model/model_1/array/best_avg_total_reward.npy",best_avg_total_reward)
        np.save("RL/head/v1/model/model_1/array/best_avg_step_reward.npy",best_avg_step_reward)

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

        plt.savefig("RL/head/v1/model/model_1/epoch_vs_reward.png")
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
        sum_of_total_reward += env.total_reward
        sum_of_total_step += env.timestep
        reward_of_case[i] = env.total_reward/env.timestep
    avg_step_reward = sum_of_total_reward / sum_of_total_step
    avg_total_reward = sum_of_total_reward / len(reward_of_case)
    print(f"\navg total reward = {round(avg_total_reward,2)}  avg step reward = {round(avg_step_reward,3)}")
    # print(f"reward of case = {round(reward_of_case[0],2)} {round(reward_of_case[1],2)} {round(reward_of_case[2],2)} {round(reward_of_case[3],2)} {round(reward_of_case[4],2)} {round(reward_of_case[5],2)}")
    env.close()

if __name__ == '__main__':
    my_env = BipedelWalkingEnv()
    best_model_path = "RL/head/v1/model/model_1/best_model.zip"
    best_step_model_path = "RL/head/v1/model/model_1/best_step_model.zip"
    current_model_path = "RL/head/v1/model/current_model.zip"
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
