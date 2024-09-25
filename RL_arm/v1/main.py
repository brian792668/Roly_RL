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
        self.truncated = False
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
        elif self.inf.timestep >= 2048:
            self.done = False
            self.truncated = True
            info = {}
            return self.observation_space, self.inf.reward, self.done, self.truncated, info
        else:
            self.inf.timestep += 1
            # print(self.inf.timestep)
            # print(self.data.time)

            for i in range(5):
                self.inf.action[i] = self.inf.action[i]*0.85 + action[i]*0.15
            self.sys.ctrlpos[5] = 0
            # self.inf.action[0] = self.inf.action[0]*0.85 + action[0]*0.15
            # self.inf.action[1] = self.inf.action[1]*0.85 + action[1]*0.15
            # self.inf.action[2] = self.inf.action[2]*0.85 + action[2]*0.15
            # self.inf.action[3] = self.inf.action[3]*0.85 + action[3]*0.15
            # self.inf.action[4] = self.inf.action[4]*0.85 + action[4]*0.15

            # for i in range(5):
            #     self.inf.action[i] += action[i]*0.5
            #     if self.inf.action[i] >=  1: self.inf.action[i] = 1
            #     if self.inf.action[i] <= -1: self.inf.action[i] = -1

            for i in range(10):
                # if self.inf.action[0] <= 0:
                #     self.sys.ctrlpos[3] = self.sys.ctrlpos[3]*0.98 + 0.0
                # else:
                #     self.sys.ctrlpos[3] = self.sys.ctrlpos[3]*0.98 + self.inf.action[0]*0.02
                # if self.inf.action[1] >= 0:
                #     self.sys.ctrlpos[4] = self.sys.ctrlpos[4]*0.98 + 0.0
                # else:
                #     self.sys.ctrlpos[4] = self.sys.ctrlpos[4]*0.98 + self.inf.action[1]*0.02
                # if self.inf.action[2] <= 0:
                #     self.sys.ctrlpos[5] = self.sys.ctrlpos[5]*0.98 + 0.0
                # else:
                #     self.sys.ctrlpos[5] = self.sys.ctrlpos[5]*0.98 + self.inf.action[2]*0.02


                # for i in range(5):
                #     if self.inf.action[i]>=0: 
                #         self.sys.ctrlpos[i+3] = self.sys.pos[i+3] + self.inf.action[i]*0.001*(self.sys.limit_high[i] - self.sys.pos[i+3])
                #     else: 
                #         self.sys.ctrlpos[i+3] = self.sys.pos[i+3] + self.inf.action[i]*0.001*(self.sys.pos[i+3] - self.sys.limit_low[i] )
                # self.sys.ctrlpos[5] = 0

                for i in range(5):
                    if self.inf.action[i]>=0: 
                        self.sys.ctrlpos[i+3] = self.sys.pos[i+3] + self.inf.action[i]*0.01*(self.sys.limit_high[i] - self.sys.pos[i+3])
                    else: 
                        self.sys.ctrlpos[i+3] = self.sys.pos[i+3] + self.inf.action[i]*0.01*(self.sys.pos[i+3] - self.sys.limit_low[i] )
                self.sys.ctrlpos[5] = 0
                

                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                
                mujoco.mj_step(self.robot, self.data)
                # print(f"{self.data.time:2f}", mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_GEOM, 'trunk'))

                for i, con in enumerate(self.data.contact):
                    geom1_id = con.geom1
                    geom2_id = con.geom2
                    if geom1_id == 32 or geom2_id == 32:
                        self.done = False
                        self.truncated = True
                        info = {}
                        return self.observation_space, self.inf.reward, self.done, self.truncated, info

            
            self.inf.reward = self.get_reward()
            self.get_state()
            self.head_camera.get_img(self.data, rgb=True, depth=True)
            self.head_camera.get_target(depth = False)
            self.head_camera.show(rgb=True)
            self.sys.ctrlpos[1:3] = self.head_camera.track(self.sys.ctrlpos[1:3], self.data, speed=0.2 )

            self.viewer.sync()
            self.observation_space = np.concatenate([self.obs.joint_camera,
                                                     [self.obs.cam2target]*3, 
                                                     self.obs.joint_arm,
                                                     self.inf.action]).astype(np.float32)
            info = {}
            self.truncated = False
            return self.observation_space, self.inf.reward, self.done, self.truncated, info
    
    def reset(self, seed=None, **kwargs):
        if self.viewer.is_running() == False:
            self.close()
        else:
            mujoco.mj_resetData(self.robot, self.data)
            self.inf.reset()
            self.sys.reset()
            self.obs.reset()
            self.head_camera.track_done = False

            for i in range(50):
                self.sys.ctrlpos[2] = self.sys.ctrlpos[2]*0.95 + np.radians(-60)*0.05
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)
                # self.head_camera.get_img(self.data, rgb=True, depth=True)
                # self.head_camera.show(rgb=True)

            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

            self.get_state()
            self.observation_space = np.concatenate([ self.obs.joint_camera, 
                                                     [self.obs.cam2target]*3,
                                                      self.obs.joint_arm,
                                                      self.inf.action]).astype(np.float32)
            self.done = False
            self.truncated = False
            return self.observation_space, {}

    def get_reward(self):
        # self.sys.pos_target = self.data.qpos[36:39].copy()
        # self.sys.pos_hand = self.data.site_xpos[42].copy()

        self.sys.pos_target = self.data.qpos[16:19].copy()
        self.sys.pos_hand = self.data.site_xpos[-1].copy()
        # print(self.data.site_xpos[-1])

        new_dis  = (self.sys.pos_target[0]-self.sys.pos_hand[0])**2
        new_dis += (self.sys.pos_target[1]-self.sys.pos_hand[1])**2
        new_dis += (self.sys.pos_target[2]-self.sys.pos_hand[2])**2
        new_dis = new_dis ** 0.5

        reward_of_getting_close = new_dis - self.sys.hand2target
        if reward_of_getting_close <= 0:
            self.inf.reward = np.exp(-5*self.sys.hand2target)
        else:
            self.inf.reward = np.exp(-5*self.sys.hand2target) - 500*reward_of_getting_close
        self.inf.total_reward += self.inf.reward
        self.sys.hand2target = new_dis
        print(self.inf.reward)
        return self.inf.reward
 
    def get_state(self):
        if self.inf.timestep%500 == 0:
            if self.inf.timestep > 0 and self.sys.hand2target >= 0.1:
                self.reset()
            else:
                self.inf.reward += 10
                self.head_camera.track_done = False
                while self.head_camera.track_done != True:
                    distoshoulder = 0.5
                    while distoshoulder >= 0.4:
                        self.data.qpos[16] = random.uniform( 0.10, 0.50)
                        self.data.qpos[17] = random.uniform(-0.50, 0.00)
                        self.data.qpos[18] = random.uniform( 0.92, 1.35)

                        self.sys.pos_target = self.data.qpos[16:19].copy()
                        distoshoulder  = (self.sys.pos_target[0]-0.00)**2
                        distoshoulder += (self.sys.pos_target[1]+0.25)**2
                        distoshoulder += (self.sys.pos_target[2]-1.35)**2
                        distoshoulder = distoshoulder ** 0.5
                    mujoco.mj_step(self.robot, self.data)
                    for i in range(100):
                        # print("tracking", f"{self.data.time:2f}")
                        self.head_camera.get_img(self.data, rgb=True, depth=True)
                        self.head_camera.get_target(depth = True)
                        self.sys.ctrlpos[1:3] = self.head_camera.track(self.sys.ctrlpos[1:3], self.data, speed=0.5 )
                        self.sys.pos = [self.data.qpos[i] for i in controlList]
                        self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                        self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                        mujoco.mj_step(self.robot, self.data)
                        # self.head_camera.show(rgb=True)

        self.obs.joint_camera = self.data.qpos[8:10].copy()
        self.obs.joint_arm    = self.data.qpos[10:15].copy()
        self.obs.cam2target   = self.head_camera.target_depth

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
            while env.truncated == False:
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
        while env.truncated == False:
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
    current_model_path = "RL/RL_arm/v1/model/model_1/current_model.zip"
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
