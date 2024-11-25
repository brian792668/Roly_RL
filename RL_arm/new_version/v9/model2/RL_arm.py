import mujoco
import mujoco.viewer
import cv2
import gymnasium as gym
import numpy as np
import random
from stable_baselines3 import SAC
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports.Camera import *
from imports.state_action import *
from imports.RL_info import *

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(64, 128) # 隱藏層64輸出128
        self.fc3 = nn.Linear(128, 4)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第1層用 ReLU
        x = F.relu(self.fc2(x))  # 第2層用 ReLU
        x = self.fc3(x)          # 最後一層直接輸出
        return x

class RL_arm(gym.Env):
    def __init__(self):
        self.done = False
        self.truncated = False
        self.robot = mujoco.MjModel.from_xml_path('Roly/Roly_XML2/Roly.xml')
        self.data = mujoco.MjData(self.robot)
        self.action_space = gym.spaces.box.Box( low  = act_low,      # action (rad)
                                                high = act_high,
                                                dtype = np.float32)
        self.observation_space = gym.spaces.box.Box(low  = obs_low,
                                                    high = obs_high,
                                                    dtype = np.float32 )
        
        self.renderer = mujoco.Renderer(self.robot, )
        self.inf = RL_inf()
        self.sys = RL_sys(Hz=50)
        self.obs = RL_obs()

        self.head_camera = Camera(renderer=self.renderer, camID=0)
        self.hand_camera = Camera(renderer=self.renderer, camID=2)

        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200

        self.model1 = SAC.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model1.zip"))
        self.IK = IKMLP()
        self.IK.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "IKmodel_v7.pth")))
        self.IK.eval()
        
    def step(self, action): 
        # self.inf.truncated = False
        if self.viewer.is_running() == False:
            self.close()
        elif self.inf.timestep > 2000:
            self.inf.timestep = 0
            # self.inf.reward += 100
            self.inf.done = False
            self.inf.truncated = True
            self.inf.info = {}
            return self.observation_space, self.inf.reward, self.done, self.truncated, self.inf.info
        else:
            self.inf.timestep += 1
            self.inf.totaltimestep += 1

            action_from_model1, _ = self.model1.predict(self.observation_space, deterministic=True)
            self.inf.action[0] = self.inf.action[0]*0.8 + action_from_model1[0]*0.2
            self.inf.action[1] = self.inf.action[1]*0.5 +             action[0]*0.5
            self.inf.action[2] = self.inf.action[2]*0.8 + action_from_model1[1]*0.2
            self.inf.action[3] = self.inf.action[3]*0.8 + action_from_model1[2]*0.2

            with torch.no_grad():  # 不需要梯度計算，因為只做推論
                desire_joints = self.IK(torch.tensor(self.obs.obj_to_neck_xyz, dtype=torch.float32)).tolist()
                desire_joints = np.radians(desire_joints)

            for i in range(int(1/self.sys.Hz/0.001)):
                self.sys.ctrlpos[3] = self.sys.pos[3] + self.inf.action[0]*0.002
                # self.sys.ctrlpos[4] = self.sys.pos[4] + self.inf.action[1]*0.002
                self.sys.ctrlpos[4] = self.sys.pos[4] + 0.002*np.tanh(desire_joints[1] - self.sys.pos[4])
                self.sys.ctrlpos[5] = 0
                self.sys.ctrlpos[6] = self.sys.pos[6] + self.inf.action[2]*0.002
                self.sys.ctrlpos[7] = self.sys.pos[7] + self.inf.action[3]*0.002
                if   self.sys.ctrlpos[3] > self.sys.limit_high[0]: self.sys.ctrlpos[3] = self.sys.limit_high[0]
                elif self.sys.ctrlpos[3] < self.sys.limit_low[0] : self.sys.ctrlpos[3] = self.sys.limit_low[0]
                if   self.sys.ctrlpos[4] > self.sys.limit_high[1]: self.sys.ctrlpos[4] = self.sys.limit_high[1]
                elif self.sys.ctrlpos[4] < self.sys.limit_low[1] : self.sys.ctrlpos[4] = self.sys.limit_low[1]
                if   self.sys.ctrlpos[6] > self.sys.limit_high[2]: self.sys.ctrlpos[6] = self.sys.limit_high[2]
                elif self.sys.ctrlpos[6] < self.sys.limit_low[2] : self.sys.ctrlpos[6] = self.sys.limit_low[2]
                if   self.sys.ctrlpos[7] > self.sys.limit_high[3]: self.sys.ctrlpos[7] = self.sys.limit_high[3]
                elif self.sys.ctrlpos[7] < self.sys.limit_low[3] : self.sys.ctrlpos[7] = self.sys.limit_low[3]

                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                
                mujoco.mj_step(self.robot, self.data)
                self.render()
                # print(f"{self.data.time:2f}", mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_GEOM, 'trunk'))

                for i, con in enumerate(self.data.contact):
                    geom1_id = con.geom1
                    geom2_id = con.geom2
                    if geom1_id == 32 or geom2_id == 32 or geom1_id == 33 or geom2_id == 33:
                        self.inf.done = False
                        self.inf.truncated = True
                        self.inf.info = {}
                        return self.observation_space, self.inf.reward, self.inf.done, self.inf.truncated, self.inf.info

            self.inf.reward = self.get_reward()
            self.get_state()
            self.observation_space = np.concatenate([self.obs.obj_to_hand_xyz, self.obs.joint_arm]).astype(np.float32)
            self.inf.info = {}
            return self.observation_space, self.inf.reward, self.inf.done, self.inf.truncated, self.inf.info
    
    def reset(self, seed=None, **kwargs): 
        if self.viewer.is_running() == False:
            self.close()
        else:
            mujoco.mj_resetData(self.robot, self.data)
            self.inf.reset()
            self.sys.reset()
            self.obs.reset()
            self.head_camera.track_done = False

            self.data.qpos[16:19] = [0.2, -0.25, 1.3] # position of target

            dummy_random = np.radians(random.uniform( 0, 1)**2)*60
            self.sys.random_arm_pos = [ -dummy_random,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2*dummy_random,
                                        0.0 ]
            for i in range(100):
                self.sys.ctrlpos[3] = self.sys.pos[3] + 0.01*np.tanh(self.sys.random_arm_pos[0] - self.sys.pos[3])
                self.sys.ctrlpos[4] = self.sys.pos[4] + 0.01*np.tanh(self.sys.random_arm_pos[1] - self.sys.pos[4])
                self.sys.ctrlpos[5] = self.sys.pos[5] + 0.01*np.tanh(self.sys.random_arm_pos[2] - self.sys.pos[5])
                self.sys.ctrlpos[6] = self.sys.pos[6] + 0.01*np.tanh(self.sys.random_arm_pos[3] - self.sys.pos[6])
                self.sys.ctrlpos[7] = self.sys.pos[7] + 0.01*np.tanh(self.sys.random_arm_pos[4] - self.sys.pos[7])
                self.sys.ctrlpos[8] = self.sys.pos[8] + 0.01*np.tanh(self.sys.random_arm_pos[5] - self.sys.pos[8])
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)
                self.render()

            self.head_camera.get_img(self.data, rgb=True, depth=True)
            self.get_state()
            self.observation_space = np.concatenate([self.obs.obj_to_hand_xyz, self.obs.joint_arm]).astype(np.float32)
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self):
        self.sys.pos_target = self.data.qpos[16:19].copy()
        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        # print(self.sys.pos_hand)

        new_dis  = (self.sys.pos_target[0]-self.sys.pos_hand[0])**2
        new_dis += (self.sys.pos_target[1]-self.sys.pos_hand[1])**2
        new_dis += (self.sys.pos_target[2]-self.sys.pos_hand[2])**2
        new_dis = new_dis ** 0.5
        self.sys.hand2target = new_dis
        # print(new_dis)

        # r0: reward of position
        r0 = np.exp(-100*new_dis**2)

        mse = 0
        with torch.no_grad():  # 不需要梯度計算，因為只做推論
            desire_joints = self.IK(torch.tensor(self.obs.obj_to_neck_xyz, dtype=torch.float32)).tolist()
            desire_joints = np.radians(desire_joints)
            actual_joints = self.obs.joint_arm[0:4]
            # mse = -sum((a - b)**2 for a, b in zip(desire_joints, actual_joints)) / 4
            mse = np.exp(-(20*(actual_joints[1]-desire_joints[1]))**2)
        
        self.inf.reward = r0*mse
        self.inf.total_reward += self.inf.reward
        # print(self.inf.reward)
        return self.inf.reward
 
    def get_state(self):
        # update position of target
        self.data.qpos[16] += 0.02*np.tanh((self.sys.pos_target0[0] - self.data.qpos[16]))
        self.data.qpos[17] += 0.02*np.tanh((self.sys.pos_target0[1] - self.data.qpos[17]))
        self.data.qpos[18] += 0.02*np.tanh((self.sys.pos_target0[2] - self.data.qpos[18]))

        self.hand_camera.get_img(self.data, rgb=True, depth=False)
        self.hand_camera.get_target(depth = False)

        self.obs.joint_arm[0:2] = self.data.qpos[10:12].copy()
        self.obs.joint_arm[2:4] = self.data.qpos[13:15].copy()

        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        neck_xyz = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()

        self.obs.obj_to_neck_xyz = [self.data.qpos[16]-neck_xyz[0],          self.data.qpos[17]-neck_xyz[1],          self.data.qpos[18]-neck_xyz[2]]
        self.obs.obj_to_hand_xyz = [self.data.qpos[16]-self.sys.pos_hand[0], self.data.qpos[17]-self.sys.pos_hand[1], self.data.qpos[18]-self.sys.pos_hand[2]]
        # print(f"{self.obs.obj_to_hand_xyz[0]:.2f}, {self.obs.obj_to_hand_xyz[1]:.2f}, {self.obs.obj_to_hand_xyz[2]:.2f}")

        if self.inf.timestep%int(3*self.sys.Hz) == 0:
            self.spawn_new_point()

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

    def render(self, speed=1):
        if int(1000*self.data.time)%int(450*speed+50) == 0: # 50ms render 一次
            self.viewer.sync()
            self.viewer.cam.azimuth += 0.05 

    def check_reachable(self, point):
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        distoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        # distoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if distoshoulder >= 0.4 or distoshoulder <= 0.25:
            return False
        elif (point[0]<0.12 and point[1] > -0.20):
            return False
        else:
            return True
        
    def spawn_new_point(self):
        hand_camera_center = 2.0
        if np.isnan(self.hand_camera.target[0]) == False:
            hand_camera_center = (self.hand_camera.target[0]**2 + self.hand_camera.target[1]**2)**0.5
        if self.inf.timestep == 0 or self.sys.hand2target <= 0.05 or hand_camera_center <= 0.2:
            self.inf.reward += 10
            # self.sys.random_arm_pos[2] = np.radians(random.uniform( -60, 6.8))
            self.sys.random_arm_pos[1] = np.radians(-60+66.8*(1-random.uniform( 0, 1)**2))
            reachable = False
            while reachable == False:
                self.sys.pos_target0[0] = random.uniform( 0.02, 0.50)
                self.sys.pos_target0[1] = random.uniform(-0.70, 0.00)
                self.sys.pos_target0[2] = random.uniform( 0.90, 1.35)

                # self.data.qpos[16] = random.uniform( 0.02, 0.50)
                # self.data.qpos[17] = random.uniform(-0.70, 0.00)
                # self.data.qpos[18] = random.uniform( 0.90, 1.35)
                # self.sys.pos_target0 = self.data.qpos[16:19].copy()
                reachable = self.check_reachable(self.sys.pos_target0)

            neck_xyz = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
            # self.obs.obj_to_neck_xyz = [self.sys.pos_target0[0]-neck_xyz[0], self.sys.pos_target0[1]-neck_xyz[1], self.sys.pos_target0[2]-neck_xyz[2]]
            # self.obs.obj_to_hand_xyz = [self.sys.pos_target0[0]-self.sys.pos_hand[0], self.sys.pos_target0[1]-self.sys.pos_hand[1], self.sys.pos_target0[2]-self.sys.pos_hand[2]]
            self.obs.obj_to_neck_xyz = [self.data.qpos[16]-neck_xyz[0],          self.data.qpos[17]-neck_xyz[1],          self.data.qpos[18]-neck_xyz[2]]
            self.obs.obj_to_hand_xyz = [self.data.qpos[16]-self.sys.pos_hand[0], self.data.qpos[17]-self.sys.pos_hand[1], self.data.qpos[18]-self.sys.pos_hand[2]]

            new_dis = (self.obs.obj_to_hand_xyz[0]**2 + self.obs.obj_to_hand_xyz[1]**2 + self.obs.obj_to_hand_xyz[2]**2) **0.5
            self.sys.hand2target  = new_dis
            self.sys.hand2target0 = new_dis
            mujoco.mj_step(self.robot, self.data)
        else:
            self.reset()