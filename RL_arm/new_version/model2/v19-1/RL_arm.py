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
from imports.model1 import RLmodel

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第1層用 ReLU
        x = F.relu(self.fc2(x))  # 第2層用 ReLU
        x = F.relu(self.fc3(x))  # 第2層用 ReLU
        x = self.fc4(x)          # 最後一層直接輸出
        return x

class RL_arm(gym.Env):
    def __init__(self):
        self.done = False
        self.truncated = False
        self.robot = mujoco.MjModel.from_xml_path('Roly/Roly_XML2-2/Roly.xml')
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
        # self.hand_camera = Camera(renderer=self.renderer, camID=2)

        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200

        self.model1 = RLmodel()
        self.IK = IKMLP()
        self.IK.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "IKmodel_v9.pth")))
        self.IK.eval()
        
    def step(self, action): 
        # self.inf.truncated = False
        if self.viewer.is_running() == False:
            self.close()
        elif self.inf.timestep > 2000:
            self.inf.timestep = 0
            self.inf.done = False
            self.inf.truncated = True
            self.inf.info = {}
            return self.observation_space, self.inf.reward, self.done, self.truncated, self.inf.info
        else:
            self.inf.timestep += 1
            self.inf.totaltimestep += 1
            self.inf.action[0] = self.inf.action[0]*0.9 + action[0]*0.1
            self.inf.action[1] = self.inf.action[1]*0.9 + action[1]*0.1
            self.inf.action[2] = self.inf.action[2]*0.9 + action[2]*0.1
            self.inf.action[3] = self.inf.action[3]*0.9 + action[3]*0.1
            # dx = - self.sys.reaching_dis*np.cos(np.radians(self.inf.action[1]))*np.cos(np.radians(self.inf.action[0]))
            # dy = self.sys.reaching_dis*np.cos(np.radians(self.inf.action[1]))*np.sin(np.radians(self.inf.action[0]))
            # dz = - self.sys.reaching_dis*np.sin(np.radians(self.inf.action[1]))
            # self.sys.pos_target = [self.sys.pos_target0[0]+dx, self.sys.pos_target0[1]+dy, self.sys.pos_target0[2]+dz] 

            self.sys.pos_target = [ self.sys.pos_target[0]+self.inf.action[0]*0.005,
                                    self.sys.pos_target[1]+self.inf.action[1]*0.005,
                                    self.sys.pos_target[2]+self.inf.action[2]*0.005] 

            action_from_model1 = self.model1.predict()
            self.sys.joints_increment[0] = self.sys.joints_increment[0]*0.9 + action_from_model1[0]*0.1
            self.sys.joints_increment[1] = self.sys.joints_increment[1]*0.9 + action_from_model1[1]*0.1
            self.sys.joints_increment[2] = self.inf.action[3]
            self.sys.joints_increment[3] = self.sys.joints_increment[3]*0.9 + action_from_model1[2]*0.1
            # alpha = 1-0.5*np.exp(-300*self.sys.hand2target**2)

            # with torch.no_grad():  # 不需要梯度計算，因為只做推論
            #     desire_joints = self.IK(torch.tensor(self.sys.vec_target2neck, dtype=torch.float32)).tolist()
            #     desire_joints[2] += 10
            #     desire_joints = np.radians(desire_joints)

            for i in range(int(1.0/self.sys.Hz/0.005)):
                self.sys.ctrlpos[0] = self.sys.pos[0] + np.tanh(10*(self.sys.neck_target_pos[0] - self.sys.pos[0]))*0.05
                self.sys.ctrlpos[2] = self.sys.ctrlpos[2] + self.sys.joints_increment[0]*0.01
                self.sys.ctrlpos[3] = self.sys.ctrlpos[3] + self.sys.joints_increment[1]*0.01
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = self.sys.ctrlpos[5] + self.sys.joints_increment[2]*0.01
                self.sys.ctrlpos[6] = self.sys.ctrlpos[6] + self.sys.joints_increment[3]*0.01
                self.control_and_step(render=True)

                # for i, con in enumerate(self.data.contact):
                #     geom1_id = con.geom1
                #     geom2_id = con.geom2
                #     if geom1_id == 32 or geom2_id == 32 or geom1_id == 33 or geom2_id == 33:
                #         self.inf.done = False
                #         self.inf.truncated = True
                #         self.inf.info = {}
                #         return self.observation_space, self.inf.reward, self.inf.done, self.inf.truncated, self.inf.info

            self.inf.reward = self.get_reward()
            self.get_state()
            self.observation_space = np.concatenate([self.sys.vec_target2neck, self.obs.joint_arm]).astype(np.float32)
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
            # self.head_camera.track_done = False

            self.data.qpos[15:18] = [0.2, -0.25, 1.3] # position of target

            dummy_random = np.radians(random.uniform( 0, 1)**2)*60
            self.sys.arm_target_pos = [ -dummy_random,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2*dummy_random,
                                        0.0 ]
            for i in range(100):
                self.sys.ctrlpos[0] = self.sys.pos[0] + np.tanh(10*(self.sys.neck_target_pos[0] - self.sys.pos[0]))*0.02
                self.sys.ctrlpos[1] = self.sys.pos[1] + np.tanh(10*(self.sys.neck_target_pos[1] - self.sys.pos[1]))*0.02
                self.sys.ctrlpos[2] = self.sys.pos[2] + np.tanh(10*(self.sys.arm_target_pos[0] - self.sys.pos[2]))*0.01
                self.sys.ctrlpos[3] = self.sys.pos[3] + np.tanh(10*(self.sys.arm_target_pos[1] - self.sys.pos[3]))*0.01
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = self.sys.pos[5] + np.tanh(10*(self.sys.arm_target_pos[3] - self.sys.pos[5]))*0.01
                self.sys.ctrlpos[6] = self.sys.pos[6] + np.tanh(10*(self.sys.arm_target_pos[4] - self.sys.pos[6]))*0.01
                self.sys.ctrlpos[7] = 0
                self.control_and_step(render=True)

            # self.head_camera.get_img(self.data, rgb=True, depth=True)
            self.get_state()
            self.observation_space = np.concatenate([self.sys.vec_target2neck, self.obs.joint_arm]).astype(np.float32)
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self):
        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        # print(self.sys.pos_hand)

        new_dis = (self.sys.pos_target[0]-self.sys.pos_hand[0])**2 + (self.sys.pos_target[1]-self.sys.pos_hand[1])**2 + (self.sys.pos_target[2]-self.sys.pos_hand[2])**2
        new_dis = new_dis ** 0.5
        self.sys.hand2target = new_dis
        # print(new_dis)

        dis2 = (self.sys.pos_target[0]-self.sys.pos_target0[0])**2 + (self.sys.pos_target[1]-self.sys.pos_target0[1])**2 + (self.sys.pos_target[2]-self.sys.pos_target0[2])**2
        dis2 = dis2**0.5
        r1 = np.exp(-1000*dis2**2)
        
        self.inf.reward = r1
        self.inf.total_reward += self.inf.reward
        # print(self.inf.reward)
        return self.inf.reward
 
    def get_state(self):
        # position of hand, neck, elbow
        self.sys.pos_hand   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        self.sys.pos_neck   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
        self.sys.pos_elbow  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_elbow_marker")].copy()

        # vectors
        self.sys.vec_target2neck  = [self.sys.pos_target[0] - self.sys.pos_neck[0],  self.sys.pos_target[1] - self.sys.pos_neck[1],  self.sys.pos_target[2] - self.sys.pos_neck[2]]
        self.sys.vec_target2hand  = [self.sys.pos_target[0] - self.sys.pos_hand[0],  self.sys.pos_target[1] - self.sys.pos_hand[1],  self.sys.pos_target[2] - self.sys.pos_hand[2]]
        self.sys.vec_target2elbow = [self.sys.pos_target[0] - self.sys.pos_elbow[0], self.sys.pos_target[1] - self.sys.pos_elbow[1], self.sys.pos_target[2] - self.sys.pos_elbow[2]]
        self.sys.vec_hand2elbow   = [self.sys.pos_hand[0]   - self.sys.pos_elbow[0], self.sys.pos_hand[1]   - self.sys.pos_elbow[1], self.sys.pos_hand[2]   - self.sys.pos_elbow[2]]
        
        # model1
        self.model1.obs_target_pos_to_neck = self.sys.vec_target2neck.copy()
        if self.sys.hand2target > 0.02:
            self.model1.obs_target_pos_to_hand_norm[0] = self.sys.vec_target2hand[0]*0.02/self.sys.hand2target
            self.model1.obs_target_pos_to_hand_norm[1] = self.sys.vec_target2hand[1]*0.02/self.sys.hand2target
            self.model1.obs_target_pos_to_hand_norm[2] = self.sys.vec_target2hand[2]*0.02/self.sys.hand2target
        self.model1.obs_joints[0:2] = self.data.qpos[9:11].copy()
        self.model1.obs_joints[2:4] = self.data.qpos[12:14].copy()

        # ----------------------------------------------------------------------------------
        # update camera

        self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "pos_target")] = self.sys.pos_target.copy()
        self.robot.site_quat[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle3")] = self.sys.obstacle_orientation.copy()
        self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle3")] = self.sys.obstacle_position.copy()
        self.head_camera.get_img(self.data, rgb=True, depth=True)
        self.head_camera.get_target(depth = False)
        self.head_camera.show(rgb=True, depth=True)
        self.sys.ctrlpos[0:2] = self.head_camera.track(self.sys.ctrlpos[0:2])

        if self.sys.hand2target <= 0.05:
            self.sys.reaching_dis *= 0.98
        else:
            self.sys.reaching_dis *= 1.05
            if self.sys.reaching_dis >= 0.1:
                self.sys.reaching_dis = 0.1

        if self.inf.timestep%int(5*self.sys.Hz) == 0:
            self.spawn_new_point()

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

    def render(self, speed=1):
        if int(1000*self.data.time+1)%int(450*speed+50) == 0: # 50ms render 一次
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "pos_target")] = self.sys.pos_target.copy()
            self.robot.site_quat[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle3")] = self.sys.obstacle_orientation.copy()
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle3")] = self.sys.obstacle_position.copy()
            # mujoco.mj_step(self.robot, self.data)
            self.viewer.sync()
            self.viewer.cam.azimuth += 0.05 

    def check_reachable(self, point):
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        distoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        # distoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if distoshoulder >= 0.45 or distoshoulder <= 0.35:
            return False
        elif (point[0]<0.12 and point[1] > -0.20):
            return False
        else:
            return True
        
    def spawn_new_point(self):
        d1 = (   self.sys.vec_hand2elbow[0]**2 +   self.sys.vec_hand2elbow[1]**2 +   self.sys.vec_hand2elbow[2]**2 ) **0.5
        d2 = ( self.sys.vec_target2elbow[0]**2 + self.sys.vec_target2elbow[1]**2 + self.sys.vec_target2elbow[2]**2 ) **0.5
        hand_camera_center = np.degrees(np.arccos(np.dot(self.sys.vec_hand2elbow, self.sys.vec_target2elbow)/(d1*d2)))
        # print(hand_camera_center)

        if self.inf.timestep == 0 or self.sys.hand2target <= 0.05 or hand_camera_center <= 5:
            self.inf.reward += 10
            reachable = False
            while reachable == False:
                self.sys.pos_target0[0] = random.uniform( 0.15, 0.45)
                self.sys.pos_target0[1] = random.uniform(-0.75, 0.00)
                self.sys.pos_target0[2] = random.uniform( 0.90, 1.40)
                reachable = self.check_reachable(self.sys.pos_target0)
            self.data.qpos[15:18] = self.sys.pos_target0.copy()
            self.sys.obstacle_hand_pos_and_quat = self.random_quaternion_and_pos(name="human_hand")
            self.sys.obstacle_table_pos_and_quat = self.random_quaternion_and_pos(name="table")
            self.data.qpos[22:29] = self.sys.obstacle_hand_pos_and_quat.copy()
            self.data.qpos[29:36] = self.sys.obstacle_table_pos_and_quat.copy()

            self.sys.reaching_dis = 0.1

            self.sys.pos_hand   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
            self.sys.pos_neck   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
            self.sys.pos_target = self.sys.pos_hand.copy()
            self.sys.vec_target2neck = [self.data.qpos[15]-self.sys.pos_neck[0], self.data.qpos[16]-self.sys.pos_neck[1], self.data.qpos[17]-self.sys.pos_neck[2]]
            self.sys.vec_target2hand = [self.data.qpos[15]-self.sys.pos_hand[0], self.data.qpos[16]-self.sys.pos_hand[1], self.data.qpos[17]-self.sys.pos_hand[2]]

            new_dis = (self.sys.vec_target2hand[0]**2 + self.sys.vec_target2hand[1]**2 + self.sys.vec_target2hand[2]**2) **0.5
            self.sys.hand2target  = new_dis
            self.sys.neck_target_pos[0] = np.arctan2(self.sys.vec_target2neck[1], self.sys.vec_target2neck[0])
            self.sys.neck_target_pos[1] = np.arctan2(self.sys.vec_target2neck[2], self.sys.vec_target2neck[0]) - np.deg2rad(10)
            for i in range(100):
                self.sys.ctrlpos[0] = self.sys.pos[0] + np.tanh(10*(self.sys.neck_target_pos[0] - self.sys.pos[0]))*0.02
                self.sys.ctrlpos[1] = self.sys.pos[1] + np.tanh(10*(self.sys.neck_target_pos[1] - self.sys.pos[1]))*0.02
                self.control_and_step(render=True)

            mujoco.mj_step(self.robot, self.data)
        else:
            self.reset()

    def control_and_step(self, render=True):
        # check motor limits
        if   self.sys.ctrlpos[2] > self.sys.limit_high[0]: self.sys.ctrlpos[2] = self.sys.limit_high[0]
        elif self.sys.ctrlpos[2] < self.sys.limit_low[0] : self.sys.ctrlpos[2] = self.sys.limit_low[0]
        if   self.sys.ctrlpos[3] > self.sys.limit_high[1]: self.sys.ctrlpos[3] = self.sys.limit_high[1]
        elif self.sys.ctrlpos[3] < self.sys.limit_low[1] : self.sys.ctrlpos[3] = self.sys.limit_low[1]
        if   self.sys.ctrlpos[5] > self.sys.limit_high[2]: self.sys.ctrlpos[5] = self.sys.limit_high[2]
        elif self.sys.ctrlpos[5] < self.sys.limit_low[2] : self.sys.ctrlpos[5] = self.sys.limit_low[2]
        if   self.sys.ctrlpos[6] > self.sys.limit_high[3]: self.sys.ctrlpos[6] = self.sys.limit_high[3]
        elif self.sys.ctrlpos[6] < self.sys.limit_low[3] : self.sys.ctrlpos[6] = self.sys.limit_low[3]

        # PID control
        self.sys.pos = [self.data.qpos[i] for i in controlList]
        self.sys.vel = [self.data.qvel[i-1] for i in controlList]
        self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)

        # fix obstacle      
        self.data.qpos[22:29] = self.sys.obstacle_hand_pos_and_quat.copy()
        self.data.qpos[29:36] = self.sys.obstacle_table_pos_and_quat.copy()

        # step & render
        mujoco.mj_step(self.robot, self.data)
        if render == True:
            self.render()

    def random_quaternion_and_pos(self, name):

        quat = np.array([1.0, 0, 0, 0])
        pos = self.data.qpos[15:18].copy()
        if name == "human_hand":
            euler = np.array([np.pi*random.uniform( 0.0, 1.5), 0, np.pi*random.uniform( -0.25, 1.25)])
            mujoco.mju_euler2Quat(quat, euler, "zyx")

        if name == "table":
            pos[0] += random.uniform( 0.20, 0.70)
            pos[2] -= random.uniform( 0.20, 0.50)
    
        return [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]