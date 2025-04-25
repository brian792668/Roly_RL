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
import time

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
        self.robot = mujoco.MjModel.from_xml_path('Roly/Roly_XML/Roly.xml')
        self.data = mujoco.MjData(self.robot)
        self.action_space = gym.spaces.box.Box( low  = act_low,      # action (rad)
                                                high = act_high,
                                                dtype = np.float32)
        self.observation_space = gym.spaces.box.Box(low  = obs_low,
                                                    high = obs_high,
                                                    dtype = np.float32 )
        
        self.renderer = mujoco.Renderer(self.robot)
        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200
        self.render_speed = 0
        self.inf = RL_inf()
        self.sys = RL_sys(Hz=50)
        self.obs = RL_obs()
        self.head_camera = Camera(renderer=self.renderer, camID=0)
        # self.hand_camera = Camera(renderer=self.renderer, camID=2)
        self.obstacle_hand_ID = mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_hand")
        self.obstacle_table_ID = mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_table")

        self.model1 = RLmodel()
        self.IK = IKMLP()
        self.IK.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "IKmodel_v9.pth"), weights_only=True))
        self.IK.eval()
    
    def render(self):
        if self.inf.timestep%int(48*self.render_speed+2) ==0:
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "predicted_grasp_point")] = self.sys.pos_grasp_point.copy()
            # self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "pos_target")] = self.sys.pos_guide.copy()
            # self.robot.site_quat[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle_hand")] = self.sys.obstacle_hand_pos_and_quat[3:7].copy()
            # self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle_hand")] = self.sys.obstacle_hand_pos_and_quat[0:3].copy()
            # mujoco.mj_step(self.robot, self.data)
            self.viewer.sync()

    def step(self, action): 
        if self.viewer.is_running() == False:
            self.close()
        else:
            self.inf.action[0] = action[0]
            self.inf.action[1] = action[1]
            self.inf.action[2] = action[2]
            self.get_reward()

            self.inf.timestep += 1
            self.inf.totaltimestep += 1
            self.inf.truncated = False
            self.inf.info = {}
            if self.inf.timestep > 2048 :
                self.inf.truncated = True

            
            action_from_model1 = self.model1.predict()
            self.sys.joints_increment[0] = self.sys.joints_increment[0]*0.9 + action_from_model1[0]*0.1
            self.sys.joints_increment[1] = self.sys.joints_increment[1]*0.9 + action_from_model1[1]*0.1
            self.sys.joints_increment[2] = np.tanh(self.sys.guide_arm_joints[3] - self.sys.pos[5])
            self.sys.joints_increment[3] = self.sys.joints_increment[3]*0.9 + action_from_model1[2]*0.1
            self.sys.joints_increment[4] = 0
            alpha = 1-0.8*np.exp(-300*self.sys.hand2guide**2)
            for i in range(int(1.0/self.sys.Hz/0.005)):
                self.sys.ctrlpos[2] = self.sys.ctrlpos[2] + self.sys.joints_increment[0]*0.01*alpha
                self.sys.ctrlpos[3] = self.sys.ctrlpos[3] + self.sys.joints_increment[1]*0.01*alpha
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = self.sys.ctrlpos[5] + self.sys.joints_increment[2]*0.01
                self.sys.ctrlpos[6] = self.sys.ctrlpos[6] + self.sys.joints_increment[3]*0.01*alpha
                self.sys.ctrlpos[7] = self.sys.ctrlpos[7] + self.sys.joints_increment[4]*0.01
                self.control_and_step()
            self.render()

            self.get_state()
            self.observation_space = np.concatenate([self.sys.vec_temp_to_neck, 
                                                     [self.sys.pos[5]]]).astype(np.float32)
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

            self.control_and_step()
            self.render()
            self.get_state()
            self.observation_space = np.concatenate([self.sys.vec_temp_to_neck, 
                                                     [self.sys.pos[5]]]).astype(np.float32)
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self):
        dis = ((self.sys.pos_grasp_point[0]-self.sys.pos_hand[0])**2 + (self.sys.pos_grasp_point[1]-self.sys.pos_hand[1])**2 + (self.sys.pos_grasp_point[2]-self.sys.pos_hand[2])**2 )**0.5
        self.inf.reward = np.exp(-35*dis)
        self.inf.reward = (1-dis/0.10)
        self.inf.total_reward += self.inf.reward
        # print(f"dis: {dis:3f},  reward: {self.inf.reward:.2f}")
 
    def get_state(self):
        if self.inf.timestep%int(1.5*self.sys.Hz) == 0:
            self.spawn_new_point()

        # position of hand, neck, elbow
        self.sys.pos_hand   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        self.sys.pos_neck   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"origin_marker")].copy()
        self.sys.pos_elbow  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_elbow_marker")].copy()
        # self.sys.pos_guide = [self.sys.pos_hand[0]*0.5 + self.sys.pos_target[0]*0.5,  self.sys.pos_hand[1]*0.5 + self.sys.pos_target[1]*0.5,  self.sys.pos_hand[2]*0.5 + self.sys.pos_target[2]*0.5]

        # vectors
        self.sys.vec_guide2neck   = [self.sys.pos_guide[0] - self.sys.pos_neck[0] ,   self.sys.pos_guide[1] - self.sys.pos_neck[1] ,  self.sys.pos_guide[2] - self.sys.pos_neck[2]]
        self.sys.vec_guide2hand   = [self.sys.pos_guide[0] - self.sys.pos_hand[0] ,   self.sys.pos_guide[1] - self.sys.pos_hand[1] ,  self.sys.pos_guide[2] - self.sys.pos_hand[2]]
        self.sys.vec_guide2elbow  = [self.sys.pos_guide[0] - self.sys.pos_elbow[0],   self.sys.pos_guide[1] - self.sys.pos_elbow[1],  self.sys.pos_guide[2] - self.sys.pos_elbow[2]]
        self.sys.vec_target2hand  = [self.sys.pos_target[0]- self.sys.pos_hand[0] ,   self.sys.pos_target[1]- self.sys.pos_hand[1] ,  self.sys.pos_target[2]- self.sys.pos_hand[2]]
        self.sys.vec_target2neck  = [self.sys.pos_target[0]- self.sys.pos_neck[0] ,   self.sys.pos_target[1]- self.sys.pos_neck[1] ,  self.sys.pos_target[2]- self.sys.pos_neck[2]]
        self.sys.vec_target2elbow = [self.sys.pos_target[0]- self.sys.pos_elbow[0],   self.sys.pos_target[1]- self.sys.pos_elbow[1],  self.sys.pos_target[2]- self.sys.pos_elbow[2]]
        self.sys.vec_target2guide = [self.sys.pos_target[0]- self.sys.pos_guide[0],  self.sys.pos_target[1]- self.sys.pos_guide[1], self.sys.pos_target[2]- self.sys.pos_guide[2]]
        self.sys.vec_hand2neck    = [self.sys.pos_hand[0]   - self.sys.pos_neck[0] ,   self.sys.pos_hand[1]   - self.sys.pos_neck[1],   self.sys.pos_hand[2]   - self.sys.pos_neck[2]]
        self.sys.vec_hand2elbow   = [self.sys.pos_hand[0]   - self.sys.pos_elbow[0],   self.sys.pos_hand[1]   - self.sys.pos_elbow[1],  self.sys.pos_hand[2]   - self.sys.pos_elbow[2]]


        # distance
        self.sys.hand2guide  = ( self.sys.vec_guide2hand[0]**2  + self.sys.vec_guide2hand[1]**2  + self.sys.vec_guide2hand[2]**2 )  ** 0.5
        self.sys.hand2target = ( self.sys.vec_target2hand[0]**2 + self.sys.vec_target2hand[1]**2 + self.sys.vec_target2hand[2]**2 ) ** 0.5
        
        # model1
        self.model1.obs_guide_to_neck = self.sys.vec_guide2neck.copy()
        self.model1.obs_guide_to_hand_norm = self.sys.vec_guide2hand.copy()
        self.model1.obs_guide_arm_joint = self.sys.guide_arm_joints[3]
        if self.sys.hand2guide > 0.05:
            self.model1.obs_guide_to_hand_norm[0] *= 0.05/self.sys.hand2guide
            self.model1.obs_guide_to_hand_norm[1] *= 0.05/self.sys.hand2guide
            self.model1.obs_guide_to_hand_norm[2] *= 0.05/self.sys.hand2guide
        self.model1.obs_joints[0:2] = self.data.qpos[9:11].copy()
        self.model1.obs_joints[2:4] = self.data.qpos[12:14].copy()
        self.model1.action[0] = self.sys.joints_increment[0]
        self.model1.action[1] = self.sys.joints_increment[1]
        self.model1.action[2] = self.sys.joints_increment[3]

        # model2
        self.obs.joint_arm[0:2] = self.data.qpos[9:11].copy()
        self.obs.joint_arm[2:4] = self.data.qpos[12:14].copy()

        # ----------------------------------------------------------------------------------
        # update camera

        self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "pos_target")] = self.sys.pos_guide.copy()
        self.robot.site_quat[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle_hand")] = self.sys.obstacle_hand_pos_and_quat[3:7].copy()
        self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle_hand")] = self.sys.obstacle_hand_pos_and_quat[0:3].copy()
        # self.head_camera.get_img(self.data, rgb=True, depth=True)
        # self.head_camera.get_guide(depth = False)
        # self.head_camera.depth_feature()
        # self.obs.feature_points = self.obs.feature_points*0.9 + self.head_camera.feature_points*0.1
        # self.head_camera.show(rgb=False, depth=True)
        # self.sys.ctrlpos[0:2] = self.head_camera.track(self.sys.ctrlpos[0:2])


        self.sys.pos_neck   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"origin_marker")]
        self.sys.pos_temp_target = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"temp_target")]
        self.sys.vec_temp_to_neck = [self.sys.pos_temp_target[0]-self.sys.pos_neck[0],  self.sys.pos_temp_target[1]-self.sys.pos_neck[1],  self.sys.pos_temp_target[2]-self.sys.pos_neck[2]]
        self.sys.pos_grasp_point[0] = self.sys.pos_temp_target[0] + 0.08*self.inf.action[0]
        self.sys.pos_grasp_point[1] = self.sys.pos_temp_target[1] + 0.08*self.inf.action[1]
        self.sys.pos_grasp_point[2] = self.sys.pos_temp_target[2] + 0.08*self.inf.action[2]
        # print(self.sys.pos_temp_target, self.sys.pos_hand)

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

    def check_reachable(self, point):
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        distoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        # distoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if distoshoulder >= 0.57 or distoshoulder <= 0.39:
            return False
        # elif point[2]>shoulder_pos[2]:
        #     return False
        # elif (point[0]<0.12 and point[1] > -0.20):
        #     return False
        else:
            return True
        
    def spawn_new_point(self):
        self.sys.guide_arm_joints[3] = np.radians(random.uniform( -90, 90))
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        reachable = False
        while reachable == False:
            self.sys.pos_target[0] = shoulder_pos[0] + random.uniform(-0.10, 0.65)
            self.sys.pos_target[1] = shoulder_pos[1] + random.uniform(-0.65, 0.65)
            self.sys.pos_target[2] = shoulder_pos[2] + random.uniform(-0.65, 0.10)
            reachable = self.check_reachable(self.sys.pos_target.copy())
        self.data.qpos[15:18] = self.sys.pos_target.copy()
        self.sys.pos_guide = self.sys.pos_target.copy()

        self.sys.pos_neck = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"origin_marker")].copy()
        self.sys.vec_target2neck = [self.sys.pos_target[0]-self.sys.pos_neck[0], self.sys.pos_target[1]-self.sys.pos_neck[1], self.sys.pos_target[2]-self.sys.pos_neck[2]]
        self.sys.vec_target2hand = [self.sys.pos_target[0]-self.sys.pos_hand[0], self.sys.pos_target[1]-self.sys.pos_hand[1], self.sys.pos_target[2]-self.sys.pos_hand[2]]
        self.sys.hand2target = ( self.sys.vec_target2hand[0]**2 + self.sys.vec_target2hand[1]**2 + self.sys.vec_target2hand[2]**2 ) ** 0.5

        mujoco.mj_step(self.robot, self.data)

    def control_and_step(self):
    
        # check motor limits
        if   self.sys.ctrlpos[2] > self.sys.limit_high[0]: self.sys.ctrlpos[2] = self.sys.limit_high[0]
        elif self.sys.ctrlpos[2] < self.sys.limit_low[0] : self.sys.ctrlpos[2] = self.sys.limit_low[0]
        if   self.sys.ctrlpos[3] > self.sys.limit_high[1]: self.sys.ctrlpos[3] = self.sys.limit_high[1]
        elif self.sys.ctrlpos[3] < self.sys.limit_low[1] : self.sys.ctrlpos[3] = self.sys.limit_low[1]
        if   self.sys.ctrlpos[5] > self.sys.limit_high[2]: self.sys.ctrlpos[5] = self.sys.limit_high[2]
        elif self.sys.ctrlpos[5] < self.sys.limit_low[2] : self.sys.ctrlpos[5] = self.sys.limit_low[2]
        if   self.sys.ctrlpos[6] > self.sys.limit_high[3]: self.sys.ctrlpos[6] = self.sys.limit_high[3]
        elif self.sys.ctrlpos[6] < self.sys.limit_low[3] : self.sys.ctrlpos[6] = self.sys.limit_low[3]
        if   self.sys.ctrlpos[7] > self.sys.limit_high[4]: self.sys.ctrlpos[7] = self.sys.limit_high[4]
        elif self.sys.ctrlpos[7] < self.sys.limit_low[4] : self.sys.ctrlpos[7] = self.sys.limit_low[4]

        # PID control
        self.sys.pos = [self.data.qpos[i] for i in controlList]
        self.sys.vel = [self.data.qvel[i-1] for i in controlList]
        self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)

        # step & render
        mujoco.mj_step(self.robot, self.data)
    
    def check_collision(self):
        id1 = mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_GEOM, f"R_shoulder")
        id2 = mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_GEOM, f"R finger3-2")
        for i, con in enumerate(self.data.contact):
            if id1 <= con.geom1 <= id2 or id1 <= con.geom2 <= id2:
                # print(mujoco.mj_id2name(self.robot, mujoco.mjtObj.mjOBJ_GEOM, con.geom1), mujoco.mj_id2name(self.robot, mujoco.mjtObj.mjOBJ_GEOM, con.geom2))
                return True
        return False
    
    def compensate(self):
        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        new_guide = [0.0, 0.0, 0.0]
        new_guide[0] = self.sys.pos_target[0] - self.sys.grasping_dis*np.cos(np.pi/2*self.inf.action[1])*np.cos(np.pi/2*self.inf.action[0])
        new_guide[1] = self.sys.pos_target[1] - self.sys.grasping_dis*np.cos(np.pi/2*self.inf.action[1])*np.sin(np.pi/2*self.inf.action[0])
        new_guide[2] = self.sys.pos_target[2] + self.sys.grasping_dis*np.sin(np.pi/2*self.inf.action[1])        
        self.sys.pos_guide = new_guide.copy()
        self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]
        # if self.check_reachable(point=new_guide) == True:
        #     self.sys.pos_guide = new_guide.copy()
        #     self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]
        # else:
        #     self.sys.pos_guide = self.sys.pos_hand.copy()
        #     self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]

