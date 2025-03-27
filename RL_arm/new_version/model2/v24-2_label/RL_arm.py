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

class CB(nn.Module):
    def __init__(self): # 3 -> 32 -> 16 -> 2
        super(CB, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x) * 1.6
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

        self.EE_xyz_label = np.array([])
        self.collision_label = np.array([])
        self.CBmodel = CB()
        self.CBmodel.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "collision_bound_2048000points_v1.pth"), weights_only=True))
        self.CBmodel.eval()
    
    def render(self):
        if self.inf.timestep%int(48*self.render_speed+2) ==0:
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "pos_target")] = self.sys.pos_guide.copy()
            self.robot.site_quat[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle_hand")] = self.sys.obstacle_hand_pos_and_quat[3:7].copy()
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "obstacle_hand")] = self.sys.obstacle_hand_pos_and_quat[0:3].copy()
            # mujoco.mj_step(self.robot, self.data)
            self.viewer.sync()

    def step(self, action): 
        if self.viewer.is_running() == False:
            self.close()
        else:
            self.get_state()
            self.compensate()
            self.inf.timestep += 1
            self.inf.totaltimestep += 1
            self.inf.reward = 0
            self.inf.truncated = False
            self.inf.info = {}
            if self.inf.timestep > 2048 :
                self.inf.truncated = True

            self.inf.action[0] = action[0]
            self.inf.action[1] = action[1]
            
            action_from_model1 = self.model1.predict()
            self.sys.joints_increment[0] = self.sys.joints_increment[0]*0.9 + action_from_model1[0]*0.1
            self.sys.joints_increment[1] = self.sys.joints_increment[1]*0.9 + action_from_model1[1]*0.1
            self.sys.joints_increment[2] = np.tanh(self.sys.guide_arm_joints[3]- self.sys.pos[5])
            # if (self.sys.limit_high[2]-self.sys.guide_arm_joints[3])*(self.sys.limit_low[2]-self.sys.guide_arm_joints[3]) <0:
            #     self.sys.joints_increment[2] = np.tanh(self.sys.guide_arm_joints[3]- self.sys.pos[5])
            # elif self.sys.guide_arm_joints[3] < self.sys.limit_low[2]:
            #     self.sys.joints_increment[2] = np.tanh(self.sys.limit_low[2]- self.sys.pos[5])
            # else: 
            #     self.sys.joints_increment[2] = np.tanh(self.sys.limit_high[2]- self.sys.pos[5])
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

            self.inf.reward = self.get_reward()
            self.observation_space = np.array(self.sys.vec_hand2neck.copy(), dtype=np.float32) 
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

            self.control_and_step()
            self.render()
            self.get_state()
            self.observation_space = np.array(self.sys.vec_hand2neck.copy(), dtype=np.float32) 
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self):
        # r1: grasping distance
        pos_elbow = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_elbow_marker")].copy()
        pos_arm1  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker1")].copy()
        pos_arm2  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker2")].copy()
        pos_arm3  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker3")].copy()
        pos_arm4  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker4")].copy()
        pos_arm5  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker5")].copy()
        pos_arm6  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker6")].copy()
        pos_arm7  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker7")].copy()
        pos_arm8  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker8")].copy()
        pos_arm9  = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker9")].copy()
        pos_arm10 = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker10")].copy()
        pos_arm11 = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_arm_marker11")].copy()
        collision = 1.0
        collision *= 1 - ((-np.tanh(500 * (pos_elbow[0] - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_elbow[1] + 0.20)) + 1) / 2) #0.08, 0.17
        collision *= 1 - ((-np.tanh(500 * (pos_arm1[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm1[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm2[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm2[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm3[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm3[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm4[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm4[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm5[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm5[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm6[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm6[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm7[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm7[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm8[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm8[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm9[0]  - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm9[1]  + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm10[0] - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm10[1] + 0.20)) + 1) / 2)
        collision *= 1 - ((-np.tanh(500 * (pos_arm11[0] - 0.10)) + 1) / 2) * ((np.tanh(500 * (pos_arm11[1] + 0.20)) + 1) / 2)

        # from MLP
        input_tensor = torch.tensor(np.array(self.sys.vec_guide2neck.copy(), dtype=np.float32) ).unsqueeze(0)
        with torch.no_grad():
            CBoutput = self.CBmodel(input_tensor)
        low_bound, high_bound = CBoutput[0].numpy()
        if low_bound > high_bound:
            low_bound, high_bound = high_bound, low_bound

        if low_bound > 1.57 or high_bound < -1.57:
            self.sys.limit_high[2] = 0.90*self.sys.limit_high[2] + 0.10*1.57
            self.sys.limit_low[2] = 0.90*self.sys.limit_low[2] + 0.10*-1.57
        else:
            self.sys.limit_high[2] = 0.90*self.sys.limit_high[2] + 0.10*1.57
            self.sys.limit_low[2] = 0.90*self.sys.limit_low[2] + 0.10*high_bound

        future_elbow_yaw = self.sys.pos[5] + 20*self.sys.joints_increment[2]
        if future_elbow_yaw > high_bound:
            self.sys.joints_increment[2] = (high_bound - future_elbow_yaw)/20.0
        elif future_elbow_yaw < low_bound:
            self.sys.joints_increment[2] = (low_bound - future_elbow_yaw)/20.0

        input_tensor = torch.tensor(np.array(self.sys.vec_hand2neck.copy(), dtype=np.float32) ).unsqueeze(0)
        with torch.no_grad():
            CBoutput = self.CBmodel(input_tensor)
        low_bound, high_bound = CBoutput[0].numpy()
        if low_bound > high_bound:
            low_bound, high_bound = high_bound, low_bound


        # boundary function
        x = self.sys.pos[5]
        # boundary_value = (x-self.inf.action[0])*(x-self.inf.action[1])
        boundary_value = (x-low_bound)*(x-high_bound)
        normalized = 1/(1+np.exp(-500*np.clip(boundary_value, -0.02, 0.02)))

        # reward
        error = abs(collision-normalized)
        self.inf.reward = 0.9*np.exp(-20*error) + 0.1*np.exp(-5*error)
        if self.inf.action[0] < self.inf.action[1]:
            self.inf.reward = 0
        self.inf.total_reward += self.inf.reward
        # self.print_scale(self.inf.action[0], self.inf.action[1], self.sys.pos[5], collision, self.inf.reward)
        self.print_scale(low_bound, high_bound, self.sys.pos[5], collision, self.inf.reward)


        # # label
        # if self.inf.totaltimestep%10 == 0:
        #     label = [self.sys.pos[5], collision]
        #     if self.EE_xyz_label.size == 0:
        #         self.EE_xyz_label = np.array([self.sys.vec_hand2neck.copy()])
        #         self.collision_label = np.array([label])
        #     else:
        #         self.EE_xyz_label = np.concatenate((self.EE_xyz_label, [self.sys.vec_hand2neck.copy()]), axis=0)
        #         self.collision_label = np.concatenate((self.collision_label, [label.copy()]), axis=0)

        return self.inf.reward
 
    def get_state(self):
        if self.inf.timestep%int(2*self.sys.Hz) == 0:
            self.spawn_new_point()

        # position of hand, neck, elbow
        self.sys.pos_hand   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        self.sys.pos_neck   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
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
        # # self.head_camera.get_guide(depth = False)
        # self.head_camera.depth_feature()
        # self.obs.feature_points = self.obs.feature_points*0.9 + self.head_camera.feature_points*0.1
        # self.head_camera.show(rgb=False, depth=True)
        # self.sys.ctrlpos[0:2] = self.head_camera.track(self.sys.ctrlpos[0:2])

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

    def check_reachable(self, point):
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        distoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        # distoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if distoshoulder >= 0.5 or distoshoulder <= 0.25:
            return False
        elif point[2]>shoulder_pos[2]:
            return False
        elif (point[0]<0.12 and point[1] > -0.20):
            return False
        else:
            return True
        
    def spawn_new_point(self):
        if self.inf.timestep == 0 or self.sys.hand2guide <= 0.05:
            reachable = False
            while reachable == False:
                self.sys.pos_target[0] = random.uniform( -0.05, 0.50) # -0.05~0.50
                self.sys.pos_target[1] = -0.75*random.uniform(0.0, 1.0)**2      # -0.75~0.0
                self.sys.pos_target[2] = random.uniform( 0.90, 1.40)
                reachable = self.check_reachable(self.sys.pos_target.copy())
            self.data.qpos[15:18] = self.sys.pos_target.copy()
            # self.sys.guide_arm_joints[3] = np.radians(random.uniform( -90, 90))
            self.model1.obs_hand_dis = 0.0
            self.robot.site_pos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")][2] = 0.22 + self.model1.obs_hand_dis
            mujoco.mj_forward(self.robot, self.data)

            self.sys.guide_arm_joints[3] = np.radians(-90 + 180*random.uniform( 0.0, 1.0)**2)

            self.sys.pos_neck = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
            self.sys.vec_target2neck = [self.sys.pos_target[0]-self.sys.pos_neck[0], self.sys.pos_target[1]-self.sys.pos_neck[1], self.sys.pos_target[2]-self.sys.pos_neck[2]]
            self.sys.vec_target2hand = [self.sys.pos_target[0]-self.sys.pos_hand[0], self.sys.pos_target[1]-self.sys.pos_hand[1], self.sys.pos_target[2]-self.sys.pos_hand[2]]
            self.sys.hand2target = ( self.sys.vec_target2hand[0]**2 + self.sys.vec_target2hand[1]**2 + self.sys.vec_target2hand[2]**2 ) ** 0.5

            mujoco.mj_step(self.robot, self.data)         
        else:
            self.reset()

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

        # fix obstacle      
        self.data.qpos[22:29] = self.sys.obstacle_hand_pos_and_quat.copy()
        self.data.qpos[29:36] = self.sys.obstacle_table_pos_and_quat.copy()
        self.data.qpos[36:43] = self.sys.obstacle_human_pos_and_quat.copy()

        # step & render
        mujoco.mj_step(self.robot, self.data)

    def print_scale(self, low, high, elbow, collision, reward):

        low *= 180/np.pi
        high *= 180/np.pi  
        elbow *= 180/np.pi

        total_length = 40  # 總長度
        start = -95
        end = 95

        pos_a = round((low - start) / (end - start) * total_length)
        pos_b = round((high - start) / (end - start) * total_length)
        pos_c = round((elbow - start) / (end - start) * total_length)

        output = ['-'] * total_length
        output[pos_c+1:pos_a] = ['-'] * (pos_a - pos_c - 1)
        output[pos_a] = ' '
        output[pos_a+1:pos_b] = ['='] * (pos_b - pos_a - 1)
        output[pos_b] = ' '
        if collision <= 0.5:
            output[pos_c] = f' \033[93m{elbow:.0f}\033[0m '
        else:
            output[pos_c] = f' {elbow:.0f} '

        print(''.join(output), f"{reward:.1f}", "bound:", f"{low:.3f}", f"{high:.3f}")

    def compensate(self):
        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        # new_guide = [0.0, 0.0, 0.0]
        # new_guide[0] = self.sys.pos_target[0] - self.sys.grasping_dis*np.cos(np.pi/2*self.inf.action[1])*np.cos(np.pi/2*self.inf.action[0])
        # new_guide[1] = self.sys.pos_target[1] - self.sys.grasping_dis*np.cos(np.pi/2*self.inf.action[1])*np.sin(np.pi/2*self.inf.action[0])
        # new_guide[2] = self.sys.pos_target[2] + self.sys.grasping_dis*np.sin(np.pi/2*self.inf.action[1])        
        # self.sys.pos_guide = new_guide.copy()
        self.sys.pos_guide = self.sys.pos_target.copy()
        self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]
        # if self.check_reachable(point=new_guide) == True:
        #     self.sys.pos_guide = new_guide.copy()
        #     self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]
        # else:
        #     self.sys.pos_guide = self.sys.pos_hand.copy()
        #     self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]