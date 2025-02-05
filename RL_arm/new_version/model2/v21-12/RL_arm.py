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
            self.compensate()
            self.get_state()
            self.inf.timestep += 1
            self.inf.totaltimestep += 1
            self.inf.reward = 0
            self.inf.truncated = False
            self.inf.info = {}
            if self.inf.timestep > 2000 :
                self.inf.truncated = True
            if self.check_collision() == True:
                self.inf.reward -= 10
                self.inf.truncated = True

            self.inf.action[0] = self.inf.action[0]*0.95 + action[0]*0.05
            self.inf.action[1] = self.inf.action[1]*0.95 + action[1]*0.05
            self.inf.action[2] = self.inf.action[2]*0.9 + action[2]*0.1
   
            action_from_model1 = self.model1.predict()
            self.sys.joints_increment[0] = self.sys.joints_increment[0]*0.9 + action_from_model1[0]*0.1
            self.sys.joints_increment[1] = self.sys.joints_increment[1]*0.9 + action_from_model1[1]*0.1
            self.sys.joints_increment[2] = self.inf.action[2]
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
            self.observation_space = np.concatenate([self.sys.vec_target2neck, 
                                                     self.sys.vec_target2guide,
                                                     [self.obs.joint_arm[2]],
                                                     [self.inf.action[2]]]).astype(np.float32)
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
            self.observation_space = np.concatenate([self.sys.vec_target2neck, 
                                                     self.sys.vec_target2guide,
                                                     [self.obs.joint_arm[2]],
                                                     [self.inf.action[2]]]).astype(np.float32)
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self):
        # r1: nature pos
        with torch.no_grad():  # 不需要梯度計算，因為只做推論
            desire_joints = self.IK(torch.tensor(self.sys.vec_guide2neck, dtype=torch.float32)).tolist()
            desire_joints[2] += 10
            desire_joints = np.radians(desire_joints)
        r1 = desire_joints[2] - self.sys.pos[5]
        r1 = np.exp(-3*r1**2)

        # r2: grasping distance
        r2 = (0.1-self.sys.grasping_dis) / 0.1

        # r3: hand central
        v1 = ( self.sys.vec_hand2elbow[0] ** 2 + self.sys.vec_hand2elbow[1] ** 2 + self.sys.vec_hand2elbow[2] ** 2 ) ** 0.5
        v2 = ( self.sys.vec_target2elbow[0]**2 + self.sys.vec_target2elbow[1]**2 + self.sys.vec_target2elbow[2]**2 ) ** 0.5
        cosine = np.dot(self.sys.vec_hand2elbow, self.sys.vec_target2elbow)/(v1*v2)
        theta = np.arccos(cosine)
        if np.degrees(theta) <= 10 and v1 < v2:
            self.sys.grasping_dis*= 0.95
        else: 
            self.sys.grasping_dis = 0.1
        r3 = np.exp(-3*theta**2)

        if self.inf.timestep%int(3*self.sys.Hz) == 0 and np.degrees(theta) >= 15:
            self.inf.truncated = True

        # reachable
        reachable = self.check_reachable(self.sys.pos_guide.copy())
        if reachable == False:
            self.inf.reward -= 0.5
        
        self.inf.reward += 1.0*r1*r2*reachable + 0.3*r3
        self.inf.total_reward += self.inf.reward
        
        # # 獎勵對應部分的比例長度
        # r1r2_length = int((1.0*r2) * 20)  # r2 部分長度（最大值 0.5 對應 10）
        # r3_length = int((0.3*r3) * 20)   # r3 部分長度（最大值 0.5 對應 3）
        # r2_bar = "." * r1r2_length + " " * (20 - r1r2_length)
        # r3_bar = "." * r3_length + " " * (6 - r3_length)

        # status_bar = f"| {r2_bar}{r3_bar}|"
        # print(f"\r{status_bar}  {self.inf.reward:.2f}", end=" ")

        return self.inf.reward
 
    def get_state(self):
        if self.inf.timestep%int(3*self.sys.Hz) == 0:
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
        self.model1.obs_guide_to_neck_to_neck = self.sys.vec_guide2neck.copy()
        self.model1.obs_guide_to_hand_norm = self.sys.vec_guide2hand.copy()
        if self.sys.hand2guide > 0.02:
            self.model1.obs_guide_to_hand_norm[0] *= 0.02/self.sys.hand2guide
            self.model1.obs_guide_to_hand_norm[1] *= 0.02/self.sys.hand2guide
            self.model1.obs_guide_to_hand_norm[2] *= 0.02/self.sys.hand2guide
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

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

    def check_reachable(self, point):
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        distoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        # distoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if distoshoulder >= 0.45 or distoshoulder <= 0.30:
            return False
        elif (point[0]< 0 or point[1]> 0 or point[2]>shoulder_pos[2]) :
            return False
        elif (point[0]<0.12 and point[1] > -0.20):
            return False
        else:
            return True
        
    def spawn_new_point(self):
        if self.inf.timestep == 0 or self.sys.hand2guide <= 0.05:
            reachable = False
            while reachable == False:
                self.sys.pos_target[0] = random.uniform(-0.05, 0.50)
                self.sys.pos_target[1] = random.uniform(-0.75, 0.00)
                self.sys.pos_target[2] = random.uniform( 0.90, 1.40)
                reachable = self.check_reachable(self.sys.pos_target.copy())
            self.data.qpos[15:18] = self.sys.pos_target.copy()

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
            
    def random_quaternion_and_pos(self, name):

        quat = np.array([1.0, 0, 0, 0])
        pos = self.data.qpos[15:18].copy()
        if name == "human_hand":
            euler = np.array([np.radians(random.uniform( 0.0, 270)), 0, np.radians(random.uniform( -45, 225))])
            mujoco.mju_euler2Quat(quat, euler, "zyx")

        if name == "table":
            euler = np.array([np.radians(random.uniform( -30, 30)), 0, 0])
            mujoco.mju_euler2Quat(quat, euler, "zyx")
            pos[0] = random.uniform( 0.35, 1.00) - 0.5
            pos[2] -= random.uniform( 0.20, 0.50)

        if name == "human":
            euler = np.array([np.radians(random.uniform( -90, 90)), 0, 0])
            mujoco.mju_euler2Quat(quat, euler, "zyx")
            pos[0] += random.uniform( 0.25, 0.70) -0.5
            pos[1] += random.uniform( -0.30, 0.30)
            pos[2] = random.uniform( -0.2, 0.75)
    
        return [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
    
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
        if self.check_reachable(point=new_guide) == True:
            self.sys.pos_guide = new_guide.copy()
            self.sys.vec_guide2hand  = [self.sys.pos_guide[0] - self.sys.pos_hand[0] , self.sys.pos_guide[1] - self.sys.pos_hand[1] , self.sys.pos_guide[2] - self.sys.pos_hand[2]]
