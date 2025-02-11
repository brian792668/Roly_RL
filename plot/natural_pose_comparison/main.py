import mujoco
import mujoco.viewer
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports.model1 import RLmodel
from imports.Settings import *
from imports.Controller import *
from imports.RL_info import *

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Roly():
    def __init__(self):
        self.robot = mujoco.MjModel.from_xml_path('Roly/Roly_XML2-2/Roly.xml')
        self.data = mujoco.MjData(self.robot)
        self.renderer = mujoco.Renderer(self.robot)
        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200
        self.render_speed = 0.5
        self.inf = RL_inf()
        self.sys = RL_sys(Hz=50)
        self.obs = RL_obs()
        self.model1 = RLmodel()
        self.IK = IKMLP()
        self.IK.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "imports/IKmodel_v9.pth"), weights_only=True))
        self.IK.eval()
        self.is_running = True
        self.reset()

        self.torque = 0
        self.plt_degree = np.array([])
        self.plt_torque = np.array([])
        self.plt_degree_nature = np.array([])
        self.plt_torque_nature = np.array([])
        self.file_path = os.path.dirname(os.path.abspath(__file__))
    
    def step(self): 
        if self.viewer.is_running() == False:
            self.close()
            self.is_running = False
        else:
            self.inf.timestep += 1
            self.get_state()
            action_from_model1 = self.model1.predict()
            self.sys.joints_increment[0] = self.sys.joints_increment[0]*0.9 + action_from_model1[0]*0.1
            self.sys.joints_increment[1] = self.sys.joints_increment[1]*0.9 + action_from_model1[1]*0.1
            self.sys.joints_increment[2] = 0
            self.sys.joints_increment[3] = self.sys.joints_increment[3]*0.9 + action_from_model1[2]*0.1
            self.sys.joints_increment[4] = 0
            alpha = 1-0.8*np.exp(-300*self.sys.hand2guide**2)
            for i in range(int(1.0/self.sys.Hz/0.005)):
                self.sys.ctrlpos[2] = self.sys.ctrlpos[2] + self.sys.joints_increment[0]*0.01*alpha
                self.sys.ctrlpos[3] = self.sys.ctrlpos[3] + self.sys.joints_increment[1]*0.01*alpha
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[6] = self.sys.ctrlpos[6] + self.sys.joints_increment[3]*0.01*alpha
                self.sys.ctrlpos[7] = 0
                if (self.inf.timestep%(33*Robot.sys.Hz)) <= (3*Robot.sys.Hz):
                    self.sys.ctrlpos[5] += np.tanh(-np.pi/2-self.sys.ctrlpos[5])*0.01
                else:
                    self.sys.ctrlpos[5] += np.pi/(30*Robot.sys.Hz*4)
                self.control_and_step()
                if self.torque <= 10 and (self.inf.timestep%(33*Robot.sys.Hz)) > (3*Robot.sys.Hz):
                    self.plt_degree = np.append(self.plt_degree, np.degrees(self.sys.ctrlpos[5]))
                    self.plt_torque = np.append(self.plt_torque, self.torque)

            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, "pos_target")] = self.sys.pos_guide.copy()
            if self.inf.timestep%int(48*self.render_speed+2) ==0:
                self.viewer.sync()

    def get_state(self):
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
        self.torque = abs(self.data.ctrl[2]) + abs(self.data.ctrl[3]) + abs(self.data.ctrl[5]) + abs(self.data.ctrl[6])
        self.torque = abs(self.data.ctrl[2]) + abs(self.data.ctrl[3])
        # self.torque = abs(self.data.ctrl[5])

        # step & render
        mujoco.mj_step(self.robot, self.data)

    def reset(self, seed=None, **kwargs): 
        if self.viewer.is_running() == False:
            self.close()
        else:
            mujoco.mj_resetData(self.robot, self.data)
            self.inf.reset()
            self.sys.reset()
            self.obs.reset()
            self.inf.timestep = 1
            # self.head_camera.track_done = False

            self.control_and_step()
            self.get_state()

    def close(self):
        self.renderer.close()

    def spawn_new_point(self):
        reachable = False
        while reachable == False:
            self.sys.pos_target[0] = random.uniform(-0.05, 0.50)
            self.sys.pos_target[1] = random.uniform(-0.75, 0.00)
            self.sys.pos_target[2] = random.uniform( 0.90, 1.40)
            reachable = self.check_reachable(self.sys.pos_target.copy())
        self.data.qpos[15:18] = self.sys.pos_target.copy()
        self.sys.pos_guide = self.sys.pos_target.copy()
        self.sys.ctrlpos[5] = 0
        self.spawn_phase = 0
        self.get_state()
        mujoco.mj_step(self.robot, self.data)         

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
        
if __name__ == '__main__':
    Robot = Roly()
    Robot.spawn_new_point()
    while Robot.viewer.is_running() == True:
        if Robot.inf.timestep%int(33*Robot.sys.Hz) == 0:
            Robot.spawn_new_point()
            fig = plt.figure(figsize=(15, 10))
            plt.title("Torque vs Elbow_yaw")
            plt.xlabel("Elbow yaw (degree)")
            plt.ylabel("Arm total torque (Nm)")
            plt.plot(Robot.plt_degree, Robot.plt_torque, color='black', alpha=0.1)
            Robot.plt_torque = gaussian_filter1d(Robot.plt_torque, sigma=50.0)
            plt.plot(Robot.plt_degree, Robot.plt_torque, color='black', alpha=1.0)

            with torch.no_grad():  # 不需要梯度計算，因為只做推論
                desire_joints = Robot.IK(torch.tensor(Robot.sys.vec_guide2neck, dtype=torch.float32)).tolist()
            plt.axvline(x=desire_joints[2], color='red', linestyle='--', linewidth=2)
            plt.savefig(os.path.join(Robot.file_path, "Torque_vs_Elbow_yaw.png"))
            plt.close()
            Robot.torque = 0
            Robot.plt_degree = np.array([])
            Robot.plt_torque = np.array([])
        Robot.step()