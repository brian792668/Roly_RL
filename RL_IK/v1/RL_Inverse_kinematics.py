import mujoco
import mujoco.viewer
import cv2
import gymnasium as gym
import numpy as np
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports.Camera import *
from imports.state_action import *
from imports.RL_info import *
from imports.Forward import *

class RL_IK(gym.Env):
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
        self.inf = RL_inf()
        self.sys = RL_sys(Hz=50)
        self.obs = RL_obs()

        tableR = [ [    0.0, np.pi/2,     0.0,     0.0],
                   [np.pi/2, np.pi/2,     0.0,  0.2488],
                   [    0.0,     0.0, -0.1705,     0.0], 
                   [np.pi/2, np.pi/2,     0.0,     0.0],
                   [np.pi/2, np.pi/2,     0.0, -0.2003],
                   [    0.0, np.pi/2,     0.0,     0.0],
                   [    0.0, np.pi/2,     0.0,  0.1700]] # 0.22
        self.DH_R = DHtable(tableR)

        # self.head_camera = Camera(renderer=self.renderer, camID=0)
        # self.hand_camera = Camera(renderer=self.renderer, camID=2)

        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200

        self.render_speed = 0.0
        
    def step(self, action): 
        # self.inf.truncated = False
        if self.viewer.is_running() == False:
            self.close()
        else:
            self.inf.timestep += 1
            self.inf.totaltimestep += 1
            self.get_reward(action)

            for i in range(int(1/self.sys.Hz/0.005)):
                self.sys.ctrlpos[2] = self.sys.ctrlpos[2] + np.tanh(1.2*self.sys.arm_target_pos[0] - 1.2*self.sys.pos[2])*0.01
                self.sys.ctrlpos[3] = self.sys.ctrlpos[3] + np.tanh(1.2*self.sys.arm_target_pos[1] - 1.2*self.sys.pos[3])*0.01
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = self.sys.ctrlpos[5] + np.tanh(1.2*self.sys.arm_target_pos[3] - 1.2*self.sys.pos[5])*0.01
                self.sys.ctrlpos[6] = self.sys.ctrlpos[6] + np.tanh(1.2*self.sys.arm_target_pos[4] - 1.2*self.sys.pos[6])*0.01
                if   self.sys.ctrlpos[2] > self.sys.limit_high[0]: self.sys.ctrlpos[2] = self.sys.limit_high[0]
                elif self.sys.ctrlpos[2] < self.sys.limit_low[0] : self.sys.ctrlpos[2] = self.sys.limit_low[0]
                if   self.sys.ctrlpos[3] > self.sys.limit_high[1]: self.sys.ctrlpos[3] = self.sys.limit_high[1]
                elif self.sys.ctrlpos[3] < self.sys.limit_low[1] : self.sys.ctrlpos[3] = self.sys.limit_low[1]
                if   self.sys.ctrlpos[5] > self.sys.limit_high[2]: self.sys.ctrlpos[5] = self.sys.limit_high[2]
                elif self.sys.ctrlpos[5] < self.sys.limit_low[2] : self.sys.ctrlpos[5] = self.sys.limit_low[2]
                if   self.sys.ctrlpos[6] > self.sys.limit_high[3]: self.sys.ctrlpos[6] = self.sys.limit_high[3]
                elif self.sys.ctrlpos[6] < self.sys.limit_low[3] : self.sys.ctrlpos[6] = self.sys.limit_low[3]

                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                
                mujoco.mj_step(self.robot, self.data)
            self.render()
            self.get_state()
            self.observation_space = np.concatenate([self.sys.EE_to_origin.copy(),
                                                     [self.obs.joint_arm[2]],
                                                     [self.obs.hand_length]]).astype(np.float32)
            self.inf.truncated = False
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
            self.sys.pos_hand     = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")]
            self.sys.pos_origin   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"origin_marker")]
            self.sys.arm_target_pos = [ 0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0 ]
            for i in range(100):
                self.sys.ctrlpos[2] = 0
                self.sys.ctrlpos[3] = 0
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = 0
                self.sys.ctrlpos[6] = 0
                self.sys.ctrlpos[5] = 0
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)

            self.get_state()
            self.observation_space = np.concatenate([self.sys.EE_to_origin.copy(),
                                                     [self.obs.joint_arm[2]],
                                                     [self.obs.hand_length]]).astype(np.float32)
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self, action):
        r1 = np.exp(-5*abs(action[0]-self.obs.joint_arm[0]))
        r2 = np.exp(-5*abs(action[1]-self.obs.joint_arm[1]))
        r3 = np.exp(-5*abs(action[2]-self.obs.joint_arm[3]))

        self.inf.reward = (r1 + r2 + r3)/3
        self.inf.total_reward += self.inf.reward
 
    def get_state(self):

        # joints
        self.obs.joint_arm[0:2] = self.data.qpos[9:11]
        self.obs.joint_arm[2:4] = self.data.qpos[12:14]
        # for i in range(4):
        #     self.obs.joint_arm[i] += random.uniform(-0.015, 0.015) # noise


        # position of hand, neck, elbow
        self.robot.site_pos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")][2] = 0.17 + self.obs.hand_length
        mujoco.mj_forward(self.robot, self.data)
        self.sys.EE_to_origin = [self.sys.pos_hand[0] - self.sys.pos_origin[0],
                                 self.sys.pos_hand[1] - self.sys.pos_origin[1],
                                 self.sys.pos_hand[2] - self.sys.pos_origin[2]]

        if self.inf.timestep%int(5*self.sys.Hz) == 0:
            self.spawn_new_point()

    def close(self):
        self.renderer.close() 
        self.viewer.close()
        cv2.destroyAllWindows() 

    def render(self):
        if self.inf.timestep%int(48*self.render_speed+2) ==0:
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"end_effector")] = self.sys.pos_EE_predict.copy()
            self.viewer.sync()
            # self.viewer.cam.azimuth += 0.05 
        
    def spawn_new_point(self):
            
        self.obs.hand_length = random.uniform(0.0, 0.15)
        self.robot.site_pos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")][2] = 0.17 + self.obs.hand_length
        mujoco.mj_forward(self.robot, self.data)

        self.sys.arm_target_pos[0] = np.radians(random.uniform( -90, 90))
        self.sys.arm_target_pos[1] = np.radians(random.uniform( -90, 90))
        self.sys.arm_target_pos[3] = np.radians(random.uniform( -90, 90))
        self.sys.arm_target_pos[4] = np.radians(random.uniform(   0, 120))
