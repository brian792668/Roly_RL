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
        self.inf = RL_inf()
        self.sys = RL_sys(Hz=50)
        self.obs = RL_obs()

        tableR = [ [    0.0, np.pi/2,     0.0,     0.0],
                   [np.pi/2, np.pi/2,     0.0,  0.2488],
                   [    0.0,     0.0, -0.1305,     0.0], # -0.1105
                   [np.pi/2, np.pi/2,     0.0,     0.0],
                   [np.pi/2, np.pi/2,     0.0, -0.1195],
                   [    0.0, np.pi/2,     0.0,     0.0],
                   [    0.0, np.pi/2,     0.0,  0.2200]] # 0.1803
        self.DH_R = DHtable(tableR)

        # self.head_camera = Camera(renderer=self.renderer, camID=0)
        # self.hand_camera = Camera(renderer=self.renderer, camID=2)

        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200
        
    def step(self, action): 
        # self.inf.truncated = False
        if self.viewer.is_running() == False:
            self.close()
        else:
            self.inf.timestep += 1
            self.inf.totaltimestep += 1

            for i in range(len(action)):
                self.inf.action[i] = self.inf.action[i]*0.9  + action[i]*0.1
                self.inf.action_new[i] = action[i]

            # alpha1 = 1-0.8*np.exp(-300*self.sys.hand2target**2)

            for i in range(int(1/self.sys.Hz/0.005)):
                self.sys.ctrlpos[2] = self.sys.ctrlpos[2] + self.inf.action[0]*0.01
                self.sys.ctrlpos[3] = self.sys.ctrlpos[3] + self.inf.action[1]*0.01
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = self.sys.ctrlpos[5] + np.tanh(self.sys.arm_target_pos[3] - self.sys.pos[5])*0.01
                self.sys.ctrlpos[6] = self.sys.ctrlpos[6] + self.inf.action[2]*0.01
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
            # self.render()

            self.get_reward()
            self.get_state()
            self.observation_space = np.concatenate([self.obs.obj_to_neck_xyz, 
                                                     self.obs.obj_to_hand_xyz_norm, 
                                                     self.inf.action, 
                                                     self.obs.joint_arm]).astype(np.float32)
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

            self.data.qpos[15:18] = [0.2, -0.25, 1.3] # position of target

            dummy_random = np.radians(random.uniform( 0, 1)**2)*60
            self.sys.arm_target_pos = [ -dummy_random,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2*dummy_random,
                                        0.0 ]
            for i in range(100):
                self.sys.ctrlpos[2] = self.sys.pos[2] + np.tanh(10*(self.sys.arm_target_pos[0] - self.sys.pos[2]))*0.02
                self.sys.ctrlpos[3] = 0
                self.sys.ctrlpos[4] = 0
                self.sys.ctrlpos[5] = 0
                self.sys.ctrlpos[6] = self.sys.pos[6] + np.tanh(10*(self.sys.arm_target_pos[4] - self.sys.pos[6]))*0.02
                self.sys.ctrlpos[5] = 0
                self.sys.pos = [self.data.qpos[i] for i in controlList]
                self.sys.vel = [self.data.qvel[i-1] for i in controlList]
                self.data.ctrl[:] = self.sys.PIDctrl.getSignal(self.sys.pos, self.sys.vel, self.sys.ctrlpos)
                mujoco.mj_step(self.robot, self.data)

            # self.head_camera.get_img(self.data, rgb=True, depth=True)
            self.get_state()
            self.observation_space = np.concatenate([self.obs.obj_to_neck_xyz, 
                                                     self.obs.obj_to_hand_xyz_norm, 
                                                     self.inf.action, 
                                                     self.obs.joint_arm]).astype(np.float32)
            self.inf.done = False
            self.inf.truncated = False
            self.inf.info = {}
            return self.observation_space, self.inf.info

    def get_reward(self):
        # ----------------------------------
        # reward using predicted EE position
        joints_in_5_steps = self.data.qpos[9:15].copy()
        gamma = (1-0.9**7)/0.1
        joints_in_5_steps[0] += (self.inf.action[0]*gamma + self.inf.action_new[0]*(1-gamma) )*0.01*int(1/self.sys.Hz/0.005)
        joints_in_5_steps[1] += (self.inf.action[1]*gamma + self.inf.action_new[1]*(1-gamma) )*0.01*int(1/self.sys.Hz/0.005)
        joints_in_5_steps[4] += (self.inf.action[2]*gamma + self.inf.action_new[2]*(1-gamma) )*0.01*int(1/self.sys.Hz/0.005)

        self.sys.pos_EE_predict = self.DH_R.forward(angles=joints_in_5_steps.copy())
        origin_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"origin_marker")].copy()
        self.sys.pos_EE_predict[0] += origin_pos[0]
        self.sys.pos_EE_predict[1] += origin_pos[1]
        self.sys.pos_EE_predict[2] += origin_pos[2]
        new_dis = (self.data.qpos[15]-self.sys.pos_EE_predict[0])**2 + (self.data.qpos[16]-self.sys.pos_EE_predict[1])**2 + (self.data.qpos[17]-self.sys.pos_EE_predict[2])**2
        new_dis = new_dis ** 0.5

        # r0: reward of position
        r0 = np.exp(-20*new_dis**2)

        # r1: reward of handCAM central
        v1 = ( self.sys.elbow_to_hand[0] ** 2 + self.sys.elbow_to_hand[1] ** 2 + self.sys.elbow_to_hand[2] ** 2 ) ** 0.5
        v2 = ( self.sys.elbow_to_target[0]**2 + self.sys.elbow_to_target[1]**2 + self.sys.elbow_to_target[2]**2 ) ** 0.5
        r1 = np.dot(self.sys.elbow_to_hand, self.sys.elbow_to_target)/(v1*v2)
        r1 *= np.abs(r1)

        # r2: reward of detail control
        r2 = np.exp(-(50*new_dis)**4)

        self.inf.reward = 0.8*r0*r1 + r2
        self.inf.total_reward_future_state += self.inf.reward
        # print(f"reward: {self.inf.reward:.2f}  ({r0*r2:.2f} + {r3:.2f})")

        # ---------------------------------
        # reward using original EE position
        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        self.sys.hand2target = ( (self.data.qpos[15]-self.sys.pos_hand[0])**2 + (self.data.qpos[16]-self.sys.pos_hand[1])**2 + (self.data.qpos[17]-self.sys.pos_hand[2])**2 )**0.5
        r0 = np.exp(-20*self.sys.hand2target**2)
        r2 = np.exp(-(50*self.sys.hand2target)**4)
        self.inf.total_reward_standard += 0.8*r0*r1 + r2
 
    def get_state(self):
        # update position of target
        self.data.qpos[15] += 0.01*np.tanh(2*(self.sys.pos_target0[0] - self.data.qpos[15]))
        self.data.qpos[16] += 0.01*np.tanh(2*(self.sys.pos_target0[1] - self.data.qpos[16]))
        self.data.qpos[17] += 0.01*np.tanh(2*(self.sys.pos_target0[2] - self.data.qpos[17]))

        # update camera
        # self.hand_camera.get_img(self.data, rgb=True, depth=False)
        # self.hand_camera.get_target(depth = False)

        # joints
        self.obs.joint_arm[0:2] = self.data.qpos[9:11]
        self.obs.joint_arm[2:4] = self.data.qpos[12:14]

        # position of hand, neck, elbow
        self.sys.pos_hand = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")].copy()
        neck_xyz = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
        elbow_xyz = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_elbow_marker")].copy()

        # vectors
        self.obs.obj_to_neck_xyz = [self.data.qpos[15]-neck_xyz[0],          self.data.qpos[16]-neck_xyz[1],          self.data.qpos[17]-neck_xyz[2]]
        self.obs.obj_to_hand_xyz = [self.data.qpos[15]-self.sys.pos_hand[0], self.data.qpos[16]-self.sys.pos_hand[1], self.data.qpos[17]-self.sys.pos_hand[2]]
        self.obs.obj_to_hand_xyz_norm = self.obs.obj_to_hand_xyz.copy()
        if self.sys.hand2target >= 0.02:
            self.obs.obj_to_hand_xyz_norm[0] *= 0.02/self.sys.hand2target
            self.obs.obj_to_hand_xyz_norm[1] *= 0.02/self.sys.hand2target
            self.obs.obj_to_hand_xyz_norm[2] *= 0.02/self.sys.hand2target
        self.sys.elbow_to_hand   = [self.sys.pos_hand[0] - elbow_xyz[0], self.sys.pos_hand[1] - elbow_xyz[1], self.sys.pos_hand[2] - elbow_xyz[2]]
        self.sys.elbow_to_target = [self.data.qpos[15]   - elbow_xyz[0], self.data.qpos[16]   - elbow_xyz[1], self.data.qpos[17]   - elbow_xyz[2]]
        # self.obs.obj_to_hand_xyz = [self.sys.pos_target0[0]-self.sys.pos_hand[0], self.sys.pos_target0[1]-self.sys.pos_hand[1], self.sys.pos_target0[2]-self.sys.pos_hand[2]]
        # print(f"{self.obs.obj_to_hand_xyz[0]:.2f}, {self.obs.obj_to_hand_xyz[1]:.2f}, {self.obs.obj_to_hand_xyz[2]:.2f}")

        if self.inf.timestep%int(5*self.sys.Hz) == 0:
            self.spawn_new_point()

    def close(self):
        self.renderer.close() 
        cv2.destroyAllWindows() 

    def render(self, speed=1):
        if self.inf.timestep%int(48*speed+2) ==0:
            self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"end_effector")] = self.sys.pos_EE_predict.copy()
            self.viewer.sync()
            self.viewer.cam.azimuth += 0.05 

    def check_reachable(self, point):
        shoulder_pos = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
        distoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        if distoshoulder >= 0.45 or distoshoulder <= 0.30:
            return False
        elif (point[0]<0.12 and point[1] > -0.20):
            return False
        else:
            return True
        
    def spawn_new_point(self):
        v1 = (   self.sys.elbow_to_hand[0]**2 +   self.sys.elbow_to_hand[1]**2 +   self.sys.elbow_to_hand[2]**2 ) **0.5
        v2 = ( self.sys.elbow_to_target[0]**2 + self.sys.elbow_to_target[1]**2 + self.sys.elbow_to_target[2]**2 ) **0.5
        hand_camera_center = np.degrees(np.arccos(np.dot(self.sys.elbow_to_hand, self.sys.elbow_to_target)/(v1*v2)))
        # print(hand_camera_center)

        if self.inf.timestep == 0 or self.sys.hand2target <= 0.05 or hand_camera_center <= 5:
            self.inf.reward += 10
            self.sys.arm_target_pos[3] = np.radians(random.uniform( -90, 90))
            reachable = False
            while reachable == False:
                self.sys.pos_target0[0] = random.uniform(-0.05, 0.50)
                self.sys.pos_target0[1] = random.uniform(-0.75, 0.00)
                self.sys.pos_target0[2] = random.uniform( 0.90, 1.40)
                reachable = self.check_reachable(self.sys.pos_target0)
            self.data.qpos[15:18] = self.sys.pos_target0.copy()

            reachable = False
            while reachable == False:
                self.sys.pos_target0[0] = random.uniform(-0.05, 0.50)
                self.sys.pos_target0[1] = random.uniform(-0.75, 0.00)
                self.sys.pos_target0[2] = random.uniform( 0.90, 1.40)
                reachable = self.check_reachable(self.sys.pos_target0)
            
            neck_xyz = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"neck_marker")].copy()
            self.obs.obj_to_neck_xyz = [self.data.qpos[15]-neck_xyz[0],          self.data.qpos[16]-neck_xyz[1],          self.data.qpos[17]-neck_xyz[2]]
            self.obs.obj_to_hand_xyz = [self.data.qpos[15]-self.sys.pos_hand[0], self.data.qpos[16]-self.sys.pos_hand[1], self.data.qpos[17]-self.sys.pos_hand[2]]

            new_dis = (self.obs.obj_to_hand_xyz[0]**2 + self.obs.obj_to_hand_xyz[1]**2 + self.obs.obj_to_hand_xyz[2]**2) **0.5
            self.sys.hand2target  = new_dis
            self.sys.hand2target0 = new_dis
            mujoco.mj_step(self.robot, self.data)
        else:
            self.reset()