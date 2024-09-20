import numpy as np
from Settings import *
from Controller import *
from Camera import *

class RL_inf():
    def __init__(self):
        self.timestep = 0
        self.action = [0.0, 0.0, 0.0]
        self.reward = 0.0
        self.total_reward = 0.0
    def reset(self):
        self.timestep = 0
        self.action = [0.0, 0.0, 0.0]
        self.reward = 0.0
        self.total_reward = 0.0

class RL_obs():
    def __init__(self):
        self.joint_camera = [0.0, 0.0]
        self.joint_arm = [0.0, 0.0, 0.0]
        self.cam2target = 0.0

    def reset(self):
        self.joint_camera = [0.0, 0.0]
        self.joint_arm = [0.0, 0.0, 0.0]
        self.cam2target = 0.0

class RL_sys():
    def __init__(self):
        # self.pos = initPos
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)
        self.pos_target = [0.0, 0.0, 0.0]
        self.pos_hand   = [0.0, 0.0, 0.0]
        self.hand2target = 1.0
        self.limit_high = [ 1.58, 0.00,  0.00]
        self.limit_low  = [ 0.00,-1.58, -3.00]

    def reset(self):
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.pos_target = [0.0, 0.0, 0.0]
        self.pos_hand   = [0.0, 0.0, 0.0]
        self.hand2target = 1.0
        self.limit_high = [ 1.58, 0.00,  3.00]
        self.limit_low  = [ 0.00,-1.58,  0.00]