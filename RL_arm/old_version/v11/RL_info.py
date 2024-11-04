import numpy as np
from Settings import *
from Controller import *
from Camera import *

class RL_inf():
    def __init__(self):
        self.timestep = 0
        self.totaltimestep = 0
        self.action = [0.0, 0.0, 0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        self.total_reward = 0.0
    def reset(self):
        self.timestep = 0
        self.action = [0.0, 0.0, 0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        # self.total_reward = 0.0

class RL_obs():
    def __init__(self):
        self.joint_camera = [0.0, 0.0]
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]
        self.vel_arm = [0.0, 0.0, 0.0, 0.0]
        self.cam2target = 0.0

    def reset(self):
        self.joint_camera = [0.0, 0.0]
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]
        self.vel_arm = [0.0, 0.0, 0.0, 0.0]
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
        self.hand2target  = 1.0
        self.hand2target0 = 1.0
        self.limit_high = [ 1.57, 0.12, 0.79, 1.57, 2.10]
        self.limit_low  = [-0.79,-1.10,-0.79,-1.10, 0.00]

    def reset(self):
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.pos_target = [0.0, 0.0, 0.0]
        self.pos_hand   = [0.0, 0.0, 0.0]
        self.hand2target  = 1.0
        self.hand2target0 = 1.0