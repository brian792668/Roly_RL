import numpy as np
from Settings import *
from Controller import *
from Camera import *

class RL_inf():
    def __init__(self):
        self.timestep = 0
        self.reward = 0.0
        self.total_reward = 0.0
    def reset(self):
        self.timestep = 0
        self.reward = 0.0
        self.total_reward = 0.0

class RL_obs():
    def __init__(self):
        self.pos_camera = [0.0, np.radians(-45)]
        self.pos_arm = [0.0, 0.0, 0.0]
        self.dis_target = 0.0

    def reset(self):
        self.pos_camera = [0.0, np.radians(-45)]
        self.pos_arm = [0.0, 0.0, 0.0]
        self.dis_target = 0.0

class RL_sys():
    def __init__(self):
        self.pos = initPos
        self.vel = initPos
        self.ctrlpos = initTarget
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)

    def reset(self):
        self.pos = initPos
        self.vel = initPos
        self.ctrlpos = initTarget