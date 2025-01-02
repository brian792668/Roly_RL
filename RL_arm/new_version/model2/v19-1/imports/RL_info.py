from imports.Settings import *
from imports.Controller import *

class RL_inf():
    def __init__(self):
        self.timestep = 0
        self.totaltimestep = 0
        self.action = [0.0, 0.0, 0.0]  # horizontal angle,  vertical angle,  elbow angle
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        self.total_reward = 0.0
    def reset(self):
        self.timestep = 0
        self.action = [0.0, 0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        # self.total_reward = 0.0

class RL_obs():
    def __init__(self):
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]

class RL_sys():
    def __init__(self, Hz = 50):
        # self.pos = initPos
        self.Hz = Hz
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)
        self.limit_high = [ 1.57, 0.12, 1.57, 2.10]
        self.limit_low  = [-1.05,-1.57,-1.57, 0.00]

        self.pos_target  = [0.0, 0.0, 0.0]
        self.pos_target0 = [0.0, 0.0, 0.0]
        self.pos_hand    = [0.0, 0.0, 0.0]
        self.pos_neck    = [0.0, 0.0, 0.0]
        self.pos_elbow   = [0.0, 0.0, 0.0]
        self.hand2target  = 1.0
        self.hand2target0 = 1.0
        self.vec_target2neck  = [0.0, 0.0, 0.0]
        self.vec_target2hand  = [0.0, 0.0, 0.0]
        self.vec_hand2elbow   = [0.0, 0.0, 0.0]
        self.vec_target2elbow = [0.0, 0.0, 0.0]
        self.random_arm_pos = [0.0, 0.0, 0.0, 0.0]
        self.arm_target_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.neck_target_pos  = [0.0, 0.0]

        self.joints_increment = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.pos_target  = [0.0, 0.0, 0.0]
        self.pos_target0 = [0.0, 0.0, 0.0]
        self.pos_hand    = [0.0, 0.0, 0.0]
        self.pos_neck    = [0.0, 0.0, 0.0]
        self.pos_elbow   = [0.0, 0.0, 0.0]
        self.hand2target  = 1.0
        self.hand2target0 = 1.0
        self.vec_target2neck  = [0.0, 0.0, 0.0]
        self.vec_target2hand  = [0.0, 0.0, 0.0]
        self.vec_hand2elbow   = [0.0, 0.0, 0.0]
        self.vec_target2elbow = [0.0, 0.0, 0.0]
        self.random_arm_pos  = [0.0, 0.0, 0.0, 0.0]
        self.arm_target_pos  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.neck_target_pos  = [0.0, 0.0]

        self.joints_increment = [0.0, 0.0, 0.0, 0.0]