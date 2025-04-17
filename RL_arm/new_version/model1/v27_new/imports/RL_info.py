from imports.Settings import *
from imports.Controller import *

class RL_inf():
    def __init__(self):
        self.timestep = 0
        self.totaltimestep = 0
        self.action = [0.0, 0.0, 0.0]
        self.action_new = [0.0, 0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        self.total_reward_standard  = 0.0
        self.total_reward_future_state = 0.0
    def reset(self):
        self.timestep = 0
        self.action = [0.0, 0.0, 0.0]
        self.action_new = [0.0, 0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}

class RL_obs():
    def __init__(self):
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]
        self.obj_to_neck_xyz = [0.0, 0.0, 0.0]
        self.obj_to_hand_xyz = [0.0, 0.0, 0.0]
        self.obj_to_hand_xyz_norm = [0.0, 0.0, 0.0]
        self.hand_length = 0.0

    def reset(self):
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]
        self.obj_to_neck_xyz = [0.0, 0.0, 0.0]
        self.obj_to_hand_xyz = [0.0, 0.0, 0.0]
        self.obj_to_hand_xyz_norm = [0.0, 0.0, 0.0]
        self.hand_length = 0.0

class RL_sys():
    def __init__(self, Hz = 50):
        # self.pos = initPos
        self.Hz = Hz
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)
        self.pos_target0 = [0.0, 0.0, 0.0]
        self.pos_target  = [0.0, 0.0, 0.0]
        self.pos_hand     = [0.0, 0.0, 0.0]
        self.pos_origin   = [0.0, 0.0, 0.0]
        self.pos_shoulder = [0.0, 0.0, 0.0]
        self.pos_elbow    = [0.0, 0.0, 0.0]
        self.hand2target  = 1.0
        self.hand2target0 = 1.0
        self.elbow_to_hand = [0.0, 0.0, 0.0]
        self.elbow_to_target = [0.0, 0.0, 0.0]
        self.limit_high = [ 1.57, 1.57, 1.57, 1.95]
        self.limit_low  = [-3.10,-1.57,-1.57, 0.00]
        self.arm_target_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pos_EE_predict = [0.0, 0.0, 0.0]

    def reset(self):
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.pos_target0 = [0.0, 0.0, 0.0]
        self.pos_target  = [0.0, 0.0, 0.0]
        self.pos_hand     = [0.0, 0.0, 0.0]
        self.pos_origin   = [0.0, 0.0, 0.0]
        self.pos_shoulder = [0.0, 0.0, 0.0]
        self.pos_elbow    = [0.0, 0.0, 0.0]
        self.hand2target  = 1.0
        self.hand2target0 = 1.0
        self.elbow_to_hand = [0.0, 0.0, 0.0]
        self.elbow_to_target = [0.0, 0.0, 0.0]
        self.arm_target_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pos_EE_predict = [0.0, 0.0, 0.0]