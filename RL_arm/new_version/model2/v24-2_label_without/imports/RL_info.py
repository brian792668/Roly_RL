from imports.Settings import *
from imports.Controller import *

class RL_inf():
    def __init__(self):
        self.timestep = 0
        self.totaltimestep = 0
        self.action = [0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        self.total_reward = 0.0
    def reset(self):
        self.timestep = 0
        self.action = [0.0, 0.0]
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        # self.total_reward = 0.0

class RL_obs():
    def __init__(self):
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]
        self.feature_points = np.array([1.0]*225)

    def reset(self):
        self.joint_arm = [0.0, 0.0, 0.0, 0.0]
        self.feature_points = np.array([1.0]*225)

class RL_sys():
    def __init__(self, Hz = 50):
        # self.pos = initPos
        self.Hz = Hz
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)
        self.limit_high = [ 1.57, 1.95, 1.57, 2.10, 3.14]
        self.limit_low  = [-1.57,-1.95,-1.57, 0.00, 0.00]

        self.pos_guide  = [0.0, 0.0, 0.0]
        self.pos_target = [0.0, 0.0, 0.0]
        self.pos_hand    = [0.0, 0.0, 0.0]
        self.pos_neck    = [0.0, 0.0, 0.0]
        self.pos_elbow   = [0.0, 0.0, 0.0]
        self.hand2guide  = 1.0
        self.hand2target = 1.0
        self.vec_guide2neck    = [0.0, 0.0, 0.0]
        self.vec_guide2hand    = [0.0, 0.0, 0.0]
        self.vec_target2neck   = [0.0, 0.0, 0.0]
        self.vec_target2hand   = [0.0, 0.0, 0.0]
        self.vec_target2elbow  = [0.0, 0.0, 0.0]
        self.vec_target2guide = [0.0, 0.0, 0.0]
        self.vec_hand2neck      = [0.0, 0.0, 0.0]
        self.vec_hand2elbow     = [0.0, 0.0, 0.0]
        self.vec_guide2elbow   = [0.0, 0.0, 0.0]
        self.guide_arm_joints  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.guide_neck_joints  = [0.0, 0.0]

        self.joints_increment = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.obstacle_hand_pos_and_quat = [0, 0, 0, 1, 0, 0, 0]
        self.obstacle_table_pos_and_quat = [0, 0, 0, 1, 0, 0, 0]
        self.obstacle_human_pos_and_quat = [0, 0, 0, 1, 0, 0, 0]
        self.compensate_angle = [0, 0]
        self.grasping_dis = 0.1

    def reset(self):
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.pos_guide  = [0.0, 0.0, 0.0]
        self.pos_target = [0.0, 0.0, 0.0]
        self.pos_hand    = [0.0, 0.0, 0.0]
        self.pos_neck    = [0.0, 0.0, 0.0]
        self.pos_elbow   = [0.0, 0.0, 0.0]
        self.hand2guide  = 1.0
        self.hand2target = 1.0
        self.vec_guide2neck    = [0.0, 0.0, 0.0]
        self.vec_guide2hand    = [0.0, 0.0, 0.0]
        self.vec_target2neck   = [0.0, 0.0, 0.0]
        self.vec_target2hand   = [0.0, 0.0, 0.0]
        self.vec_target2elbow  = [0.0, 0.0, 0.0]
        self.vec_target2guide = [0.0, 0.0, 0.0]
        self.vec_hand2elbow     = [0.0, 0.0, 0.0]
        self.vec_hand2neck      = [0.0, 0.0, 0.0]
        self.vec_guide2elbow   = [0.0, 0.0, 0.0]
        self.guide_arm_joints  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.guide_neck_joints  = [0.0, 0.0]

        self.joints_increment = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.obstacle_hand_pos_and_quat = [0, 0, 0, 1, 0, 0, 0]
        self.obstacle_table_pos_and_quat = [0, 0, 0, 1, 0, 0, 0]
        self.obstacle_human_pos_and_quat = [0, 0, 0, 1, 0, 0, 0]
        self.compensate_angle = [0, 0]
        self.grasping_dis = 0.1