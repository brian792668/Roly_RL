from imports.Settings import *
from imports.Controller import *

class info():
    def __init__(self, Hz = 50):
        self.Hz = Hz
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
        self.PIDctrl = PIDcontroller(controlParameter, self.ctrlpos)

        self.limit_high = [ 1.57, 1.57, 1.57, 1.95]
        self.limit_low  = [-3.10,-1.57,-1.57, 0.00]

        self.arm_target_pos = [0] * 5
        self.hand_length = 0.0
        self.pos_hand = [0] * 3
        self.pos_origin = [0] * 3
        self.joints = [0] * 4

    def reset(self):
        self.pos = [0] * len(controlList)
        self.vel = [0] * len(controlList)
        self.ctrlpos = initTarget.copy()
