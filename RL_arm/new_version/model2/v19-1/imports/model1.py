import sys
import os
import numpy as np
from stable_baselines3 import SAC

class RLmodel:
    def __init__(self):
        self.model = SAC.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model1.zip"))
        self.state = [0]*10
        self.action = [0]*3

        self.obs_target_pos_to_neck = [0, 0, 0]
        self.obs_target_pos_to_hand_norm = [0, 0, 0]
        self.obs_joints = [0, 0, 0, 0]
        
    def predict(self):
        self.state = np.concatenate([self.obs_target_pos_to_neck.copy(), self.obs_target_pos_to_hand_norm.copy(), self.obs_joints.copy()]).astype(np.float32)
        self.action, _ = self.model.predict(self.state.copy(), deterministic=True)
        return(self.action.copy())