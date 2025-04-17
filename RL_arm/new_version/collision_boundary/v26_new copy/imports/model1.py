import sys
import os
import torch
import numpy as np
from stable_baselines3 import SAC

class RL_moving_model:
    def __init__(self):
        self.model = SAC.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "moving_model.zip"))
        self.model.policy.to("cpu")
        print("Model 1 : CPU")
        # if torch.cuda.is_available():
        #     self.model.policy.to("cuda")
        #     print("Model 2 : CUDA")
        # else:
        #     self.model.policy.to("cpu")
        #     print("Model 2 : CPU")
        self.state = [0]*14
        self.action = [0]*3

        self.obs_guide_to_neck = [0, 0, 0]
        self.obs_guide_to_hand_norm = [0, 0, 0]
        self.obs_joints = [0, 0, 0, 0]
        self.obs_arm_target_pos = 0.0
        self.obs_hand_dis = 0.0
        
    def predict(self):
        self.state = np.concatenate([self.obs_guide_to_neck.copy(), self.obs_guide_to_hand_norm.copy(), self.action.copy(), self.obs_joints.copy(), [self.obs_arm_target_pos]]).astype(np.float32)
        self.action, _ = self.model.predict(self.state.copy(), deterministic=True)
        return(self.action.copy())