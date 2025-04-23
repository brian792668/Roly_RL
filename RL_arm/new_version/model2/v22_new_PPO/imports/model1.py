import sys
import os
import torch
import numpy as np
from stable_baselines3 import SAC

class RLmodel:
    def __init__(self):
        self.model = SAC.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model1.zip"))
        self.model.policy.to("cpu")
        print("Model 1 : CPU")
        # if torch.cuda.is_available():
        #     self.model.policy.to("cuda")
        #     print("Model 2 : CUDA")
        # else:
        #     self.model.policy.to("cpu")
        #     print("Model 2 : CPU")
        self.state = [0]*13
        self.action = [0]*3

        self.obs_guide_to_neck = [0, 0, 0]
        self.obs_guide_to_hand_norm = [0, 0, 0]
        self.obs_joints = [0, 0, 0, 0]
        self.obs_guide_arm_joint = 0
        self.obs_grasp_dis = 0.0
        
    def predict(self):
        self.state = np.concatenate([self.obs_guide_to_neck.copy(), self.obs_guide_to_hand_norm.copy(), self.action.copy(), self.obs_joints.copy(), [self.obs_guide_arm_joint], [self.obs_grasp_dis]]).astype(np.float32)
        self.action, _ = self.model.predict(self.state.copy(), deterministic=True)
        return(self.action.copy())