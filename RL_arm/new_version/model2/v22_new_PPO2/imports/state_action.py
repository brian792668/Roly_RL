import numpy as np

act_high = np.array([ 1,  1,  1], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
act_low  = np.array([-1, -1, -1], dtype=np.float32)  

#                            target xyz            joint
obs_high = np.concatenate([[ 0.65,  0.40,  0.10], [ 1.57]]).astype(np.float32)
obs_low  = np.concatenate([[-0.10, -0.90, -0.65], [-1.57]]).astype(np.float32)