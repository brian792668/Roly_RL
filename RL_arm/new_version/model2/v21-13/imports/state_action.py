import numpy as np

act_high = np.array([ 1,  1], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
act_low  = np.array([-1, -1], dtype=np.float32)  

#                          target xyz            target to guide                       action[2]
obs_high = np.concatenate([[0.50,  0.00,  0.00], [ 0.10,  0.10,  0.10]]).astype(np.float32)
obs_low  = np.concatenate([[0.00, -0.75, -0.60], [-0.10, -0.10, -0.10]]).astype(np.float32)