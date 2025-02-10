import numpy as np

act_high = np.array([ 1,  1,  1,  1], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
act_low  = np.array([-1, -1, -1, -1], dtype=np.float32)  

#                          target xyz            hand xyz               joints                       actions
obs_high = np.concatenate([[0.50,  0.00,  0.00], [0.50,  0.00,  0.00], [1.57,  1.57,  1.57,  1.95],  [ 1,  1,  1,  1]]).astype(np.float32)
obs_low  = np.concatenate([[0.00, -0.75, -0.60], [0.00, -0.75, -0.60], [1.57,  1.57,  1.57,  1.95],  [-1, -1, -1, -1]]).astype(np.float32)