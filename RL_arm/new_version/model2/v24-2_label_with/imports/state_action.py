import numpy as np

act_high = np.array([  1.6,   1.6], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
act_low  = np.array([ -1.6,  -1.6], dtype=np.float32)  

#                          end effector position
obs_high = np.array([  0.60,  0.00,  0.00], dtype=np.float32)
obs_low  = np.array([ -0.20, -0.85, -0.70], dtype=np.float32)