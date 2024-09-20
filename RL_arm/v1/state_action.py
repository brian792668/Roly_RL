import numpy as np

# act_low  = np.array([-1.58, -1.58, -3.00], dtype=np.float32)   
# act_high = np.array([ 1.58,  1.58,  3.00], dtype=np.float32)
act_low  = np.array([-1, -1, -1], dtype=np.float32)   
act_high = np.array([ 1,  1,  1], dtype=np.float32)

obs_low  = np.array([-1.58, -1.58,  # pos of camera
                      0.00,         # dis to target
                      0.00, -1.58,  0.00 ], dtype=np.float32) # pos of arm
obs_high = np.array([ 1.58,  0.00,
                      100, 
                      1.58,  0.00,  3.00 ], dtype=np.float32)