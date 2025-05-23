import numpy as np
 
act_high = np.array([ 1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1], dtype=np.float32)  

obs_high = np.array([   0.70,  0.45,  0.00,         # target xyz
                        0.05,  0.05,  0.05,         # target to hand xyz
                        1.00,  1.00,  1.00,         # action
                        1.95,  1.95,  1.57,  3.05,  # pos of arm
                        0], dtype=np.float32)
obs_low  = np.array([  -0.05, -0.95, -0.80,     
                       -0.05, -0.05, -0.05,
                       -1.00, -1.00, -1.00,
                       -1.95, -1.95, -1.57,  0.00,
                       0.15], dtype=np.float32) 