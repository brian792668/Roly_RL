import numpy as np
 
act_high = np.array([ 1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1], dtype=np.float32)  

obs_high = np.array([   0.60,  0.10,  0.10,                            # target xyz
                        0.05,  0.05,  0.05,                                   # target to hand xyz
                        1.00,  1.00,  1.00,                                   # action
                        1.57,  1.57,  1.57,  1.95], dtype=np.float32)   # pos of arm
obs_low  = np.array([  -0.05, -0.85, -0.70,     
                       -0.05, -0.05, -0.05,
                       -1.00, -1.00, -1.00,
                       -3.10, -1.57, -1.57,  0.00], dtype=np.float32) 