import numpy as np
 
act_high = np.array([ 1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1], dtype=np.float32)  

obs_high = np.array([   0.60,  0.00,  0.00,                            # target xyz
                        0.02,  0.02,  0.02,                                   # target to hand xyz
                        1.00,  1.00,  1.00,                                   # action
                        1.57,  1.95,  1.57,  1.95], dtype=np.float32)   # pos of arm
obs_low  = np.array([   0.00, -0.85, -0.70,     
                       -0.02, -0.02, -0.02,
                       -1.00, -1.00, -1.00,
                       -1.57, -1.95, -1.57,  0.00], dtype=np.float32) 