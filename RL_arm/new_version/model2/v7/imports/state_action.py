import numpy as np
 
act_high = np.array([ 1], dtype=np.float32)
act_low  = np.array([-1], dtype=np.float32)  

obs_high = np.array([   0.50,  0.60,  0.00,                           # target to neck xyz
                        0.60,  1.20,  0.60,                           # target to hand xyz
                        1.57,  0.12,  1.57,  2.10], dtype=np.float32) # pos of arm
obs_low  = np.array([   0.00, -0.60, -0.60,     
                       -0.60, -1.20, -0.60,
                       -1.05, -1.57, -1.57,  0.0], dtype=np.float32) 