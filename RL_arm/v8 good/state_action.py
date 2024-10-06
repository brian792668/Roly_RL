import numpy as np
 
act_high = np.array([ 1,  1,  1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  

obs_high = np.array([   1.58,  0.00,
                        1.58,  0.00,
                        60,  60,  60,  60,  60,  
                        0.7, 0.7, 0.7, 0.7, 0.7,
                        0.79,  0.17,  0.79,  1.57,  2.10], dtype=np.float32)
obs_low  = np.array([   -1.58, -1.58,  # pos of camera
                        -1.58, -1.58,
                        15, 15, 15, 15, 15,         # dis to target
                        0,  0,  0,  0,  0, 
                        -0.79, -0.79, -0.79, -1.10,  0.00], dtype=np.float32) # pos of arm