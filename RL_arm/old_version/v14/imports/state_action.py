import numpy as np
 
act_high = np.array([ 1,  1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1, -1], dtype=np.float32)  

obs_high = np.array([   1.58,  0.00,    # pos of camera
                        1.58,  0.00,
                        60,  60,  60,  60,  60,     # dis to target
                        1.57,  0.12,  1.57,  2.10], dtype=np.float32) # pos of arm
obs_low  = np.array([   -1.58, -1.58,  
                        -1.58, -1.58,
                        15, 15, 15, 15, 15,         
                        -0.79, -1.10, -1.10,  0.00], dtype=np.float32) 