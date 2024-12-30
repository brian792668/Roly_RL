import numpy as np
 
act_high = np.array([ 90,  90,  1], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
act_low  = np.array([-90, -90, -1], dtype=np.float32)  

obs_high = np.array([   0.50,  0.00,  0.00,                           # target xyz
                        1.57,  1.05,  1.57,  2.10], dtype=np.float32) # pos of arm
obs_low  = np.array([   0.00, -0.75, -0.60,     
                       -1.05, -1.57, -1.57,  0.0], dtype=np.float32) 