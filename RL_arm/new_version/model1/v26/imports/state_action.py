import numpy as np
 
act_high = np.array([ 1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1], dtype=np.float32)  

obs_high = np.array([   0.50,  0.25,  0.10,         # target xyz
                        0.05,  0.05,  0.05,         # target to hand xyz
                        1.00,  1.00,  1.00,         # action
                        1.57,  1.57,  1.57,  1.95,  # pos of arm
                        1.57                        # target elbow pos
                    ], dtype=np.float32)   
obs_low  = np.array([  -0.10, -0.75, -0.50,     
                       -0.05, -0.05, -0.05,
                       -1.00, -1.00, -1.00,
                       -3.10, -1.57, -1.57,  0.00,
                       -1.57
                    ], dtype=np.float32) 