import numpy as np
 
act_high = np.array([ 1.57,  1.57, 1.95], dtype=np.float32)
act_low  = np.array([-1.57, -1.57, 0], dtype=np.float32)  

obs_high = np.array([   0.65,  0.40,  0.10, # target xyz
                        1.57,               # target elbow pos
                        0                   # hand length
                    ], dtype=np.float32)   
obs_low  = np.array([  -0.10, -0.90, -0.65,
                       -1.57,
                       0.15
                    ], dtype=np.float32) 