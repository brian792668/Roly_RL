import numpy as np
 
# act_high = np.array([ 90,  90,  1], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
# act_low  = np.array([-90, -90, -1], dtype=np.float32)  

act_high = np.array([ 1,  1,  1,  1], dtype=np.float32)   # horizontal angle,  vertical angle,  elbow angle
act_low  = np.array([-1, -1, -1, -1], dtype=np.float32)  

# obs_high = np.array([   0.50,  0.00,  0.00,                           # target xyz
#                         1.57,  1.05,  1.57,  2.10], dtype=np.float32) # pos of arm
# obs_low  = np.array([   0.00, -0.75, -0.60,     
#                        -1.05, -1.57, -1.57,  0.0], dtype=np.float32) 

#                          target xyz            hand xyz              hand to elbow          actions           joints
obs_high = np.concatenate([[0.50,  0.00,  0.00], [0.50,  0.00,  0.00], [ 0.22,  0.22,  0.22], [ 1,  1,  1,  1], [ 1.57,  1.57,  1.57,  2.10,  3.14] ]).astype(np.float32)
obs_low  = np.concatenate([[0.00, -0.75, -0.60], [0.00, -0.75, -0.60], [-0.22, -0.22, -0.22], [-1, -1, -1, -1], [-1.57, -1.57, -1.57,  0.00,  0.00] ]).astype(np.float32)