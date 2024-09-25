import numpy as np
  
# act_high = np.array([ 1.58,  1.58,  3.00], dtype=np.float32) 
# act_low  = np.array([-1.58, -1.58, -3.00], dtype=np.float32) 
act_high = np.array([ 1,  1,  1,  1,  1], dtype=np.float32)
act_low  = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  

# obs_high = np.array([ 1.58,  0.00, 
#                       1.58,  0.00, 
#                       1.58,  0.00, 
#                       70, 
#                       70, 
#                       70, 
#                       0.79,  0.00,  0.79,  0.79,  2.60, 
#                       0.79,  0.00,  0.79,  0.79,  2.60, 
#                       0.79,  0.00,  0.79,  0.79,  2.60, ], dtype=np.float32)
# obs_low  = np.array([-1.58, -1.58, 
#                      -1.58, -1.58, 
#                      -1.58, -1.58,  # pos of camera
#                       20, 
#                       20, 
#                       20,         # dis to target
#                       -0.79, -1.58, -0.79, 0.00,  0.00, 
#                       -0.79, -1.58, -0.79, 0.00,  0.00,  
#                       -0.79, -1.58, -0.79, 0.00,  0.00, ], dtype=np.float32) # pos of arm

obs_high = np.array([ 1.58,  0.00,
                      70, 
                      70, 
                      70,  
                      0.79,  0.00,  0.79,  1.10,  2.60,
                      1.00,  1.00,  1.00,  1.00,  1.00 ], dtype=np.float32)
obs_low  = np.array([ -1.58, -1.58,  # pos of camera
                      20, 
                      20, 
                      20,         # dis to target
                      -0.79, -1.58, -0.79, -0.17,  0.00,
                      -1.00, -1.00, -1.00, -1.00, -1.00 ], dtype=np.float32) # pos of arm