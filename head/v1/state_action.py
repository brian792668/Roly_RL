import numpy as np

act_low  = [-90.0, -45.0]    
act_high = [ 90.0,  45.0]

obs_low  = [ -1.58, -1.58, 
             -1.00, -1.00
             ] 
obs_high = [  1.58,  0.00,
              1.00,  1.00,
             ] 
act_low_flat  = np.array(act_low, dtype=np.float32)
act_high_flat = np.array(act_high, dtype=np.float32)
obs_low_flat  = np.array(obs_low, dtype=np.float32)
obs_high_flat = np.array(obs_high, dtype=np.float32)
# act_low_flat  = np.array(act_low).astype(np.float32)
# act_high_flat = np.array(act_high).astype(np.float32)
# obs_low_flat  = np.concatenate(obs_low).astype(np.float32)
# obs_high_flat = np.concatenate(obs_high).astype(np.float32)


