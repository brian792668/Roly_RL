import numpy as np
initPos = [ 
    0, 0, 0.83,     # 機器人base起始位置 (x,y,z)
    -1, 0, 0, 0,

    # 0,  # 7  trunk        
    0,  # 8  neck
    0,  # 9  camera

    0,  # 10 R_shoulder
    0,  # 11 R_arm1
    0,  # 12 R_arm2
    0,  # 13 R_arm3
    0,  # 14 R_arm4
    0,  # 15 R_palm
    # 0,  # 16 R finger1-1
    # 0,  # 17 R finger1-2
    # 0,  # 18 R finger2-1
    # 0,  # 19 R finger2-2
    # 0,  # 20 R finger3-1
    # 0,  # 21 R finger3-2
    # 0,  # 22 R gripper

    # 0,  # 23 L_shoulder
    # 0,  # 24 L_arm1
    # 0,  # 25 L_arm2
    # 0,  # 26 L_arm3
    # 0,  # 27 L_arm4
    # 0,  # 28 L_palm
    # 0,  # 29 L finger1-1
    # 0,  # 30 L finger1-2
    # 0,  # 31 L finger2-1
    # 0,  # 32 L finger2-2
    # 0,  # 33 L finger3-1
    # 0,  # 34 L finger3-2
    # 0,  # 35 L gripper

    0, 0, 0, 1, 0, 0, 0,  # red ball pos & orientation
    # 0, 0, 0, 1, 0, 0, 0,  # red box pos & orientation
]
initPos = [ 
    0, 0, 0.83,     # 機器人base起始位置 (x,y,z)
    -1, 0, 0, 0,
       
    0,  # 7  neck
    0,  # 8  camera

    0,  # 9 R_shoulder
    0,  # 10 R_arm1
    0,  # 11 R_arm2
    0,  # 12 R_arm3
    0,  # 13 R_arm4
    0,  # 14 R_palm

    0, 0, 0, 1, 0, 0, 0,  # red ball pos & orientation
]

initPos[7:] = list(np.pi/180*(np.array(initPos[7:])))
# controlList = [  7,  8,  9, 
#                 10, 11, 12, 13, 14, 15]
controlList = [  7,  8,  
                 9, 10, 11, 12, 13, 14]


initTarget = [
    # 0,  # R_hip_yaw
    # 0,  # R_hip_roll
    # 0,  # R_hip_pitch
    # 0,  # R_knee
    # 0,  # R_ankle_pitch 
    # 0,  # R_ankle_roll

    # 0,  # L_hip_yaw
    # 0,  # L_hip_roll
    # 0,  # L_hip_pitch
    # 0,  # L_knee
    # 0,  # L_ankle_pitch
    # 0,  # L_ankle_roll

    # 0,  # trunk
    0,  # neck                  
    0,  # camera

    0,  # R_shoulder
    0,  # R_arm1
    0,  # R_arm2
    0,  # R_arm3
    0,  # R_arm4
    0,  # R_palm
    # 0,  # R_gripper

    # 0,  # L_shoulder
    # 0,  # L_arm1
    # 0,  # L_arm2
    # 0,  # L_arm3
    # 0,  # L_arm4
    # 0,  # L_palm
    # 0,  # L_gripper
]

controlParameter = [
    # [1000.0, 1.0, 0.1],      # R_hip_yaw
    # [1000.0, 1.0, 0.1],      # R_hip_roll
    # [2000.0, 1.0, 0.1],     # R_hip_pitch
    # [2000.0, 1.0, 0.1],     # R_knee
    # [1000.0, 1.0, 0.1],      # R_ankle_pitch 
    # [1000.0, 1.0, 0.1],      # R_ankle_roll

    # [1000.0, 1.0, 0.1],      # L_hip_yaw
    # [1000.0, 1.0, 0.1],      # L_hip_roll
    # [2000.0, 1.0, 0.1],     # L_hip_pitch
    # [2000.0, 1.0, 0.1],     # L_knee
    # [1000.0, 1.0, 0.1],      # L_ankle_pitch
    # [1000.0, 1.0, 0.1],      # L_ankle_roll

    # [1000.0, 1.0, 0.1],      # trunck
    [100.0, 0.05, 0.1],   # neck                  
    [100.0, 0.05, 0.1],   # camera

    [500.0, 0.0, 0.1],
    [500.0, 0.0, 0.1],
    [500.0, 0.0, 0.1],
    [500.0, 0.0, 0.1],
    [500.0, 0.0, 0.1],
    [500.0, 0.0, 0.1]
    # [5000.0, 0.0, 0.1],  # R_gripper

    # [500.0, 0.0, 0.5],
    # [500.0, 0.0, 0.5],
    # [500.0, 0.0, 0.5],
    # [300.0, 0.0, 0.1],
    # [200.0, 0.0, 0.1],
    # [100.0, 0.0, 0.1],
    # [5000.0, 0.0, 0.1]   # L_gripper
]



# initSensor = [0, 0, 0, 0]

# camName = ['head_camera']
# frameDic = {'head_camera':np.zeros((240, 320, 3), dtype=np.uint8)}
# widthDic = {'head_camera':960}
# heightDic = {'head_camera':540}
# posxDic = {'head_camera':480}
# posyDic = {'head_camera':540}
# camName = ['head_camera', 'R_hand_camera', 'L_hand_camera']
# frameDic = {'head_camera':np.zeros((240, 320, 3), dtype=np.uint8), 
#             'R_hand_camera':np.zeros((240, 320, 3), dtype=np.uint8), 
#             'L_hand_camera':np.zeros((240, 320, 3), dtype=np.uint8)}
# widthDic = {'head_camera':960, 'R_hand_camera':960, 'L_hand_camera':960}
# heightDic = {'head_camera':540, 'R_hand_camera':540, 'L_hand_camera':540}
# posxDic = {'head_camera':480, 'R_hand_camera':960, 'L_hand_camera':1}
# posyDic = {'head_camera':540, 'R_hand_camera':0, 'L_hand_camera':0}