import numpy as np

initPos = [         # radious
    0, 0, 0.83,     # 機器人base起始位置 (x,y,z)
    # 0.7071, 0, 0, 0.7071,
    -1, 0, 0, 0,

    # 0,  # 7 R_hip_yaw    
    # 0,  # 8 R_hip_roll   
    # 0,  # 9 R_hip_pitch  
    # 0,  # 10 R_knee
    # 0,  # 11 R_ankle_pitch
    # 0,  # 12 R_ankle_roll 
    # 0,  # 13 R_knee_driver
    # 0,  # 14 R_knee_linkage

    # 0,  # 15 L_hip_yaw     
    # 0,  # 16 L_hip_roll   
    # 0,  # 17 L_hip_pitch 
    # 0,  # 18 L_knee 
    # 0,  # 19 L_ankle_pitch
    # 0,  # 20 L_ankle_roll 
    # 0,  # 21 L_knee_driver
    # 0,  # 22 L_knee_linkage

    0,  # 23 trunk        
    0,  # 24 neck
    0,  # 25 camera

    0,  # 26 R_shoulder
    0,  # 27 R_arm1
    0,  # 28 R_arm2
    0,  # 29 R_arm3
    0,  # 30 R_arm4
    0,  # 31 R_palm
    0,  # 32 R finger1-1
    0,  # 33 R finger1-2
    0,  # 34 R finger2-1
    0,  # 35 R finger2-2
    0,  # 36 R finger3-1
    0,  # 37 R finger3-2
    0.02,  # 38 R gripper

    0,  # 39 L_shoulder
    0,  # 40 L_arm1
    0,  # 41 L_arm2
    0,  # 42 L_arm3
    0,  # 43 L_arm4
    0,  # 44 L_palm
    0,  # 45 L finger1-1
    0,  # 46 L finger1-2
    0,  # 47 L finger2-1
    0,  # 48 L finger2-2
    0,  # 49 L finger3-1
    0,  # 50 L finger3-2
    0,  # 51 L gripper

    0, 0, 0, 1, 0, 0, 0,  # red ball pos & orientation
    # 0, 0, 0, 1, 0, 0, 0,  # red box pos & orientation
]
initPos[7:] = list(np.pi/180*(np.array(initPos[7:])))

# controlList = [  7,  8,  9, 10, 11, 12,
#                 15, 16, 17, 18, 19, 20,   
#                 23, 24, 25,
#                 26, 27, 28, 29, 30, 31, 38,
#                 39, 40, 41, 42, 43, 44, 51 ]
controlList = [  7,  8,  9, 
                10, 11, 12, 13, 13, 15, 22,
                23, 24, 25, 26, 27, 28, 35 ]

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

    0,  # trunk
    0,  # neck                  
    0,  # camera

    0,  # R_shoulder
    0,  # R_arm1
    0,  # R_arm2
    0,  # R_arm3
    0,  # R_arm4
    0,  # R_palm
    0,  # R_gripper

    0,  # L_shoulder
    0,  # L_arm1
    0,  # L_arm2
    0,  # L_arm3
    0,  # L_arm4
    0,  # L_palm
    0,  # L_gripper
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

    [1000.0, 1.0, 0.1],      # trunck
    [100.0, 0.05, 0.1],   # neck                  
    [100.0, 0.05, 0.1],   # camera

    [500.0, 0.05, 0.1],
    [500.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [5000.0, 0.0, 0.1],  # R_gripper

    [500.0, 0.05, 0.1],
    [500.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [200.0, 0.05, 0.1],
    [5000.0, 0.0, 0.1]   # L_gripper
]

# initSensor = [0, 0, 0, 0]

camName = ['head_camera']
frameDic = {'head_camera':np.zeros((240, 320, 3), dtype=np.uint8)}
widthDic = {'head_camera':960}
heightDic = {'head_camera':540}
posxDic = {'head_camera':480}
posyDic = {'head_camera':540}
# camName = ['head_camera', 'R_hand_camera', 'L_hand_camera']
# frameDic = {'head_camera':np.zeros((240, 320, 3), dtype=np.uint8), 
#             'R_hand_camera':np.zeros((240, 320, 3), dtype=np.uint8), 
#             'L_hand_camera':np.zeros((240, 320, 3), dtype=np.uint8)}
# widthDic = {'head_camera':960, 'R_hand_camera':960, 'L_hand_camera':960}
# heightDic = {'head_camera':540, 'R_hand_camera':540, 'L_hand_camera':540}
# posxDic = {'head_camera':480, 'R_hand_camera':960, 'L_hand_camera':1}
# posyDic = {'head_camera':540, 'R_hand_camera':0, 'L_hand_camera':0}