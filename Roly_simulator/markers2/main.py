# 往外轉為正
# 往外抬為正
# 往前踢為正

import mujoco
import mujoco.viewer
import numpy as np
import os
import random
import glfw

from imports.Settings import *
from imports.Controller import *
from imports.Draw_joint_info import *
from imports.Show_camera_view import *
from imports.Camera import *

if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/Roly_simulator/markers2/RolyURDF2/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    # ============================================================================ PID2
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R gripper")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand_marker")
    # pos2 = data.xpos[body_id].copy()
    pos2 = data.site_xpos[site_id].copy()
    targetpoint = pos2.copy()
    vel2 = [0.0, 0.0, 0.0]
    controlParameter2 = [
        [100.0, 0.0, 0.02],  # x
        [100.0, 0.0, 0.02],  # y
        [100.0, 0.0, 0.02]  # z
    ]
    PIDctrl2 = PIDcontroller(controlParameter2, targetpoint)
    # -----------------------------------------------------------------------------


    # ======================================================== 200 points
    target200point = [[0, 0, 0] for _ in range(200)]
    label200point = [[0, 0, 0] for _ in range(200)]
    joint200point = [[0, 0, 0, 0] for _ in range(200)]
    for i in range(len(target200point)):
        distoshoulder = 0.5
        while distoshoulder >= 0.42:
            target200point[i][0] = random.uniform(0.1, 0.5)
            target200point[i][1] = random.uniform(-0.6, 0.0)
            target200point[i][2] = random.uniform(1.35, 0.9)
            distoshoulder  = (target200point[i][0]-0.00)**2
            distoshoulder += (target200point[i][1]+0.25)**2
            distoshoulder += (target200point[i][2]-1.35)**2
            distoshoulder = distoshoulder ** 0.5
    sorted_target200point = sorted(target200point, key=lambda x: x[2])
    point_index = 0
    stabletime = 0
    model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker0")] = [0, 1, 0, 1]
    model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker0")] = 0.01
    neckpos = [0.0, 0.0, 0.0]
    # --------------------------------------------------------

    # initialize
    pos = initPos
    target = initTarget
    PIDctrl = PIDcontroller(controlParameter, target)
    head_camera = Camera(renderer=renderer, camID=0)
    # if base is free joint
    data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]


    mujoco.mj_resetData(model, data)  # Reset state and time.
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.lookat = [0.0, -0.25, 1.0]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    
    step = 0
    direction = 0

    for i in range(1000):
        mujoco.mj_step(model, data)
        if i%50 == 0:
            viewer.sync()
    pos2 = data.site_xpos[site_id].copy()
    vel2[0] = vel2[0]*0.6+0.4*(data.site_xpos[site_id][0].copy() - pos2[0] )/ 0.001
    vel2[1] = vel2[1]*0.6+0.4*(data.site_xpos[site_id][1].copy() - pos2[1] )/ 0.001
    vel2[2] = vel2[2]*0.6+0.4*(data.site_xpos[site_id][2].copy() - pos2[2] )/ 0.001
    targetpoint = pos2.copy()

    neckpos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "neck_marker")].copy()
    for i in range(len(target200point)):
        label200point[i][0] = sorted_target200point[i][0]-neckpos[0]
        label200point[i][1] = sorted_target200point[i][1]-neckpos[1]
        label200point[i][2] = sorted_target200point[i][2]-neckpos[2]
    

    while viewer.is_running():
        step += 1
        if stabletime == 2000:
            joint200point[point_index] = [np.degrees(pos[3]), np.degrees(pos[4]), np.degrees(pos[6]), np.degrees(pos[7])]
            print(f"index = {point_index}: {np.degrees(pos[3]):.2f}, {np.degrees(pos[4]):.2f}, {np.degrees(pos[6]):.2f}, {np.degrees(pos[7]):.2f} --- label: [{label200point[point_index][0]:.2f} {label200point[point_index][1]:.2f} {label200point[point_index][2]:.2f}]")
            point_index += 1
            stabletime = 0
            # if point_index == len(sorted_target200point):
            if point_index == 10:
                renderer.close()
                break
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index}")] = [0, 1, 0, 1]
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index-1}")] = [1, 0, 1, 0.5]
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index}")] = 0.01
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index-1}")] = 0.005
        if (np.abs(targetpoint[0] - sorted_target200point[point_index][0]) < 0.002) and (np.abs(targetpoint[1] - sorted_target200point[point_index][1]) < 0.002) and (np.abs(targetpoint[2] - sorted_target200point[point_index][2]) < 0.002):
            stabletime +=1
        else:
            stabletime = 0
            
        targetpoint[0] += 0.0001*np.tanh(100*(sorted_target200point[point_index][0]-targetpoint[0]))
        targetpoint[1] += 0.0001*np.tanh(100*(sorted_target200point[point_index][1]-targetpoint[1]))
        targetpoint[2] += 0.0001*np.tanh(100*(sorted_target200point[point_index][2]-targetpoint[2]))

        # if base is free joint
        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        # # if base is not free joint
        # pos = [data.qpos[i-7] for i in controlList]
        # vel = [data.qvel[i-8] for i in controlList]

        vel2 = [(data.site_xpos[site_id][i].copy()-pos2[i] )/0.001 for i in range(len(vel2))]
        pos2 = data.site_xpos[site_id].copy()
        data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)

        data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
        # data.ctrl[3:9] = [0]*6
        data.ctrl[3:5] = [0]*2
        data.ctrl[6:8] = [0]*2
        mujoco.mj_step(model, data)

        if step%1000 == 0:
            for i in range(len(sorted_target200point)):
                data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{i}")] = sorted_target200point[i]
            viewer.sync()
            # viewer.cam.azimuth += 0.1


    renderer.close() 
    cv2.destroyAllWindows() 

    save_dir = 'Roly/Inverse_kinematics/'
    label200point_np = np.array(label200point)
    joint200point_np = np.array(joint200point)
    np.save(os.path.join(save_dir, 'label200point.npy'), label200point_np)
    np.save(os.path.join(save_dir, 'joint200point.npy'), joint200point_np)
    print("done\n")
