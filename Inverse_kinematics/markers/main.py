import mujoco
import mujoco.viewer
import numpy as np
import os
import random

from imports.Settings import *
from imports.Controller import *
from imports.Draw_joint_info import *
from imports.Show_camera_view import *
from imports.Camera import *

def lebal_Roly_IK():
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/Inverse_kinematics/markers/RolyURDF2/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    # ========================  PID2: control end effector position =========================
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R gripper")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand_marker")
    # pos2 = data.xpos[body_id].copy()
    # pos2 = data.site_xpos[site_id].copy()
    pos2 = [0.0, 0.0, 0.0]
    vel2 = [0.0, 0.0, 0.0]
    targetpoint = pos2.copy()
    controlParameter2 = [
        [100.0, 0.0, 0.02],  # x
        [100.0, 0.0, 0.02],  # y
        [100.0, 0.0, 0.02]  # z
    ]
    PIDctrl2 = PIDcontroller(controlParameter2, targetpoint)


    # ================================= 400 points ======================================
    def check(point):
        dtoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if dtoshoulder >= 0.4 or dtoshoulder <= 0.22:
            return False
        elif (point[0]<0.1 and point[1] > -0.20):
            return False
        # elif (point[0]<0.1 and point[2] > 1.1):
        #     return False
        else:
            return True

    numberofpoints = 200
    targetpoints = [[0, 0, 0] for _ in range(numberofpoints)]
    xyzs = [[0, 0, 0] for _ in range(numberofpoints)]
    joints = [[0, 0, 0, 0] for _ in range(numberofpoints)]
    # for i in range(len(targetpoints)):
    #     distoshoulder = 0.5
    #     while distoshoulder >= 0.42:
    #         targetpoints[i][0] = random.uniform(0.1, 0.5)
    #         targetpoints[i][1] = random.uniform(-0.6, 0.0)
    #         targetpoints[i][2] = random.uniform(1.35, 0.9)
    #         distoshoulder  = (targetpoints[i][0]-0.00)**2
    #         distoshoulder += (targetpoints[i][1]+0.25)**2
    #         distoshoulder += (targetpoints[i][2]-1.35)**2
    #         distoshoulder = distoshoulder ** 0.5
    for i in range(len(targetpoints)):
        reachable = False
        while reachable == False:
            targetpoints[i][0] = random.uniform(0.02, 0.5)
            targetpoints[i][1] = random.uniform(-0.7, 0.0)
            targetpoints[i][2] = random.uniform(1.35, 0.9)
            reachable = check(targetpoints[i])
    sorted_targetpoints = sorted(targetpoints, key=lambda x: x[2])
    point_index = 0
    numberofdelete = 0
    stabletime = 0
    unstabletime = 0
    model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker0")] = [0, 1, 0, 1]
    model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker0")] = 0.01
    neckpos = [0.0, 0.0, 0.0]

    # ======================== Initialize Robot ========================
    pos = initPos
    target = initTarget
    PIDctrl = PIDcontroller(controlParameter, target)
    # head_camera = Camera(renderer=renderer, camID=0)
    # if base is free joint
    data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    mujoco.mj_resetData(model, data)  # Reset state and time.
    viewer = mujoco.viewer.launch_passive(model, data, show_right_ui= False)
    viewer.cam.distance = 1.5
    viewer.cam.lookat = [0.0, -0.25, 1.2]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    
    # ======================== Start simulatioln, wait for stabling ========================
    for i in range(1000):
        mujoco.mj_step(model, data)
        if i%50 == 0:
            viewer.sync()
    pos2 = data.site_xpos[site_id].copy()
    vel2 = [0.0, 0.0, 0.0]
    targetpoint = pos2.copy()
    # for i in range(5000):
    #     pos = [data.qpos[i] for i in controlList]
    #     vel = [data.qvel[i-1] for i in controlList]
    #     vel2 = [(data.site_xpos[site_id][i].copy()-pos2[i] )/0.001 for i in range(len(vel2))]
    #     pos2 = data.site_xpos[site_id].copy()
    #     data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)
    #     targetpoint[0] += 0.0001*np.tanh(100*(0.3-targetpoint[0]))
    #     targetpoint[1] += 0.0001*np.tanh(100*(-0.25-targetpoint[1]))
    #     targetpoint[2] += 0.0001*np.tanh(100*(1.2-targetpoint[2]))
    #     data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
    #     data.ctrl[3:5] = [0]*2
    #     data.ctrl[6:8] = [0]*2
    #     mujoco.mj_step(model, data)
    #     if i%500 == 0:
    #         viewer.sync()

    # ======================== label all points wrt. neck (x,y,z)========================
    neckpos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "neck_marker")].copy()
    for i in range(len(targetpoints)):
        xyzs[i][0] = sorted_targetpoints[i][0]-neckpos[0]
        xyzs[i][1] = sorted_targetpoints[i][1]-neckpos[1]
        xyzs[i][2] = sorted_targetpoints[i][2]-neckpos[2]
    
    step = 0
    while viewer.is_running():
        step += 1
        if stabletime == 1000:
            for i in range(len(sorted_targetpoints)):
                data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{i}")] = sorted_targetpoints[i]
            viewer.sync()
            joints[point_index] = [np.degrees(pos[3]), np.degrees(pos[4]), np.degrees(pos[6]), np.degrees(pos[7])]
            print(f"index: {point_index} -- xyz: [{xyzs[point_index][0]:.2f}, {xyzs[point_index][1]:.2f}, {xyzs[point_index][2]:.2f}] -- joints: [{np.degrees(pos[3]):.1f}, {np.degrees(pos[4]):.1f}, {np.degrees(pos[6]):.1f}, {np.degrees(pos[7]):.1f}] ")
            point_index += 1
            stabletime = 0
            unstabletime = 0
            if point_index == len(xyzs):
                renderer.close()
                break
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = [0, 1, 0, 0.5]
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete-1}")] = [1, 0, 1, 0.5]
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = 0.005
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete-1}")] = 0.002
        if unstabletime >= 40000:
            del xyzs[point_index]
            del joints[point_index]
            unstabletime = 0
            numberofdelete += 1
            print(f"Point {point_index} got delete")
            # mujoco.mj_resetData(model, data)  # Reset state and time.
            # for i in range(100):
            #     mujoco.mj_step(model, data)
            #     if i%50 == 0:
            #         viewer.sync()
            # pos2 = data.site_xpos[site_id].copy()
            # vel2 = [0.0, 0.0, 0.0]
            # targetpoint = pos2.copy()
            for i in range(5000):
                pos = [data.qpos[i] for i in controlList]
                vel = [data.qvel[i-1] for i in controlList]
                vel2 = [(data.site_xpos[site_id][i].copy()-pos2[i] )/0.001 for i in range(len(vel2))]
                pos2 = data.site_xpos[site_id].copy()
                data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)
                targetpoint[0] += 0.0001*np.tanh(100*(0.3-targetpoint[0]))
                targetpoint[1] += 0.0001*np.tanh(100*(-0.25-targetpoint[1]))
                targetpoint[2] += 0.0001*np.tanh(100*(1.2-targetpoint[2]))
                data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
                data.ctrl[3:5] = [0]*2
                data.ctrl[6:8] = [0]*2
                mujoco.mj_step(model, data)
                if i%500 == 0:
                    viewer.sync()
        if data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "neck_marker")][2] < 1:
            mujoco.mj_resetData(model, data)  # Reset state and time.
            for i in range(100):
                mujoco.mj_step(model, data)
                if i%50 == 0:
                    viewer.sync()
            pos2 = data.site_xpos[site_id].copy()
            vel2 = [0.0, 0.0, 0.0]
            targetpoint = pos2.copy()
            for i in range(5000):
                pos = [data.qpos[i] for i in controlList]
                vel = [data.qvel[i-1] for i in controlList]
                vel2 = [(data.site_xpos[site_id][i].copy()-pos2[i] )/0.001 for i in range(len(vel2))]
                pos2 = data.site_xpos[site_id].copy()
                data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)
                targetpoint[0] += 0.0001*np.tanh(100*(0.3-targetpoint[0]))
                targetpoint[1] += 0.0001*np.tanh(100*(-0.25-targetpoint[1]))
                targetpoint[2] += 0.0001*np.tanh(100*(1.2-targetpoint[2]))
                data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
                data.ctrl[3:5] = [0]*2
                data.ctrl[6:8] = [0]*2
                mujoco.mj_step(model, data)
                if i%500 == 0:
                    viewer.sync()
            
        if (np.abs(pos2[0] - sorted_targetpoints[point_index+numberofdelete][0]) < 0.02) and (np.abs(pos2[1] - sorted_targetpoints[point_index+numberofdelete][1]) < 0.02) and (np.abs(pos2[2] - sorted_targetpoints[point_index+numberofdelete][2]) < 0.015):
            stabletime +=1
        else:
            stabletime = 0
            unstabletime += 1
            
            
        targetpoint[0] += 0.0001*np.tanh(100*(sorted_targetpoints[point_index+numberofdelete][0]-targetpoint[0]))
        targetpoint[1] += 0.0001*np.tanh(100*(sorted_targetpoints[point_index+numberofdelete][1]-targetpoint[1]))
        targetpoint[2] += 0.0001*np.tanh(100*(sorted_targetpoints[point_index+numberofdelete][2]-targetpoint[2]))

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

        # data.ctrl[3:8] = [0]*5
        mujoco.mj_step(model, data)

        if step%5000 == 0:
            for i in range(len(sorted_targetpoints)):
                data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{i}")] = sorted_targetpoints[i]
            viewer.sync()
            viewer.cam.azimuth += 0.1


    renderer.close() 
    cv2.destroyAllWindows() 

    xyzs_np = np.array(xyzs)
    joints_np = np.array(joints)
    np.save(os.path.join(f'Roly/Inverse_kinematics/datasets/{len(xyzs)}points_xyz.npy'), xyzs_np)
    np.save(os.path.join(f'Roly/Inverse_kinematics/datasets/{len(xyzs)}points_joints.npy'), joints_np)
    print("Done labeling\n")
    print(f"length of dataest: {len(xyzs)}\n\n")

if __name__ == '__main__':
    lebal_Roly_IK()
    # from ..IK_train import *
    # train(numberofpoints=10, version="v1")
