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

def lebal_Roly_IK(numberofpoints = 2000):
    # Add xml path
    file_path = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(file_path, "Roly_XML/Roly.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    # ========================  PID2: control end effector position =========================
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_finger1")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "R_hand_marker")
    # site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand_marker")
    # pos2 = data.xpos[body_id].copy()
    # pos2 = data.site_xpos[site_id].copy()
    pos2 = [0.0, 0.0, 0.0]
    vel2 = [0.0, 0.0, 0.0]
    targetpoint = pos2.copy()
    controlParameter2 = [
        [50.0, 0.0, 0.02],  # x
        [50.0, 0.0, 0.02],  # y
        [50.0, 0.0, 0.02]  # z
    ]
    PIDctrl2 = PIDcontroller(controlParameter2, targetpoint)

    # ======================== Initialize Robot ========================
    pos = initPos
    target = initTarget
    PIDctrl = PIDcontroller(controlParameter, target)
    # data.qpos[:] = pos[:]

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




    # ================================= n points ======================================
    shoulder_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"R_shoulder_marker")].copy()
    def check(point):
        dtoshoulder = ( (point[0]-shoulder_pos[0])**2 + (point[1]-shoulder_pos[1])**2 + (point[2]-shoulder_pos[2])**2 ) **0.5
        if dtoshoulder >= 0.54 or dtoshoulder <= 0.39:
            return False
        elif (point[0]<0.12 and point[1] > -0.20):
            return False
        # elif (point[0]<0.1 and point[2] > 1.1):
        #     return False
        else:
            return True

    targetpoints = [[0, 0, 0] for _ in range(numberofpoints)]
    xyzs = [[0, 0, 0] for _ in range(numberofpoints)]
    joints = [[0, 0, 0, 0] for _ in range(numberofpoints)]

    for i in range(len(targetpoints)):
        reachable = False
        while reachable == False:
            targetpoints[i][0] = shoulder_pos[0] + random.uniform(-0.00, 0.65)
            targetpoints[i][1] = shoulder_pos[1] + random.uniform(-0.65, 0.25)
            targetpoints[i][2] = shoulder_pos[2] + random.uniform(-0.65, 0.00)
            reachable = check(targetpoints[i])

    # 設定範圍
    # x_min, x_max = -0.05, 0.5
    # y_min, y_max = -0.7, 0.0
    # z_min, z_max = 0.8, 1.33
    x_min, x_max = shoulder_pos[0]-0.00, shoulder_pos[0]+0.65
    y_min, y_max = shoulder_pos[1]-0.65, shoulder_pos[1]+0.25
    z_min, z_max = shoulder_pos[2]-0.65, shoulder_pos[2]+0.00

    # 計算 x, y, z 軸上應該分幾個點，取近似的三次方根
    num_x = int(round(numberofpoints ** (1/3)))
    num_y = int(round(numberofpoints ** (1/3)))
    num_z = int(round(numberofpoints ** (1/3)))

    # 產生均勻分佈的點
    x_vals = np.linspace(x_min, x_max, num_x)
    y_vals = np.linspace(y_min, y_max, num_y)
    z_vals = np.linspace(z_min, z_max, num_z)

    # 建立 3D 網格
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # 轉成 (N, 3) 形狀

    # 檢查可到達點
    targetpoints = [point.tolist() for point in grid_points if check(point)]

    # 如果可到達的點數不足，補上
    while len(targetpoints) < numberofpoints:
        extra_point = [
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ]
        if check(extra_point):
            targetpoints.append(extra_point)

    # 確保點的數量剛好為 10000
    targetpoints = targetpoints[:numberofpoints]


    # sorted_targetpoints = sorted(targetpoints, key=lambda x: x[2])
    sorted_targetpoints = targetpoints
    point_index = 0
    numberofdelete = 0
    stabletime = 0
    unstabletime = 0
    model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker0")] = [0, 1, 0, 1]
    model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker0")] = 0.002
    origin_pos = [0.0, 0.0, 0.0]

    # ======================== label all points wrt. neck (x,y,z)========================
    origin_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "origin_marker")].copy()
    for i in range(len(targetpoints)):
        xyzs[i][0] = sorted_targetpoints[i][0]-origin_pos[0]
        xyzs[i][1] = sorted_targetpoints[i][1]-origin_pos[1]
        xyzs[i][2] = sorted_targetpoints[i][2]-origin_pos[2]
    
    step = 0
    while viewer.is_running():
        step += 1
        if stabletime == 2000:
            for i in range(len(sorted_targetpoints)):
                data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{i}")] = sorted_targetpoints[i]
            viewer.sync()
            joints[point_index] = [np.degrees(pos[3]), np.degrees(pos[4]), np.degrees(pos[6]), np.degrees(pos[7])]
            print(f"index: {point_index} -- xyz: [{xyzs[point_index][0]:.2f}, {xyzs[point_index][1]:.2f}, {xyzs[point_index][2]:.2f}] -- joints: [{np.degrees(pos[3]):.1f}, {np.degrees(pos[4]):.1f}, {np.degrees(pos[6]):.1f}, {np.degrees(pos[7]):.1f}] ")
            
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = [1, 0, 1, 0.5]
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = 0.002
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete+1}")] = [0, 1, 0, 0.5]
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete+1}")] = 0.005
            point_index += 1
            stabletime = 0
            unstabletime = 0
            if point_index == len(xyzs):
                renderer.close()
                break
            # model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = [0, 1, 0, 0.5]
            # model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete-1}")] = [1, 0, 1, 0.5]
            # model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = 0.005
            # model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete-1}")] = 0.002
        if unstabletime >= 20000:
            del xyzs[point_index]
            del joints[point_index]
            model.site_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = [0, 1, 0, 0.5]
            model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{point_index+numberofdelete}")] = 0.002
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

            # mujoco.mj_resetData(model, data)  # Reset state and time.
            # for i in range(100):
            #     mujoco.mj_step(model, data)
            # pos2 = data.site_xpos[site_id].copy()
            # vel2 = [0.0, 0.0, 0.0]
            # targetpoint = pos2.copy()

            # for i in range(5000):
            #     pos = [data.qpos[i] for i in controlList]
            #     vel = [data.qvel[i-1] for i in controlList]
            #     vel2 = [((data.site_xpos[site_id][i].copy()-pos2[i] )/0.005)*0.4 + vel2[i]*0.6 for i in range(len(vel2))]
            #     pos2 = data.site_xpos[site_id].copy()
            #     data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)
            #     targetpoint[0] += 0.0001*np.tanh(1000*(1.2* 0.30 - 1.2*targetpoint[0]))
            #     targetpoint[1] += 0.0001*np.tanh(1000*(1.2*-0.25 - 1.2*targetpoint[1]))
            #     targetpoint[2] += 0.0001*np.tanh(1000*(1.2* 0.90 - 1.2*targetpoint[2]))
            #     data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
            #     data.ctrl[3:5] = [0]*2
            #     data.ctrl[6:8] = [0]*2
            #     mujoco.mj_step(model, data)
            #     if i%100 == 0:
            #         viewer.sync()
        if data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "origin_marker")][2] < 1 or np.abs(pos[3]) >= 1.56:
            mujoco.mj_resetData(model, data)  # Reset state and time.
            for i in range(100):
                mujoco.mj_step(model, data)
            pos2 = data.site_xpos[site_id].copy()
            vel2 = [0.0, 0.0, 0.0]
            targetpoint = pos2.copy()
            # for i in range(5000):
            #     pos = [data.qpos[i] for i in controlList]
            #     vel = [data.qvel[i-1] for i in controlList]
            #     vel2 = [((data.site_xpos[site_id][i].copy()-pos2[i] )/0.005)*0.4 + vel2[i]*0.6 for i in range(len(vel2))]
            #     pos2 = data.site_xpos[site_id].copy()
            #     data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)
            #     targetpoint[0] += 0.0001*np.tanh(1000*(1.2* 0.30 - 1.2*targetpoint[0]))
            #     targetpoint[1] += 0.0001*np.tanh(1000*(1.2*-0.25 - 1.2*targetpoint[1]))
            #     targetpoint[2] += 0.0001*np.tanh(1000*(1.2* 0.90 - 1.2*targetpoint[2]))
            #     data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
            #     data.ctrl[3:5] = [0]*2
            #     data.ctrl[6:8] = [0]*2
            #     mujoco.mj_step(model, data)
            #     if i%100 == 0:
            #         viewer.sync()
            
        if (np.abs(pos2[0] - sorted_targetpoints[point_index+numberofdelete][0]) < 0.01) and (np.abs(pos2[1] - sorted_targetpoints[point_index+numberofdelete][1]) < 0.01) and (np.abs(pos2[2] - sorted_targetpoints[point_index+numberofdelete][2]) < 0.01):
            stabletime +=1
        else:
            stabletime = 0
            unstabletime += 1
            
            
        targetpoint[0] += 0.0001*np.tanh(1000*(1.2*sorted_targetpoints[point_index+numberofdelete][0]-1.2*targetpoint[0]))
        targetpoint[1] += 0.0001*np.tanh(1000*(1.2*sorted_targetpoints[point_index+numberofdelete][1]-1.2*targetpoint[1]))
        targetpoint[2] += 0.0001*np.tanh(1000*(1.2*sorted_targetpoints[point_index+numberofdelete][2]-1.2*targetpoint[2]))

        # if base is free joint
        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        # # if base is not free joint
        # pos = [data.qpos[i-7] for i in controlList]
        # vel = [data.qvel[i-8] for i in controlList]

        vel2 = [((data.site_xpos[site_id][i].copy()-pos2[i] )/0.005)*0.4 + vel2[i]*0.6 for i in range(len(vel2))]
        pos2 = data.site_xpos[site_id].copy()
        data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, targetpoint)
        # data.xfrc_applied[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_arm2")][2] = -5
        data.xfrc_applied[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_arm2")][0] = -0.5
        

        data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
        # data.ctrl[3:9] = [0]*6
        data.ctrl[3:5] = [0]*2
        data.ctrl[6:8] = [0]*2

        # data.ctrl[3:8] = [0]*5
        mujoco.mj_step(model, data)

        # if step%100 == 0:
        #     for i in range(len(sorted_targetpoints)):
        #         data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker{i}")] = sorted_targetpoints[i]
        #     viewer.sync()


    renderer.close() 
    cv2.destroyAllWindows() 

    xyzs_np = np.array(xyzs)
    joints_np = np.array(joints)
    folder_path = os.path.join(file_path, f"../datasets/new/{len(xyzs)}points")
    os.makedirs(folder_path, exist_ok=True)
    np.save(os.path.join(folder_path, "xyz.npy"), xyzs_np)
    np.save(os.path.join(folder_path, "joints.npy"), joints_np)
    print("Done labeling\n")
    print(f"length of dataest: {len(xyzs)}\n\n")

if __name__ == '__main__':
    lebal_Roly_IK(numberofpoints = 10000)
    # from ..IK_train import *
    # train(numberofpoints=10, version="v1")
