# 往外轉為正
# 往外抬為正
# 往前踢為正

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
from imports.Forward_kinematics import *

if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/foward_kinematics/RolyURDF2/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)


    # initialize
    pos = initPos
    target = initTarget
    # sensor = initSensor
    PIDctrl = PIDcontroller(controlParameter, target)
    head_camera = Camera(renderer=renderer, camID=0)

    # if base is free joint
    data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    mujoco.mj_resetData(model, data)  # Reset state and time.
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 2.5
    viewer.cam.lookat = [0.0, 0.0, 0.8]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    
    step = 0
    while viewer.is_running():
        step += 1

        ## if base is free joint
        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        # if base is not free joint
        # pos = [data.qpos[i-7] for i in controlList]
        # vel = [data.qvel[i-8] for i in controlList]

        data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
        data.ctrl[3:8] = [0, 0, 0, 0, 0]
        mujoco.mj_step(model, data)

        if step%50 == 0:
            DH = DHtable(pos[3:9])
            T01  = T(DH, 0)
            T12  = T(DH, 1)
            T23  = T(DH, 2)
            T34  = T(DH, 3)
            T45  = T(DH, 4)
            T56  = T(DH, 5)
            T6E  = T(DH, 6)

            T02 = np.dot(T01, T12)
            T03 = np.dot(T02, T23)
            T04 = np.dot(T03, T34)
            T05 = np.dot(T04, T45)
            T06 = np.dot(T05, T56)
            T0E = np.dot(T06, T6E)
            EE  = np.dot(T0E, np.array([[0], [0], [0], [1]]))

            data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"marker2")] = [EE[0][0]+0.0, EE[1][0]+0.0, EE[2][0]+1.34]
            viewer.sync()


    renderer.close() 
    cv2.destroyAllWindows() 
