# 往外轉為正
# 往外抬為正
# 往前踢為正

import mujoco
import mujoco.viewer
import os

from imports.Settings import *
from imports.Controller import *
from imports.Draw_joint_info import *
from imports.Show_camera_view import *
from imports.Camera import *
from Forward_kinematics import *

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
    # data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    mujoco.mj_resetData(model, data)  # Reset state and time.
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 2.5
    viewer.cam.lookat = [0.0, 0.0, 0.8]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    
    tableR = [[    0.0, np.pi/2,     0.0,     0.0],
              [np.pi/2, np.pi/2,     0.0,  0.2488],
              [    0.0,     0.0, -0.1105,     0.0],
              [np.pi/2, np.pi/2,     0.0,     0.0],
              [np.pi/2, np.pi/2,     0.0, -0.1195],
              [    0.0, np.pi/2,     0.0,     0.0],
              [    0.0, np.pi/2,     0.0,  0.1803]]
    tableL = [[    0.0, np.pi/2,     0.0,     0.0],
              [np.pi/2, np.pi/2,     0.0, -0.2488],
              [    0.0,     0.0, -0.1105,     0.0],
              [np.pi/2, np.pi/2,     0.0,     0.0],
              [np.pi/2, np.pi/2,     0.0, -0.1195],
              [    0.0, np.pi/2,     0.0,     0.0],
              [    0.0, np.pi/2,     0.0,  0.1803]]
    DH_R = DHtable(tableR)
    DH_L = DHtable(tableL)
    
    step = 0
    while viewer.is_running():
        step += 1

        ## if base is free joint
        posR = [data.qpos[i] for i in controlList]
        velR = [data.qvel[i-1] for i in controlList]
        posL = data.qpos[16:22]
        # if base is not free joint
        # pos = [data.qpos[i-7] for i in controlList]
        # vel = [data.qvel[i-8] for i in controlList]

        data.ctrl[:] = PIDctrl.getSignal(posR, velR, target)
        data.ctrl[3:8] = [0, 0, 0, 0, 0]
        mujoco.mj_step(model, data)

        if step%50 == 0:
            EE_R = DH_R.forward(angles=posR[3:9])
            EE_L = DH_L.forward(angles=posL)
            data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"markerR")] = [EE_R[0]+0.0, EE_R[1]+0.0, EE_R[2]+1.34]
            data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"markerL")] = [EE_L[0]+0.0, EE_L[1]+0.0, EE_L[2]+1.34]
            viewer.sync()

    renderer.close() 
    cv2.destroyAllWindows() 
