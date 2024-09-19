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



if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'RL/RolyURDF2/Roly.xml'
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
    viewer.cam.azimuth = 160
    
    step = 0
    while viewer.is_running():
        step += 1
        if step%1000 == 0:
            data.qpos[36] = random.uniform(-0.6, 0.8)
            data.qpos[37] = random.uniform(-0.6, 0.6)

        ## if base is free joint
        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        # if base is not free joint
        # pos = [data.qpos[i-7] for i in controlList]
        # vel = [data.qvel[i-8] for i in controlList]

        if step%20 == 0:
            target[1:3] = head_camera.track(target[1:3], data, speed = 1.0)

        data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
        mujoco.mj_step(model, data)

        if step%50 == 0:
            viewer.sync()
            head_camera.get_img(data, rgb=True, depth=True)
            head_camera.get_target()
            head_camera.show(rgb=True, depth=False)


    renderer.close() 
    cv2.destroyAllWindows() 
