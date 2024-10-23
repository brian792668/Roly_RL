# 往外轉為正
# 往外抬為正
# 往前踢為正

import mujoco
import mujoco_viewer
import mujoco.viewer
import numpy as np
import math
import os
import time

from imports.Settings import *
from imports.Controller import *
from imports.Draw_joint_info import *
from imports.Show_camera_view import *


if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/Roly_simulator/whole body/RolyURDF2/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    # create the viewer object
    viewer = mujoco_viewer.MujocoViewer(model, data)
    init_viewer(viewer)

    # initialize
    pos = initPos
    target = initTarget
    sensor = initSensor

    # set PID controller parameter
    PIDctrl = PIDcontroller(controlParameter, target)

    # if base is free joint
    data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    # camera view setting
    showCameraView = ShowCameraView(renderer, camName)
    showCameraView.setParameter(widthDic, heightDic, posxDic, posyDic)
    last_render_t = 0.0
    mujoco.mj_resetData(model, data)  # Reset state and time.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        step = 0
        while viewer.is_running():
            step += 1

            # Get robot simu data
            ## if base is free joint
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]
            # if base is not free joint
            # pos = [data.qpos[i-7] for i in controlList]
            # vel = [data.qvel[i-8] for i in controlList]

            data.ctrl[:] = 29*[0]
            # data.ctrl[8] = deg2rad(57.3)
            # data.ctrl[14] = deg2rad(57.3)
            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            # max_value = np.max(data.ctrl)

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            if step%20 == 0:
                # Example modification of a viewer option: toggle contact points every two seconds.
                # with viewer.lock():
                #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
    
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step) 

        renderer.close() 
        cv2.destroyAllWindows() 
