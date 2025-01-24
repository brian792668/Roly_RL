# 往外轉為正
# 往外抬為正
# 往前踢為正

import mujoco
import mujoco.viewer
import numpy as np
import os

from imports.Settings import *
from imports.Controller import *
from imports.Draw_joint_info import *
from imports.Show_camera_view import *
from imports.Camera import *


if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/Roly_simulator/whole body/Roly_XML4/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)


    # initialize
    pos = initPos
    target = initTarget
    sensor = initSensor

    # set PID controller parameter
    PIDctrl = PIDcontroller(controlParameter, target)

    # if base is free joint
    # data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    head_camera = Camera(renderer=renderer, camID=0)
    head_camera = Camera(renderer=renderer, camID=0)

    mujoco.mj_resetData(model, data)  # Reset state and time.
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 2.5
    viewer.cam.lookat = [0.0, 0.0, 0.8]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    # Close the viewer automatically after 30 wall-seconds.
    step = 0

    for i in range(50):
        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
        mujoco.mj_step(model, data)
        # viewer.sync()

    # target[0]    # R hip yaw
    # target[0+1]  # R hip roll
    # target[0+2]  # R hip pitch
    # target[0+3]  # R knee
    # target[0+4]  # R ankle pitch
    # target[0+5]  # R ankle roll

    # target[6]    # L hip yaw
    # target[6+1]  # L hip roll
    # target[6+2]  # L hip pitch
    # target[6+3]  # L knee
    # target[6+4]  # L ankle pitch
    # target[6+5]  # L ankle roll

    time0 = data.time
    while data.time <= 2.0:
        print(data.time - time0)
        # viewer.sync()
        target[0+2] += -0.0050 # R hip pitch
        target[0+3] += +0.0120 # R knee
        target[0+4] += -0.0070 # R ankle pitch

        target[6+2] += -0.0050 # L hip pitch
        target[6+3] += +0.0120 # L knee
        target[6+4] += -0.0070 # L ankle pitch
        for i in range(20):
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]

            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            mujoco.mj_step(model, data)

    while data.time <= 2.50:
        print(data.time - time0)
        viewer.sync()
        target[0+1] += -0.0060  # R hip roll
        target[0+5] += -0.0060  # R ankle roll
        target[6+1] += -0.0060  # R hip roll
        target[6+5] += -0.0060  # R ankle roll
        for i in range(20):
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]
            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            mujoco.mj_step(model, data)
    
    while data.time <= 2.75:
        print(data.time - time0)
        viewer.sync()

        target[0+1] += -0.0060  # R hip roll
        target[0+5] += -0.0060  # R ankle roll
        target[6+1] += -0.0060  # R hip roll
        target[6+5] += -0.0060  # R ankle roll

        target[0+2] += -0.050*0.4 # R hip pitch
        target[0+3] += +0.120*0.4 # R knee
        # target[0+4] += -0.060 # R ankle pitch
        for i in range(20):
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]
            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            mujoco.mj_step(model, data)

    while data.time <= 3.00:
        print(data.time - time0)
        viewer.sync()

        target[0+1] += -0.0060  # R hip roll
        target[0+5] += -0.0060  # R ankle roll
        target[6+1] += -0.0060  # R hip roll
        target[6+5] += -0.0060  # R ankle roll

        target[0+2] += +0.050*0.1 # R hip pitch
        target[0+3] += -0.120*0.1 # R knee
        target[0+4] += -0.060*0.4 # R ankle pitch
        for i in range(20):
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]
            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            mujoco.mj_step(model, data)

    

    while viewer.is_running():
        viewer.sync()
        for i in range(20):
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]
            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            mujoco.mj_step(model, data)

    renderer.close() 
    cv2.destroyAllWindows() 
