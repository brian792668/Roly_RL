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

    # set PID controller parameter
    PIDctrl = PIDcontroller(controlParameter, target)

    # if base is free joint
    data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    head_camera = Camera(renderer=renderer, camID=0)

    mujoco.mj_resetData(model, data)  # Reset state and time.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        step = 0
        while viewer.is_running():
            step += 1

            if step%1000 == 0:
                data.qpos[36] = random.uniform(-0.6, 0.8)
                data.qpos[37] = random.uniform(-0.6, 0.6)

            # Get robot simu data
            ## if base is free joint
            pos = [data.qpos[i] for i in controlList]
            vel = [data.qvel[i-1] for i in controlList]
            # if base is not free joint
            # pos = [data.qpos[i-7] for i in controlList]
            # vel = [data.qvel[i-8] for i in controlList]

            target[0]=  0.20*(np.cos(step*2*np.pi/8000)-1) # trunk
            # target[1]= -0.20*(np.cos(step*2*np.pi/8000)-1) # neck
            # target[2]=  0.30*(np.cos(step*2*np.pi/8000)-1) # camera

            target[3]  = -0.20*(np.cos(step*2*np.pi/8000)-1) # R shoulder
            target[3+1]=  1.00*(np.cos(step*2*np.pi/8000)-1) # R arm1
            target[3+2]= -1.00*(np.cos(step*2*np.pi/8000)-1) # R arm2
            # target[3+3]= -0.50*(np.cos(step*2*np.pi/8000)-1) # R arm3
            # target[3+4]= -0.50*(np.cos(step*2*np.pi/8000)-1) # R arm4
            # target[3+5]= -0.50*(np.cos(step*2*np.pi/8000)-1) # R palm
            target[3+6]= -0.01*(np.cos(step*2*np.pi/8000)-1) # R gripper

            target[10]  = -0.20*(np.cos(step*2*np.pi/8000)-1) # L shoulder
            target[10+1]= -1.00*(np.cos(step*2*np.pi/8000)-1) # L arm1
            target[10+2]=  1.00*(np.cos(step*2*np.pi/8000)-1) # L arm2
            # target[10+3]=  0.50*(np.cos(step*2*np.pi/8000)-1) # L arm3
            # target[10+4]= -0.50*(np.cos(step*2*np.pi/8000)-1) # L arm4
            # target[10+5]= -0.50*(np.cos(step*2*np.pi/8000)-1) # L palm
            target[10+6]= -0.01*(np.cos(step*2*np.pi/8000)-1) # L gripper

            if step%20 == 0:
                target[1:3] = head_camera.track(target[1:3], data, speed = 1.0)

            data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
            mujoco.mj_step(model, data)

            if step%100 == 0:
                viewer.sync()
                head_camera.get_img(data, rgb=True, depth=True)
                head_camera.get_target()
                head_camera.show(rgb=True, depth=True)


        renderer.close() 
        cv2.destroyAllWindows() 
