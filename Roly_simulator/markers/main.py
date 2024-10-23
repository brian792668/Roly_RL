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

def key_callback(key):
    global direction
    global target2
    
    if key == glfw.KEY_UP:
        target2[direction] += 0.01
        # print("Up key pressed")
    elif key == glfw.KEY_DOWN:
        target2[direction] -= 0.01
        # print("Down key pressed")
    elif key == glfw.KEY_7:
        direction = 0
        # print("0 pressed")
    elif key == glfw.KEY_RIGHT:
        target2[1] += 0.01
        # print("1 pressed")
    elif key == glfw.KEY_LEFT:
        target2[1] -= 0.01
        # print("1 pressed")
    elif key == glfw.KEY_9:
        direction = 2
        # print("2 pressed")
    elif key == glfw.KEY_I:
        target2 = [0,0,0]
    print(f"{target2[0]:.2f}, {target2[1]:.2f}, {target2[2]:.2f}")

if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/Roly_simulator/markers/RolyURDF2/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)


    ######
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R gripper")
    pos2 = data.xpos[body_id].copy()
    target2 = pos2.copy()
    vel2 = [0.0, 0.0, 0.0]
    controlParameter2 = [
        [100.0, 0.0, 0.02],  # x
        [100.0, 0.0, 0.02],  # y
        [100.0, 0.0, 0.02]  # z
    ]
    PIDctrl2 = PIDcontroller(controlParameter2, target2)
    ######


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
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer.cam.distance = 2.5
    viewer.cam.lookat = [0.0, 0.0, 0.8]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 180
    
    step = 0
    direction = 0

    for i in range(1000):
        mujoco.mj_step(model, data)
        if i%50 == 0:
            viewer.sync()
    pos2 = data.xpos[body_id].copy()
    vel2[0] = vel2[0]*0.6+0.4*(data.xpos[body_id][0].copy() - pos2[0] )/ 0.001
    vel2[1] = vel2[1]*0.6+0.4*(data.xpos[body_id][1].copy() - pos2[1] )/ 0.001
    vel2[2] = vel2[2]*0.6+0.4*(data.xpos[body_id][2].copy() - pos2[2] )/ 0.001
    target2 = pos2.copy()

    while viewer.is_running():
        step += 1
        if step%1000 == 0:
            data.qpos[16] = random.uniform(0.1, 0.8)
            data.qpos[17] = random.uniform(-0.6, 0.6)
            data.qpos[18] = random.uniform(1.2, 0.9)

        # if base is free joint
        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        # # if base is not free joint
        # pos = [data.qpos[i-7] for i in controlList]
        # vel = [data.qvel[i-8] for i in controlList]

        vel2[0] = (data.xpos[body_id][0].copy() - pos2[0] )/ 0.001
        vel2[1] = (data.xpos[body_id][1].copy() - pos2[1] )/ 0.001
        vel2[2] = (data.xpos[body_id][2].copy() - pos2[2] )/ 0.001
        pos2 = data.xpos[body_id].copy()
        data.xfrc_applied[body_id][:3] = PIDctrl2.getSignal(pos2, vel2, target2)
        

        if step%20 == 0:
            target[1:3] = head_camera.track(target[1:3], data, speed = 1.0)
            

        data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
        # data.ctrl[3:9] = [0]*6
        data.ctrl[3:5] = [0]*2
        data.ctrl[6:8] = [0]*2
        mujoco.mj_step(model, data)

        if step%20 == 0:
            # data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "marker1")] = target2
            viewer.sync()
            head_camera.get_img(data, rgb=True, depth=True)
            head_camera.get_target()
            # head_camera.show(rgb=True, depth=False)
            # viewer.cam.azimuth = 180+10*np.sin(step*0.0001)


    renderer.close() 
    cv2.destroyAllWindows() 
