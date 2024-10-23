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
    xml_path = 'Roly/Roly_simulator/whole body/RolyURDF2/Roly.xml'
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
    data.qpos[:] = pos[:]
    # # if base is fixed
    # data.qpos[:] = pos[7:]

    head_camera = Camera(renderer=renderer, camID=0)
    head_camera = Camera(renderer=renderer, camID=0)

    mujoco.mj_resetData(model, data)  # Reset state and time.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.lookat = [0.0, 0.0, 0.8]
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 180
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

            # target[0]   =  0.50*(np.cos(step*2*np.pi/8000)-1) # R hip yaw
            # target[0+1] =  0.50*(np.cos(step*2*np.pi/8000)-1) # R hip roll
            target[0+2] =  0.45*(np.cos(step*2*np.pi/8000)-1) # R hip pitch
            target[0+3] = -1.00*(np.cos(step*2*np.pi/8000)-1) # R knee
            target[0+4] =  0.55*(np.cos(step*2*np.pi/8000)-1) # R ankle pitch
            # target[0+5] =  0.50*(np.cos(step*2*np.pi/8000)-1) # R ankle roll

            # target[6]   =  0.50*(np.cos(step*2*np.pi/8000)-1) # L hip yaw
            # target[6+1] =  0.50*(np.cos(step*2*np.pi/8000)-1) # L hip roll
            target[6+2] =  0.45*(np.cos(step*2*np.pi/8000)-1) # L hip pitch
            target[6+3] = -1.00*(np.cos(step*2*np.pi/8000)-1) # L knee
            target[6+4] =  0.55*(np.cos(step*2*np.pi/8000)-1) # L ankle pitch
            # target[6+5] =  0.50*(np.cos(step*2*np.pi/8000)-1) # L ankle roll

            target[12]=  0.20*(np.cos(step*2*np.pi/8000)-1) # trunk
            target[13]= -0.20*(np.cos(step*2*np.pi/8000)-1) # neck
            target[14]=  0.30*(np.cos(step*2*np.pi/8000)-1) # camera

            target[15]  = -0.20*(np.cos(step*2*np.pi/8000)-1) # R shoulder
            target[15+1]=  1.00*(np.cos(step*2*np.pi/8000)-1) # R arm1
            target[15+2]= -1.00*(np.cos(step*2*np.pi/8000)-1) # R arm2
            # target[15+3]= -0.50*(np.cos(step*2*np.pi/8000)-1) # R arm3
            # target[15+4]= -0.50*(np.cos(step*2*np.pi/8000)-1) # R arm4
            # target[15+5]= -0.50*(np.cos(step*2*np.pi/8000)-1) # R palm
            target[15+6]= -0.01*(np.cos(step*2*np.pi/8000)-1) # R gripper

            target[22]  = -0.20*(np.cos(step*2*np.pi/8000)-1) # L shoulder
            target[22+1]= -1.00*(np.cos(step*2*np.pi/8000)-1) # L arm1
            target[22+2]=  1.00*(np.cos(step*2*np.pi/8000)-1) # L arm2
            # target[22+3]=  0.50*(np.cos(step*2*np.pi/8000)-1) # L arm3
            # target[22+4]= -0.50*(np.cos(step*2*np.pi/8000)-1) # L arm4
            # target[22+5]= -0.50*(np.cos(step*2*np.pi/8000)-1) # L palm
            target[22+6]= -0.01*(np.cos(step*2*np.pi/8000)-1) # L gripper
            data.ctrl[:] = PIDctrl.getSignal(pos, vel, initTarget)
            # max_value = np.max(data.ctrl)

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            if step%100 == 0:
                # Example modification of a viewer option: toggle contact points every two seconds.
                # with viewer.lock():
                #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
    
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                head_camera.get_img(data, rgb=True, depth=True)
                head_camera.get_target()
                head_camera.show(rgb=True, depth=True)
                viewer.cam.azimuth = 180+10*np.sin(step*0.0002)

            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = model.opt(step - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step) 

        renderer.close() 
        cv2.destroyAllWindows() 
