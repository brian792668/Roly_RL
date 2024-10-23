# 往外轉為正
# 往外抬為正
# 往前踢為正

import mujoco
import mujoco_viewer
import numpy as np
import math
import time
import os

from imports.Settings import *
from imports.Controller import *
from imports.Draw_joint_info import *


if __name__ == '__main__':
    # Add xml path
    current_dir = os.getcwd()
    xml_path = 'Roly/Roly_simulator/whole body/RolyURDF2/Roly.xml'
    xml_path = os.path.join(current_dir, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

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

    # simulate and render
    timestep = 0
    mujoco.mjMAXLINEPNT = 200
    while viewer.is_alive:
        timestep += 1

        # target[0]   =  0.50*(np.cos(timestep*2*np.pi/8000)-1) # R hip yaw
        # target[0+1] =  0.50*(np.cos(timestep*2*np.pi/8000)-1) # R hip roll
        target[0+2] =  0.45*(np.cos(timestep*2*np.pi/8000)-1) # R hip pitch
        target[0+3] = -1.00*(np.cos(timestep*2*np.pi/8000)-1) # R knee
        target[0+4] =  0.55*(np.cos(timestep*2*np.pi/8000)-1) # R ankle pitch
        # target[0+5] =  0.50*(np.cos(timestep*2*np.pi/8000)-1) # R ankle roll

        # target[6]   =  0.50*(np.cos(timestep*2*np.pi/8000)-1) # L hip yaw
        # target[6+1] =  0.50*(np.cos(timestep*2*np.pi/8000)-1) # L hip roll
        target[6+2] =  0.45*(np.cos(timestep*2*np.pi/8000)-1) # L hip pitch
        target[6+3] = -1.00*(np.cos(timestep*2*np.pi/8000)-1) # L knee
        target[6+4] =  0.55*(np.cos(timestep*2*np.pi/8000)-1) # L ankle pitch
        # target[6+5] =  0.50*(np.cos(timestep*2*np.pi/8000)-1) # L ankle roll

        target[12]=  0.20*(np.cos(timestep*2*np.pi/8000)-1) # trunk
        target[13]= -0.20*(np.cos(timestep*2*np.pi/8000)-1) # neck
        target[14]=  0.30*(np.cos(timestep*2*np.pi/8000)-1) # camera

        target[15]  = -0.20*(np.cos(timestep*2*np.pi/8000)-1) # R shoulder
        target[15+1]=  1.00*(np.cos(timestep*2*np.pi/8000)-1) # R arm1
        target[15+2]= -1.00*(np.cos(timestep*2*np.pi/8000)-1) # R arm2
        # target[15+3]= -0.50*(np.cos(timestep*2*np.pi/8000)-1) # R arm3
        # target[15+4]= -0.50*(np.cos(timestep*2*np.pi/8000)-1) # R arm4
        # target[15+5]= -0.50*(np.cos(timestep*2*np.pi/8000)-1) # R palm
        target[15+6]= -0.01*(np.cos(timestep*2*np.pi/8000)-1) # R gripper

        target[22]  = -0.20*(np.cos(timestep*2*np.pi/8000)-1) # L shoulder
        target[22+1]= -1.00*(np.cos(timestep*2*np.pi/8000)-1) # L arm1
        target[22+2]=  1.00*(np.cos(timestep*2*np.pi/8000)-1) # L arm2
        # target[22+3]=  0.50*(np.cos(timestep*2*np.pi/8000)-1) # L arm3
        # target[22+4]= -0.50*(np.cos(timestep*2*np.pi/8000)-1) # L arm4
        # target[22+5]= -0.50*(np.cos(timestep*2*np.pi/8000)-1) # L palm
        target[22+6]= -0.01*(np.cos(timestep*2*np.pi/8000)-1) # L gripper
        

        pos = [data.qpos[i] for i in controlList]
        vel = [data.qvel[i-1] for i in controlList]
        data.ctrl[:] = PIDctrl.getSignal(pos, vel, target)
        mujoco.mj_step(model, data)
        
        # render every 25ms
        if int(data.time*1000)%25 == 0:  
            # viewer.add_marker(
            #       label=f"Clock: {data.time:.1f} sec",
            #       pos=[-0.2, 0, 1.5],
            #       type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #       size=0.0,
            #       rgba=[0, 1, 0, 0.99],
            #       emission=0.99)
            draw_fig_to_viewer(viewer, target, data, sensor)
            viewer.render()
        if not viewer.is_alive:
            break

    # close
    viewer.close()
