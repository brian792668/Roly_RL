from mujoco import mj_name2id, mjtObj
import numpy as np
# from Controller import *

class DrawJointInfo(object):
    def __init__(self, viewer):
        self.viewer = viewer

    def addLine(self, draw_dic:dict):
        for line_name, fig_idx in zip(draw_dic.keys(), draw_dic.values()):
            self.viewer.add_line_to_fig(line_name=line_name, fig_idx=fig_idx)

    def setParameter(self, fig_idx:int, fig_title:str, r=0.2, g=0, b=0, alpha=0.2):
        fig = self.viewer.figs[fig_idx]
        fig.title = fig_title
        fig.flg_legend = True
        fig.xlabel = "Timesteps"
        fig.figurergba[0] = r
        fig.figurergba[3] = alpha
        fig.gridsize[0] = 5 # x axis
        fig.gridsize[1] = 5 # y axis

    def drawInfo(self):
        pass
    def __repr__(self) -> str:
        return f"Object used to create Info figure"


class DrawPosInfo(DrawJointInfo):
    def __init__(self, viewer):
        super().__init__(viewer)

    def drawInfo(self, draw_dic:dict, model:object, data:object):
        for line_name, fig_idx in zip(draw_dic.keys(), draw_dic.values()):
            self.viewer.add_data_to_line(line_name=line_name, fig_idx=fig_idx,
                    line_data=data.qpos[mj_name2id(model, mjtObj.mjOBJ_JOINT, line_name)])
    
    def __repr__(self) -> str:
        return f"Object used to create Pos Info figure"

class DrawVelInfo(DrawJointInfo):
    def __init__(self, viewer):
        pass
    def drawInfo(self):
        pass
    def __repr__(self) -> str:
        return f"Object used to create Vel Info figure"
    
def init_viewer(viewer):
    viewer.cam.distance = 2.5
    viewer.cam.lookat = [0.0, 0.0, 0.8]
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 160
    viewer._render_every_frame = False
    viewer._convex_hull_rendering = True
    viewer._run_speed = 0.125

    viewer.figs[0].title = "postion err"
    viewer.add_line_to_fig(fig_idx=0, line_name="R hip p")
    viewer.add_line_to_fig(fig_idx=0, line_name="R knee")
    viewer.add_line_to_fig(fig_idx=0, line_name="R ankle p")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R ankle r")
    # viewer.add_line_to_fig(fig_idx=0, line_name="trunk")
    # viewer.add_line_to_fig(fig_idx=0, line_name="camera")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R shoulder")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R arm1")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R arm2")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R arm3")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R arm4")
    # viewer.add_line_to_fig(fig_idx=0, line_name="R gripper")
    # viewer.add_line_to_fig(fig_idx=0, line_name="L gripper")

    viewer.figs[1].title = "torque"
    viewer.add_line_to_fig(fig_idx=1, line_name="R_hip_pitch")
    viewer.add_line_to_fig(fig_idx=1, line_name="R_knee")
    viewer.add_line_to_fig(fig_idx=1, line_name="R_ankle_pitch")

    viewer.figs[2].title = "sensor"
    viewer.add_line_to_fig(fig_idx=2, line_name="R_toe")
    viewer.add_line_to_fig(fig_idx=2, line_name="R_heel")
    viewer.add_line_to_fig(fig_idx=2, line_name="L_toe")
    viewer.add_line_to_fig(fig_idx=2, line_name="L_heel")

def draw_fig_to_viewer(viewer, target, data, sensor):
    for i in range(4):
        sensor[i] = 0.8*sensor[i] + 0.2*sum(data.sensordata[4*i:4*i+4]) 
    viewer.add_data_to_line(line_name="R hip p",    fig_idx=0, line_data=np.pi/180*(target[2] - data.qpos[9]))
    viewer.add_data_to_line(line_name="R knee",     fig_idx=0, line_data=np.pi/180*(target[3] - data.qpos[10]))
    viewer.add_data_to_line(line_name="R ankle p",  fig_idx=0, line_data=np.pi/180*(target[4] - data.qpos[11]))
    # viewer.add_data_to_line(line_name="R ankle r",  fig_idx=0, line_data=np.pi/180*(target[5] - data.qpos[12]))
    # viewer.add_data_to_line(line_name="trunk",      fig_idx=0, line_data=np.pi/180*(target[12] - data.qpos[23]))
    # viewer.add_data_to_line(line_name="camera",     fig_idx=0, line_data=np.pi/180*(target[14] - data.qpos[25]))
    # viewer.add_data_to_line(line_name="R shoulder", fig_idx=0, line_data=np.pi/180*(target[15] - data.qpos[26]))
    # viewer.add_data_to_line(line_name="R arm1",     fig_idx=0, line_data=np.pi/180*(target[16] - data.qpos[27]))
    # viewer.add_data_to_line(line_name="R arm2",     fig_idx=0, line_data=np.pi/180*(target[17] - data.qpos[28]))
    # viewer.add_data_to_line(line_name="R arm3",     fig_idx=0, line_data=np.pi/180*(target[18] - data.qpos[29]))
    # viewer.add_data_to_line(line_name="R arm4",     fig_idx=0, line_data=np.pi/180*(target[19] - data.qpos[30]))
    # viewer.add_data_to_line(line_name="R gripper",  fig_idx=0, line_data=100*(target[21]-data.qpos[38]))
    # viewer.add_data_to_line(line_name="L gripper",  fig_idx=0, line_data=100*(target[28]-data.qpos[51]))
    
    viewer.add_data_to_line(line_name="R_hip_pitch",    fig_idx=1, line_data=data.ctrl[2])
    viewer.add_data_to_line(line_name="R_knee",         fig_idx=1, line_data=data.ctrl[3])
    viewer.add_data_to_line(line_name="R_ankle_pitch",  fig_idx=1, line_data=data.ctrl[4])
    
    viewer.add_data_to_line(line_name="R_toe",          fig_idx=2, line_data=sensor[0])
    viewer.add_data_to_line(line_name="R_heel",         fig_idx=2, line_data=sensor[1])
    viewer.add_data_to_line(line_name="L_toe",          fig_idx=2, line_data=sensor[2])
    viewer.add_data_to_line(line_name="L_heel",         fig_idx=2, line_data=sensor[3])
