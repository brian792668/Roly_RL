import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info

class Roly_motor(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [1, 2, 10, 11, 12, 13, 14, 15], 
                        "model": [X_series_info] * 8 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [180.0] * 8
        self.joints_axis = [1,   -1,   1,   1,   1,   -1,   -1,   1]

        self.joints = [0] * 8
        self.joints_increment = [0] * 8
        self.initial_pos = [-20, -45, 2, -20, 0, 40, 92, 0]
        self.limit_high = [ 1.57, 0.12, 1.57, 1.90]
        self.limit_low  = [-1.05,-1.57,-1.57, 0.00]

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[200, 200, 50, 50, 50, 50, 50, 50])
        time.sleep(0.1)

    def to_pose(self, pose):
        current_angles = self.readAllMotorPosition()
        current_angles = [(resolution2degree(current_angles[i])-self.joints_bias[i])*self.joints_axis[i] for i in range(len(current_angles))]
        final_angles = current_angles
        if pose == "initial":   final_angles = self.initial_pos.copy()
        if pose == "shut down": final_angles = [0]*8

        t=0
        while t <= 1.0:
            ctrlpos, t = self.smooth_transition(t, initial_angles=current_angles, final_angles=final_angles, speed=0.005)
            self.writeAllMotorPosition(self.toRolyctrl(ctrlpos.copy()))
            time.sleep(0.001)

        current_angles = self.readAllMotorPosition()
        current_angles = [(resolution2degree(current_angles[i])-self.joints_bias[i])*self.joints_axis[i] for i in range(len(current_angles))]
        self.joints = current_angles.copy()
        time.sleep(0.1)


    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]
    
    def smooth_transition(self, t, initial_angles, final_angles, speed=0.001):

        np_initial_angles = np.array(initial_angles)
        np_final_angles = np.array(final_angles)

        progress = min(t, 1)
        progress = ((1 - np.cos(np.pi * progress)) / 2)
        current_angles = np_initial_angles*(1-progress) + np_final_angles*progress

        t_next = t + speed
        return current_angles.tolist(), t_next


# def init_motor():
#     X_series_info = X_Motor_Info()
#     # P_series_info = P_Motor_Info()

#     DEVICENAME = "/dev/ttyUSB0"
#     DXL_MODELS = {
#         "id": [1, 2, 10, 11, 12, 13, 14, 15], 
#         "model": [X_series_info] * 8
#     }

#     motor = DXL_Motor(DEVICENAME, DXL_MODELS, BAUDRATE=115200)
#     motor.changeAllMotorOperatingMode(OP_MODE=3)
#     motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[100]*len(motor.pos_ctrl))
#     time.sleep(0.1)
#     return motor

# def initial_pos(motor):
    
#     motor.changeAllMotorOperatingMode(OP_MODE=3)
#     motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[50]*len(motor.pos_ctrl))
#     time.sleep(0.1)
#     initial_angles = motor.readAllMotorPosition()
#     final_angles=[-20, -45, 2, -20, 0, 40, 92, 0]

#     # initial
#     t=0
#     while t <= 1.0:
#         joints, t = smooth_transition(t, initial_angles=initial_angles, final_angles=final_angles.copy(), speed=0.005)
#         motor.writeAllMotorPosition(motor.toRolyctrl(joints.copy()))
#         time.sleep(0.001)
#     motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[200, 200, 50, 50, 50, 50, 50, 50])
#     time.sleep(0.1)


#     # motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[10]*len(motor.pos_ctrl))
#     # for i in range(1000):
#     #     motor.writeAllMotorPosition(motor.toRolyctrl(final_angles.copy()))
#     #     time.sleep(0.001)
#     #     print(i)
    
#     return final_angles.copy()
       

# def smooth_transition(t, initial_angles, final_angles, speed=0.001):

#     np_initial_angles = np.array(initial_angles)
#     np_final_angles = np.array(final_angles)

#     progress = min(t, 1)
#     progress = ((1 - np.cos(np.pi * progress)) / 2)
#     current_angles = np_initial_angles*(1-progress) + np_final_angles*progress

#     t_next = t + speed
#     return current_angles.tolist(), t_next