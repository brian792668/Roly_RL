import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info

def init_motor():
    X_series_info = X_Motor_Info()
    # P_series_info = P_Motor_Info()

    DEVICENAME = "/dev/ttyUSB0"
    DXL_MODELS = {
        "id": [1, 2, 10, 11, 12, 13, 14, 15], 
        "model": [X_series_info] * 8
    }

    motor = DXL_Motor(DEVICENAME, DXL_MODELS, BAUDRATE=115200)
    motor.changeAllMotorOperatingMode(OP_MODE=3)
    motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[100]*len(motor.pos_ctrl))
    time.sleep(0.1)
    # initial_angles = motor.readAllMotorPosition()
    # final_angles=[-20, -45, 10, 4, 0, 5, 78, 0]

    # # initial
    # t=0
    # while t <= 1.0:
    #     joints, t = smooth_transition(t, initial_angles=initial_angles.copy(), final_angles=final_angles.copy(), speed=0.005)
    #     motor.writeAllMotorPosition(motor.toRolyctrl(joints.copy()))
    #     time.sleep(0.01)
    # motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[200, 200, 50, 50, 50, 50, 50, 50])
    # time.sleep(0.1)
    
    # return motor, final_angles.copy()
    return motor

def initial_pos(motor):
    
    motor.changeAllMotorOperatingMode(OP_MODE=3)
    motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[100]*len(motor.pos_ctrl))
    time.sleep(0.1)
    initial_angles = motor.readAllMotorPosition()
    final_angles=[-20, -45, 2, 5.5, 0, 0, 92, 0]

    # initial
    t=0
    while t <= 1.0:
        joints, t = smooth_transition(t, initial_angles=initial_angles.copy(), final_angles=final_angles.copy(), speed=0.005)
        motor.writeAllMotorPosition(motor.toRolyctrl(joints.copy()))
        time.sleep(0.01)
    motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[200, 200, 50, 50, 50, 50, 50, 50])
    time.sleep(0.1)
    
    return final_angles.copy()
       

def smooth_transition(t, initial_angles, final_angles, speed=0.001):

    np_initial_angles = np.array(initial_angles)
    np_final_angles = np.array(final_angles)

    progress = min(t, 1)
    progress = ((1 - np.cos(np.pi * progress)) / 2)
    current_angles = np_initial_angles*(1-progress) + np_final_angles*progress

    t_next = t + speed
    return current_angles.tolist(), t_next